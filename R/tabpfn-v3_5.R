# Defaults of Python `TabPFNV3p5Config` (tabpfn 9.1.0).
tabpfn_v3_5_config_defaults <- list(
  max_num_classes = -1L,
  num_buckets = -1L,
  name = "TabPFN-v3.5",
  embed_dim = 128L,
  dist_embed_num_blocks = 3L,
  dist_embed_num_heads = 8L,
  dist_embed_num_inducing_points = 128L,
  feature_group_size = 3L,
  feat_agg_num_blocks = 3L,
  feat_agg_num_heads = 8L,
  feat_agg_num_cls_tokens = 8L,
  feat_agg_rope_base = 100000,
  nlayers = 24L,
  icl_num_heads = 16L,
  icl_num_kv_heads = NULL,
  icl_num_kv_heads_test = 1L,
  decoder_head_dim = 64L,
  decoder_num_heads = 6L,
  decoder_use_softmax_scaling = TRUE,
  ff_factor = 2L,
  softmax_scaling_mlp_hidden_dim = 64L,
  fourier_encoding_num_frequencies = 32L,
  cell_ecdf_num_frequencies = 4L,
  cell_ecdf_num_buckets = 8192L,
  cell_embed_row_chunk_size = 2048L,
  inference_chunk_cells = 1572864L,
  max_batched_estimator_rows = 32768L,
  max_batched_estimator_cells = 768000000L,
  inference_col_chunk_size = 4L
)

# Keys in the v3.5 checkpoints that the Python architecture ignores because
# the behavior is hard-coded.
tabpfn_v3_5_config_fixed <- list(
  feat_agg_use_softmax_scaling = FALSE,
  icl_bf16 = FALSE,
  layernorm_elementwise_affine = TRUE,
  use_fourier_cell_embedding = TRUE,
  use_mlp_heads = TRUE,
  use_nan_indicators = TRUE,
  use_qk_norm = TRUE,
  use_rope = TRUE,
  y_encoder_layernorm = TRUE
)

tabpfn_v3_5_parse_config <- function(config, call = caller_env()) {
  tabpfn_parse_config(
    config,
    defaults = tabpfn_v3_5_config_defaults,
    fixed = tabpfn_v3_5_config_fixed,
    free = "inference_row_chunk_size",
    architecture = "tabpfn_v3_5",
    call = call
  )
}

# ------------------------------------------------------------------------------

# Modules specific to the TabPFN v3.5 architecture (Python
# `tabpfn/architectures/tabpfn_v3_5.py`): the Fourier cell embedder and the
# multitask output heads. Modules shared with v3 are in `modules.R`.

# ---- Cell embedding --------------------------------------------------------

# Fourier features of the grouped cell values: (..., G) -> (..., E).
tabpfn_v3_5_fourier_embedder <- torch::nn_module(
  "tabpfn_v3_5_fourier_embedder",
  initialize = function(group_size, embed_dim, num_freq) {
    self$frequencies <- torch::nn_parameter(torch::torch_zeros(
      group_size,
      num_freq
    ))
    self$in_linear <- torch::nn_linear(num_freq * 2, embed_dim, bias = FALSE)
  },
  forward = function(x) {
    dt <- x$dtype
    compute <- tabpfn_at_least_fp32(dt)
    proj <- x$unsqueeze(-1)$to(dtype = compute) *
      self$frequencies$to(dtype = compute)
    feats <- torch::torch_cat(list(proj$sin(), proj$cos()), dim = -1)$to(
      dtype = dt
    )
    self$in_linear(feats$sum(dim = -2))
  }
)

tabpfn_at_least_fp32 <- function(dtype) {
  if (dtype == torch::torch_float64()) {
    dtype
  } else {
    torch::torch_float32()
  }
}

# Sin/cos features of ECDF values in [0, 1]: (...) -> (..., 2K).
tabpfn_v3_5_ecdf_fourier <- function(u, num_frequencies) {
  k <- torch::torch_arange(
    1,
    num_frequencies,
    dtype = u$dtype,
    device = u$device
  )
  phase <- u$unsqueeze(-1) * (pi * k)
  torch::torch_cat(list(phase$sin(), phase$cos()), dim = -1)
}

# Grouped cell tensor (B, R, C, 3G) -> (B, R, C, E). The input holds the
# standard-scaled values, the NaN/Inf indicators, and the raw ECDF ranks,
# G channels each.
tabpfn_v3_5_cell_embedder <- torch::nn_module(
  "tabpfn_v3_5_cell_embedder",
  initialize = function(
    group_size,
    embed_dim,
    num_freq,
    ecdf_num_frequencies,
    row_chunk_size = NULL
  ) {
    self$group_size <- group_size
    self$ecdf_num_frequencies <- ecdf_num_frequencies
    self$row_chunk_size <- row_chunk_size
    self$fourier <- tabpfn_v3_5_fourier_embedder(
      group_size,
      embed_dim,
      num_freq
    )
    metadata_width <- group_size * 2 + group_size * 2 * ecdf_num_frequencies
    self$metadata_linear <- torch::nn_linear(
      metadata_width,
      embed_dim,
      bias = FALSE
    )
    self$layernorm <- torch::nn_layer_norm(embed_dim)
  },
  forward = function(x) {
    rows <- x$size(2)
    chunk <- self$row_chunk_size
    if (is.null(chunk) || rows <= chunk) {
      return(self$embed(x))
    }
    starts <- seq(1, rows, by = chunk)
    parts <- lapply(starts, function(s) {
      out <- self$embed(x$narrow(2, s, min(chunk, rows - s + 1)))
      tabpfn_release()
      out
    })
    torch::torch_cat(parts, dim = 2)
  },
  embed = function(x) {
    dt <- x$dtype
    g <- self$group_size
    width <- x$size(-1)
    fourier_out <- self$fourier(x[.., 1:g])
    ranks <- x[.., (width - g + 1):width]
    metadata <- torch::torch_cat(
      list(
        x[.., 1:(width - g)],
        tabpfn_v3_5_ecdf_fourier(ranks, self$ecdf_num_frequencies)$flatten(
          start_dim = -2
        )
      ),
      dim = -1
    )
    compute <- tabpfn_at_least_fp32(dt)
    metadata_out <- torch::nnf_linear(
      metadata$to(dtype = compute),
      self$metadata_linear$weight$to(dtype = compute)
    )
    self$layernorm((fourier_out$to(dtype = compute) + metadata_out)$to(
      dtype = dt
    ))
  }
)

# ---- Output heads --------------------------------------------------------------

tabpfn_v3_5_prehead_mlp <- torch::nn_module(
  "tabpfn_v3_5_prehead_mlp",
  initialize = function(emsize, dim_feedforward) {
    self$norm <- tabpfn_rms_norm(emsize)
    self$mlp <- tabpfn_mlp(emsize, dim_feedforward)
  },
  forward = function(x) {
    x + self$mlp(self$norm(x))
  }
)

# Classification (many-class decoder) and regression (bar logits) heads,
# each behind its own residual pre-head MLP.
tabpfn_v3_5_multitask_heads <- torch::nn_module(
  "tabpfn_v3_5_multitask_heads",
  initialize = function(
    input_size,
    max_num_classes,
    num_buckets,
    decoder_head_dim,
    decoder_num_heads,
    decoder_softmax_scaling_layer,
    mlp_dim_feedforward
  ) {
    self$many_class_decoder <- tabpfn_many_class_decoder(
      max_num_classes,
      input_size,
      decoder_head_dim,
      decoder_num_heads,
      softmax_scaling_layer = decoder_softmax_scaling_layer
    )
    self$output_projection <- torch::nn_linear(input_size, num_buckets)
    self$mlp_classification <- tabpfn_v3_5_prehead_mlp(
      input_size,
      mlp_dim_feedforward
    )
    self$mlp_regression <- tabpfn_v3_5_prehead_mlp(
      input_size,
      mlp_dim_feedforward
    )
    self$regression_borders <- torch::nn_buffer(torch::torch_zeros(
      num_buckets + 1
    ))
  },
  project_decoder_keys = function(train_emb) {
    self$many_class_decoder$project_keys(self$mlp_classification(train_emb))
  },
  # test_emb: (B, M, D). Returns (M, B, max_num_classes) or (M, B, num_buckets).
  forward = function(
    train_keys,
    test_emb,
    y_train,
    task_type,
    num_present_classes = NULL
  ) {
    if (task_type == "regression") {
      test_emb <- self$mlp_regression(test_emb)
      return(self$output_projection(test_emb$transpose(1, 2)))
    }
    test_emb <- self$mlp_classification(test_emb)
    self$many_class_decoder(train_keys, test_emb, y_train, num_present_classes)
  }
)

# ------------------------------------------------------------------------------

# The TabPFN v3.5 model (Python `TabPFNV3p5`). One checkpoint serves both
# classification ("multiclass") and regression.
#
# Pipeline, for x of shape (Ri, B, C) (rows, ensemble members, columns) and
# y of shape (N) or (N, B) holding the N train targets:
#   0. NaN/Inf indicators, mean imputation, standard scaling, and per-cell
#      ECDF ranks against the train rows.
#   1. Feature grouping (circular shifts) and cell embedding, plus the
#      target embedding on train rows.
#   2. Per-column induced self-attention, then per-row column aggregation
#      into CLS tokens.
#   3. ICL transformer over rows (keys/values from train rows only) and the
#      task head.
#
# Not ported: the KV cache, torch.compile, bf16 ICL, and OOM-driven chunking.

tabpfn_v3_5_model <- torch::nn_module(
  "tabpfn_v3_5_model",
  initialize = function(config) {
    self$config <- config
    e <- config$embed_dim
    self$icl_emsize <- e * config$feat_agg_num_cls_tokens
    self$max_num_classes <- config$max_num_classes
    self$feature_group_size <- config$feature_group_size
    self$ecdf_num_buckets <- config$cell_ecdf_num_buckets

    self$x_embed <- tabpfn_v3_5_cell_embedder(
      config$feature_group_size,
      e,
      config$fourier_encoding_num_frequencies,
      ecdf_num_frequencies = config$cell_ecdf_num_frequencies,
      row_chunk_size = config$cell_embed_row_chunk_size
    )
    self$col_y_encoder <- torch::nn_module_dict(list(
      multiclass = tabpfn_class_embedding(config$max_num_classes, e),
      regression = torch::nn_linear(1, e)
    ))
    self$col_y_layernorm <- torch::nn_layer_norm(e)

    scaling <- function(num_heads, head_dim) {
      tabpfn_softmax_scaling_mlp(
        num_heads,
        head_dim,
        config$softmax_scaling_mlp_hidden_dim
      )
    }
    self$feature_distribution_embedder <- tabpfn_feature_distribution_embedder(
      e,
      config$dist_embed_num_heads,
      config$dist_embed_num_inducing_points,
      e * config$ff_factor,
      config$dist_embed_num_blocks,
      softmax_scaling_factory = function() {
        scaling(config$dist_embed_num_heads, e %/% config$dist_embed_num_heads)
      }
    )
    self$column_aggregator <- tabpfn_column_aggregator(
      e,
      config$feat_agg_num_heads,
      config$feat_agg_num_blocks,
      e * config$ff_factor,
      config$feat_agg_num_cls_tokens,
      config$feat_agg_rope_base
    )
    d <- self$icl_emsize
    self$icl_y_encoder <- torch::nn_module_dict(list(
      multiclass = tabpfn_class_embedding(config$max_num_classes, d),
      regression = torch::nn_linear(1, d)
    ))
    self$icl_y_layernorm <- torch::nn_layer_norm(d)
    self$icl_blocks <- torch::nn_module_list(lapply(
      seq_len(config$nlayers),
      function(i) {
        tabpfn_icl_block(
          d,
          config$icl_num_heads,
          d * config$ff_factor,
          softmax_scaling_layer = scaling(
            config$icl_num_heads,
            d %/% config$icl_num_heads
          ),
          num_kv_heads = config$icl_num_kv_heads,
          num_kv_heads_test = config$icl_num_kv_heads_test
        )
      }
    ))
    self$output_norm <- tabpfn_rms_norm(d)
    decoder_scaling <- NULL
    if (isTRUE(config$decoder_use_softmax_scaling)) {
      decoder_scaling <- scaling(
        config$decoder_num_heads,
        config$decoder_head_dim
      )
    }
    self$heads <- tabpfn_v3_5_multitask_heads(
      input_size = d,
      max_num_classes = config$max_num_classes,
      num_buckets = config$num_buckets,
      decoder_head_dim = config$decoder_head_dim,
      decoder_num_heads = config$decoder_num_heads,
      decoder_softmax_scaling_layer = decoder_scaling,
      mlp_dim_feedforward = d * config$ff_factor
    )
  },

  borders = function() {
    self$heads$regression_borders
  },

  forward = function(x, y, task_type = c("multiclass", "regression")) {
    if (y$dim() == 3 && y$size(3) == 1) {
      y <- y$squeeze(3)
    }
    num_train <- y$size(1)
    b <- x$size(2)
    num_present_classes <- NULL
    if (task_type == "multiclass") {
      num_present_classes <- tabpfn_num_present_classes(y, self$max_num_classes)
    }

    x <- self$stages_0_to_2(x, y, num_train, task_type)

    # Stage 3: ICL.
    x <- x$flatten(start_dim = -2)
    y_icl <- tabpfn_prepare_y(y, num_train, task_type)
    y_icl_emb <- self$embed_y(
      self$icl_y_encoder,
      self$icl_y_layernorm,
      y_icl,
      task_type
    )
    x[, 1:num_train] <- x[, 1:num_train] + y_icl_emb
    for (i in seq_along(self$icl_blocks)) {
      x <- self$icl_blocks[[i]](x, num_train)
      tabpfn_release()
    }
    x <- self$output_norm(x)

    r <- x$size(2)
    test_emb <- x[, (num_train + 1):r]
    train_emb <- x[, 1:num_train]
    train_keys <- NULL
    if (task_type == "multiclass") {
      train_keys <- self$heads$project_decoder_keys(train_emb)
    }
    if (y$dim() == 2) {
      y_bn <- y$transpose(1, 2)
    } else {
      y_bn <- y$unsqueeze(1)
    }
    out <- self$heads(
      train_keys,
      test_emb,
      y_bn[, 1:num_train],
      task_type,
      num_present_classes
    )
    out$nan_to_num(nan = 0)
  },

  # Stages 0-2: (Ri, B, C) -> (B, Ri, num_cls_tokens, E).
  stages_0_to_2 = function(x, y, num_train, task_type) {
    prep <- self$preprocess(x, num_train)
    y_col <- tabpfn_prepare_y(y, num_train, task_type)
    y_col_emb <- self$embed_y(
      self$col_y_encoder,
      self$col_y_layernorm,
      y_col,
      task_type
    )
    grouped <- tabpfn_group_features(
      list(prep$x, prep$nan_ind, prep$ecdf),
      self$feature_group_size
    )

    tabpfn_stages_1_2(
      self,
      grouped,
      y_col_emb,
      num_train,
      tabpfn_chunk_cells(self$config)
    )
  },

  # NaN/Inf indicators, imputation, scaling, and ECDF ranks; all (B, Ri, C).
  preprocess = function(x, num_train) {
    nan_ind <- tabpfn_nan_indicator(x)$transpose(1, 2)
    imputed <- tabpfn_impute_mean(x, num_train)
    stats <- tabpfn_scaler_fit(imputed$x[1:num_train])
    x <- torch::torch_where(
      imputed$is_finite,
      imputed$x,
      stats$mean$unsqueeze(1)$expand_as(imputed$x)
    )
    x_imputed <- x$transpose(1, 2)
    context <- tabpfn_v3_5_ecdf_context(
      x_imputed,
      num_train,
      self$ecdf_num_buckets
    )
    ecdf <- tabpfn_v3_5_in_context_ecdf(x_imputed, context)
    x <- tabpfn_scaler_transform(x, stats)$transpose(1, 2)
    list(
      x = x,
      nan_ind = nan_ind,
      ecdf = ecdf,
      stats = stats,
      context = context
    )
  },

  # Target embedding: (B, N) -> (B, N, width).
  embed_y = function(encoders, layernorm, y_bn, task_type) {
    if (task_type == "multiclass") {
      emb <- encoders$multiclass(y_bn)
    } else {
      emb <- encoders$regression(y_bn$reshape(c(-1, 1)))
      emb <- emb$reshape(c(y_bn$shape, emb$size(-1)))
    }
    layernorm(emb)
  }
)

# ---- Per-cell ECDF -----------------------------------------------------------

# Summarise the train rows of each (member, column) into K = min(num_buckets,
# N) bucket edges: (3, B, C, K) holding the edge values and the counts of
# train values below and at most each edge. Python `_build_ecdf_context`.
tabpfn_v3_5_ecdf_context <- function(x, num_train, num_buckets) {
  f32 <- torch::torch_float32()
  sorted <- x[, 1:num_train]$transpose(2, 3)$to(dtype = f32)$contiguous()$sort(
    dim = -1
  )[[1]]
  n <- sorted$size(-1)
  k <- min(num_buckets, n)

  # 0-based index of each sorted value among the column's distinct values.
  is_new <- torch::torch_ones_like(sorted, dtype = torch::torch_int32())
  if (n > 1) {
    is_new[.., 2:n] <- (sorted[.., 2:n] != sorted[.., 1:(n - 1)])$to(
      dtype = torch::torch_int32()
    )
  }
  distinct_idx <- is_new$cumsum(dim = -1)$sub(1L)$to(
    dtype = torch::torch_int32()
  )
  num_distinct <- distinct_idx[.., n:n] + 1L

  steps <- torch::torch_arange(0, k - 1, dtype = f32, device = x$device)
  lead <- c(sorted$size(1), sorted$size(2))
  if (k > 1) {
    distinct_targets <- steps * ((num_distinct - 1L)$to(dtype = f32) / (k - 1))
    rows_k <- (steps * ((n - 1) / (k - 1)))$round()$to(
      dtype = torch::torch_long()
    )
    targets <- torch::torch_where(
      num_distinct <= k,
      distinct_targets$round()$to(dtype = torch::torch_int32()),
      distinct_idx$gather(
        -1,
        (rows_k$expand(c(lead, k)) + 1L)$to(dtype = torch::torch_long())
      )
    )$contiguous()
  } else {
    targets <- steps$round()$to(dtype = torch::torch_int32())$expand(c(
      lead,
      k
    ))$contiguous()
  }

  below <- torch::torch_searchsorted(distinct_idx, targets, side = "left")
  at_most <- torch::torch_searchsorted(distinct_idx, targets, side = "right")
  edges <- sorted$gather(-1, (below + 1L)$to(dtype = torch::torch_long()))
  torch::torch_stack(list(
    edges,
    below$to(dtype = f32),
    at_most$to(dtype = f32)
  ))
}

# Midrank of each value against the bucket edges, as a train-row count;
# values outside the train range get the midrank of the nearest extreme.
# values: (B, C, R). Python `_ecdf_midrank_counts`.
tabpfn_v3_5_ecdf_counts <- function(values, context) {
  edges <- context[1]
  below <- context[2]
  at_most <- context[3]
  k <- edges$size(-1)
  left <- torch::torch_searchsorted(edges, values, side = "left")
  right <- torch::torch_searchsorted(edges, values, side = "right")

  long <- torch::torch_long()
  lo_idx <- ((left - 1L)$clamp(min = 0L) + 1L)$to(dtype = long)
  hi_idx <- (left$clamp(max = k - 1L) + 1L)$to(dtype = long)
  edge_lo <- edges$gather(-1, lo_idx)
  edge_hi <- edges$gather(-1, hi_idx)
  at_most_lo <- at_most$gather(-1, lo_idx)
  below_hi <- below$gather(-1, hi_idx)

  zeros <- torch::torch_zeros_like(values)
  width <- edge_hi - edge_lo
  inside <- width > 0
  weight <- torch::torch_where(
    inside,
    (values - edge_lo) /
      torch::torch_where(inside, width, torch::torch_ones_like(width)),
    zeros
  )
  counts <- at_most_lo + weight * (below_hi - at_most_lo)
  is_edge <- right > left
  exact <- 0.5 * (below_hi + at_most$gather(-1, hi_idx))
  counts <- torch::torch_where(is_edge, exact, counts)
  counts <- torch::torch_where((left == 0) & !is_edge, zeros, counts)

  mid_lo <- 0.5 * (below[.., 1:1] + at_most[.., 1:1])
  mid_hi <- 0.5 * (below[.., k:k] + at_most[.., k:k])
  torch::torch_minimum(torch::torch_maximum(counts, mid_lo), mid_hi)
}

# ECDF value in [0, 1] of every cell: (B, Ri, C) -> (B, Ri, C).
tabpfn_v3_5_in_context_ecdf <- function(x, context) {
  at_most <- context[3]
  n <- at_most[.., at_most$size(-1):at_most$size(-1)]
  values <- x$transpose(2, 3)$to(dtype = torch::torch_float32())$contiguous()
  counts <- tabpfn_v3_5_ecdf_counts(values, context)
  (counts / n)$transpose(2, 3)$to(dtype = x$dtype)
}
