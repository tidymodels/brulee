# Defaults of Python `TabPFNV3Config` (tabpfn 9.1.0).
tabpfn_v3_config_defaults <- list(
  max_num_classes = -1L,
  num_buckets = -1L,
  name = "TabPFN-v3",
  embed_dim = 128L,
  dist_embed_num_blocks = 3L,
  dist_embed_num_heads = 8L,
  dist_embed_num_inducing_points = 128L,
  feature_group_size = 3L,
  feat_agg_num_blocks = 3L,
  feat_agg_num_heads = 8L,
  feat_agg_num_cls_tokens = 4L,
  feat_agg_rope_base = 100000,
  use_rope = TRUE,
  nlayers = 24L,
  icl_num_heads = 8L,
  icl_num_kv_heads = NULL,
  icl_num_kv_heads_test = NULL,
  decoder_head_dim = 64L,
  decoder_num_heads = 6L,
  decoder_use_softmax_scaling = FALSE,
  ff_factor = 2L,
  dropout = 0,
  softmax_scaling_mlp_hidden_dim = 64L,
  layernorm_elementwise_affine = TRUE,
  use_nan_indicators = TRUE,
  inference_chunk_cells = 1572864L,
  max_batched_estimator_rows = 32768L,
  max_batched_estimator_cells = 768000000L,
  inference_col_chunk_size = 4L
)

tabpfn_v3_parse_config <- function(config, call = caller_env()) {
  parsed <- tabpfn_parse_config(
    config,
    defaults = tabpfn_v3_config_defaults,
    free = "inference_row_chunk_size",
    architecture = "tabpfn_v3",
    call = call
  )
  # Settings the Python architecture supports that no released checkpoint
  # uses, and that the R port therefore doesn't implement.
  unsupported <- c(
    layernorm_elementwise_affine = !isTRUE(parsed$layernorm_elementwise_affine),
    use_nan_indicators = !isTRUE(parsed$use_nan_indicators)
  )
  if (any(unsupported)) {
    cli::cli_abort(
      "brulee doesn't support {.field {names(unsupported)[unsupported]}} =
       {.val FALSE} for {.val tabpfn_v3}.",
      call = call
    )
  }
  parsed
}

# ------------------------------------------------------------------------------

# The TabPFN v3 model (Python `TabPFNV3` in `tabpfn_v3.py`). Unlike v3.5,
# each checkpoint serves one task: classification when the config has
# classes, regression otherwise.
#
# Pipeline, for x of shape (Ri, B, C) and train targets y of shape (N) or
# (N, B):
#   0. NaN/Inf indicators, mean imputation, and standard scaling.
#   1. Feature grouping and a linear cell embedding, plus the target
#      embedding on train rows.
#   2. Per-column induced self-attention, then per-row column aggregation.
#   3. ICL transformer and the task head (many-class decoder, or an MLP
#      giving bar-distribution logits).
#
# Not ported: the KV cache, torch.compile, and OOM-driven chunking.

tabpfn_v3_model <- torch::nn_module(
  "tabpfn_v3_model",
  initialize = function(config) {
    self$config <- config
    self$task_type <- if (config$max_num_classes > 0) {
      "multiclass"
    } else {
      "regression"
    }
    e <- config$embed_dim
    d <- e * config$feat_agg_num_cls_tokens
    self$icl_emsize <- d
    self$max_num_classes <- config$max_num_classes
    self$feature_group_size <- config$feature_group_size
    classification <- self$task_type == "multiclass"

    self$x_embed <- torch::nn_linear(config$feature_group_size * 2, e)
    self$col_y_encoder <- if (classification) {
      tabpfn_class_embedding(config$max_num_classes, e)
    } else {
      torch::nn_linear(1, e)
    }

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
      },
      qk_norm = FALSE
    )
    self$column_aggregator <- tabpfn_column_aggregator(
      e,
      config$feat_agg_num_heads,
      config$feat_agg_num_blocks,
      e * config$ff_factor,
      config$feat_agg_num_cls_tokens,
      config$feat_agg_rope_base,
      use_rope = isTRUE(config$use_rope),
      qk_norm = FALSE
    )
    self$icl_y_encoder <- if (classification) {
      tabpfn_class_embedding(config$max_num_classes, d)
    } else {
      torch::nn_linear(1, d)
    }
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
          num_kv_heads_test = config$icl_num_kv_heads_test,
          qk_norm = FALSE
        )
      }
    ))
    self$output_norm <- tabpfn_rms_norm(d)
    if (classification) {
      self$many_class_decoder <- tabpfn_many_class_decoder(
        config$max_num_classes,
        d,
        config$decoder_head_dim,
        config$decoder_num_heads,
        softmax_scaling_layer = if (
          isTRUE(config$decoder_use_softmax_scaling)
        ) {
          scaling(config$decoder_num_heads, config$decoder_head_dim)
        }
      )
    } else {
      self$output_projection <- torch::nn_sequential(
        torch::nn_linear(d, d * config$ff_factor),
        torch::nn_gelu(),
        torch::nn_linear(d * config$ff_factor, config$num_buckets)
      )
    }
    self$regression_borders <- torch::nn_buffer(torch::torch_zeros(
      config$num_buckets + 1
    ))
  },

  borders = function() {
    self$regression_borders
  },

  forward = function(x, y, task_type = self$task_type) {
    if (!identical(task_type, self$task_type)) {
      cli::cli_abort(
        "This TabPFN v3 checkpoint is for {self$task_type}, not {task_type}."
      )
    }
    if (y$dim() == 3 && y$size(3) == 1) {
      y <- y$squeeze(3)
    }
    num_train <- y$size(1)
    classification <- task_type == "multiclass"
    num_present_classes <- NULL
    if (classification) {
      num_present_classes <- tabpfn_num_present_classes(y, self$max_num_classes)
    }

    x <- self$stages_0_to_2(x, y, num_train, task_type)

    x <- x$flatten(start_dim = -2)
    y_icl <- tabpfn_prepare_y(y, num_train, task_type)
    x[, 1:num_train] <- x[, 1:num_train] +
      self$embed_y(self$icl_y_encoder, y_icl, task_type)
    for (i in seq_along(self$icl_blocks)) {
      x <- self$icl_blocks[[i]](x, num_train)
      tabpfn_release()
    }
    x <- self$output_norm(x)

    r <- x$size(2)
    test_emb <- x[, (num_train + 1):r]
    if (classification) {
      train_keys <- self$many_class_decoder$project_keys(x[, 1:num_train])
      y_bn <- if (y$dim() == 2) y$transpose(1, 2) else y$unsqueeze(1)
      out <- self$many_class_decoder(
        train_keys,
        test_emb,
        y_bn[, 1:num_train],
        num_present_classes
      )
    } else {
      out <- self$output_projection(test_emb$transpose(1, 2))
    }
    out$nan_to_num(nan = 0)
  },

  # Stages 0-2: (Ri, B, C) -> (B, Ri, num_cls_tokens, E).
  stages_0_to_2 = function(x, y, num_train, task_type) {
    prep <- self$preprocess(x, num_train)
    y_col <- tabpfn_prepare_y(y, num_train, task_type)
    grouped <- tabpfn_group_features(
      list(prep$x, prep$nan_ind),
      self$feature_group_size
    )
    tabpfn_stages_1_2(
      self,
      grouped,
      self$embed_y(self$col_y_encoder, y_col, task_type),
      num_train,
      tabpfn_chunk_cells(self$config)
    )
  },

  # NaN/Inf indicators, imputation, and scaling; both (B, Ri, C).
  preprocess = function(x, num_train) {
    nan_ind <- tabpfn_nan_indicator(x)$transpose(1, 2)
    imputed <- tabpfn_impute_mean(x, num_train)$x
    stats <- tabpfn_scaler_fit(imputed[1:num_train])
    x <- tabpfn_scaler_transform(imputed, stats)$transpose(1, 2)
    list(x = x, nan_ind = nan_ind, stats = stats)
  },

  # Target embedding: (B, N) -> (B, N, width). No LayerNorm in v3.
  embed_y = function(encoder, y_bn, task_type) {
    if (task_type == "multiclass") {
      return(encoder(y_bn))
    }
    emb <- encoder(y_bn$reshape(c(-1, 1)))
    emb$reshape(c(y_bn$shape, emb$size(-1)))
  }
)
