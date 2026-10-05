# Building blocks shared by the TabPFN architectures. Module and parameter
# names follow the Python modules exactly, so that `state_dict()` names match
# the checkpoint keys (see `tabpfn_load_state()`).
#
# Index conventions differ from PyTorch: R torch dimensions are 1-based,
# `gather()`, `nn_embedding()`, and `nnf_one_hot()` take 1-based indices, and
# `torch_searchsorted()` returns the same 0-based counts as PyTorch.
#
# Parts adapted from brulee's TabICL port (commit b6adc31): the SDPA wrapper
# (`tabicl_sdpa`) and the query-aware softmax scaling (`tabicl_qassmax`).

# RMSNorm over the last dimension. As in PyTorch's `nn.RMSNorm(eps = None)`,
# eps is the machine epsilon of the input's dtype.
tabpfn_rms_norm <- torch::nn_module(
  "tabpfn_rms_norm",
  initialize = function(dim) {
    self$weight <- torch::nn_parameter(torch::torch_ones(dim))
  },
  forward = function(x) {
    eps <- torch::torch_finfo(x$dtype)$eps
    x *
      torch::torch_rsqrt(x$pow(2)$mean(dim = -1, keepdim = TRUE) + eps) *
      self$weight$to(dtype = x$dtype)
  }
)

# Two-layer bias-free GELU MLP; Python `MLP(nn.Sequential)`, so the weights
# are `0.weight` and `2.weight`.
tabpfn_mlp <- function(emsize, dim_feedforward) {
  torch::nn_sequential(
    torch::nn_linear(emsize, dim_feedforward, bias = FALSE),
    torch::nn_gelu(),
    torch::nn_linear(dim_feedforward, emsize, bias = FALSE)
  )
}

# Query-aware attention scaling:
#   q * base_mlp(log n) * (1 + tanh(query_mlp(q)))
# where n is the number of keys.
tabpfn_softmax_scaling_mlp <- torch::nn_module(
  "tabpfn_softmax_scaling_mlp",
  initialize = function(num_heads, head_dim, n_hidden = 64) {
    self$num_heads <- num_heads
    self$head_dim <- head_dim
    self$base_mlp <- torch::nn_sequential(
      torch::nn_linear(1, n_hidden),
      torch::nn_gelu(),
      torch::nn_linear(n_hidden, num_heads * head_dim)
    )
    self$query_mlp <- torch::nn_sequential(
      torch::nn_linear(head_dim, n_hidden),
      torch::nn_gelu(),
      torch::nn_linear(n_hidden, head_dim)
    )
  },
  # q: (B, S, H, D)
  forward = function(q, n) {
    logn <- torch::torch_tensor(
      log(max(n, 1)),
      dtype = torch::torch_float32(),
      device = q$device
    )$to(dtype = q$dtype)$reshape(c(1, 1))
    base_scales <- self$base_mlp(logn)$view(c(
      1,
      1,
      self$num_heads,
      self$head_dim
    ))
    modulation <- 1 + torch::torch_tanh(self$query_mlp(q))
    q * (base_scales * modulation)
  }
)

# Scaled dot-product attention on (B, S, H, D) tensors, with optional query
# scaling and grouped-query attention (fewer key/value heads than query heads,
# repeated to match).
tabpfn_sdpa <- function(q, k, v, scaling = NULL) {
  if (!is.null(scaling)) {
    q <- scaling(q, k$size(2))
  }
  q <- q$permute(c(1, 3, 2, 4))
  k <- k$permute(c(1, 3, 2, 4))
  v <- v$permute(c(1, 3, 2, 4))
  num_q_heads <- q$size(2)
  num_kv_heads <- k$size(2)
  if (num_q_heads != num_kv_heads) {
    rep <- num_q_heads %/% num_kv_heads
    k <- k$repeat_interleave(rep, dim = 2)
    v <- v$repeat_interleave(rep, dim = 2)
  }
  if (q$device$type == "mps") {
    # MPS attention gives wrong results for non-contiguous (here, permuted)
    # inputs when the query sequence is short (R torch 0.17, macOS 27).
    q <- q$contiguous()
    k <- k$contiguous()
    v <- v$contiguous()
    out <- tabpfn_sdpa_chunked(q, k, v)
  } else {
    out <- torch::torch_scaled_dot_product_attention(q, k, v)
  }
  out$permute(c(1, 3, 2, 4))
}

# The libtorch bundled with R torch has no memory-efficient attention kernel
# for MPS (PyTorch added one in 2.13), so attention there materializes the
# full (B, H, queries, keys) score matrix: about 10 GB for one ICL layer with
# 5,000 training rows. Each query row's attention is independent of the
# others, so splitting the queries into chunks gives the same result while
# bounding the score matrix to `brulee.tabpfn_mps_attention_cells` elements.
tabpfn_sdpa_chunked <- function(q, k, v) {
  n_q <- q$size(3)
  per_row <- q$size(1) * q$size(2) * k$size(3)
  budget <- getOption("brulee.tabpfn_mps_attention_cells", 2^27)
  chunk <- max(1, floor(budget / per_row))
  if (chunk >= n_q) {
    return(torch::torch_scaled_dot_product_attention(q, k, v))
  }
  parts <- lapply(seq(1, n_q, by = chunk), function(start) {
    rows <- q$narrow(3, start, min(chunk, n_q - start + 1))
    out <- torch::torch_scaled_dot_product_attention(rows, k, v)
    tabpfn_release()
    out
  })
  torch::torch_cat(parts, dim = 3)
}

# Rotary position embedding along the second-to-last dimension, rotating
# the two halves of the last dimension against each other (non-interleaved).
# `freqs` is a non-trainable parameter, as in Python, so it's in the
# checkpoint.
tabpfn_rope <- torch::nn_module(
  "tabpfn_rope",
  initialize = function(dim, theta = 10000) {
    inv_freq <- 1 /
      (theta^(torch::torch_arange(
        0,
        dim - 1,
        2,
        dtype = torch::torch_float32()
      ) /
        dim))
    self$freqs <- torch::nn_parameter(inv_freq, requires_grad = FALSE)
  },
  rotate = function(t) {
    dtype <- t$dtype
    seq_len <- t$size(-2)
    positions <- torch::torch_arange(
      0,
      seq_len - 1,
      dtype = self$freqs$dtype,
      device = t$device
    )
    freqs <- positions$unsqueeze(2) * self$freqs$unsqueeze(1)
    cos <- freqs$cos()
    sin <- freqs$sin()
    cos <- torch::torch_cat(list(cos, cos), dim = -1)
    sin <- torch::torch_cat(list(sin, sin), dim = -1)
    d <- t$size(-1)
    half <- d %/% 2
    rotated <- torch::torch_cat(
      list(-t[.., (half + 1):d], t[.., 1:half]),
      dim = -1
    )
    (t * cos + rotated * sin)$to(dtype = dtype)
  }
)

# Mean and sample standard deviation over the first dimension, ignoring NaN
# (Python `tabpfn.preprocessing.torch.ops.torch_nanmean`/`torch_nanstd`).
tabpfn_nanmean <- function(x) {
  is_nan <- torch::torch_isnan(x)
  zeros <- torch::torch_zeros_like(x)
  num_valid <- torch::torch_where(is_nan, zeros, torch::torch_ones_like(x))$sum(
    dim = 1
  )
  value_sum <- torch::torch_where(is_nan, zeros, x)$sum(dim = 1)
  value_sum / num_valid$clamp(min = 1)
}

tabpfn_nanstd <- function(x) {
  is_nan <- torch::torch_isnan(x)
  zeros <- torch::torch_zeros_like(x)
  num_valid <- torch::torch_where(is_nan, zeros, torch::torch_ones_like(x))$sum(
    dim = 1
  )
  mean <- torch::torch_where(is_nan, zeros, x)$sum(dim = 1) /
    num_valid$clamp(min = 1)
  sq_diff <- torch::torch_where(
    is_nan,
    zeros,
    (x - mean$unsqueeze(1)$expand_as(x))$square()
  )$sum(dim = 1)
  torch::torch_sqrt(sq_diff / (num_valid - 1)$clamp(min = 1))
}

# Standard scaler with NaN handling (Python `TorchStandardScaler`).
tabpfn_scaler_fit <- function(x) {
  mean <- tabpfn_nanmean(x)
  std <- tabpfn_nanstd(x)
  std <- torch::torch_where(std == 0, torch::torch_ones_like(std), std)
  if (x$size(1) == 1) {
    std <- torch::torch_ones_like(std)
  }
  list(mean = mean, std = std)
}

tabpfn_scaler_transform <- function(x, stats) {
  eps <- torch::torch_finfo(stats$std$dtype)$eps
  out <- (x - stats$mean) / (stats$std + eps)
  torch::torch_clip(out, min = -100, max = 100)
}

# ------------------------------------------------------------------------------

# Attention modules and blocks shared by the v3 and v3.5 architectures
# (Python `tabpfn_v3.py` and `tabpfn_v3_5.py`). The two differ only in the
# per-head RMSNorm on queries and keys, which v3.5 adds (`qk_norm`), and in
# whether the column aggregator uses RoPE (always in v3.5). Both
# architectures' parity fixtures run through these modules.
#
# Tensor-name suffixes follow the Python code: B batch (ensemble members),
# R/Ri rows, N train rows, M test rows, C columns, E embedding, G group, H
# heads, D head dim.

tabpfn_qkv <- function(self, embedding_size, num_heads, head_dim, qk_norm) {
  self$num_heads <- num_heads
  self$head_dim <- head_dim
  self$qk_norm <- qk_norm
  width <- head_dim * num_heads
  self$q_projection <- torch::nn_linear(embedding_size, width, bias = FALSE)
  self$k_projection <- torch::nn_linear(embedding_size, width, bias = FALSE)
  self$v_projection <- torch::nn_linear(embedding_size, width, bias = FALSE)
  self$out_projection <- torch::nn_linear(width, embedding_size, bias = FALSE)
  if (qk_norm) {
    self$q_norm <- tabpfn_rms_norm(head_dim)
    self$k_norm <- tabpfn_rms_norm(head_dim)
  }
}

tabpfn_rotate <- function(rope, t) {
  if (is.null(rope)) {
    return(t)
  }
  rope$rotate(t$transpose(2, 3))$transpose(2, 3)
}

# Self-attention over the sequence dimension of (B, S, E), with optional RoPE.
tabpfn_attention <- torch::nn_module(
  "tabpfn_attention",
  initialize = function(embedding_size, num_heads, head_dim, qk_norm = TRUE) {
    tabpfn_qkv(self, embedding_size, num_heads, head_dim, qk_norm)
  },
  forward = function(x, rope = NULL) {
    b <- x$size(1)
    s <- x$size(2)
    shape <- c(b, s, -1, self$head_dim)
    q <- tabpfn_rotate(rope, self$q_projection(x)$view(shape))
    k <- tabpfn_rotate(rope, self$k_projection(x)$view(shape))
    v <- self$v_projection(x)$view(shape)
    if (self$qk_norm) {
      q <- self$q_norm(q)
      k <- self$k_norm(k)
    }
    out <- tabpfn_sdpa(q, k, v)$reshape(c(b, s, self$head_dim * self$num_heads))
    self$out_projection(out)
  }
)

# Cross-attention: queries (B, Q, E) attend to context (B, V, E).
tabpfn_cross_attention <- torch::nn_module(
  "tabpfn_cross_attention",
  initialize = function(
    embedding_size,
    num_heads,
    head_dim,
    softmax_scaling_layer = NULL,
    qk_norm = TRUE
  ) {
    tabpfn_qkv(self, embedding_size, num_heads, head_dim, qk_norm)
    if (!is.null(softmax_scaling_layer)) {
      self$softmax_scaling_layer <- softmax_scaling_layer
    }
  },
  forward = function(x_q, x_kv) {
    b <- x_q$size(1)
    nq <- x_q$size(2)
    nv <- x_kv$size(2)
    q <- self$q_projection(x_q)$view(c(b, nq, -1, self$head_dim))
    k <- self$k_projection(x_kv)$view(c(b, nv, -1, self$head_dim))
    v <- self$v_projection(x_kv)$view(c(b, nv, -1, self$head_dim))
    if (self$qk_norm) {
      q <- self$q_norm(q)
      k <- self$k_norm(k)
    }
    out <- tabpfn_sdpa(q, k, v, scaling = self$softmax_scaling_layer)
    self$out_projection(out$reshape(c(b, nq, self$head_dim * self$num_heads)))
  }
)

# ICL attention: every row attends to the train rows only. Test rows use
# only the first `num_kv_heads_test` key/value heads (multi-query attention).
tabpfn_icl_attention <- torch::nn_module(
  "tabpfn_icl_attention",
  initialize = function(
    embedding_size,
    num_heads,
    head_dim,
    softmax_scaling_layer = NULL,
    num_kv_heads = NULL,
    num_kv_heads_test = NULL,
    qk_norm = TRUE
  ) {
    self$num_heads <- num_heads
    self$head_dim <- head_dim
    self$qk_norm <- qk_norm
    self$num_kv_heads <- num_kv_heads %||% num_heads
    self$num_kv_heads_test <- num_kv_heads_test
    kv_dim <- self$num_kv_heads * head_dim
    self$q_projection <- torch::nn_linear(
      embedding_size,
      head_dim * num_heads,
      bias = FALSE
    )
    self$out_projection <- torch::nn_linear(
      head_dim * num_heads,
      embedding_size,
      bias = FALSE
    )
    self$k_projection <- torch::nn_linear(embedding_size, kv_dim, bias = FALSE)
    self$v_projection <- torch::nn_linear(embedding_size, kv_dim, bias = FALSE)
    if (qk_norm) {
      self$q_norm <- tabpfn_rms_norm(head_dim)
      self$k_norm <- tabpfn_rms_norm(head_dim)
    }
    if (!is.null(softmax_scaling_layer)) {
      self$softmax_scaling_layer <- softmax_scaling_layer
    }
  },
  forward = function(x, num_train) {
    b <- x$size(1)
    r <- x$size(2)
    q <- self$q_projection(x)$view(c(b, r, self$num_heads, self$head_dim))
    x_train <- x[, 1:num_train]
    kv_shape <- c(b, num_train, self$num_kv_heads, self$head_dim)
    k <- self$k_projection(x_train)$view(kv_shape)
    v <- self$v_projection(x_train)$view(kv_shape)
    if (self$qk_norm) {
      q <- self$q_norm(q)
      k <- self$k_norm(k)
    }
    scaling <- self$softmax_scaling_layer
    if (!is.null(self$num_kv_heads_test) && num_train < r) {
      out_train <- tabpfn_sdpa(q[, 1:num_train], k, v, scaling = scaling)
      h <- self$num_kv_heads_test
      out_test <- tabpfn_sdpa(
        q[, (num_train + 1):r],
        k[,, 1:h],
        v[,, 1:h],
        scaling = scaling
      )
      out <- torch::torch_cat(list(out_train, out_test), dim = 2)
    } else {
      out <- tabpfn_sdpa(q, k, v, scaling = scaling)
    }
    self$out_projection(out$reshape(c(b, r, self$head_dim * self$num_heads)))
  }
)

# Pre-norm cross-attention block with MLP.
tabpfn_cross_attention_block <- torch::nn_module(
  "tabpfn_cross_attention_block",
  initialize = function(
    emsize,
    nhead,
    dim_feedforward,
    softmax_scaling_layer = NULL,
    qk_norm = TRUE
  ) {
    self$attn <- tabpfn_cross_attention(
      emsize,
      nhead,
      emsize %/% nhead,
      softmax_scaling_layer = softmax_scaling_layer,
      qk_norm = qk_norm
    )
    self$mlp <- tabpfn_mlp(emsize, dim_feedforward)
    self$layernorm_q <- tabpfn_rms_norm(emsize)
    self$layernorm_kv <- tabpfn_rms_norm(emsize)
    self$layernorm2 <- tabpfn_rms_norm(emsize)
  },
  forward = function(x, context) {
    x <- x + self$attn(self$layernorm_q(x), self$layernorm_kv(context))
    x + self$mlp(self$layernorm2(x))
  }
)

# Pre-norm transformer block of the column aggregator, on (B, R, C, E):
# attention runs over C within each row.
tabpfn_transformer_block <- torch::nn_module(
  "tabpfn_transformer_block",
  initialize = function(emsize, nhead, dim_feedforward, qk_norm = TRUE) {
    self$attention <- tabpfn_attention(
      emsize,
      nhead,
      emsize %/% nhead,
      qk_norm = qk_norm
    )
    self$layernorm <- tabpfn_rms_norm(emsize)
    self$layernorm_mlp <- tabpfn_rms_norm(emsize)
    self$mlp <- tabpfn_mlp(emsize, dim_feedforward)
  },
  forward = function(x, rope = NULL) {
    flat <- x$flatten(start_dim = 1, end_dim = 2)
    x <- x + self$attention(self$layernorm(flat), rope)$view(x$shape)
    tabpfn_release()
    x + self$mlp(self$layernorm_mlp(x))
  },
  # The last block: queries (B, R, Q, E) attend to the context (B, R, V, E).
  forward_cross = function(query, context, rope = NULL) {
    b <- query$size(1)
    r <- query$size(2)
    nq <- query$size(3)
    nv <- context$size(3)
    e <- context$size(4)
    att <- self$attention
    q_flat <- self$layernorm(query)$view(c(b * r, nq, e))
    c_flat <- self$layernorm(context)$view(c(b * r, nv, e))
    q <- tabpfn_rotate(
      rope,
      att$q_projection(q_flat)$view(c(b * r, nq, -1, att$head_dim))
    )
    k <- tabpfn_rotate(
      rope,
      att$k_projection(c_flat)$view(c(b * r, nv, -1, att$head_dim))
    )
    v <- att$v_projection(c_flat)$view(c(b * r, nv, -1, att$head_dim))
    if (att$qk_norm) {
      q <- att$q_norm(q)
      k <- att$k_norm(k)
    }
    out <- tabpfn_sdpa(q, k, v)$reshape(c(
      b * r,
      nq,
      att$head_dim * att$num_heads
    ))
    out <- att$out_projection(out)$view(c(b, r, nq, e))
    x <- query + out
    tabpfn_release()
    x + self$mlp(self$layernorm_mlp(x))
  }
)

# ICL transformer block on (B, R, E).
tabpfn_icl_block <- torch::nn_module(
  "tabpfn_icl_block",
  initialize = function(
    emsize,
    nhead,
    dim_feedforward,
    softmax_scaling_layer = NULL,
    num_kv_heads = NULL,
    num_kv_heads_test = NULL,
    qk_norm = TRUE
  ) {
    self$icl_attention <- tabpfn_icl_attention(
      emsize,
      nhead,
      emsize %/% nhead,
      softmax_scaling_layer = softmax_scaling_layer,
      num_kv_heads = num_kv_heads,
      num_kv_heads_test = num_kv_heads_test,
      qk_norm = qk_norm
    )
    self$layernorm <- tabpfn_rms_norm(emsize)
    self$layernorm_mlp <- tabpfn_rms_norm(emsize)
    self$mlp <- tabpfn_mlp(emsize, dim_feedforward)
  },
  forward = function(x, num_train) {
    x <- x + self$icl_attention(self$layernorm(x), num_train)
    tabpfn_release()
    x + self$mlp(self$layernorm_mlp(x))
  }
)

# Induced self-attention, applied per column: inducing points attend to the
# train rows, then all rows attend to the inducing points.
tabpfn_isab <- torch::nn_module(
  "tabpfn_isab",
  initialize = function(
    emsize,
    nhead,
    num_inducing_points,
    dim_feedforward,
    softmax_scaling_layer = NULL,
    qk_norm = TRUE
  ) {
    self$cross_attn_block1 <- tabpfn_cross_attention_block(
      emsize,
      nhead,
      dim_feedforward,
      softmax_scaling_layer = softmax_scaling_layer,
      qk_norm = qk_norm
    )
    self$cross_attn_block2 <- tabpfn_cross_attention_block(
      emsize,
      nhead,
      dim_feedforward,
      qk_norm = qk_norm
    )
    self$inducing_vectors <- torch::nn_parameter(torch::torch_zeros(
      num_inducing_points,
      emsize
    ))
  },
  # The inducing points' states after attending to the train rows of
  # x_flat (B * C, N, E).
  induce = function(x_flat) {
    ind <- self$inducing_vectors$unsqueeze(1)$expand(c(x_flat$size(1), -1, -1))
    self$cross_attn_block1(ind, x_flat)
  },
  # x: (B, R, C, E). `hidden` (from `induce()`) is given when the rows are
  # processed in chunks; otherwise it is computed from the train rows of x.
  forward = function(x, num_train, hidden = NULL) {
    b <- x$size(1)
    r <- x$size(2)
    nc <- x$size(3)
    e <- x$size(4)
    x_flat <- x$transpose(2, 3)$contiguous()$reshape(c(b * nc, r, e))
    hidden <- hidden %||% self$induce(x_flat[, 1:num_train])
    out <- self$cross_attn_block2(x_flat, hidden)
    out$reshape(c(b, nc, r, e))$transpose(2, 3)$contiguous()
  }
)

tabpfn_feature_distribution_embedder <- torch::nn_module(
  "tabpfn_feature_distribution_embedder",
  initialize = function(
    emsize,
    nhead,
    num_inducing_points,
    dim_feedforward,
    num_layers,
    softmax_scaling_factory,
    qk_norm = TRUE
  ) {
    self$layers <- torch::nn_module_list(lapply(
      seq_len(num_layers),
      function(i) {
        tabpfn_isab(
          emsize,
          nhead,
          num_inducing_points,
          dim_feedforward,
          softmax_scaling_layer = softmax_scaling_factory(),
          qk_norm = qk_norm
        )
      }
    ))
  },
  # `hidden` is the list from `inducing_hidden()` when the rows are
  # processed in chunks.
  forward = function(x, num_train, hidden = NULL) {
    for (i in seq_along(self$layers)) {
      x <- self$layers[[i]](x, num_train, hidden = hidden[[i]])
      tabpfn_release()
    }
    x
  },
  # Each layer's inducing-point states from the train rows alone (B, N, C, E),
  # as the full forward computes them, `col_chunk` columns at a time to bound
  # memory (Python `_compute_all_inducing_hidden`). Returns one
  # (B * C, inducing points, E) tensor per layer.
  inducing_hidden = function(x_train, col_chunk = 4L) {
    b <- x_train$size(1)
    n <- x_train$size(2)
    nc <- x_train$size(3)
    e <- x_train$size(4)
    n_layers <- length(self$layers)
    per_chunk <- lapply(seq(1, nc, by = col_chunk), function(start) {
      cols <- start:min(start + col_chunk - 1, nc)
      x_flat <- x_train$narrow(3, start, length(cols))$transpose(
        2,
        3
      )$contiguous()$reshape(c(
        b * length(cols),
        n,
        e
      ))
      hidden <- vector("list", n_layers)
      for (i in seq_len(n_layers)) {
        layer <- self$layers[[i]]
        h <- layer$induce(x_flat)
        if (i < n_layers) {
          x_flat <- layer$cross_attn_block2(x_flat, h)
        }
        hidden[[i]] <- h$reshape(c(b, length(cols), h$size(2), e))
        tabpfn_release()
      }
      hidden
    })
    lapply(seq_len(n_layers), function(i) {
      h <- torch::torch_cat(lapply(per_chunk, `[[`, i), dim = 2)
      h$reshape(c(b * nc, h$size(3), e))
    })
  }
)

# Prepends CLS tokens to each row's feature embeddings, runs transformer
# blocks over the features, and reads out the CLS tokens with the last
# block: (B, R, C, E) -> (B, R, num_cls_tokens, E).
tabpfn_column_aggregator <- torch::nn_module(
  "tabpfn_column_aggregator",
  initialize = function(
    emsize,
    nhead,
    num_layers,
    dim_feedforward,
    num_cls_tokens,
    rope_base,
    use_rope = TRUE,
    qk_norm = TRUE
  ) {
    self$num_cls_tokens <- num_cls_tokens
    self$blocks <- torch::nn_module_list(lapply(
      seq_len(num_layers),
      function(i) {
        tabpfn_transformer_block(
          emsize,
          nhead,
          dim_feedforward,
          qk_norm = qk_norm
        )
      }
    ))
    if (use_rope) {
      self$rope <- tabpfn_rope(emsize %/% nhead, theta = as.integer(rope_base))
    }
    self$cls_tokens <- torch::nn_parameter(torch::torch_zeros(
      num_cls_tokens,
      emsize
    ))
    self$out_ln <- tabpfn_rms_norm(emsize)
  },
  forward = function(x) {
    b <- x$size(1)
    r <- x$size(2)
    e <- x$size(4)
    cls <- self$cls_tokens$expand(c(b, r, self$num_cls_tokens, e))
    x <- torch::torch_cat(list(cls, x), dim = 3)
    n_blocks <- length(self$blocks)
    for (i in seq_len(n_blocks - 1)) {
      x <- self$blocks[[i]](x, self$rope)
      tabpfn_release()
    }
    cls_part <- x[,, 1:self$num_cls_tokens]
    cls_out <- self$blocks[[n_blocks]]$forward_cross(cls_part, x, self$rope)
    self$out_ln(cls_out)
  }
)

# Class embeddings (Python `TrainableOrthogonalEmbedding`); labels are
# 0-based, R's embedding is 1-based.
tabpfn_class_embedding <- torch::nn_module(
  "tabpfn_class_embedding",
  initialize = function(num_classes, embed_dim) {
    self$embedding <- torch::nn_embedding(num_classes, embed_dim)
  },
  forward = function(y) {
    self$embedding(y$to(dtype = torch::torch_long()) + 1L)
  }
)

# Attention-based retrieval over one-hot train targets: each test row's class
# probabilities are an attention-weighted average of the train rows' one-hot
# labels, averaged over heads. Returns logits (M, B, max_num_classes).
#
# v3 one-hot encodes the targets directly; v3.5 first zeroes rows with
# non-finite targets. The pipeline only passes finite class labels, where the
# two agree.
tabpfn_many_class_decoder <- torch::nn_module(
  "tabpfn_many_class_decoder",
  initialize = function(
    max_num_classes,
    input_size,
    head_dim,
    num_heads,
    softmax_scaling_layer = NULL
  ) {
    self$max_num_classes <- max_num_classes
    self$head_dim <- head_dim
    self$num_heads <- num_heads
    self$q_projection <- torch::nn_linear(input_size, head_dim * num_heads)
    self$k_projection <- torch::nn_linear(input_size, head_dim * num_heads)
    if (!is.null(softmax_scaling_layer)) {
      self$softmax_scaling_layer <- softmax_scaling_layer
    }
  },
  project_keys = function(train_emb) {
    k <- self$k_projection(train_emb)
    k$view(c(k$size(1), k$size(2), self$num_heads, self$head_dim))$contiguous()
  },
  # train_keys: (B, N, H, D); test_emb: (B, M, E); targets: (B, N), 0-based
  forward = function(train_keys, test_emb, targets, num_present_classes) {
    b <- test_emb$size(1)
    m <- test_emb$size(2)
    q <- self$q_projection(test_emb)$view(c(
      b,
      m,
      self$num_heads,
      self$head_dim
    ))
    is_finite <- torch::torch_isfinite(targets)
    labels <- torch::torch_where(
      is_finite,
      targets,
      torch::torch_zeros_like(targets)
    )$to(dtype = torch::torch_long())
    one_hot <- torch::nnf_one_hot(labels + 1L, num_present_classes)
    one_hot <- torch::torch_where(
      is_finite$unsqueeze(-1),
      one_hot,
      torch::torch_zeros_like(one_hot)
    )$to(dtype = q$dtype)
    values <- one_hot$unsqueeze(3)$expand(c(
      -1,
      -1,
      self$num_heads,
      -1
    ))$contiguous()
    out <- tabpfn_chunked_class_attention(
      q,
      train_keys,
      values,
      self$softmax_scaling_layer
    )
    out <- out$mean(dim = 3)
    missing <- self$max_num_classes - num_present_classes
    if (missing > 0) {
      out <- torch::nnf_pad(out, c(0, missing))
    }
    out <- out$transpose(1, 2)
    torch::torch_log(torch::torch_clamp(out, min = 1e-5) + 3e-5)
  }
)

# Attention whose values (one-hot classes, width T) may be wider than the
# head dim D: V is split into D-wide chunks folded into the batch.
tabpfn_chunked_class_attention <- function(q, k, v, scaling = NULL) {
  b <- q$size(1)
  s <- q$size(2)
  h <- q$size(3)
  d <- q$size(4)
  n_classes <- v$size(4)
  num_chunks <- ceiling(n_classes / d)
  pad <- num_chunks * d - n_classes
  if (pad > 0) {
    v <- torch::nnf_pad(v, c(0, pad))
  }
  j <- v$size(2)
  v_folded <- v$reshape(c(b, j, h, num_chunks, d))$permute(c(
    1,
    4,
    2,
    3,
    5
  ))$reshape(
    c(b * num_chunks, j, h, d)
  )$contiguous()
  q_folded <- q$unsqueeze(2)$expand(c(-1, num_chunks, -1, -1, -1))$reshape(c(
    b * num_chunks,
    s,
    h,
    d
  ))$contiguous()
  k_folded <- k$unsqueeze(2)$expand(c(-1, num_chunks, -1, -1, -1))$reshape(c(
    b * num_chunks,
    j,
    h,
    d
  ))$contiguous()
  out <- tabpfn_sdpa(q_folded, k_folded, v_folded, scaling = scaling)
  out <- out$reshape(c(b, num_chunks, s, h, d))$permute(c(
    1,
    3,
    4,
    2,
    5
  ))$reshape(
    c(b, s, h, num_chunks * d)
  )
  out[.., 1:n_classes]
}

# Number of classes in 0-based train targets: the highest label plus one.
tabpfn_num_present_classes <- function(y, max_num_classes) {
  n <- as.integer(y$nan_to_num(nan = 0)$max()$item() + 1)
  if (n > max_num_classes) {
    cli::cli_abort("The model supports at most {max_num_classes} classes.")
  }
  n
}

# Train targets as (B or 1, N), with non-finite targets imputed (rounded up
# for classification). Python `_prepare_y` / `_impute_target_nan_and_inf`.
tabpfn_prepare_y <- function(y, num_train, task_type) {
  y <- y[1:num_train]
  y_nb1 <- if (y$dim() == 1) y$view(c(num_train, 1, 1)) else y$unsqueeze(-1)
  imputed <- tabpfn_impute_mean(y_nb1, num_train)
  y_nb1 <- imputed$x
  if (task_type == "multiclass") {
    y_nb1 <- torch::torch_where(imputed$is_finite, y_nb1, y_nb1$ceil())
  }
  y_nb1$squeeze(-1)$transpose(1, 2)
}

tabpfn_nan_indicator <- function(x) {
  (torch::torch_isnan(x) *
    -2 +
    torch::torch_isposinf(x) * 2 +
    torch::torch_isneginf(x) * 4)$to(dtype = x$dtype)
}

# Replace NaN and +/-Inf by the column mean of the finite train values (0 for
# a column without any).
tabpfn_impute_mean <- function(x, num_train) {
  is_finite <- torch::torch_isfinite(x)
  train <- x[1:num_train]
  train <- torch::torch_where(
    is_finite[1:num_train],
    train,
    torch::torch_full_like(train, NaN)
  )
  means <- train$nanmean(dim = 1)$nan_to_num(nan = 0)
  list(
    x = torch::torch_where(is_finite, x, means$unsqueeze(1)$expand_as(x)),
    is_finite = is_finite
  )
}

# Each column is grouped with the columns 1, 2, 4, ... positions after it
# (circularly): each (B, Ri, C) tensor becomes (B, Ri, C, G), concatenated
# along the last dimension.
tabpfn_group_features <- function(tensors, group_size) {
  shifts <- -(2^(seq_len(group_size) - 1))
  stacked <- lapply(tensors, function(t) {
    torch::torch_stack(
      lapply(shifts, function(s) torch::torch_roll(t, shifts = s, dims = 3)),
      dim = -1
    )
  })
  torch::torch_cat(stacked, dim = -1)
}

# Cells per row chunk of stages 1-2. Python's default (`inference_chunk_cells`,
# about 1.6 million) bounds memory for PyTorch, which frees intermediates at
# once; R keeps them until the next garbage collection, so a smaller budget is
# used unless the option `brulee.tabpfn_chunk_cells` says otherwise.
tabpfn_chunk_cells <- function(config) {
  getOption(
    "brulee.tabpfn_chunk_cells",
    min(config$inference_chunk_cells, 262144)
  )
}

# R torch frees a tensor only when R's garbage collector runs, unlike
# PyTorch's reference counting, so the large intermediates of each layer
# accumulate across layers. A minor collection between layers keeps the peak
# memory to about one layer's worth.
tabpfn_release <- function() {
  invisible(gc(verbose = FALSE, full = TRUE))
}

# Stages 1-2 for both architectures, for the grouped cells (B, Ri, C, G): cell
# embedding plus the train rows' target embedding, the per-column induced
# self-attention, and the column aggregator. When B * Ri * C exceeds
# `chunk_cells`, the inducing-point states are computed from the train rows
# first and the rows are then processed in chunks (Python's chunked path,
# which gives the same result with less memory).
tabpfn_stages_1_2 <- function(
  model,
  grouped,
  y_col_emb,
  num_train,
  chunk_cells,
  col_chunk = 4L
) {
  b <- grouped$size(1)
  rows <- grouped$size(2)
  cols <- grouped$size(3)
  fde <- model$feature_distribution_embedder
  embed <- function(start, end) {
    emb <- model$x_embed(grouped[, start:end])
    n_train_rows <- max(0, min(num_train - start + 1, end - start + 1))
    if (n_train_rows > 0) {
      emb$narrow(2, 1, n_train_rows)$add_(
        y_col_emb$narrow(2, start, n_train_rows)$unsqueeze(3)
      )
    }
    emb
  }
  chunk <- max(1, floor(chunk_cells / (b * cols)))
  if (chunk >= rows) {
    emb <- fde(embed(1, rows), num_train)
    return(model$column_aggregator(emb))
  }
  hidden <- fde$inducing_hidden(embed(1, num_train), col_chunk)
  starts <- seq(1, rows, by = chunk)
  parts <- lapply(starts, function(start) {
    end <- min(start + chunk - 1, rows)
    emb <- fde(embed(start, end), num_train, hidden = hidden)
    out <- model$column_aggregator(emb)
    tabpfn_release()
    out
  })
  torch::torch_cat(parts, dim = 2)
}
