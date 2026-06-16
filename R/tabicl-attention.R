# Attention primitives for TabICL: scalable softmax (SSMax), scaled
# dot-product attention, and multi-head attention with optional RoPE / SSMax.
#
# Ported from `src/tabicl/_model/ssmax.py`, `attention.py`, and the
# `MultiheadAttention` class in `layers.py`. The released v2 checkpoints only
# use the `qassmax-mlp-elementwise` scalable-softmax variant, so that is the
# only one implemented here; the other SSMax types in the reference are unused.
#
# Flash Attention 3 is not bound: the reference falls back to
# `F.scaled_dot_product_attention` whenever FA3 is unavailable (e.g. CPU, or no
# `flash_attn_interface`), and on CUDA libtorch's fused SDPA kernels compute the
# same math. We always take the SDPA path.

# ------------------------------------------------------------------------------
# Scalable softmax: query-aware, MLP-based, elementwise (qassmax-mlp-elementwise)

# scale = base_mlp(log n) * (1 + tanh(query_mlp(q)))  applied as  q * scale.
# `base_mlp` maps log(seq_len) to a per-(head, head_dim) base scale; `query_mlp`
# adds a query-dependent, zero-centred modulation. See QASSMaxMLP in ssmax.py.
tabicl_qassmax <- nn_module(
  "tabicl_qassmax",
  initialize = function(num_heads, head_dim, n_hidden = 64) {
    self$num_heads <- num_heads
    self$head_dim <- head_dim
    self$base_mlp <- nn_sequential(
      nn_linear(1, n_hidden),
      nn_gelu(),
      nn_linear(n_hidden, num_heads * head_dim)
    )
    self$query_mlp <- nn_sequential(
      nn_linear(head_dim, n_hidden),
      nn_gelu(),
      nn_linear(n_hidden, head_dim)
    )
  },
  forward = function(q, n) {
    # q: (flat_batch, n_heads, seq_len, head_dim); n: source sequence length.
    logn <- torch_log(torch_tensor(
      max(n, 1),
      dtype = q$dtype,
      device = q$device
    ))$reshape(c(1, 1))
    base_scales <- self$base_mlp(logn)$view(c(
      1,
      self$num_heads,
      1,
      self$head_dim
    ))
    modulation <- 1 + torch_tanh(self$query_mlp(q))
    q * (base_scales * modulation)
  }
)

# ------------------------------------------------------------------------------
# Scaled dot-product attention with flattened batch dims

# Mirrors `sdpa_with_flattened_batch()`: flatten leading batch dims so the call
# is a plain 4-D SDPA (mathematically identical to keeping them), optionally
# apply SSMax to the queries first, then SDPA with the default 1/sqrt(head_dim)
# scaling.
tabicl_sdpa <- function(q, k, v, attn_mask = NULL, ssmax_layer = NULL) {
  q_shape <- dim(q)
  qf <- q$reshape(c(-1, utils::tail(q_shape, 3)))
  kf <- k$reshape(c(-1, utils::tail(dim(k), 3)))
  vf <- v$reshape(c(-1, utils::tail(dim(v), 3)))
  if (!is.null(attn_mask)) {
    attn_mask <- attn_mask$reshape(c(-1, utils::tail(dim(attn_mask), 3)))
  }

  if (!is.null(ssmax_layer)) {
    qf <- ssmax_layer(qf, kf$size(-2))
  }

  out <- torch_scaled_dot_product_attention(
    qf,
    kf,
    vf,
    attn_mask = if (is.null(attn_mask)) list() else attn_mask
  )
  out$view(q_shape)
}

# ------------------------------------------------------------------------------
# Multi-head attention

# Packed Q/K/V input projection (matching nn.MultiheadAttention's
# `in_proj_weight` / `in_proj_bias` layout), optional RoPE applied to Q and K,
# optional SSMax on Q, then SDPA and the output projection.
tabicl_mha <- nn_module(
  "tabicl_mha",
  initialize = function(embed_dim, num_heads, ssmax = "none") {
    self$embed_dim <- embed_dim
    self$num_heads <- num_heads
    self$head_dim <- embed_dim %/% num_heads
    self$in_proj_weight <- nn_parameter(torch_empty(3 * embed_dim, embed_dim))
    self$in_proj_bias <- nn_parameter(torch_zeros(3 * embed_dim))
    self$out_proj <- nn_linear(embed_dim, embed_dim)
    self$ssmax_layer <- if (identical(ssmax, "none")) {
      NULL
    } else {
      tabicl_qassmax(num_heads, self$head_dim)
    }
  },
  forward = function(
    query,
    key = NULL,
    value = NULL,
    rope = NULL,
    attn_mask = NULL
  ) {
    if (is.null(key)) {
      key <- query
    }
    if (is.null(value)) {
      value <- query
    }

    e <- self$embed_dim
    nh <- self$num_heads
    hd <- self$head_dim
    batch_shape <- utils::head(dim(query), -2)
    tgt_len <- query$size(-2)
    src_len <- key$size(-2)

    # Split the packed projection into Q/K/V weights and biases.
    wq <- self$in_proj_weight[1:e, ]
    wk <- self$in_proj_weight[(e + 1):(2 * e), ]
    wv <- self$in_proj_weight[(2 * e + 1):(3 * e), ]
    bq <- self$in_proj_bias[1:e]
    bk <- self$in_proj_bias[(e + 1):(2 * e)]
    bv <- self$in_proj_bias[(2 * e + 1):(3 * e)]

    q <- nnf_linear(query, wq, bq)
    k <- nnf_linear(key, wk, bk)
    v <- nnf_linear(value, wv, bv)

    # (..., seq, embed) -> (..., n_heads, seq, head_dim)
    q <- q$view(c(batch_shape, tgt_len, nh, hd))$transpose(-3, -2)
    k <- k$view(c(batch_shape, src_len, nh, hd))$transpose(-3, -2)
    v <- v$view(c(batch_shape, src_len, nh, hd))$transpose(-3, -2)

    if (!is.null(rope)) {
      q <- rope(q)
      k <- rope(k)
    }

    out <- tabicl_sdpa(
      q,
      k,
      v,
      attn_mask = attn_mask,
      ssmax_layer = self$ssmax_layer
    )
    out <- out$transpose(-3, -2)$contiguous()$view(c(batch_shape, tgt_len, e))
    self$out_proj(out)
  }
)
