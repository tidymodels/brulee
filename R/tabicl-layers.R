# Transformer building blocks for TabICL: the pre-norm multi-head attention
# block and the encoder stack. Ported from `MultiheadAttentionBlock` in
# `src/tabicl/_model/layers.py` and `Encoder` in `encoders.py`.
#
# The induced-self-attention block (Set Transformer) used by the column
# embedder is added separately in the column-embedding port.

# LayerNorm with an optional bias. Equivalent to nn.LayerNorm over the last
# dimension; R torch's `nn_layer_norm` has no `bias` argument, but TabICL's
# regressor checkpoint uses bias-free LayerNorms (`bias_free_ln = TRUE`).
tabicl_layer_norm <- nn_module(
  "tabicl_layer_norm",
  initialize = function(dim, eps = 1e-5, bias = TRUE) {
    self$eps <- eps
    self$weight <- nn_parameter(torch_ones(dim))
    if (bias) {
      self$bias <- nn_parameter(torch_zeros(dim))
    } else {
      self$bias <- NULL
    }
  },
  forward = function(x) {
    mean <- x$mean(dim = -1, keepdim = TRUE)
    var <- x$var(dim = -1, unbiased = FALSE, keepdim = TRUE)
    normed <- (x - mean) / torch_sqrt(var + self$eps)
    out <- normed * self$weight
    if (!is.null(self$bias)) {
      out <- out + self$bias
    }
    out
  }
)

# Convert a boolean key-padding mask (TRUE = ignore) into an additive float mask
# broadcastable to the flattened SDPA layout (..., 1, 1, src_len). Used only
# when feature counts vary across tables (the `d` argument); the common
# inference path (d = NULL) passes NULL straight through.
tabicl_key_padding_to_attn_mask <- function(key_padding_mask, dtype) {
  if (is.null(key_padding_mask)) {
    return(NULL)
  }
  neg_inf <- -Inf
  additive <- torch_zeros_like(
    key_padding_mask,
    dtype = dtype
  )$masked_fill(key_padding_mask, neg_inf)
  # (..., src_len) -> (..., 1, 1, src_len)
  additive$unsqueeze(-2)$unsqueeze(-2)
}

# Pre-norm (or post-norm) transformer encoder layer with RoPE / SSMax support.
# Mirrors nn.TransformerEncoderLayer's parameter layout (norm1, norm2, linear1,
# linear2) with the attention replaced by `tabicl_mha`.
tabicl_mha_block <- nn_module(
  "tabicl_mha_block",
  initialize = function(
    d_model,
    nhead,
    dim_feedforward,
    activation = nnf_gelu,
    norm_first = TRUE,
    bias_free_ln = FALSE,
    ssmax = "none"
  ) {
    self$norm_first <- norm_first
    self$attn <- tabicl_mha(d_model, nhead, ssmax = ssmax)
    self$linear1 <- nn_linear(d_model, dim_feedforward)
    self$linear2 <- nn_linear(dim_feedforward, d_model)
    self$norm1 <- tabicl_layer_norm(d_model, bias = !bias_free_ln)
    self$norm2 <- tabicl_layer_norm(d_model, bias = !bias_free_ln)
    self$act <- activation
  },
  ff_block = function(x) {
    self$linear2(self$act(self$linear1(x)))
  },
  forward = function(
    q,
    k = NULL,
    v = NULL,
    key_padding_mask = NULL,
    rope = NULL,
    train_size = NULL
  ) {
    if (is.null(train_size)) {
      if (is.null(k)) {
        k <- q
      }
      if (is.null(v)) {
        v <- k
      }
    } else {
      # ICL path: queries are the full sequence, keys/values the first
      # `train_size` positions (context-only attention).
      k <- v <- q$narrow(dim = -2, start = 1, length = train_size)
    }

    attn_mask <- tabicl_key_padding_to_attn_mask(key_padding_mask, q$dtype)

    if (self$norm_first) {
      q_n <- self$norm1(q)
      if (is.null(train_size)) {
        k_n <- self$norm1(k)
        v_n <- self$norm1(v)
      } else {
        k_n <- v_n <- q_n$narrow(dim = -2, start = 1, length = train_size)
      }
      attn <- self$attn(q_n, k_n, v_n, rope = rope, attn_mask = attn_mask)
      x <- q + attn
      x <- x + self$ff_block(self$norm2(x))
    } else {
      attn <- self$attn(q, k, v, rope = rope, attn_mask = attn_mask)
      x <- self$norm1(q + attn)
      x <- self$norm2(x + self$ff_block(x))
    }
    x
  }
)

# Linear layer that passes "skip" rows through unchanged. A row whose inputs are
# all `skip_value` (a padded / empty feature slot) produces `skip_value` outputs
# instead of the linear projection. Mirrors `SkippableLinear` in layers.py;
# parameters are exposed as `weight` / `bias` to match the checkpoint keys.
tabicl_skippable_linear <- nn_module(
  "tabicl_skippable_linear",
  initialize = function(in_features, out_features, skip_value = -100) {
    self$weight <- nn_parameter(torch_empty(out_features, in_features))
    self$bias <- nn_parameter(torch_zeros(out_features))
    self$skip_value <- skip_value
  },
  forward = function(x) {
    out <- nnf_linear(x, self$weight, self$bias)
    skip <- (x == self$skip_value)$all(dim = -1, keepdim = TRUE)
    torch_where(
      skip,
      torch_scalar_tensor(
        self$skip_value,
        dtype = out$dtype,
        device = out$device
      ),
      out
    )
  }
)

# One-hot encoding followed by a linear projection, in one module. Mirrors
# `OneHotAndLinear` (an nn.Linear subclass), so parameters are `weight` / `bias`.
# Used as the classification target encoder. R torch's `nnf_one_hot` is 1-indexed,
# so the one-hot is built by comparison to keep 0-based class indices.
tabicl_onehot_linear <- nn_module(
  "tabicl_onehot_linear",
  initialize = function(num_classes, embed_dim) {
    self$num_classes <- num_classes
    self$weight <- nn_parameter(torch_empty(embed_dim, num_classes))
    self$bias <- nn_parameter(torch_zeros(embed_dim))
  },
  forward = function(src) {
    classes <- torch_arange(
      start = 0,
      end = self$num_classes - 1,
      dtype = torch_long(),
      device = src$device
    )
    one_hot <- (src$to(dtype = torch_long())$unsqueeze(-1) == classes)$to(
      dtype = self$weight$dtype
    )
    nnf_linear(one_hot, self$weight, self$bias)
  }
)

# Induced self-attention block (Set Transformer): inducing points attend to the
# input (optionally only its first `train_size` positions), then the input
# attends back to that bottleneck. Mirrors `InducedSelfAttentionBlock`.
#
# Skip handling: a batch slot whose entire (seq, d_model) slice equals
# `skip_value` is restored to `skip_value` on output. Attention treats batch
# dims independently, so running induced attention over all slots and then
# masking the skip slots is identical to the reference's masked-subset compute.
tabicl_isab <- nn_module(
  "tabicl_isab",
  initialize = function(
    d_model,
    nhead,
    dim_feedforward,
    num_inds,
    activation = nnf_gelu,
    norm_first = TRUE,
    bias_free_ln = FALSE,
    ssmax = "none",
    skip_value = -100
  ) {
    self$num_inds <- num_inds
    self$skip_value <- skip_value
    self$multihead_attn1 <- tabicl_mha_block(
      d_model,
      nhead,
      dim_feedforward,
      activation = activation,
      norm_first = norm_first,
      bias_free_ln = bias_free_ln,
      ssmax = ssmax
    )
    self$multihead_attn2 <- tabicl_mha_block(
      d_model,
      nhead,
      dim_feedforward,
      activation = activation,
      norm_first = norm_first,
      bias_free_ln = bias_free_ln,
      ssmax = "none"
    )
    self$ind_vectors <- nn_parameter(torch_empty(num_inds, d_model))
  },
  induced_attention = function(src, train_size = NULL) {
    sizes <- dim(src)
    d_model <- sizes[length(sizes)]
    batch_shape <- utils::head(sizes, -2)
    ind <- self$ind_vectors$expand(c(batch_shape, self$num_inds, d_model))
    if (is.null(train_size)) {
      ctx <- src
    } else {
      ctx <- src$narrow(dim = -2, start = 1, length = train_size)
    }
    hidden <- self$multihead_attn1(ind, ctx, ctx)
    self$multihead_attn2(src, hidden, hidden)
  },
  forward = function(src, train_size = NULL) {
    out <- self$induced_attention(src, train_size)
    skip <- (src == self$skip_value)$all(dim = -1)$all(dim = -1)
    skip <- skip$unsqueeze(-1)$unsqueeze(-1)
    torch_where(
      skip,
      torch_scalar_tensor(
        self$skip_value,
        dtype = out$dtype,
        device = out$device
      ),
      out
    )
  }
)

# A stack of induced self-attention blocks. Mirrors `SetTransformer`.
tabicl_set_transformer <- nn_module(
  "tabicl_set_transformer",
  initialize = function(
    num_blocks,
    d_model,
    nhead,
    dim_feedforward,
    num_inds,
    activation = nnf_gelu,
    norm_first = TRUE,
    bias_free_ln = FALSE,
    ssmax = "none"
  ) {
    self$blocks <- purrr::map(seq_len(num_blocks), \(i) {
      tabicl_isab(
        d_model = d_model,
        nhead = nhead,
        dim_feedforward = dim_feedforward,
        num_inds = num_inds,
        activation = activation,
        norm_first = norm_first,
        bias_free_ln = bias_free_ln,
        ssmax = ssmax
      )
    }) |>
      nn_module_list()
  },
  forward = function(src, train_size = NULL) {
    out <- src
    for (i in seq_along(self$blocks)) {
      out <- self$blocks[[i]](out, train_size = train_size)
    }
    out
  }
)

# A stack of `tabicl_mha_block`s plus an optional shared RoPE module. Mirrors
# `Encoder`: `blocks` (a module list) and `rope`. The plain `forward` runs each
# block in sequence; the row interactor drives the blocks directly for its
# CLS-token aggregation, so it reaches into `self$blocks` / `self$rope`.
tabicl_encoder <- nn_module(
  "tabicl_encoder",
  initialize = function(
    num_blocks,
    d_model,
    nhead,
    dim_feedforward,
    activation = nnf_gelu,
    norm_first = TRUE,
    bias_free_ln = FALSE,
    ssmax = "none",
    use_rope = FALSE,
    rope_base = 100000
  ) {
    self$blocks <- purrr::map(seq_len(num_blocks), \(i) {
      tabicl_mha_block(
        d_model = d_model,
        nhead = nhead,
        dim_feedforward = dim_feedforward,
        activation = activation,
        norm_first = norm_first,
        bias_free_ln = bias_free_ln,
        ssmax = ssmax
      )
    }) |>
      nn_module_list()
    if (use_rope) {
      self$rope <- tabicl_rope(dim = d_model %/% nhead, theta = rope_base)
    } else {
      self$rope <- NULL
    }
  },
  forward = function(src, train_size = NULL) {
    out <- src
    for (i in seq_along(self$blocks)) {
      out <- self$blocks[[i]](
        q = out,
        train_size = train_size,
        rope = self$rope
      )
    }
    out
  }
)
