# TODO:
# - serialize model file/bundle method
# - hardhat as_tibble( wide = TRUE) method

# ------------------------------------------------------------------------------
# utilities

nan_to_num <- function(x, nan = 0.0) {
  torch_where(
    torch_isnan(x),
    torch_tensor(nan, dtype = x$dtype, device = x$device),
    x
  )
}

torch_nanmean <- function(x, dim = -1L, keepdim = FALSE) {
  mask <- !torch_isnan(x)
  x_zeroed <- nan_to_num(x, nan = 0.0)
  total <- x_zeroed$sum(dim = dim, keepdim = keepdim)
  count <- mask$to(dtype = torch_float32())$sum(dim = dim, keepdim = keepdim)
  count <- torch_clamp(count, min = 1.0)
  total / count
}

torch_nansum <- function(x, dim = -1L, keepdim = FALSE) {
  x_zeroed <- nan_to_num(x, nan = 0.0)
  x_zeroed$sum(dim = dim, keepdim = keepdim)
}

left_pad_and_stack <- function(tensors) {
  max_len <- max(purrr::map_int(tensors, \(t) t$size(1)))
  padded <- lapply(tensors, function(t) {
    len <- t$size(1)
    if (len < max_len) {
      pad <- torch_full(max_len - len, fill_value = NaN)
      torch_cat(list(pad, t))
    } else {
      t
    }
  })
  torch_stack(padded)
}

left_pad_and_cat_2D <- function(tensor_list) {
  max_len <- max(purrr::map_int(tensor_list, \(t) t$size(-1)))
  padded <- lapply(tensor_list, function(t) {
    n_rows <- t$size(1)
    len <- t$size(-1)
    if (len < max_len) {
      pad <- torch_full(
        c(n_rows, max_len - len),
        fill_value = NaN,
        dtype = t$dtype,
        device = t$device
      )
      torch_cat(list(pad, t), dim = 2)
    } else {
      t
    }
  })
  torch_cat(padded, dim = 1)
}

# ------------------------------------------------------------------------------
# Preprocessing

# Instance normalization with optional arcsinh transform
chronos2_instance_norm <- nn_module(
  "Chronos2InstanceNorm",
  initialize = function(eps = 1e-5, use_arcsinh = FALSE) {
    self$eps <- eps
    self$use_arcsinh <- use_arcsinh
  },
  forward = function(x, loc_scale = NULL) {
    x <- x$to(dtype = torch_float32())
    if (is.null(loc_scale)) {
      loc <- torch_nanmean(x, dim = -1L, keepdim = TRUE)
      loc <- nan_to_num(loc, nan = 0.0)
      scale <- nan_to_num(
        torch_nanmean((x - loc)$pow(2), dim = -1L, keepdim = TRUE)$sqrt(),
        nan = 1.0
      )
      scale <- torch_where(scale == 0, self$eps, scale)
    } else {
      loc <- loc_scale[[1]]
      scale <- loc_scale[[2]]
    }

    scaled_x <- (x - loc) / scale
    if (self$use_arcsinh) {
      scaled_x <- torch_asinh(scaled_x)
    }
    list(scaled_x, list(loc, scale))
  },
  inverse = function(x, loc_scale) {
    x <- x$to(dtype = torch_float32())
    loc <- loc_scale[[1]]
    scale <- loc_scale[[2]]
    if (self$use_arcsinh) {
      x <- torch_sinh(x)
    }
    x * scale + loc
  }
)

# Patch: split time series into patches, left-pad with NaN if needed
chronos2_patch <- nn_module(
  "Chronos2Patch",
  initialize = function(patch_size, patch_stride) {
    self$patch_size <- patch_size
    self$patch_stride <- patch_stride
  },
  forward = function(x) {
    len <- x$size(-1)
    if (len %% self$patch_size != 0) {
      padding_size <- self$patch_size - (len %% self$patch_size)
      pad_shape <- c(x$size(1), padding_size)
      padding <- torch_full(
        pad_shape,
        fill_value = NaN,
        dtype = x$dtype,
        device = x$device
      )
      x <- torch_cat(list(padding, x), dim = -1)
    }
    # unfold uses 0-indexed dims in R torch; dim=1 is the length axis for [batch, length]
    x$unfold(dimension = 1, size = self$patch_size, step = self$patch_stride)
  }
)

# Prepare patched context: normalize, patch, build time encoding, concatenate
prepare_patched_context <- function(
  context,
  context_mask,
  instance_norm,
  patch,
  config
) {
  batch_size <- context$size(1)

  # Truncate to context_length
  if (context$size(2) > config$context_length) {
    start <- context$size(2) - config$context_length + 1
    context <- context[, start:context$size(2)]
    context_mask <- context_mask[, start:context_mask$size(2)]
  }

  # Instance normalization
  norm_result <- instance_norm(context)
  context_scaled <- norm_result[[1]]
  loc_scale <- norm_result[[2]]

  context_mask <- context_mask$to(dtype = context_scaled$dtype)

  # Patching
  patched_context <- patch(context_scaled)
  patched_mask <- nan_to_num(patch(context_mask), nan = 0.0)
  patched_context <- torch_where(patched_mask > 0.0, patched_context, 0.0)

  # attention_mask: 1 if at least one item in the patch is observed
  attention_mask <- patched_mask$sum(dim = -1) > 0

  num_patches <- attention_mask$size(2)

  # Time encoding: arange(-final_context_length, 0) / time_encoding_scale
  final_context_length <- num_patches * config$input_patch_size
  time_enc <- torch_arange(
    0,
    final_context_length - 1,
    dtype = torch_float32(),
    device = context$device
  ) -
    final_context_length

  # Reshape to [batch, num_patches, patch_size]
  time_enc <- time_enc$view(c(
    1,
    num_patches,
    config$input_patch_size
  ))$expand(c(batch_size, -1, -1))$div(config$time_encoding_scale)$to(
    dtype = context_scaled$dtype
  )

  # Concatenate [time_enc, patches, mask] along last dim -> [batch, num_patches, patch_size*3]
  patched_context <- torch_cat(
    list(time_enc, patched_context, patched_mask),
    dim = -1
  )

  list(
    patched_context = patched_context,
    attention_mask = attention_mask,
    loc_scale = loc_scale,
    num_patches = num_patches
  )
}

# Prepare future patches with optional covariate values
prepare_patched_future <- function(
  num_output_patches,
  config,
  batch_size,
  device,
  dtype,
  future_covariates = NULL,
  future_covariates_mask = NULL,
  loc_scale = NULL,
  instance_norm = NULL
) {
  output_patch_size <- config$output_patch_size
  final_future_length <- num_output_patches * output_patch_size

  if (!is.null(future_covariates)) {
    # Normalize future covariates with the same loc/scale as context
    fc_normed <- instance_norm(future_covariates, loc_scale)
    future_covariates <- fc_normed[[1]]$to(dtype = dtype)

    # Build mask from non-NaN positions if not provided
    if (is.null(future_covariates_mask)) {
      future_covariates_mask <- (!torch_isnan(future_covariates))$to(
        dtype = dtype
      )
    } else {
      future_covariates_mask <- future_covariates_mask$to(dtype = dtype)
    }

    # Zero out masked positions
    future_covariates <- torch_where(
      future_covariates_mask > 0.0,
      future_covariates,
      0.0
    )

    # Pad to full output length if needed
    fc_len <- future_covariates$size(-1)
    if (final_future_length > fc_len) {
      pad_size <- final_future_length - fc_len
      pad_shape <- c(batch_size, pad_size)
      future_covariates <- torch_cat(
        list(
          future_covariates,
          torch_zeros(pad_shape, device = device, dtype = dtype)
        ),
        dim = -1
      )
      future_covariates_mask <- torch_cat(
        list(
          future_covariates_mask,
          torch_zeros(pad_shape, device = device, dtype = dtype)
        ),
        dim = -1
      )
    }

    # Reshape to patches: (batch, final_future_length) -> (batch, num_output_patches, output_patch_size)
    patched_future_covariates <- future_covariates[,
      1:final_future_length
    ]$view(c(batch_size, num_output_patches, output_patch_size))
    patched_future_covariates_mask <- future_covariates_mask[,
      1:final_future_length
    ]$view(c(batch_size, num_output_patches, output_patch_size))
  } else {
    patched_future_covariates <- torch_zeros(
      batch_size,
      num_output_patches,
      output_patch_size,
      device = device,
      dtype = dtype
    )
    patched_future_covariates_mask <- torch_zeros(
      batch_size,
      num_output_patches,
      output_patch_size,
      device = device,
      dtype = dtype
    )
  }

  # Future time encoding: arange(0, final_future_length) / time_encoding_scale
  future_time_enc <- torch_arange(
    0,
    final_future_length - 1,
    dtype = torch_float32(),
    device = device
  )
  future_time_enc <- future_time_enc$view(c(
    1,
    num_output_patches,
    output_patch_size
  ))$expand(c(batch_size, -1, -1))$div(config$time_encoding_scale)$to(
    dtype = dtype
  )

  # Concatenate [time_enc, covariates, mask]
  patched_future <- torch_cat(
    list(
      future_time_enc,
      patched_future_covariates,
      patched_future_covariates_mask
    ),
    dim = -1
  )

  list(
    patched_future = patched_future,
    patched_future_covariates_mask = patched_future_covariates_mask
  )
}

# ------------------------------------------------------------------------------
# Layers

# T5-style RMS LayerNorm (no bias, no mean subtraction)
chronos2_layer_norm <- nn_module(
  "Chronos2LayerNorm",
  initialize = function(hidden_size, eps = 1e-6) {
    self$weight <- nn_parameter(torch_ones(hidden_size))
    self$eps <- eps
  },
  forward = function(x) {
    variance <- x$to(dtype = torch_float32())$pow(2)$mean(-1, keepdim = TRUE)
    x <- x * torch_rsqrt(variance + self$eps)
    self$weight * x
  }
)

rotate_half <- function(x) {
  half_dim <- x$size(-1) %/% 2
  x1 <- x[,,, 1:half_dim]
  x2 <- x[,,, (half_dim + 1):x$size(-1)]
  torch_cat(list(-x2, x1), dim = -1)
}

apply_rotary_pos_emb <- function(q, k, cos_val, sin_val) {
  # cos_val, sin_val: [batch, seq_len, head_dim] -> unsqueeze for heads
  cos_val <- cos_val$unsqueeze(2)
  sin_val <- sin_val$unsqueeze(2)
  q_embed <- (q * cos_val) + (rotate_half(q) * sin_val)
  k_embed <- (k * cos_val) + (rotate_half(k) * sin_val)
  list(q_embed, k_embed)
}

# Rotary Position Embeddings
rope <- nn_module(
  "RoPE",
  initialize = function(dim, base = 10000) {
    self$dim <- dim
    self$base <- base
    inv_freq <- 1.0 /
      (base^(torch_arange(0, dim - 1, step = 2, dtype = torch_float32()) / dim))
    self$register_buffer("inv_freq", inv_freq)
  },
  forward = function(x, position_ids) {
    # x: [batch, n_heads, seq_len, head_size]
    # position_ids: [batch, seq_len]
    batch_size <- position_ids$size(1)
    half_dim <- self$dim %/% 2

    # inv_freq: [half_dim] -> reshape to [1, half_dim, 1] -> expand to [batch, half_dim, 1]
    inv_freq_expanded <- self$inv_freq$view(c(1, half_dim, 1))$expand(c(
      batch_size,
      half_dim,
      1
    ))$to(dtype = torch_float32())
    # position_ids: [batch, seq_len] -> [batch, 1, seq_len]
    position_ids_expanded <- position_ids$unsqueeze(2)$to(
      dtype = torch_float32()
    )

    # freqs: [batch, half_dim, 1] @ [batch, 1, seq_len] -> [batch, half_dim, seq_len]
    # then transpose -> [batch, seq_len, half_dim]
    freqs <- torch_matmul(inv_freq_expanded, position_ids_expanded)$transpose(
      2,
      3
    )
    emb <- torch_cat(list(freqs, freqs), dim = -1)
    cos_val <- emb$cos()$to(dtype = x$dtype)
    sin_val <- emb$sin()$to(dtype = x$dtype)
    list(cos_val, sin_val)
  }
)

# MLP: wi -> act -> dropout -> wo
chronos2_mlp <- nn_module(
  "Chronos2MLP",
  initialize = function(config) {
    self$wi <- nn_linear(config$d_model, config$d_ff, bias = FALSE)
    self$wo <- nn_linear(config$d_ff, config$d_model, bias = FALSE)
    self$dropout <- nn_dropout(p = config$dropout_rate)
  },
  forward = function(x) {
    x <- self$wi(x)
    x <- nnf_relu(x)
    x <- self$dropout(x)
    x <- self$wo(x)
    x
  }
)

# FeedForward: layer_norm -> mlp -> dropout + residual
chronos2_feed_forward <- nn_module(
  "Chronos2FeedForward",
  initialize = function(config) {
    self$mlp <- chronos2_mlp(config)
    self$layer_norm <- chronos2_layer_norm(
      config$d_model,
      eps = config$layer_norm_epsilon
    )
    self$dropout <- nn_dropout(p = config$dropout_rate)
  },
  forward = function(x) {
    forwarded <- self$layer_norm(x)
    forwarded <- self$mlp(forwarded)
    x + self$dropout(forwarded)
  }
)

# ResidualBlock: hidden_layer -> act -> output_layer + residual_layer, optional layer_norm
chronos2_residual_block <- nn_module(
  "Chronos2ResidualBlock",
  initialize = function(
    in_dim,
    h_dim,
    out_dim,
    dropout_p = 0.0,
    use_layer_norm = FALSE
  ) {
    self$hidden_layer <- nn_linear(in_dim, h_dim)
    self$output_layer <- nn_linear(h_dim, out_dim)
    self$residual_layer <- nn_linear(in_dim, out_dim)
    self$dropout <- nn_dropout(p = dropout_p)
    self$use_layer_norm <- use_layer_norm
    if (use_layer_norm) {
      self$layer_norm <- chronos2_layer_norm(out_dim)
    }
  },
  forward = function(x) {
    hid <- nnf_relu(self$hidden_layer(x))
    out <- self$dropout(self$output_layer(hid))
    res <- self$residual_layer(x)
    out <- out + res
    if (self$use_layer_norm) {
      out <- self$layer_norm(out)
    }
    out
  }
)

# ------------------------------------------------------------------------------
# Attention

# Multi-Head Attention (no scaling — critical for Chronos-2)
chronos2_mha <- nn_module(
  "Chronos2MHA",
  initialize = function(config, use_rope = TRUE) {
    self$d_model <- config$d_model
    self$kv_proj_dim <- config$d_kv
    self$n_heads <- config$num_heads
    self$dropout_rate <- config$dropout_rate
    self$inner_dim <- self$n_heads * self$kv_proj_dim
    self$use_rope <- use_rope

    self$q <- nn_linear(self$d_model, self$inner_dim, bias = FALSE)
    self$k <- nn_linear(self$d_model, self$inner_dim, bias = FALSE)
    self$v <- nn_linear(self$d_model, self$inner_dim, bias = FALSE)
    self$o <- nn_linear(self$inner_dim, self$d_model, bias = FALSE)

    if (use_rope) {
      self$rope_embed <- rope(dim = self$kv_proj_dim, base = config$rope_theta)
    }
  },
  forward = function(
    hidden_states,
    mask,
    encoder_states = NULL,
    position_ids = NULL
  ) {
    batch_size <- hidden_states$size(1)
    seq_length <- hidden_states$size(2)

    # Project and reshape to [batch, n_heads, seq_len, kv_proj_dim]
    query_states <- self$q(hidden_states)$view(c(
      batch_size,
      seq_length,
      self$n_heads,
      self$kv_proj_dim
    ))$permute(c(1, 3, 2, 4))

    is_cross_attention <- !is.null(encoder_states)

    if (is_cross_attention) {
      kv_seq_length <- encoder_states$size(2)
      key_states <- self$k(encoder_states)$view(c(
        batch_size,
        kv_seq_length,
        self$n_heads,
        self$kv_proj_dim
      ))$permute(c(1, 3, 2, 4))
      value_states <- self$v(encoder_states)$view(c(
        batch_size,
        kv_seq_length,
        self$n_heads,
        self$kv_proj_dim
      ))$permute(c(1, 3, 2, 4))
    } else {
      key_states <- self$k(hidden_states)$view(c(
        batch_size,
        seq_length,
        self$n_heads,
        self$kv_proj_dim
      ))$permute(c(1, 3, 2, 4))
      value_states <- self$v(hidden_states)$view(c(
        batch_size,
        seq_length,
        self$n_heads,
        self$kv_proj_dim
      ))$permute(c(1, 3, 2, 4))

      if (self$use_rope && !is.null(position_ids)) {
        rope_out <- self$rope_embed(value_states, position_ids)
        cos_val <- rope_out[[1]]
        sin_val <- rope_out[[2]]
        rotated <- apply_rotary_pos_emb(
          query_states,
          key_states,
          cos_val,
          sin_val
        )
        query_states <- rotated[[1]]
        key_states <- rotated[[2]]
      }
    }

    # Eager attention: NO scaling (scores = Q @ K^T, not Q @ K^T / sqrt(d_k))
    scores <- torch_matmul(query_states, key_states$transpose(3, 4))
    scores <- scores + mask
    attn_weights <- nnf_softmax(
      scores$to(dtype = torch_float32()),
      dim = -1
    )$to(dtype = scores$dtype)
    if (self$training) {
      attn_weights <- nnf_dropout(attn_weights, p = self$dropout_rate)
    }
    attn_output <- torch_matmul(attn_weights, value_states)

    # Reshape back: [batch, n_heads, seq_len, kv_proj_dim] -> [batch, seq_len, inner_dim]
    attn_output <- attn_output$permute(c(1, 3, 2, 4))$contiguous()$view(c(
      batch_size,
      seq_length,
      self$inner_dim
    ))
    attn_output <- self$o(attn_output)

    attn_output
  }
)

# Time Self-Attention: layer_norm -> MHA(with RoPE) -> dropout + residual
chronos2_time_self_attention <- nn_module(
  "Chronos2TimeSelfAttention",
  initialize = function(config) {
    self$self_attention <- chronos2_mha(config, use_rope = TRUE)
    self$layer_norm <- chronos2_layer_norm(
      config$d_model,
      eps = config$layer_norm_epsilon
    )
    self$dropout <- nn_dropout(p = config$dropout_rate)
  },
  forward = function(hidden_states, attention_mask, position_ids) {
    normed <- self$layer_norm(hidden_states)
    attn_output <- self$self_attention(
      normed,
      mask = attention_mask,
      position_ids = position_ids
    )
    hidden_states + self$dropout(attn_output)
  }
)

# Group Self-Attention: transpose batch<->time, layer_norm -> MHA(no RoPE) -> dropout + residual, transpose back
chronos2_group_self_attention <- nn_module(
  "Chronos2GroupSelfAttention",
  initialize = function(config) {
    self$self_attention <- chronos2_mha(config, use_rope = FALSE)
    self$layer_norm <- chronos2_layer_norm(
      config$d_model,
      eps = config$layer_norm_epsilon
    )
    self$dropout <- nn_dropout(p = config$dropout_rate)
  },
  forward = function(hidden_states, attention_mask) {
    # hidden_states: [batch, time, d] -> [time, batch, d]
    hidden_states <- hidden_states$permute(c(2, 1, 3))
    normed <- self$layer_norm(hidden_states)
    attn_output <- self$self_attention(normed, mask = attention_mask)
    hidden_states <- hidden_states + self$dropout(attn_output)
    # [time, batch, d] -> [batch, time, d]
    hidden_states$permute(c(2, 1, 3))
  }
)

# ------------------------------------------------------------------------------
# Encoder

# Encoder Block: TimeSelfAttention -> GroupSelfAttention -> FeedForward
chronos2_encoder_block <- nn_module(
  "Chronos2EncoderBlock",
  initialize = function(config) {
    self$time_self_attn <- chronos2_time_self_attention(config)
    self$group_self_attn <- chronos2_group_self_attention(config)
    self$feed_forward <- chronos2_feed_forward(config)
  },
  forward = function(
    hidden_states,
    position_ids,
    attention_mask,
    group_time_mask
  ) {
    hidden_states <- self$time_self_attn(
      hidden_states,
      attention_mask,
      position_ids
    )
    hidden_states <- self$group_self_attn(hidden_states, group_time_mask)
    hidden_states <- self$feed_forward(hidden_states)
    hidden_states
  }
)

# Full Encoder: N blocks + final layer_norm + dropout
chronos2_encoder <- nn_module(
  "Chronos2Encoder",
  initialize = function(config) {
    self$blocks <- nn_module_list(lapply(
      seq_len(config$num_layers),
      function(i) {
        chronos2_encoder_block(config)
      }
    ))
    self$final_layer_norm <- chronos2_layer_norm(
      config$d_model,
      eps = config$layer_norm_epsilon
    )
    self$dropout <- nn_dropout(p = config$dropout_rate)
  },
  forward = function(
    inputs_embeds,
    group_ids,
    attention_mask = NULL,
    position_ids = NULL
  ) {
    batch_size <- inputs_embeds$size(1)
    seq_length <- inputs_embeds$size(2)

    if (is.null(position_ids)) {
      position_ids <- torch_arange(
        0,
        seq_length - 1,
        dtype = torch_long(),
        device = inputs_embeds$device
      )$unsqueeze(1)
    }
    if (is.null(attention_mask)) {
      attention_mask <- torch_ones(
        batch_size,
        seq_length,
        device = inputs_embeds$device,
        dtype = inputs_embeds$dtype
      )
    }

    # Expand time attention mask: [batch, seq] -> [batch, 1, 1, seq] additive mask
    extended_attention_mask <- attention_mask$unsqueeze(2)$unsqueeze(2)
    extended_attention_mask <- extended_attention_mask$to(
      dtype = inputs_embeds$dtype
    )
    extended_attention_mask <- (1.0 - extended_attention_mask) *
      torch_finfo(inputs_embeds$dtype)$min

    # Construct group_time_mask
    # group_mask: [batch, batch] — TRUE if same group
    group_mask <- (group_ids$unsqueeze(2) == group_ids$unsqueeze(1))$to(
      dtype = inputs_embeds$dtype
    )
    # group_time_mask = einsum("qb, bt -> qbt", group_mask, attention_mask)
    group_time_mask <- torch_einsum(
      "qb, bt -> qbt",
      list(group_mask, attention_mask)
    )
    # Reshape to [time, 1, query_batch, key_batch]
    group_time_mask <- group_time_mask$permute(c(3, 1, 2))$unsqueeze(2)
    group_time_mask <- (1.0 - group_time_mask) *
      torch_finfo(inputs_embeds$dtype)$min

    hidden_states <- self$dropout(inputs_embeds)

    for (i in seq_along(self$blocks)) {
      hidden_states <- self$blocks[[i]](
        hidden_states,
        position_ids = position_ids,
        attention_mask = extended_attention_mask,
        group_time_mask = group_time_mask
      )
    }

    hidden_states <- self$final_layer_norm(hidden_states)
    hidden_states <- self$dropout(hidden_states)
    hidden_states
  }
)

# ------------------------------------------------------------------------------
# Modules

chronos2_model <- nn_module(
  "Chronos2Model",
  initialize = function(config) {
    self$config <- config
    self$model_dim <- config$d_model

    # Shared embedding (vocab_size = 2 for [PAD] and [REG])
    self$shared <- nn_embedding(config$vocab_size, config$d_model)

    # Input patch embedding: patch_size * 3 (time_enc + values + mask) -> d_model
    self$input_patch_embedding <- chronos2_residual_block(
      in_dim = config$input_patch_size * 3,
      h_dim = config$d_ff,
      out_dim = config$d_model,
      dropout_p = config$dropout_rate
    )

    # Patching layer
    self$patch <- chronos2_patch(
      patch_size = config$input_patch_size,
      patch_stride = config$input_patch_stride
    )

    # Instance normalization
    self$instance_norm <- chronos2_instance_norm(
      use_arcsinh = config$use_arcsinh
    )

    # Encoder
    self$encoder <- chronos2_encoder(config)

    # Quantiles buffer
    quantiles <- torch_tensor(config$quantiles, dtype = torch_float32())
    self$register_buffer("quantiles", quantiles)
    self$num_quantiles <- length(config$quantiles)

    # Output patch embedding: d_model -> num_quantiles * output_patch_size
    self$output_patch_embedding <- chronos2_residual_block(
      in_dim = config$d_model,
      h_dim = config$d_ff,
      out_dim = self$num_quantiles * config$output_patch_size,
      dropout_p = config$dropout_rate
    )
  },

  encode = function(
    context,
    num_output_patches = 1L,
    group_ids = NULL,
    future_covariates = NULL,
    future_covariates_mask = NULL
  ) {
    batch_size <- context$size(1)

    # Create context mask from NaN values
    context_mask <- (!torch_isnan(context))$to(dtype = torch_float32())

    # Prepare patched context
    ctx <- prepare_patched_context(
      context,
      context_mask,
      self$instance_norm,
      self$patch,
      self$config
    )
    patched_context <- ctx$patched_context
    attention_mask <- ctx$attention_mask
    loc_scale <- ctx$loc_scale
    num_context_patches <- ctx$num_patches

    # Embed context patches
    input_embeds <- self$input_patch_embedding(patched_context)

    # Append [REG] token embedding
    if (self$config$use_reg_token) {
      reg_input_ids <- torch_full(
        c(batch_size, 1L),
        fill_value = self$config$reg_token_id,
        dtype = torch_long(),
        device = input_embeds$device
      )
      reg_embeds <- self$shared(reg_input_ids)
      input_embeds <- torch_cat(list(input_embeds, reg_embeds), dim = 2)
      attention_mask <- torch_cat(
        list(
          attention_mask$to(dtype = input_embeds$dtype),
          torch_ones(
            batch_size,
            1L,
            device = input_embeds$device,
            dtype = input_embeds$dtype
          )
        ),
        dim = 2
      )
    }

    # Prepare future patches (with covariates if provided)
    future_result <- prepare_patched_future(
      num_output_patches,
      self$config,
      batch_size,
      input_embeds$device,
      input_embeds$dtype,
      future_covariates = future_covariates,
      future_covariates_mask = future_covariates_mask,
      loc_scale = loc_scale,
      instance_norm = self$instance_norm
    )
    patched_future <- future_result$patched_future
    patched_future_covariates_mask <- future_result$patched_future_covariates_mask

    # Embed future patches
    future_embeds <- self$input_patch_embedding(patched_future)

    # Concatenate context and future embeddings and masks
    input_embeds <- torch_cat(list(input_embeds, future_embeds), dim = 2)
    future_attention_mask <- torch_ones(
      batch_size,
      num_output_patches,
      dtype = input_embeds$dtype,
      device = input_embeds$device
    )
    attention_mask <- torch_cat(
      list(attention_mask, future_attention_mask),
      dim = 2
    )

    # Group IDs: default to each item independent
    if (is.null(group_ids)) {
      group_ids <- torch_arange(
        0,
        batch_size - 1,
        dtype = torch_long(),
        device = input_embeds$device
      )
    }

    # Run encoder
    hidden_states <- self$encoder(
      inputs_embeds = input_embeds,
      group_ids = group_ids,
      attention_mask = attention_mask
    )

    list(
      hidden_states = hidden_states,
      loc_scale = loc_scale,
      patched_future_covariates_mask = patched_future_covariates_mask,
      num_context_patches = num_context_patches,
      num_output_patches = num_output_patches
    )
  },

  forward = function(
    context,
    num_output_patches = 1L,
    group_ids = NULL,
    future_covariates = NULL,
    future_covariates_mask = NULL
  ) {
    batch_size <- context$size(1)

    enc <- self$encode(
      context,
      num_output_patches,
      group_ids = group_ids,
      future_covariates = future_covariates,
      future_covariates_mask = future_covariates_mask
    )
    hidden_states <- enc$hidden_states
    loc_scale <- enc$loc_scale
    num_output_patches_actual <- enc$num_output_patches

    # Slice last num_output_patches hidden states for output
    total_seq <- hidden_states$size(2)
    start_idx <- total_seq - num_output_patches_actual + 1
    forecast_embeds <- hidden_states[, start_idx:total_seq, ]

    # Project to quantile predictions
    quantile_preds <- self$output_patch_embedding(forecast_embeds)
    # Shape: [batch, num_output_patches, num_quantiles * output_patch_size]
    # Rearrange to [batch, num_quantiles, num_output_patches * output_patch_size]
    output_patch_size <- self$config$output_patch_size
    quantile_preds <- quantile_preds$view(c(
      batch_size,
      num_output_patches_actual,
      self$num_quantiles,
      output_patch_size
    ))
    quantile_preds <- quantile_preds$permute(c(1, 3, 2, 4))$contiguous()$view(c(
      batch_size,
      self$num_quantiles,
      num_output_patches_actual * output_patch_size
    ))

    # Inverse instance normalization to unscale predictions
    horizon <- num_output_patches_actual * output_patch_size
    # Reshape for inverse: [batch, num_quantiles * horizon] -> apply inverse -> reshape back
    quantile_preds_flat <- quantile_preds$view(c(
      batch_size,
      self$num_quantiles * horizon
    ))
    quantile_preds_flat <- self$instance_norm$inverse(
      quantile_preds_flat,
      loc_scale
    )
    quantile_preds <- quantile_preds_flat$view(c(
      batch_size,
      self$num_quantiles,
      horizon
    ))

    quantile_preds
  }
)

# ------------------------------------------------------------------------------
# Parse the config file that comes with the model file

# The `config.json` is downloaded alongside `model.safetensors` from HuggingFace.
# Users never specify these values — they're parsed internally by
# `chronos2_parse_config()`. The config contains:
#
# - Architecture shape parameters (`d_model`, `num_heads`, etc.) needed to
#   construct the nn_modules that weights slot into
# - Training artifacts like `dropout_rate` which are dead at inference
#   (`model$eval()` disables dropout)

chronos2_parse_config <- function(path) {
  cfg <- jsonlite::fromJSON(path)

  list(
    d_model = cfg$d_model,
    d_ff = cfg$d_ff,
    d_kv = cfg$d_kv,
    num_heads = cfg$num_heads,
    num_layers = cfg$num_layers,
    dropout_rate = cfg$dropout_rate,
    layer_norm_epsilon = cfg$layer_norm_epsilon,
    rope_theta = cfg$rope_theta,
    vocab_size = cfg$vocab_size,
    pad_token_id = cfg$pad_token_id,
    reg_token_id = cfg$reg_token_id,
    context_length = cfg$chronos_config$context_length,
    input_patch_size = cfg$chronos_config$input_patch_size,
    input_patch_stride = cfg$chronos_config$input_patch_stride,
    output_patch_size = cfg$chronos_config$output_patch_size,
    max_output_patches = cfg$chronos_config$max_output_patches,
    quantiles = cfg$chronos_config$quantiles,
    use_arcsinh = cfg$chronos_config$use_arcsinh,
    use_reg_token = cfg$chronos_config$use_reg_token,
    time_encoding_scale = cfg$chronos_config$time_encoding_scale
  )
}

# ------------------------------------------------------------------------------
# Load weights

load_chronos2_weights <- function(model, safetensors_path) {
  tensors <- safetensors::safe_load_file(safetensors_path, framework = "torch")

  # Build mapping from Python state_dict keys to R model parameter paths
  assignments <- list()

  # shared embedding
  assignments[["shared.weight"]] <- model$shared$weight

  # input_patch_embedding (ResidualBlock)
  # fmt: skip
  assignments[["input_patch_embedding.hidden_layer.weight"]] <-
    model$input_patch_embedding$hidden_layer$weight
  assignments[["input_patch_embedding.hidden_layer.bias"]] <-
    model$input_patch_embedding$hidden_layer$bias
  assignments[["input_patch_embedding.output_layer.weight"]] <-
    model$input_patch_embedding$output_layer$weight
  assignments[["input_patch_embedding.output_layer.bias"]] <-
    model$input_patch_embedding$output_layer$bias
  assignments[["input_patch_embedding.residual_layer.weight"]] <-
    model$input_patch_embedding$residual_layer$weight
  assignments[["input_patch_embedding.residual_layer.bias"]] <-
    model$input_patch_embedding$residual_layer$bias

  # output_patch_embedding (ResidualBlock)
  # fmt: skip
  assignments[["output_patch_embedding.hidden_layer.weight"]] <-
    model$output_patch_embedding$hidden_layer$weight
  assignments[["output_patch_embedding.hidden_layer.bias"]] <-
    model$output_patch_embedding$hidden_layer$bias
  assignments[["output_patch_embedding.output_layer.weight"]] <-
    model$output_patch_embedding$output_layer$weight
  assignments[["output_patch_embedding.output_layer.bias"]] <-
    model$output_patch_embedding$output_layer$bias
  assignments[["output_patch_embedding.residual_layer.weight"]] <-
    model$output_patch_embedding$residual_layer$weight
  assignments[["output_patch_embedding.residual_layer.bias"]] <-
    model$output_patch_embedding$residual_layer$bias

  # encoder.final_layer_norm
  # fmt: skip
  assignments[["encoder.final_layer_norm.weight"]] <-
    model$encoder$final_layer_norm$weight

  # encoder blocks
  # fmt: skip
  num_layers <- length(model$encoder$blocks)
  for (i in seq_len(num_layers)) {
    py_i <- i - 1 # Python 0-indexed
    block <- model$encoder$blocks[[i]]
    prefix <- paste0("encoder.block.", py_i, ".")

    # layer[0] = TimeSelfAttention
    assignments[[paste0(prefix, "layer.0.layer_norm.weight")]] <-
      block$time_self_attn$layer_norm$weight
    assignments[[paste0(prefix, "layer.0.self_attention.q.weight")]] <-
      block$time_self_attn$self_attention$q$weight
    assignments[[paste0(prefix, "layer.0.self_attention.k.weight")]] <-
      block$time_self_attn$self_attention$k$weight
    assignments[[paste0(prefix, "layer.0.self_attention.v.weight")]] <-
      block$time_self_attn$self_attention$v$weight
    assignments[[paste0(prefix, "layer.0.self_attention.o.weight")]] <-
      block$time_self_attn$self_attention$o$weight

    # layer[1] = GroupSelfAttention
    # fmt: skip
    assignments[[paste0(prefix, "layer.1.layer_norm.weight")]] <-
      block$group_self_attn$layer_norm$weight
    assignments[[paste0(prefix, "layer.1.self_attention.q.weight")]] <-
      block$group_self_attn$self_attention$q$weight
    assignments[[paste0(prefix, "layer.1.self_attention.k.weight")]] <-
      block$group_self_attn$self_attention$k$weight
    assignments[[paste0(prefix, "layer.1.self_attention.v.weight")]] <-
      block$group_self_attn$self_attention$v$weight
    assignments[[paste0(prefix, "layer.1.self_attention.o.weight")]] <-
      block$group_self_attn$self_attention$o$weight

    # layer[2] = FeedForward
    # fmt: skip
    assignments[[paste0(prefix, "layer.2.layer_norm.weight")]] <-
      block$feed_forward$layer_norm$weight
    assignments[[paste0(prefix, "layer.2.mlp.wi.weight")]] <-
      block$feed_forward$mlp$wi$weight
    assignments[[paste0(prefix, "layer.2.mlp.wo.weight")]] <-
      block$feed_forward$mlp$wo$weight
  }

  # Assign weights (disable gradient tracking for in-place copy)
  loaded <- 0
  skipped <- character()
  with_no_grad({
    for (key in names(tensors)) {
      if (key %in% names(assignments)) {
        param <- assignments[[key]]
        tensor <- tensors[[key]]
        if (!identical(dim(param), dim(tensor))) {
          cli::cli_warn(
            "Shape mismatch for {.val {key}}: model={paste(dim(param), collapse='x')}, file={paste(dim(tensor), collapse='x')}"
          )
          next
        }
        param$copy_(tensor)
        loaded <- loaded + 1
      } else {
        # Skip rope inv_freq buffers (they are computed, not loaded)
        if (!grepl("rope_embed\\.inv_freq", key)) {
          skipped <- c(skipped, key)
        }
      }
    }
  })

  # TODO Add verbose argument
  # cli::cli_inform(
  #   "Loaded {loaded}/{length(assignments)} parameters. Skipped: {length(skipped)}"
  # )
  # if (length(skipped) > 0 && length(skipped) <= 10) {
  #   cli::cli_inform("Skipped keys: {.val {skipped}}")
  # } else if (length(skipped) > 10) {
  #   cli::cli_inform("First 10 skipped keys: {.val {skipped[1:10]}}")
  # }

  invisible(model)
}

# ------------------------------------------------------------------------------
# Download

# Pinned default revision for `amazon/chronos-2`. Bump this deliberately
# when we're ready to ship a new set of weights -- never let users silently
# track a moving HuggingFace branch.
chronos2_default_revision <- function() {
  "0f8a440441931157957e2be1a9bce66627d99c76"
}

# Resolve a HuggingFace revision (branch / tag / SHA) to a 40-character
# commit SHA. SHAs are returned as-is so this is a no-op for the default
# pinned revision (no network call required).
chronos2_resolve_revision <- function(model_id, revision) {
  if (grepl("^[0-9a-f]{40}$", revision)) {
    return(revision)
  }

  url <- sprintf(
    "https://huggingface.co/api/models/%s/revision/%s",
    model_id,
    utils::URLencode(revision, reserved = TRUE)
  )
  res <- tryCatch(
    curl::curl_fetch_memory(url),
    error = function(e) e
  )
  if (inherits(res, "error")) {
    cli::cli_abort(c(
      "Failed to resolve revision {.val {revision}} for {.val {model_id}}.",
      "x" = conditionMessage(res)
    ))
  }
  if (res$status_code != 200L) {
    cli::cli_abort(
      "Failed to resolve revision {.val {revision}} for {.val {model_id}} (HTTP {res$status_code})."
    )
  }
  parsed <- jsonlite::fromJSON(rawToChar(res$content))
  if (is.null(parsed$sha) || !nzchar(parsed$sha)) {
    cli::cli_abort(
      "HuggingFace API did not return a SHA for revision {.val {revision}}."
    )
  }
  parsed$sha
}

# Fetch the Content-Length for a URL via a HEAD request, returning NA when
# the server doesn't expose it (in which case we just skip size checking).
chronos2_remote_size <- function(url) {
  handle <- curl::new_handle()
  curl::handle_setopt(handle, nobody = TRUE, followlocation = TRUE)
  res <- tryCatch(
    curl::curl_fetch_memory(url, handle = handle),
    error = function(e) e
  )
  if (inherits(res, "error") || res$status_code >= 400L) {
    return(NA_real_)
  }
  hdrs <- curl::parse_headers(res$headers)
  cl <- grep("^content-length:", hdrs, ignore.case = TRUE, value = TRUE)
  if (length(cl) == 0L) {
    return(NA_real_)
  }
  as.numeric(sub("^[Cc]ontent-[Ll]ength:\\s*([0-9]+).*$", "\\1", cl[[1L]]))
}

# Download a single file with size validation and bounded retries. If the
# destination already holds a complete file (size matches the remote
# Content-Length, or HEAD didn't expose one), we keep it.
chronos2_download_file <- function(url, dest, label, max_attempts = 3L) {
  expected_size <- chronos2_remote_size(url)

  if (
    file.exists(dest) &&
      (is.na(expected_size) || file.size(dest) == expected_size)
  ) {
    return(invisible(dest))
  }

  if (file.exists(dest)) {
    cli::cli_alert_info(
      "Cached {.file {label}} is incomplete (size {.val {file.size(dest)}} of {.val {expected_size}}); re-downloading."
    )
    file.remove(dest)
  }

  for (attempt in seq_len(max_attempts)) {
    cli::cli_progress_step("Downloading {.url {url}}")
    err <- tryCatch(
      {
        curl::curl_download(url, dest, mode = "wb", quiet = TRUE)
        NULL
      },
      error = function(e) e
    )

    ok <-
      is.null(err) &&
      file.exists(dest) &&
      (is.na(expected_size) || file.size(dest) == expected_size)
    if (ok) {
      return(invisible(dest))
    }

    if (file.exists(dest)) {
      file.remove(dest)
    }
    if (attempt < max_attempts) {
      cli::cli_alert_warning(
        "Attempt {attempt}/{max_attempts} for {.val {label}} failed; retrying."
      )
    }
  }

  cli::cli_abort(c(
    "Failed to download {.url {url}} after {max_attempts} attempts.",
    "i" = "If you keep hitting this, try a different network or proxy."
  ))
}

chronos2_download <- function(
  model_id = "amazon/chronos-2",
  revision = chronos2_default_revision(),
  cache_dir = tools::R_user_dir("brulee", which = "cache"),
  confirm = FALSE,
  call = rlang::caller_env()
) {
  sha <- chronos2_resolve_revision(model_id, revision)

  model_dir <- file.path(cache_dir, gsub("/", "--", model_id), sha)
  files <- c("config.json", "model.safetensors")

  # This internal helper downloads unconditionally by default: anyone calling
  # it directly (via `:::`) is already being deliberate. The user-facing gate
  # (ask when interactive, error otherwise) only applies through
  # brulee_chronos(), which passes confirm = TRUE.
  cached <- dir.exists(model_dir) &&
    all(file.exists(file.path(model_dir, files)))
  if (!cached && confirm) {
    brulee_confirm_download(
      label = model_id,
      size = "500MB",
      fn = "brulee_chronos",
      root = cache_dir,
      hint = "Run {.fn brulee_chronos} in an interactive session to download them.",
      call = call
    )
  }

  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

  # `download.file()` defaults to a 60-second timeout, which is too tight
  # for the ~478MB safetensors file. Lift it for the duration of the call.
  old_timeout <- getOption("timeout")
  options(timeout = max(600L, old_timeout))
  on.exit(options(timeout = old_timeout), add = TRUE)

  for (f in files) {
    url <- sprintf(
      "https://huggingface.co/%s/resolve/%s/%s",
      model_id,
      sha,
      f
    )
    chronos2_download_file(url, file.path(model_dir, f), label = f)
  }

  list(model_dir = model_dir, sha = sha)
}
