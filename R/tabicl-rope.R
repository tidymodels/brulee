# Rotary positional embeddings for the TabICL row-interaction transformer.
#
# Ported from the non-interleaved path of `src/tabicl/_model/rope.py`
# (RotaryEmbedding). TabICL's released checkpoints use `interleaved = FALSE`
# and `theta = 100000`, and store the frequency vector as a *learned* parameter
# (`row_interactor.tf_row.rope.freqs`), so it is loaded from the checkpoint
# rather than recomputed. The interleaved variants, XPOS, caching, and
# interpolation in the reference are unused by these weights and are omitted.

# Split the last dim in half and rotate: [x1, x2] -> [-x2, x1]. Mirrors
# `rotate_half_contiguous()` in rope.py (non-interleaved rotation).
tabicl_rotate_half <- function(x) {
  halves <- x$chunk(2, dim = -1)
  torch_cat(list(-halves[[2]], halves[[1]]), dim = -1)
}

tabicl_rope <- nn_module(
  "tabicl_rope",
  initialize = function(dim, theta = 100000) {
    self$dim <- dim
    half <- dim %/% 2L
    # Default "lang" frequencies; overwritten by the checkpoint on load.
    idx <- torch_arange(
      start = 0,
      end = dim - 1,
      step = 2,
      dtype = torch_float32()
    )[1:half]
    freqs <- 1.0 / (theta^(idx / dim))
    self$freqs <- nn_parameter(freqs)
  },
  forward = function(x) {
    # x: (..., n_heads, seq_len, head_dim). The sequence axis is dim -2.
    seq_len <- x$size(-2)
    pos <- torch_arange(
      start = 0,
      end = seq_len - 1,
      dtype = self$freqs$dtype,
      device = x$device
    )
    # Outer product position x frequency -> (seq_len, head_dim / 2).
    fr <- torch_outer(pos, self$freqs)
    cos_f <- torch_cat(list(fr$cos(), fr$cos()), dim = -1)$to(dtype = x$dtype)
    sin_f <- torch_cat(list(fr$sin(), fr$sin()), dim = -1)$to(dtype = x$dtype)
    x * cos_f + tabicl_rotate_half(x) * sin_f
  }
)
