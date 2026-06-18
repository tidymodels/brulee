# Stage 2 of TabICL: row-wise feature interaction. Ported from `RowInteraction`
# in `src/tabicl/_model/interaction.py`.
#
# Per row, a transformer with RoPE mixes the per-feature cell embeddings.
# Learnable CLS tokens are prepended; after the stack, the CLS tokens attend to
# the whole row one last time and their outputs are concatenated into a single
# per-row representation. Input is the column-embedder output of shape
# (B, T, H + C, E): B tables, T rows, H feature slots + C CLS slots, embedding E.
# The first C slots are placeholders overwritten with the learned CLS tokens.

tabicl_row_interaction <- nn_module(
  "tabicl_row_interaction",
  initialize = function(
    embed_dim,
    num_blocks,
    nhead,
    dim_feedforward,
    num_cls = 4,
    rope_base = 100000,
    activation = "gelu",
    norm_first = TRUE,
    bias_free_ln = FALSE
  ) {
    self$embed_dim <- embed_dim
    self$num_cls <- num_cls
    self$norm_first <- norm_first

    self$tf_row <- tabicl_encoder(
      num_blocks = num_blocks,
      d_model = embed_dim,
      nhead = nhead,
      dim_feedforward = dim_feedforward,
      activation = activation,
      norm_first = norm_first,
      bias_free_ln = bias_free_ln,
      ssmax = "none",
      use_rope = TRUE,
      rope_base = rope_base
    )

    self$cls_tokens <- nn_parameter(torch_empty(num_cls, embed_dim))
    if (norm_first) {
      self$out_ln <- tabicl_layer_norm(embed_dim, bias = !bias_free_ln)
    } else {
      self$out_ln <- nn_identity()
    }
  },
  # Run the encoder blocks then aggregate via the CLS tokens. The first
  # `num_blocks - 1` blocks are full self-attention; the last has the CLS
  # tokens as queries attending to the whole row.
  aggregate = function(embeddings, key_mask = NULL) {
    rope <- self$tf_row$rope
    n_blocks <- length(self$tf_row$blocks)

    if (n_blocks > 1) {
      for (i in seq_len(n_blocks - 1)) {
        embeddings <- self$tf_row$blocks[[i]](
          q = embeddings,
          key_padding_mask = key_mask,
          rope = rope
        )
      }
    }

    last_block <- self$tf_row$blocks[[n_blocks]]
    cls_q <- embeddings$narrow(dim = -2, start = 1, length = self$num_cls)
    cls_outputs <- last_block(
      q = cls_q,
      k = embeddings,
      v = embeddings,
      key_padding_mask = key_mask,
      rope = rope
    )
    cls_outputs <- self$out_ln(cls_outputs)
    cls_outputs$flatten(start_dim = -2) # (B, T, C * E)
  },
  forward = function(embeddings, key_mask = NULL) {
    sizes <- dim(embeddings)
    b <- sizes[1]
    t <- sizes[2]
    hc <- sizes[3]

    # Overwrite the first `num_cls` slots with the learned CLS tokens. Building
    # via concatenation avoids an in-place write on the input.
    cls <- self$cls_tokens$expand(c(b, t, self$num_cls, self$embed_dim))
    features <- embeddings$narrow(
      dim = -2,
      start = self$num_cls + 1,
      length = hc - self$num_cls
    )
    embeddings <- torch_cat(list(cls, features), dim = -2)

    self$aggregate(embeddings, key_mask = key_mask)
  }
)
