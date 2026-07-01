# Stage 1 of TabICL: distribution-aware column-wise embedding. Ported from
# `ColEmbedding` in `src/tabicl/_model/embedding.py`, for the configuration the
# released v2 checkpoints use: feature_group = "same", target_aware = TRUE,
# affine = FALSE, ssmax = "qassmax-mlp-elementwise".
#
# Each scalar cell is projected, optionally combined with an embedded target,
# and passed through a shared Set Transformer. With affine = FALSE the embeddings
# are the Set Transformer output directly. `reserve_cls_tokens` empty slots are
# prepended along the feature axis for the row interactor's CLS tokens.

tabicl_col_embedding <- nn_module(
  "tabicl_col_embedding",
  initialize = function(
    embed_dim,
    num_blocks,
    nhead,
    dim_feedforward,
    num_inds,
    feature_group_size = 3,
    target_aware = TRUE,
    max_classes = 10,
    reserve_cls_tokens = 4,
    activation = nnf_gelu,
    norm_first = TRUE,
    bias_free_ln = FALSE,
    ssmax = "none"
  ) {
    self$embed_dim <- embed_dim
    self$reserve_cls_tokens <- reserve_cls_tokens
    self$feature_group_size <- feature_group_size
    self$target_aware <- target_aware
    self$max_classes <- max_classes
    self$skip_value <- -100

    # feature_group = "same" is enabled, so each cell carries feature_group_size
    # grouped values.
    self$in_linear <- tabicl_skippable_linear(feature_group_size, embed_dim)
    self$tf_col <- tabicl_set_transformer(
      num_blocks = num_blocks,
      d_model = embed_dim,
      nhead = nhead,
      dim_feedforward = dim_feedforward,
      num_inds = num_inds,
      activation = activation,
      norm_first = norm_first,
      bias_free_ln = bias_free_ln,
      ssmax = ssmax
    )

    if (target_aware) {
      if (max_classes > 0) {
        self$y_encoder <- tabicl_onehot_linear(max_classes, embed_dim)
      } else {
        self$y_encoder <- nn_linear(1, embed_dim)
      }
    }
  },
  # "same"-mode feature grouping: for group offset 2^i, gather the circularly
  # shifted columns and stack. Mirrors `feature_grouping`.
  feature_grouping = function(x) {
    h <- x$size(3)
    idxs <- torch_arange(
      start = 0,
      end = h - 1,
      dtype = torch_long(),
      device = x$device
    )
    groups <- purrr::map(seq_len(self$feature_group_size), \(i) {
      shift <- 2^(i - 1)
      perm <- ((idxs + shift) %% h) + 1L # +1: index_select is 1-based
      x$index_select(dim = 3, index = perm$to(dtype = torch_long()))
    })
    torch_stack(groups, dim = -1) # (B, T, H, group_size)
  },
  # Project cells, add the target embedding to the training rows, run the Set
  # Transformer. With affine = FALSE the result is the embedding.
  compute_embeddings = function(
    features,
    train_size,
    y_train,
    embed_with_test
  ) {
    src <- self$in_linear(features) # (..., T, E)

    if (self$target_aware) {
      if (self$max_classes > 0) {
        y_emb <- self$y_encoder(y_train$to(dtype = torch_float()))
      } else {
        y_emb <- self$y_encoder(
          y_train$to(dtype = torch_float())$unsqueeze(-1)
        )
      }
      t_total <- src$size(-2)
      train_part <- src$narrow(dim = -2, start = 1, length = train_size) + y_emb
      if (train_size < t_total) {
        rest <- src$narrow(
          dim = -2,
          start = train_size + 1,
          length = t_total - train_size
        )
        src <- torch_cat(list(train_part, rest), dim = -2)
      } else {
        src <- train_part
      }
    }

    if (embed_with_test) {
      tf_train_size <- NULL
    } else {
      tf_train_size <- train_size
    }
    self$tf_col(src, train_size = tf_train_size)
  },
  forward = function(x, y_train, embed_with_test = FALSE) {
    train_size <- y_train$size(-1)

    if (self$target_aware && self$max_classes > 0) {
      num_classes <- as.integer(as.numeric(y_train$max()$item())) + 1L
      if (num_classes > self$max_classes) {
        cli::cli_abort(
          "Mixed-radix ensembling for {num_classes} classes (> max_classes
           {self$max_classes}) is not yet implemented in the brulee port."
        )
      }
    }

    xg <- self$feature_grouping(x) # (B, T, H, group_size)
    # Prepend reserve_cls_tokens empty (skip-valued) slots along the feature axis.
    xg <- nnf_pad(
      xg,
      pad = c(0, 0, self$reserve_cls_tokens, 0),
      value = self$skip_value
    )
    features <- xg$transpose(2, 3) # (B, H + C, T, group_size)

    if (self$target_aware) {
      # (B, H + C, train_size)
      y_expanded <- y_train$unsqueeze(2)$expand(c(-1, features$size(2), -1))
    } else {
      y_expanded <- NULL
    }

    embeddings <- self$compute_embeddings(
      features,
      train_size,
      y_expanded,
      embed_with_test
    )
    embeddings$transpose(2, 3) # (B, T, H + C, E)
  }
)
