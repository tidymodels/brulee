# Stage 3 of TabICL: dataset-wise in-context learning. Ported from `ICLearning`
# in `src/tabicl/_model/learning.py`.
#
# A transformer (no RoPE, with SSMax) attends test rows to the labelled training
# rows. The training targets are embedded and added to the training-row
# representations, the stack runs with the test rows querying only the training
# context, and a small decoder MLP produces per-row outputs: class logits
# (classification) or quantile values (regression).
#
# The hierarchical-classification path (num_classes > max_classes) is not yet
# ported; the standard path (num_classes <= max_classes) covers the common case.

tabicl_icl_learning <- nn_module(
  "tabicl_icl_learning",
  initialize = function(
    max_classes,
    out_dim,
    d_model,
    num_blocks,
    nhead,
    dim_feedforward,
    activation = "gelu",
    norm_first = TRUE,
    bias_free_ln = FALSE,
    ssmax = "none"
  ) {
    self$max_classes <- max_classes
    self$norm_first <- norm_first

    self$tf_icl <- tabicl_encoder(
      num_blocks = num_blocks,
      d_model = d_model,
      nhead = nhead,
      dim_feedforward = dim_feedforward,
      activation = activation,
      norm_first = norm_first,
      bias_free_ln = bias_free_ln,
      ssmax = ssmax,
      use_rope = FALSE
    )

    if (norm_first) {
      self$ln <- tabicl_layer_norm(d_model, bias = !bias_free_ln)
    }

    self$y_encoder <- if (max_classes > 0) {
      tabicl_onehot_linear(max_classes, d_model)
    } else {
      nn_linear(1, d_model)
    }

    self$decoder <- nn_sequential(
      nn_linear(d_model, d_model * 2),
      nn_gelu(),
      nn_linear(d_model * 2, out_dim)
    )
  },
  # Embed targets, add to the training rows, run the ICL transformer (test rows
  # attend to the first train_size positions), normalize, decode. Returns
  # (B, T, out_dim) over all rows.
  icl_predictions = function(r, y_train) {
    train_size <- y_train$size(-1)
    ry_train <- if (self$max_classes > 0) {
      self$y_encoder(y_train$to(dtype = torch_float()))
    } else {
      self$y_encoder(y_train$to(dtype = torch_float())$unsqueeze(-1))
    }

    t_total <- r$size(2)
    train_part <- r$narrow(dim = 2, start = 1, length = train_size) + ry_train
    rest <- r$narrow(
      dim = 2,
      start = train_size + 1,
      length = t_total - train_size
    )
    r <- torch_cat(list(train_part, rest), dim = 2)

    src <- self$tf_icl(r, train_size = train_size)
    if (self$norm_first) {
      src <- self$ln(src)
    }
    self$decoder(src)
  },
  # Inference prediction over the test rows. Mirrors `_predict_standard` /
  # `_inference_forward` for the standard (num_classes <= max_classes) path.
  predict = function(
    r,
    y_train,
    return_logits = TRUE,
    softmax_temperature = 0.9
  ) {
    train_size <- y_train$size(-1)
    out <- self$icl_predictions(r, y_train)
    t_total <- out$size(2)
    out <- out$narrow(
      dim = 2,
      start = train_size + 1,
      length = t_total - train_size
    )

    if (self$max_classes > 0) {
      num_classes <- length(unique(as.numeric(y_train[1, ]$cpu())))
      if (num_classes > self$max_classes) {
        cli::cli_abort(
          "Hierarchical classification for {num_classes} classes (> max_classes
           {self$max_classes}) is not yet implemented in the brulee port."
        )
      }
      out <- out$narrow(dim = -1, start = 1, length = num_classes)
      if (!return_logits) {
        out <- nnf_softmax(out / softmax_temperature, dim = -1)
      }
    }
    out
  },
  forward = function(
    r,
    y_train,
    return_logits = TRUE,
    softmax_temperature = 0.9
  ) {
    self$predict(r, y_train, return_logits, softmax_temperature)
  }
)
