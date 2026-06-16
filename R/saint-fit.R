#' Fit SAINT models for tabular data
#'
#' `brulee_saint()` fits the SAINT (Self-Attention and Inter-sample Attention
#' Transformer) model from Somepalli _et al_ (2021). SAINT applies multi-head
#' self-attention across both features (column attention) and samples within a
#' batch (row/inter-sample attention) to learn complex feature interactions.
#'
#' @inheritParams brulee_mlp
#' @param num_embedding An integer for the dimension of the initial embedding
#'   layer that encodes the original predictors.  Each feature (categorical or
#'   continuous) is mapped to a vector of this dimension. Must be >= 1.
#' @param attention_type A character string for the type of attention to use.
#'   Options are:
#'   - `"column"`: Column attention only (attends across features). This is the
#'     SAINT-s variant.
#'   - `"row"`: Row/inter-sample attention only (attends across samples within
#'     a batch). This is the SAINT-i variant.
#'   - `"both"`: Alternates between column and row attention in each
#'     transformer block. This is the full SAINT model.
#' @param num_attn_heads An integer for the number of parallel attention heads
#'   used in both column and row attention. Must be >= 1.
#' @param num_attn_blocks An integer for the number of sequential transformer
#'   blocks (depth). Must be >= 1.
#' @param dropout_attn A number in `[0, 1)` for the dropout rate applied to
#'   attention weights during training.
#' @param row_attention_on_predict A logical value. Should row (inter-sample)
#'   attention be applied during prediction? Default is `FALSE`. When `FALSE`,
#'   row attention is only used during training and predictions use column
#'   attention only — this ensures that predictions for a given row are
#'   independent of what other rows are in the prediction set. This is only
#'   relevant when `attention_type` is `"row"` or `"both"`.
#' @param hidden_units An integer vector for the number of units in optional
#'   hidden layers between the transformer backbone and the output head.
#'   When `NULL` (the default), no hidden layers are added and the pooled
#'   transformer output is projected directly to the output.
#' @param hidden_activations A character vector of activation functions for the
#'   hidden layers. Must be the same length as `hidden_units` or a single value
#'   that will be recycled. See [brulee_activations()] for options.
#' @param dropout_hidden A number in `[0, 1)` for the dropout rate applied
#'   within the feed-forward layers of each transformer block.
#' @param dropout_last A number in `[0, 1)` for the dropout rate applied
#'   between the last hidden layer and the output head. Only has effect when
#'   `hidden_units` is not `NULL`. Default is 0 (no dropout).
#' @param use_target_token A logical value. When `TRUE` (the default), a
#'   learnable target token (`[CLS]` in the SAINT paper) is prepended to
#'   each sample's feature sequence and only its final-layer embedding is
#'   fed to the head. This matches the architecture described in the SAINT
#'   paper (Section 3 and Figure 1); see the **Target Token Pooling**
#'   section in **Details**. When `FALSE`, the head instead consumes the
#'   concatenation of every feature token, which matches the SAINT
#'   reference implementation at <https://github.com/somepago/saint>.
#' @details
#'
#' ## Architecture
#'
#' The SAINT architecture has three stages:
#'
#' 1. **Embedding layer**: Categorical features are mapped through per-feature
#'    embedding tables. Continuous features are passed through per-feature MLPs
#'    (1 -> 100 -> `num_embedding`). These initial embeddings are per-feature;
#'    there is a distinct embedding MLP for each predictor.
#' 2. **Transformer backbone**: A stack of `num_attn_blocks` transformer layers.
#'    Each layer contains multi-head self-attention followed by a feed-forward
#'    network with GeGLU activation. For `attention_type = "both"`, each block
#'    alternates between column attention (across features) and row attention
#'    (across samples within the batch).
#' 3. **Output head**: Pools the transformer output (either the target
#'    token's embedding or the flattened concatenation of all feature
#'    embeddings, controlled by `use_target_token`) and projects it through
#'    optional hidden layers to the output dimension.
#'
#' There is a `summary()` methods that can provide details of the architecture
#' for a specific model fit.
#'
#' Differences in this implementation and the orignal paper:
#'
#'  - Pretraining isn't supported.
#'
#' ## Attention Types
#'
#' - **Column attention** (`"column"`): Standard self-attention over features.
#'   Each feature embedding attends to all other feature embeddings.
#' - **Row attention** (`"row"`): inter-sample attention. Reshapes the batch so
#'   that each sample's full feature representation becomes a single token,
#'   then applies attention across all samples in the batch.
#' - **Both** (`"both"`): Alternates between column and row attention in each
#'   transformer block. This is the full SAINT model.
#'
#' ## Target Token Pooling
#'
#' Borrowing from BERT, SAINT prepends a learnable target token (the
#' paper calls it `[CLS]`) to each sample's feature sequence before the
#' transformer. With embeddings `E(x_i^{(1)}), ..., E(x_i^{(n)})` for the
#' `n` predictors of sample `i`, the input sequence becomes
#'
#' `[target, E(x_i^{(1)}), E(x_i^{(2)}), ..., E(x_i^{(n)})]`
#'
#' giving `n + 1` tokens of dimension `num_embedding`. The target token has
#' no input value; it is a free parameter of the model that is trained
#' alongside the rest of the network. Column attention lets every feature
#' token attend to the target and vice versa, so the target slot accumulates
#' a contextual summary of the sample. When `attention_type` is `"row"` or
#' `"both"`, inter-sample attention sees the full `n + 1` token sequence per
#' sample, so the target slot also exchanges information across samples in
#' the batch.
#'
#' After the transformer backbone, the head reads _only_ the final-layer
#' embedding of the target token (the first position) and feeds it through
#' the optional `hidden_units` MLP and the output layer. This is what the
#' paper describes in Figure 1: "We take the contextual embeddings from
#' SAINT and pass only the embedding correspond to the CLS token through an
#' MLP to obtain the final prediction."
#'
#' With `use_target_token = FALSE`, no target token is added and the head
#' instead consumes the concatenation of all `n` feature tokens. That
#' option is provided because the SAINT reference Python implementation
#' (<https://github.com/somepago/saint>) departs from the paper and uses
#' flatten-pooling; it is kept available for compatibility with that code
#' path and for users who want the original brulee behavior.
#'
#' ## Row Attention at Prediction Time
#'
#' Row attention computations adjust the internal embeddings based on the rows
#' that are available at any given time. During training, the other rows in the
#' batch are used to compute attention. After training, when `predict()` is
#' called, the default behavior is to bypass row attention. This is because the
#' predictions would depend on the other data available at the time. If this is
#' what you want, set `row_attention_on_predict` to `TRUE`.
#'
#' ## Learning Rates
#'
#' The learning rate can be set to constant (the default) or dynamically set
#' via a learning rate scheduler (via the `rate_schedule`). Using
#' `rate_schedule = 'none'` uses the `learn_rate` argument.
#'
#' ## Other Notes
#'
#' Unlike other \pkg{brulee} models, `brulee_saint()` natively handles factor
#' predictors via learned embeddings. Factor columns are automatically detected
#' and embedded, while numeric columns pass through per-feature MLPs. There is
#' _no need to pre-encode factors as indicators_.
#'
#' When the outcome is a number, the function internally standardizes the
#' outcome data to have mean zero and a standard deviation of one. The prediction
#' function creates predictions on the original scale.
#'
#' By default, training halts when the validation loss increases for at least
#' `stop_iter` iterations. If `validation = 0` the training set loss is used.
#'
#' The model objects are saved for each epoch so that the number of epochs can
#' be efficiently tuned. The [predict()] method for this model has an
#' `epoch` argument (which defaults to the epoch with the best loss value).
#'
#' @references
#'
#' Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B., &
#' Goldstein, T. (2021). SAINT: Improved Neural Networks for Tabular Data
#' via Row Attention and Contrastive Pre-Training. arXiv preprint
#' arXiv:2106.01342.
#'
#' @seealso [predict.brulee_saint()], [autoplot.brulee_saint()]
#'
#' @return
#'
#' A `brulee_saint` object with elements:
#'  * `models_obj`: a serialized raw vector for the torch module.
#'  * `estimates`: a list of matrices with the model parameter estimates per
#'                 epoch.
#'  * `best_epoch`: an integer for the epoch with the smallest loss.
#'
#'  * `loss`: A vector of loss values (MSE for regression, negative log-
#'            likelihood for classification) at each epoch.
#'  * `dim`: A list of data dimensions and feature metadata.
#'  * `y_stats`: A list of summary statistics for numeric outcomes.
#'  * `parameters`: A list of some tuning parameter values.
#'  * `device`: A character string for the device used during training.
#'  * `blueprint`: The `hardhat` blueprint data.
#'
#' @examplesIf !brulee:::is_cran_check()
#' \donttest{
#' pkgs <- c("recipes", "yardstick", "modeldata")
#' if (torch::torch_is_installed() & rlang::is_installed(pkgs)) {
#'
#'  set.seed(87261)
#'  tr_data <- modeldata::sim_regression(500, method = "worley_1987")
#'  te_data <- modeldata::sim_regression(50, method = "worley_1987")
#'
#'  library(recipes)
#'  rec <- recipe(outcome ~ ., data = te_data) |>
#'   step_normalize(all_numeric_predictors())
#'
#'  set.seed(389)
#'  fit <- brulee_saint(
#'   rec,
#'   data = te_data,
#'   hidden_unit = 5,
#'   dropout_hidden = 0.2,
#'   num_embedding = 3,
#'   num_attn_heads = 5,
#'   num_attn_blocks = 4,
#'   dropout_attn = 0.2,
#'   epochs = 50L,
#'   batch_size = 32L,
#'   learn_rate = 0.01,
#'   optimize = "SGD",
#'   verbose = TRUE
#'  )
#'
#'  autoplot(fit)
#'  summary(fit)
#'
#'  library(yardstick)
#'  predict(fit, te_data) |>
#'   dplyr::bind_cols(te_data) |>
#'   rsq(outcome, .pred)
#'
#' }
#' }
#' @export
brulee_saint <- function(x, ...) {
  UseMethod("brulee_saint")
}

#' @export
#' @rdname brulee_saint
brulee_saint.default <- function(x, ...) {
  stop(
    "`brulee_saint()` is not defined for a '",
    class(x)[1],
    "'.",
    call. = FALSE
  )
}

# XY method - data frame

#' @export
#' @rdname brulee_saint
brulee_saint.data.frame <- function(
  x,
  y,
  epochs = 100L,
  num_embedding = 32L,
  attention_type = "both",
  num_attn_heads = 8L,
  num_attn_blocks = 6L,
  dropout_attn = 0.1,
  dropout_hidden = 0.1,
  dropout_last = 0,
  row_attention_on_predict = FALSE,
  hidden_units = 5,
  hidden_activations = "relu",
  penalty = 0.001,
  mixture = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.0001,
  rate_schedule = "none",
  momentum = 0.0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  verbose = FALSE,
  device = NULL,
  use_target_token = TRUE,
  ...
) {
  processed <- hardhat::mold(x, y)

  brulee_saint_bridge(
    processed,
    epochs = epochs,
    num_embedding = num_embedding,
    attention_type = attention_type,
    num_attn_heads = num_attn_heads,
    num_attn_blocks = num_attn_blocks,
    dropout_attn = dropout_attn,
    dropout_hidden = dropout_hidden,
    dropout_last = dropout_last,
    row_attention_on_predict = row_attention_on_predict,
    hidden_units = hidden_units,
    hidden_activations = hidden_activations,
    learn_rate = learn_rate,
    rate_schedule = rate_schedule,
    penalty = penalty,
    mixture = mixture,
    validation = validation,
    optimizer = optimizer,
    momentum = momentum,
    batch_size = batch_size,
    class_weights = class_weights,
    stop_iter = stop_iter,
    verbose = verbose,
    device = device,
    use_target_token = use_target_token,
    ...
  )
}

# XY method - matrix

#' @export
#' @rdname brulee_saint
brulee_saint.matrix <- function(
  x,
  y,
  epochs = 100L,
  num_embedding = 32L,
  attention_type = "both",
  num_attn_heads = 8L,
  num_attn_blocks = 6L,
  dropout_attn = 0.1,
  dropout_hidden = 0.1,
  dropout_last = 0,
  row_attention_on_predict = FALSE,
  hidden_units = 5,
  hidden_activations = "relu",
  penalty = 0.001,
  mixture = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.0001,
  rate_schedule = "none",
  momentum = 0.0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  verbose = FALSE,
  device = NULL,
  use_target_token = TRUE,
  ...
) {
  processed <- hardhat::mold(x, y)

  brulee_saint_bridge(
    processed,
    epochs = epochs,
    num_embedding = num_embedding,
    attention_type = attention_type,
    num_attn_heads = num_attn_heads,
    num_attn_blocks = num_attn_blocks,
    dropout_attn = dropout_attn,
    dropout_hidden = dropout_hidden,
    dropout_last = dropout_last,
    row_attention_on_predict = row_attention_on_predict,
    hidden_units = hidden_units,
    hidden_activations = hidden_activations,
    learn_rate = learn_rate,
    rate_schedule = rate_schedule,
    momentum = momentum,
    penalty = penalty,
    mixture = mixture,
    validation = validation,
    optimizer = optimizer,
    batch_size = batch_size,
    class_weights = class_weights,
    stop_iter = stop_iter,
    verbose = verbose,
    device = device,
    use_target_token = use_target_token,
    ...
  )
}

# Formula method

#' @export
#' @rdname brulee_saint
brulee_saint.formula <- function(
  formula,
  data,
  epochs = 100L,
  num_embedding = 32L,
  attention_type = "both",
  num_attn_heads = 8L,
  num_attn_blocks = 6L,
  dropout_attn = 0.1,
  dropout_hidden = 0.1,
  dropout_last = 0,
  row_attention_on_predict = FALSE,
  hidden_units = 5,
  hidden_activations = "relu",
  penalty = 0.001,
  mixture = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.0001,
  rate_schedule = "none",
  momentum = 0.0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  verbose = FALSE,
  device = NULL,
  use_target_token = TRUE,
  ...
) {
  processed <- hardhat::mold(
    formula,
    data,
    blueprint = hardhat::default_formula_blueprint(indicators = "none")
  )

  brulee_saint_bridge(
    processed,
    epochs = epochs,
    num_embedding = num_embedding,
    attention_type = attention_type,
    num_attn_heads = num_attn_heads,
    num_attn_blocks = num_attn_blocks,
    dropout_attn = dropout_attn,
    dropout_hidden = dropout_hidden,
    dropout_last = dropout_last,
    row_attention_on_predict = row_attention_on_predict,
    hidden_units = hidden_units,
    hidden_activations = hidden_activations,
    learn_rate = learn_rate,
    rate_schedule = rate_schedule,
    momentum = momentum,
    penalty = penalty,
    mixture = mixture,
    validation = validation,
    optimizer = optimizer,
    batch_size = batch_size,
    class_weights = class_weights,
    stop_iter = stop_iter,
    verbose = verbose,
    device = device,
    use_target_token = use_target_token,
    ...
  )
}

# Recipe method

#' @export
#' @rdname brulee_saint
brulee_saint.recipe <- function(
  x,
  data,
  epochs = 100L,
  num_embedding = 32L,
  attention_type = "both",
  num_attn_heads = 8L,
  num_attn_blocks = 6L,
  dropout_attn = 0.1,
  dropout_hidden = 0.1,
  dropout_last = 0,
  row_attention_on_predict = FALSE,
  hidden_units = 5,
  hidden_activations = "relu",
  penalty = 0.001,
  mixture = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.0001,
  rate_schedule = "none",
  momentum = 0.0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  verbose = FALSE,
  device = NULL,
  use_target_token = TRUE,
  ...
) {
  processed <- hardhat::mold(x, data)

  brulee_saint_bridge(
    processed,
    epochs = epochs,
    num_embedding = num_embedding,
    attention_type = attention_type,
    num_attn_heads = num_attn_heads,
    num_attn_blocks = num_attn_blocks,
    dropout_attn = dropout_attn,
    dropout_hidden = dropout_hidden,
    dropout_last = dropout_last,
    row_attention_on_predict = row_attention_on_predict,
    hidden_units = hidden_units,
    hidden_activations = hidden_activations,
    learn_rate = learn_rate,
    rate_schedule = rate_schedule,
    momentum = momentum,
    penalty = penalty,
    mixture = mixture,
    validation = validation,
    optimizer = optimizer,
    batch_size = batch_size,
    class_weights = class_weights,
    stop_iter = stop_iter,
    verbose = verbose,
    device = device,
    use_target_token = use_target_token,
    ...
  )
}

# ------------------------------------------------------------------------------
# Bridge

brulee_saint_bridge <- function(
  processed,
  epochs,
  num_embedding,
  attention_type,
  num_attn_heads,
  num_attn_blocks,
  dropout_attn,
  dropout_hidden,
  dropout_last,
  row_attention_on_predict,
  hidden_units,
  hidden_activations,
  learn_rate,
  rate_schedule,
  momentum,
  penalty,
  mixture,
  class_weights,
  validation,
  optimizer,
  batch_size,
  stop_iter,
  verbose,
  device,
  use_target_token,
  ...,
  call = rlang::caller_env()
) {
  if (!torch::torch_is_installed()) {
    cli::cli_abort(
      "The torch backend has not been installed; use {.run torch::install_torch()}.",
      call = call
    )
  }

  device <- guess_brulee_device(device)

  saint_validated <- validate_saint_args(
    num_embedding = num_embedding,
    attention_type = attention_type,
    num_attn_heads = num_attn_heads,
    num_attn_blocks = num_attn_blocks,
    dropout_attn = dropout_attn,
    dropout_hidden = dropout_hidden,
    use_target_token = use_target_token,
    call = call
  )

  hidden_validated <- validate_hidden_args(
    hidden_units = hidden_units,
    hidden_activations = hidden_activations,
    call = call
  )

  check_double(
    dropout_last,
    single = TRUE,
    0,
    1,
    incl = c(TRUE, FALSE),
    call = call
  )
  check_bool(row_attention_on_predict, call = call)

  if (!is.null(batch_size) & optimizer != "LBFGS") {
    if (is.numeric(batch_size) & !is.integer(batch_size)) {
      batch_size <- as.integer(batch_size)
    }
    check_integer(batch_size, single = TRUE, 1, call = call)
  }
  if (is.null(batch_size) & optimizer != "LBFGS") {
    batch_size <- 256L
  }

  validated <- validate_common_args(
    epochs = epochs,
    batch_size = batch_size,
    penalty = penalty,
    mixture = mixture,
    validation = validation,
    momentum = momentum,
    learn_rate = learn_rate,
    verbose = verbose,
    call = call
  )

  epochs <- validated$epochs
  batch_size <- validated$batch_size

  ## ---------------------------------------------------------------------------

  predictors <- processed$predictors
  split <- split_predictors_auto_int(predictors, call = call)

  ## ---------------------------------------------------------------------------

  outcome <- validate_mlp_outcome(processed$outcomes[[1]], call = call)

  # ----------------------------------------------------------------------------

  lvls <- levels(outcome)
  xtab <- table(outcome)
  class_weights <- check_class_weights(class_weights, lvls, xtab, call = call)

  ## ---------------------------------------------------------------------------

  fit <- saint_fit_imp(
    x_cat = split$x_cat,
    x_cont = split$x_cont,
    pred_lvls = split$pred_lvls,
    cat_names = split$cat_names,
    cont_names = split$cont_names,
    y = outcome,
    epochs = epochs,
    num_embedding = saint_validated$num_embedding,
    attention_type = saint_validated$attention_type,
    num_attn_heads = saint_validated$num_attn_heads,
    num_attn_blocks = saint_validated$num_attn_blocks,
    dropout_attn = saint_validated$dropout_attn,
    dropout_hidden = saint_validated$dropout_hidden,
    dropout_last = dropout_last,
    row_attention_on_predict = row_attention_on_predict,
    hidden_units = hidden_validated$hidden_units,
    hidden_activations = hidden_validated$hidden_activations,
    learn_rate = learn_rate,
    rate_schedule = rate_schedule,
    momentum = momentum,
    penalty = penalty,
    mixture = mixture,
    validation = validation,
    optimizer = optimizer,
    batch_size = batch_size,
    class_weights = class_weights,
    stop_iter = stop_iter,
    verbose = verbose,
    device = device,
    use_target_token = saint_validated$use_target_token,
    ...
  )

  new_brulee_saint(
    model_obj = fit$model_obj,
    estimates = fit$estimates,
    best_epoch = fit$best_epoch,
    loss = fit$loss,
    dims = fit$dims,
    y_stats = fit$y_stats,
    output_type = fit$output_type,
    parameters = fit$parameters,
    device = fit$device,
    blueprint = processed$blueprint
  )
}

# ------------------------------------------------------------------------------
# Validation

validate_saint_args <- function(
  num_embedding,
  attention_type,
  num_attn_heads,
  num_attn_blocks,
  dropout_attn,
  dropout_hidden,
  use_target_token,
  call = rlang::caller_env()
) {
  if (is.numeric(num_embedding) & !is.integer(num_embedding)) {
    num_embedding <- as.integer(num_embedding)
  }
  check_integer(num_embedding, single = TRUE, 1, call = call)

  if (is.numeric(num_attn_heads) & !is.integer(num_attn_heads)) {
    num_attn_heads <- as.integer(num_attn_heads)
  }
  check_integer(num_attn_heads, single = TRUE, 1, call = call)

  if (is.numeric(num_attn_blocks) & !is.integer(num_attn_blocks)) {
    num_attn_blocks <- as.integer(num_attn_blocks)
  }
  check_integer(num_attn_blocks, single = TRUE, 1, call = call)

  attn_choices <- c("column", "row", "both")
  if (!(attention_type %in% attn_choices)) {
    cli::cli_abort(
      "{.arg attention_type} should be one of: {.val {attn_choices}}.",
      call = call
    )
  }

  check_bool(use_target_token, call = call)

  check_double(dropout_attn, single = TRUE, 0, call = call)
  if (dropout_attn >= 1) {
    cli::cli_abort("{.arg dropout_attn} must be less than 1.", call = call)
  }

  check_double(dropout_hidden, single = TRUE, 0, call = call)
  if (dropout_hidden >= 1) {
    cli::cli_abort("{.arg dropout_hidden} must be less than 1.", call = call)
  }

  list(
    num_embedding = num_embedding,
    attention_type = attention_type,
    num_attn_heads = num_attn_heads,
    num_attn_blocks = num_attn_blocks,
    dropout_attn = dropout_attn,
    dropout_hidden = dropout_hidden,
    use_target_token = use_target_token
  )
}

# ------------------------------------------------------------------------------
# Constructor

new_brulee_saint <- function(
  model_obj,
  estimates,
  best_epoch,
  loss,
  dims,
  y_stats,
  output_type,
  parameters,
  device,
  blueprint
) {
  if (!inherits(model_obj, "raw")) {
    cli::cli_abort("{.arg model_obj} should be a raw vector.", call = NULL)
  }
  if (!is.list(estimates)) {
    cli::cli_abort("{.arg estimates} should be a list", call = NULL)
  }
  if (!is.vector(best_epoch) || !is.integer(best_epoch)) {
    cli::cli_abort("{.arg best_epoch} should be an integer", call = NULL)
  }
  if (!is.vector(loss) || !is.numeric(loss)) {
    cli::cli_abort("{.arg loss} should be a numeric vector", call = NULL)
  }
  if (!is.list(dims)) {
    cli::cli_abort("{.arg dims} should be a list", call = NULL)
  }
  if (!is.list(y_stats)) {
    cli::cli_abort("{.arg y_stats} should be a list", call = NULL)
  }
  if (!is.list(parameters)) {
    cli::cli_abort("{.arg parameters} should be a list", call = NULL)
  }
  if (!inherits(blueprint, "hardhat_blueprint")) {
    cli::cli_abort(
      "{.arg blueprint} should be a hardhat blueprint",
      call = NULL
    )
  }

  num_items <- purrr::map_int(estimates, length)
  estimates <- estimates[num_items > 0]

  hardhat::new_model(
    model_obj = model_obj,
    estimates = estimates,
    best_epoch = best_epoch,
    loss = loss,
    dims = dims,
    y_stats = y_stats,
    output_type = output_type,
    parameters = parameters,
    device = device,
    blueprint = blueprint,
    class = "brulee_saint"
  )
}

## -----------------------------------------------------------------------------
# Fit implementation

saint_fit_imp <- function(
  x_cat,
  x_cont,
  pred_lvls,
  cat_names,
  cont_names,
  y,
  epochs = 100L,
  batch_size = 256L,
  num_embedding = 32L,
  attention_type = "both",
  num_attn_heads = 8L,
  num_attn_blocks = 6L,
  dropout_attn = 0.1,
  dropout_hidden = 0.1,
  dropout_last = 0,
  row_attention_on_predict = FALSE,
  hidden_units = NULL,
  hidden_activations = NULL,
  penalty = 0.001,
  mixture = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.0001,
  rate_schedule = "none",
  momentum = 0.0,
  class_weights = NULL,
  stop_iter = 5,
  verbose = FALSE,
  device = "cpu",
  use_target_token = TRUE,
  ...
) {
  start_seed <- sample.int(10^5, 1)
  torch::torch_manual_seed(start_seed)

  ## ---------------------------------------------------------------------------

  n <- length(y)
  p_cat <- length(pred_lvls)
  p_cont <- if (is.null(x_cont)) 0L else ncol(x_cont)
  p <- p_cat + p_cont
  all_features <- c(cat_names, cont_names)

  ## ---------------------------------------------------------------------------

  if (is.factor(y)) {
    lvls <- levels(y)
    y_dim <- length(lvls)
    loss_fn <- function(input, target, wts = NULL) {
      torch::nnf_cross_entropy(
        input = input,
        target = target,
        weight = weights_to_tensor(wts, device = input$device)
      )
    }
  } else {
    y_dim <- 1
    lvls <- NULL
    loss_fn <- function(input, target, wts = NULL) {
      nnf_mse_loss(input, target$view(c(-1, 1)))
    }
  }

  ## ---------------------------------------------------------------------------

  if (validation > 0) {
    in_val <- sample(seq_len(n), floor(n * validation))
    x_cat_val <- if (!is.null(x_cat)) x_cat[in_val, , drop = FALSE] else NULL
    x_cont_val <- if (!is.null(x_cont)) x_cont[in_val, , drop = FALSE] else NULL
    y_val <- y[in_val]
    x_cat <- if (!is.null(x_cat)) x_cat[-in_val, , drop = FALSE] else NULL
    x_cont <- if (!is.null(x_cont)) x_cont[-in_val, , drop = FALSE] else NULL
    y <- y[-in_val]
  } else {
    x_cat_val <- NULL
    x_cont_val <- NULL
    y_val <- NULL
  }

  ## ---------------------------------------------------------------------------

  if (!is.factor(y)) {
    y_stats <- scale_stats(y)
    y <- scale_y(y, y_stats)
    if (validation > 0) {
      y_val <- scale_y(y_val, y_stats)
    }
    loss_label <- "\tLoss (scaled):"
  } else {
    y_stats <- list(mean = NA_real_, sd = NA_real_)
    loss_label <- "\tLoss:"
  }

  ## ---------------------------------------------------------------------------

  if (optimizer == "LBFGS") {
    batch_size <- length(y)
  }
  batch_size <- min(batch_size, length(y))

  ## ---------------------------------------------------------------------------

  or_dtype <- torch::torch_get_default_dtype()
  on.exit(torch::torch_set_default_dtype(or_dtype))
  torch::torch_set_default_dtype(torch::torch_float32())

  # Build the module on the CPU, then move it to `device`. See the
  # "Device-handling notes" comment block at the top of R/0_utils.R for the
  # full rationale. Saint's embedding/transformer stack contains many
  # `nn_linear` and `nn_layer_norm` modules whose parameters are initialized
  # at construction time; running construction inside `with_device(mps, ...)`
  # would route those `nn_init_*` calls to the MPS RNG, which
  # `torch_manual_seed()` does NOT reliably reset. CPU init drives every
  # `nn_init_*` from the properly-seeded CPU RNG, then `model$to(device)`
  # moves the assembled module to the target backend, giving reproducible
  # initial weights from the same seed on every backend.
  torch::torch_manual_seed(start_seed + 1)
  model <- saint_module(
    pred_lvls = pred_lvls,
    n_continuous = p_cont,
    num_embedding = num_embedding,
    attention_type = attention_type,
    num_attn_heads = num_attn_heads,
    num_attn_blocks = num_attn_blocks,
    dropout_attn = dropout_attn,
    dropout_hidden = dropout_hidden,
    dropout_last = dropout_last,
    hidden_units = hidden_units,
    hidden_activations = hidden_activations,
    y_dim = y_dim,
    use_target_token = use_target_token
  )
  model$to(device = device)

  training_output <- torch::with_device(device = device, {
    # NOTE: every torch_tensor() / float_32() call below passes `device`
    # explicitly. `with_device(...)` looks like it should set a default
    # device for these calls, but it does NOT propagate to torch_tensor() --
    # see the "Device-handling notes" at the top of R/0_utils.R. Without
    # the explicit `device =` arg these tensors would land on the CPU and
    # later trigger device-mismatch errors when fed to the MPS/CUDA model.
    make_saint_tensors <- function(xc, xn, yv) {
      t_cat <- if (!is.null(xc)) {
        torch::torch_tensor(xc, dtype = torch::torch_long(), device = device)
      } else {
        NULL
      }
      t_cont <- if (!is.null(xn)) float_32(xn, device = device) else NULL
      if (is.factor(yv)) {
        t_y <- torch::torch_tensor(
          as.numeric(yv),
          dtype = torch::torch_long(),
          device = device
        )
      } else {
        t_y <- float_32(yv, device = device)
      }
      list(x_cat = t_cat, x_cont = t_cont, y = t_y)
    }

    train_tensors <- make_saint_tensors(x_cat, x_cont, y)

    if (!is.null(train_tensors$x_cat) && !is.null(train_tensors$x_cont)) {
      ds <- torch::tensor_dataset(
        x_cat = train_tensors$x_cat,
        x_cont = train_tensors$x_cont,
        y = train_tensors$y
      )
    } else if (!is.null(train_tensors$x_cat)) {
      ds <- torch::tensor_dataset(
        x_cat = train_tensors$x_cat,
        y = train_tensors$y
      )
    } else {
      ds <- torch::tensor_dataset(
        x_cont = train_tensors$x_cont,
        y = train_tensors$y
      )
    }
    dl <- torch::dataloader(ds, batch_size = batch_size, shuffle = TRUE)

    dl_val <- NULL
    if (validation > 0) {
      val_tensors <- make_saint_tensors(x_cat_val, x_cont_val, y_val)
      if (!is.null(val_tensors$x_cat) && !is.null(val_tensors$x_cont)) {
        ds_val <- torch::tensor_dataset(
          x_cat = val_tensors$x_cat,
          x_cont = val_tensors$x_cont,
          y = val_tensors$y
        )
      } else if (!is.null(val_tensors$x_cat)) {
        ds_val <- torch::tensor_dataset(
          x_cat = val_tensors$x_cat,
          y = val_tensors$y
        )
      } else {
        ds_val <- torch::tensor_dataset(
          x_cont = val_tensors$x_cont,
          y = val_tensors$y
        )
      }
      dl_val <- torch::dataloader(ds_val)
    }

    ## -------------------------------------------------------------------------

    res <- list(
      dims = list(
        p = p,
        n = n,
        p_cat = p_cat,
        p_cont = p_cont,
        pred_lvls = pred_lvls,
        cat_names = cat_names,
        cont_names = cont_names,
        y = y_dim,
        levels = lvls,
        features = all_features
      ),
      y_stats = y_stats,
      output_type = "logits",
      parameters = list(
        num_embedding = num_embedding,
        attention_type = attention_type,
        num_attn_heads = num_attn_heads,
        num_attn_blocks = num_attn_blocks,
        dropout_attn = dropout_attn,
        dropout_hidden = dropout_hidden,
        dropout_last = dropout_last,
        row_attention_on_predict = row_attention_on_predict,
        hidden_units = hidden_units,
        hidden_activations = hidden_activations,
        learn_rate = learn_rate,
        class_weights = as.numeric(class_weights),
        penalty = penalty,
        mixture = mixture,
        validation = validation,
        optimizer = optimizer,
        batch_size = batch_size,
        momentum = momentum,
        stop_iter = stop_iter,
        sched = rate_schedule,
        use_target_token = use_target_token,
        sched_opt = list(...)
      )
    )

    ## -------------------------------------------------------------------------
    # Loss and optimizer (model now lives on the target device)

    mixture <- check_mixture(mixture, optimizer)
    loss_fn <- make_penalized_loss(loss_fn, model, penalty, mixture, optimizer)

    optimizer_obj <- set_optimizer(
      optimizer,
      model,
      learn_rate,
      momentum,
      penalty,
      mixture
    )

    ## -------------------------------------------------------------------------

    best_epoch <- 0L
    loss_vec <- rep(NA_real_, epochs + 1)

    get_batch_tensors <- function(dataset) {
      tens <- dataset$tensors
      list(
        x_cat = tens$x_cat,
        x_cont = tens$x_cont,
        y = tens$y
      )
    }

    if (validation > 0) {
      bt <- get_batch_tensors(dl_val$dataset)
    } else {
      bt <- get_batch_tensors(dl$dataset)
    }
    pred <- model(bt$x_cat, bt$x_cont)
    loss <- loss_fn(pred, bt$y, class_weights)

    loss_vec[1] <- loss$item()
    loss_min <- loss_vec[1]

    if (verbose) {
      epoch_chr <- format_epoch_labels(0:epochs)
      cli::cli_inform(
        "epoch: {epoch_chr[1]}, learn rate: {sprintf('%.5f', learn_rate)}, {loss_label} {signif(loss_vec[1], 3)}"
      )
      epoch_chr <- epoch_chr[-1]
    }

    param_per_epoch <- vector(mode = "list", length = epochs + 1)
    param_per_epoch[[1]] <-
      lapply(model$state_dict(), function(x) torch::as_array(x$cpu()))

    res$model_obj <- model_to_raw(model)
    res$estimates <- param_per_epoch[[1]]
    res$loss <- loss_vec[1]
    res$best_epoch <- best_epoch

    ## -------------------------------------------------------------------------
    # Training loop

    training_result <- run_saint_training_loop(
      model = model,
      dl = dl,
      dl_val = dl_val,
      loss_fn = loss_fn,
      optimizer_obj = optimizer_obj,
      epochs = epochs,
      learn_rate = learn_rate,
      stop_iter = stop_iter,
      validation = validation,
      class_weights = class_weights,
      loss_label = loss_label,
      verbose = verbose,
      rate_schedule = rate_schedule,
      ...
    )

    param_per_epoch <- c(
      list(param_per_epoch[[1]]),
      training_result$param_per_epoch
    )
    loss_vec <- c(loss_vec[1], training_result$loss_vec)
    best_epoch <- training_result$best_epoch

    res$model_obj <- model_to_raw(model)
    res$estimates <- param_per_epoch
    res$loss <- loss_vec
    res$best_epoch <- best_epoch

    ## -------------------------------------------------------------------------

    res$parameters$class_weights <- as.numeric(class_weights)
    names(res$parameters$class_weights) <- lvls

    res
  })

  training_output$device <- device

  training_output
}


# ------------------------------------------------------------------------------
# Training loop

run_saint_training_loop <- function(
  model,
  dl,
  dl_val,
  loss_fn,
  optimizer_obj,
  epochs,
  learn_rate,
  stop_iter,
  validation,
  class_weights = NULL,
  loss_label = "\tLoss:",
  verbose = FALSE,
  rate_schedule = "none",
  ...
) {
  loss_prev <- 10^38
  loss_min <- loss_prev
  poor_epoch <- 0
  best_epoch <- 1L
  loss_vec <- rep(NA_real_, epochs)
  param_per_epoch <- list()

  if (verbose) {
    epoch_chr <- format_epoch_labels(1:epochs)
  }

  for (epoch in 1:epochs) {
    learn_rate <- set_learn_rate(
      epoch - 1,
      learn_rate,
      type = rate_schedule,
      ...
    )
    for (i in seq_along(optimizer_obj$param_groups)) {
      optimizer_obj$param_groups[[i]]$lr <- learn_rate
    }

    model$train()

    coro::loop(
      for (batch in dl) {
        cl <- function() {
          optimizer_obj$zero_grad()
          pred <- model(batch$x_cat, batch$x_cont)

          if (is.null(class_weights)) {
            loss <- loss_fn(pred, batch$y)
          } else {
            loss <- loss_fn(pred, batch$y, class_weights)
          }

          loss$backward()
          loss
        }
        optimizer_obj$step(cl)
      }
    )

    model$eval()

    if (validation > 0) {
      bt <- dl_val$dataset$tensors
      pred <- model(bt$x_cat, bt$x_cont)
      if (is.null(class_weights)) {
        loss <- loss_fn(pred, bt$y)
      } else {
        loss <- loss_fn(pred, bt$y, class_weights)
      }
    } else {
      bt <- dl$dataset$tensors
      pred <- model(bt$x_cat, bt$x_cont)
      if (is.null(class_weights)) {
        loss <- loss_fn(pred, bt$y)
      } else {
        loss <- loss_fn(pred, bt$y, class_weights)
      }
    }

    loss_curr <- loss$item()
    loss_vec[epoch] <- loss_curr

    if (is.nan(loss_curr)) {
      cli::cli_warn(
        "Early stopping occurred at epoch {epoch} due to numerical overflow of the loss function."
      )
      break()
    }

    if (loss_curr >= loss_min) {
      poor_epoch <- poor_epoch + 1
    } else {
      loss_min <- loss_curr
      poor_epoch <- 0
      best_epoch <- as.integer(epoch)
    }

    loss_prev <- loss_curr

    param_per_epoch[[epoch]] <-
      lapply(model$state_dict(), function(x) torch::as_array(x$cpu()))

    if (verbose) {
      cli::cli_inform(
        "epoch: {epoch_chr[epoch]}, learn rate: {sprintf('%.5f', learn_rate)}, {loss_label} {signif(loss_curr, 3)}"
      )
    }

    if (poor_epoch == stop_iter) {
      break()
    }
  }

  list(
    param_per_epoch = param_per_epoch,
    loss_vec = loss_vec[1:length(param_per_epoch)],
    best_epoch = best_epoch
  )
}

# ------------------------------------------------------------------------------
# Torch modules

saint_geglu_module <- torch::nn_module(
  "saint_geglu",
  initialize = function() {},
  forward = function(x) {
    chunks <- x$chunk(2L, dim = -1L)
    chunks[[1]] * torch::nnf_gelu(chunks[[2]])
  }
)

saint_feedforward_module <- torch::nn_module(
  "saint_feedforward",
  initialize = function(dim, mult = 4L, dropout = 0) {
    self$net <- torch::nn_sequential(
      torch::nn_linear(dim, dim * mult * 2L),
      saint_geglu_module(),
      torch::nn_dropout(p = dropout),
      torch::nn_linear(dim * mult, dim)
    )
  },
  forward = function(x) {
    self$net(x)
  }
)

saint_attention_module <- torch::nn_module(
  "saint_attention",
  initialize = function(dim, heads = 8L, dim_head = 16L, dropout = 0) {
    inner_dim <- dim_head * heads
    self$heads <- heads
    self$scale <- dim_head^(-0.5)

    self$to_qkv <- torch::nn_linear(dim, inner_dim * 3L, bias = FALSE)
    self$to_out <- torch::nn_linear(inner_dim, dim)
    self$dropout <- torch::nn_dropout(p = dropout)
  },
  forward = function(x) {
    h <- self$heads
    b <- x$shape[1]
    n <- x$shape[2]

    qkv <- self$to_qkv(x)$chunk(3L, dim = -1L)
    q <- qkv[[1]]$reshape(c(b, n, h, -1L))$permute(c(1L, 3L, 2L, 4L))
    k <- qkv[[2]]$reshape(c(b, n, h, -1L))$permute(c(1L, 3L, 2L, 4L))
    v <- qkv[[3]]$reshape(c(b, n, h, -1L))$permute(c(1L, 3L, 2L, 4L))

    sim <- torch::torch_matmul(q, k$transpose(3L, 4L)) * self$scale
    attn <- torch::nnf_softmax(sim, dim = -1L)
    attn <- self$dropout(attn)
    out <- torch::torch_matmul(attn, v)

    out <- out$permute(c(1L, 3L, 2L, 4L))$reshape(c(b, n, -1L))
    self$to_out(out)
  }
)

saint_col_transformer_module <- torch::nn_module(
  "saint_col_transformer",
  initialize = function(dim, depth, heads, dim_head, attn_dropout, ff_dropout) {
    self$layers <- torch::nn_module_list()
    for (i in seq_len(depth)) {
      self$layers$append(torch::nn_module_list(list(
        torch::nn_layer_norm(dim),
        saint_attention_module(
          dim,
          heads = heads,
          dim_head = dim_head,
          dropout = attn_dropout
        ),
        torch::nn_layer_norm(dim),
        saint_feedforward_module(dim, dropout = ff_dropout)
      )))
    }
  },
  forward = function(x) {
    for (i in seq_along(self$layers)) {
      layer <- self$layers[[i]]
      norm1 <- layer[[1]]
      attn <- layer[[2]]
      norm2 <- layer[[3]]
      ff <- layer[[4]]
      x <- x + attn(norm1(x))
      x <- x + ff(norm2(x))
    }
    x
  }
)

saint_rowcol_transformer_module <- torch::nn_module(
  "saint_rowcol_transformer",
  initialize = function(
    dim,
    nfeats,
    depth,
    heads,
    dim_head,
    attn_dropout,
    ff_dropout,
    style = "both"
  ) {
    self$style <- style
    self$nfeats <- nfeats
    self$dim <- dim
    self$use_row_attention <- TRUE
    self$layers <- torch::nn_module_list()

    row_dim <- dim * nfeats

    for (i in seq_len(depth)) {
      if (style == "both") {
        self$layers$append(torch::nn_module_list(list(
          torch::nn_layer_norm(dim),
          saint_attention_module(
            dim,
            heads = heads,
            dim_head = dim_head,
            dropout = attn_dropout
          ),
          torch::nn_layer_norm(dim),
          saint_feedforward_module(dim, dropout = ff_dropout),
          torch::nn_layer_norm(row_dim),
          saint_attention_module(
            row_dim,
            heads = heads,
            dim_head = 64L,
            dropout = attn_dropout
          ),
          torch::nn_layer_norm(row_dim),
          saint_feedforward_module(row_dim, dropout = ff_dropout)
        )))
      } else {
        self$layers$append(torch::nn_module_list(list(
          torch::nn_layer_norm(row_dim),
          saint_attention_module(
            row_dim,
            heads = heads,
            dim_head = 64L,
            dropout = attn_dropout
          ),
          torch::nn_layer_norm(row_dim),
          saint_feedforward_module(row_dim, dropout = ff_dropout)
        )))
      }
    }
  },
  forward = function(x) {
    n <- self$nfeats
    d <- self$dim

    if (self$style == "both") {
      for (i in seq_along(self$layers)) {
        layer <- self$layers[[i]]
        norm1 <- layer[[1]]
        attn1 <- layer[[2]]
        norm2 <- layer[[3]]
        ff1 <- layer[[4]]
        norm3 <- layer[[5]]
        attn2 <- layer[[6]]
        norm4 <- layer[[7]]
        ff2 <- layer[[8]]

        x <- x + attn1(norm1(x))
        x <- x + ff1(norm2(x))

        if (self$use_row_attention) {
          b <- x$shape[1]
          x_row <- x$reshape(c(1L, b, n * d))
          x_row <- x_row + attn2(norm3(x_row))
          x_row <- x_row + ff2(norm4(x_row))
          x <- x_row$reshape(c(b, n, d))
        }
      }
    } else {
      if (self$use_row_attention) {
        for (i in seq_along(self$layers)) {
          layer <- self$layers[[i]]
          norm1 <- layer[[1]]
          attn1 <- layer[[2]]
          norm2 <- layer[[3]]
          ff1 <- layer[[4]]

          b <- x$shape[1]
          x_row <- x$reshape(c(1L, b, n * d))
          x_row <- x_row + attn1(norm1(x_row))
          x_row <- x_row + ff1(norm2(x_row))
          x <- x_row$reshape(c(b, n, d))
        }
      }
    }
    x
  }
)

saint_embedding_module <- torch::nn_module(
  "saint_embedding",
  initialize = function(
    pred_lvls,
    n_continuous,
    num_embedding,
    use_target_token = FALSE
  ) {
    self$n_cat <- length(pred_lvls)
    self$n_cont <- n_continuous
    self$num_embedding <- num_embedding
    self$use_target_token <- isTRUE(use_target_token)

    if (self$n_cat > 0) {
      self$cat_embeddings <- torch::nn_module_list(lapply(
        pred_lvls,
        function(card) {
          torch::nn_embedding(
            num_embeddings = card,
            embedding_dim = num_embedding
          )
        }
      ))
    }

    if (self$n_cont > 0) {
      self$cont_mlps <- torch::nn_module_list(lapply(
        seq_len(n_continuous),
        function(i) {
          torch::nn_sequential(
            torch::nn_linear(1L, 100L),
            torch::nn_relu(),
            torch::nn_linear(100L, num_embedding)
          )
        }
      ))
    }

    if (self$use_target_token) {
      self$target_token <- torch::nn_parameter(
        torch::torch_randn(1L, 1L, num_embedding)
      )
    }
  },
  forward = function(x_cat = NULL, x_cont = NULL) {
    parts <- list()

    if (self$n_cat > 0 && !is.null(x_cat)) {
      cat_embeds <- lapply(seq_len(self$n_cat), function(i) {
        self$cat_embeddings[[i]](x_cat[, i])
      })
      parts <- c(parts, list(torch::torch_stack(cat_embeds, dim = 2L)))
    }

    if (self$n_cont > 0 && !is.null(x_cont)) {
      cont_embeds <- lapply(seq_len(self$n_cont), function(i) {
        self$cont_mlps[[i]](x_cont[, i, drop = FALSE]$unsqueeze(-1L))$squeeze(
          2L
        )
      })
      parts <- c(parts, list(torch::torch_stack(cont_embeds, dim = 2L)))
    }

    feats <- torch::torch_cat(parts, dim = 2L)

    if (self$use_target_token) {
      batch <- feats$shape[1]
      tgt <- self$target_token$expand(c(batch, 1L, self$num_embedding))
      feats <- torch::torch_cat(list(tgt, feats), dim = 2L)
    }

    feats
  }
)

saint_module <- torch::nn_module(
  "saint_module",
  initialize = function(
    pred_lvls,
    n_continuous,
    num_embedding,
    attention_type,
    num_attn_heads,
    num_attn_blocks,
    dropout_attn,
    dropout_hidden,
    dropout_last,
    hidden_units,
    hidden_activations,
    y_dim,
    use_target_token = FALSE
  ) {
    num_features <- length(pred_lvls) + n_continuous
    self$num_features <- num_features
    self$num_embedding <- num_embedding
    self$use_target_token <- isTRUE(use_target_token)
    seq_len <- num_features + as.integer(self$use_target_token)

    self$embedding <- saint_embedding_module(
      pred_lvls = pred_lvls,
      n_continuous = n_continuous,
      num_embedding = num_embedding,
      use_target_token = self$use_target_token
    )

    if (attention_type == "column") {
      self$backbone <- saint_col_transformer_module(
        dim = num_embedding,
        depth = num_attn_blocks,
        heads = num_attn_heads,
        dim_head = 16L,
        attn_dropout = dropout_attn,
        ff_dropout = dropout_hidden
      )
    } else {
      self$backbone <- saint_rowcol_transformer_module(
        dim = num_embedding,
        nfeats = seq_len,
        depth = num_attn_blocks,
        heads = num_attn_heads,
        dim_head = 16L,
        attn_dropout = dropout_attn,
        ff_dropout = dropout_hidden,
        style = attention_type
      )
    }

    head_input_dim <- if (self$use_target_token) {
      num_embedding
    } else {
      seq_len * num_embedding
    }

    if (!is.null(hidden_units)) {
      hidden_layers <- list()
      input_dim <- head_input_dim
      for (i in seq_along(hidden_units)) {
        hidden_layers[[length(hidden_layers) + 1]] <-
          torch::nn_linear(input_dim, hidden_units[i])
        hidden_layers[[length(hidden_layers) + 1]] <-
          get_activation_fn(hidden_activations[i])
        input_dim <- hidden_units[i]
      }
      self$hidden <- torch::nn_sequential(!!!hidden_layers)
      if (dropout_last > 0) {
        self$hidden_drop <- torch::nn_dropout(p = dropout_last)
      } else {
        self$hidden_drop <- NULL
      }
      self$output_head <- torch::nn_linear(
        hidden_units[length(hidden_units)],
        y_dim
      )
    } else {
      self$hidden <- NULL
      self$hidden_drop <- NULL
      self$output_head <- torch::nn_linear(head_input_dim, y_dim)
    }

    self$y_dim <- y_dim
  },
  forward = function(x_cat = NULL, x_cont = NULL) {
    embeds <- self$embedding(x_cat, x_cont)
    h <- self$backbone(embeds)

    if (self$use_target_token) {
      h_pooled <- h[, 1L, ]
    } else {
      h_pooled <- h$reshape(c(h$shape[1], -1L))
    }

    if (!is.null(self$hidden)) {
      h_pooled <- self$hidden(h_pooled)
      if (!is.null(self$hidden_drop)) {
        h_pooled <- self$hidden_drop(h_pooled)
      }
    }
    # Classification returns raw logits; softmax is applied at predict time
    # so the loss can use nnf_cross_entropy (numerically stable).
    self$output_head(h_pooled)
  }
)

## -----------------------------------------------------------------------------

get_num_saint_coef <- function(x) {
  length(unlist(x$estimates[[1]]))
}

#' @export
print.brulee_saint <- function(x, ...) {
  cat(cli::style_bold("SAINT network"), "\n\n", sep = "")

  lvl <- get_levels(x)
  n <- format(x$dims$n, big.mark = ",")
  p <- format(x$dims$p, big.mark = ",")

  data_lst <- c(
    " " = "Samples: {n}",
    " " = "Predictors: {p} ({x$dims$p_cat} factor, {x$dims$p_cont} numeric)"
  )
  if (!is.null(lvl)) {
    data_lst <- c(data_lst, " " = "Classes: {.val {lvl}}")
  }
  cli::cli_bullets(data_lst)

  cat("\n")

  param_lst <- c(
    " " = "Attention type: {.val {x$parameters$attention_type}}",
    " " = "Embedding dim: {x$parameters$num_embedding}",
    " " = "Attention: {x$parameters$num_attn_heads} heads, {x$parameters$num_attn_blocks} block{?s}"
  )
  if (!is.null(x$parameters$hidden_units)) {
    units_str <- paste(x$parameters$hidden_units, collapse = ", ")
    hidden_info <- "Hidden layers: {units_str} units, {.val {unique(x$parameters$hidden_activations)}} activation"
    if (x$parameters$dropout_last > 0) {
      hidden_info <- paste0(
        hidden_info,
        ", dropout={signif(x$parameters$dropout_last, 3)}"
      )
    }
    param_lst <- c(param_lst, " " = hidden_info)
  }
  param_lst <- c(
    param_lst,
    " " = "Learning Rate: {signif(x$parameters$learn_rate, 3)}, Schedule: {.val {x$parameters$sched}}",
    " " = "Stopping iterations: {x$parameters$stop_iter}"
  )
  if (x$parameters$validation > 0) {
    param_lst <- c(
      param_lst,
      " " = "% Validation: {signif(x$parameters$validation, 3)}"
    )
  }
  if (x$parameters$dropout_attn > 0 || x$parameters$dropout_hidden > 0) {
    param_lst <- c(
      param_lst,
      " " = "Dropout: attention={signif(x$parameters$dropout_attn, 3)}, hidden={signif(x$parameters$dropout_hidden, 3)}"
    )
  }
  if (x$parameters$penalty > 0) {
    param_lst <- c(
      param_lst,
      " " = "Penalty: {signif(x$parameters$penalty, 3)}, {round(x$parameters$mixture * 100, 1)}% L1"
    )
  }
  param_lst <- c(param_lst, " " = "Optimizer: {.val {x$parameters$optimizer}}")
  if (x$parameters$optimizer != "LBFGS") {
    param_lst <- c(param_lst, " " = "Batch Size: {x$parameters$batch_size}")
  }
  if (!is.null(x$device)) {
    param_lst <- c(param_lst, " " = "Device: {.val {x$device}}")
  }

  cli::cli_bullets(param_lst)

  # Can take a long time
  # n_params <- format(get_num_saint_coef(x), big.mark = ",")
  # res_list <- c(" " = "# Parameters: {n_params}")
  res_list <- character(0)

  if (!is.null(x$loss)) {
    it <- x$best_epoch
    loss_val <- signif(x$loss[it], 3)
    epoch_str <- cli::pluralize("{it} epoch{?s}")

    if (x$parameters$validation > 0) {
      if (is.na(x$y_stats$mean)) {
        res_list <- c(
          res_list,
          " " = "validation loss after {epoch_str}: {loss_val}"
        )
      } else {
        res_list <- c(
          res_list,
          " " = "scaled validation loss after {epoch_str}: {loss_val}"
        )
      }
    } else {
      if (is.na(x$y_stats$mean)) {
        res_list <- c(
          res_list,
          " " = "training set loss after {epoch_str}: {loss_val}"
        )
      } else {
        res_list <- c(
          res_list,
          " " = "scaled training set loss after {epoch_str}: {loss_val}"
        )
      }
    }
  }

  cat("\n")
  cli::cli_bullets(res_list)

  invisible(x)
}
