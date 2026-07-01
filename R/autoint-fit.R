#' Fit AutoInt models for tabular data
#'
#' `brulee_auto_int()` fits AutoInt from Song _at al_ (2019) that use multi-head
#' columnar self-attention to help exploit how combinations of embeddings can be
#' used to improve specific predictions.
#'
#' @inheritParams brulee_mlp
#' @param num_embedding An integer for the embedding dimension. Each feature
#'   (categorical or continuous) is mapped to a vector of this dimension.
#'   Must be >= 1.
#' @param num_attn_feat An integer for the per-head attention dimension. The
#'   total attention dimension is `num_attn_feat * num_attn_heads`. Must be >= 1.
#' @param num_attn_heads An integer for the number of attention heads. Each head
#'   learns different interaction patterns in parallel. Must be >= 1.
#' @param num_attn_blocks An integer for the number of stacked self-attention
#'   layers. More layers capture higher-order interactions. Must be >= 1.
#' @param dropout A number in `[0, 1)` for the dropout rate applied between
#'   the last hidden layer and the output head. Only has effect when
#'   `hidden_units` is not `NULL`. Default is 0 (no dropout).
#' @param dropout_attn A number in `[0, 1)` for the dropout rate applied to
#'   attention weights during training.
#' @param dropout_embedding A number in `[0, 1)` for the dropout rate applied
#'   to the embedding layer during training.
#' @param activation A single character string for the activation function used
#'   in the self-attention backbone (applied after each residual connection in
#'   each attention block). This does not affect the optional hidden layers; use
#'   `hidden_activations` for those. See [brulee_activations()] for options.
#' @param hidden_units An integer vector for the number of units in optional
#'   hidden layers between the attention backbone and the output head. For
#'   example, `c(64L, 32L)` adds two hidden layers with 64 and 32 units.
#'   When `NULL` (the default), no hidden layers are added.
#' @param hidden_activations A character vector of activation functions for the
#'   hidden layers. Must be the same length as `hidden_units` or a single value
#'   that will be recycled. When `NULL` (the default), no hidden layers are
#'   added. See [brulee_activations()] for options.
#'
#' @details
#'
#' ## What is Being Estimated
#'
#' In statistics, an interaction occurs when two or more predictors jointly
#' predict the outcome. You need to know the values of all predictors within
#' the interaction effect to appropriately model the data. AutoInt is often
#' described as "automatically learning feature interactions," but that is not
#' an accurate description.
#'
#' In neural networks, the original predictors are converted to _embeddings_,
#'  which are often the hidden units of the network.
#'
#' AutoInt uses _column attention_ to change how embeddings are represented. It
#' learns how to make the embeddings more relevant to the outcome by creating
#' mixtures of them. For example, if we predict a data point in one part of the
#' predictor space, attention will refocus (i.e., transform) the embedding to
#' be more relevant to that part of the space.
#'
#' ## Architecture
#'
#' The AutoInt architecture has three stages:
#'
#' 1. **Embedding layer**: Maps every feature (categorical or continuous) into
#'    a shared vector space of dimension `num_embedding`.
#' 2. **Self-attention backbone**: A stack of `num_attn_blocks` multi-head
#'    self-attention layers. After all blocks, a residual connection from
#'    the original embeddings is added and an activation is applied.
#' 3. **Hidden layers** (optional): If `hidden_units` is specified, one or more
#'    fully-connected layers with activations process the flattened attention
#'    output before the output head.
#' 4. **Output head**: Projects to the output dimension via a linear layer.
#'
#' Unlike other \pkg{brulee} models, `brulee_auto_int()` natively handles factor
#' predictors via learned embeddings. Factor columns are automatically detected
#' and embedded, while numeric columns use a scaled embedding. There is _no need
#' to pre-encode factors as indicators_.
#'
#' ## Attention Parameters
#'
#' The self-attention backbone has several tuning parameters that control its
#' capacity and regularization:
#'
#' - `num_attn_heads`: The number of attention heads that operate **in parallel**
#'   within each attention block. Each head independently learns which features
#'   interact, giving the model multiple "views" of the feature relationships.
#'   The total attention dimension per block is `num_attn_feat * num_attn_heads`.
#'
#' - `num_attn_feat`: The per-head attention dimension. Each head projects
#'   features into a space of this size to compute attention scores. Larger
#'   values give each head more capacity to represent complex interactions.
#'
#' - `num_attn_blocks`: The number of attention layers stacked **sequentially**.
#'   Each block's output feeds into the next, allowing the model to build
#'   higher-order interactions (e.g., block 1 captures pairwise interactions,
#'   block 2 can combine those into three-way interactions, etc.).
#'
#' - `activation`: The activation function applied after the residual connection
#'   at the end of the attention backbone.
#'
#' - `dropout_attn`: Dropout applied to the attention weight matrix within each
#'   block, which randomly zeroes out attention connections during training.
#'
#' ## Learning Rates
#'
#' The learning rate can be set to constant (the default) or dynamically set
#' via a learning rate scheduler (via the `rate_schedule`). Using
#' `rate_schedule = 'none'` uses the `learn_rate` argument. Otherwise, any
#' arguments to the schedulers can be passed via `...`.
#'
#' ## Other Notes
#'
#' When the outcome is a number, the function internally standardizes the
#' outcome data to have mean zero and a standard deviation of one. The prediction
#' function creates predictions on the original scale.
#'
#' By default, training halts when the validation loss increases for at least
#' `stop_iter` iterations. If `validation = 0` the training set loss is used.
#'
#' The model objects are saved for each epoch so that the number of epochs can
#' be efficiently tuned. Both the [predict()] method for this model has an
#' `epoch` argument (which defaults to the epoch with the best loss value).
#'
#' The use of the L1 penalty (a.k.a. the lasso penalty) does _not_ force
#' parameters to be strictly zero (as it does in packages such as \pkg{glmnet}).
#' The zeroing out of parameters is a specific feature the optimization method
#' used in those packages.
#'
#' @references
#'
#' Song, W., Shi, C., Xiao, Z., Duan, Z., Xu, Y., Zhang, M., & Tang, J.
#' (2019). AutoInt: Automatic Feature Interaction Learning via Self-Attentive
#' Neural Networks. In _Proceedings of the 28th ACM International Conference on
#' Information and Knowledge Management (CIKM)_.
#'
#' @seealso [predict.brulee_auto_int()], [autoplot.brulee_auto_int()]
#'
#' @return
#'
#' A `brulee_auto_int` object with elements:
#'  * `models_obj`: a serialized raw vector for the torch module.
#'  * `estimates`: a list of matrices with the model parameter estimates per
#'                 epoch. The first element is epoch zero (the randomly
#'                 initialized parameters before training), so the list has
#'                 `epochs + 1` elements.
#'  * `best_epoch`: an integer for the epoch with the smallest loss. Since
#'                  `estimates` and `loss` include epoch zero, this epoch's
#'                  values are at position `best_epoch + 1` in those objects.
#'  * `loss`: A vector of loss values (MSE for regression, negative log-
#'            likelihood for classification) at each epoch, starting with
#'            epoch zero.
#'  * `dim`: A list of data dimensions and feature metadata.
#'  * `top_interactions`: A tibble containing the top 10 two-way feature
#'                        interactions.
#'  * `y_stats`: A list of summary statistics for numeric outcomes.
#'  * `parameters`: A list of some tuning parameter values.
#'  * `device`: A character string for the device used during training.
#'  * `blueprint`: The `hardhat` blueprint data.
#'
#' @examplesIf !brulee:::is_cran_check()
#' \donttest{
#' pkgs <- c("recipes", "yardstick", "modeldata")
#' if (torch::torch_is_installed() && rlang::is_installed(pkgs)) {
#'
#'   set.seed(87261)
#'   tr_data <- modeldata::sim_regression(500)
#'   te_data <- modeldata::sim_regression(50)
#'
#'   set.seed(2)
#'   fit <- brulee_auto_int(outcome ~ ., data = tr_data,
#'                          epochs = 50L, batch_size = 64L, stop_iter = 10L,
#'                          learn_rate = 0.01, penalty = 0.01)
#'   fit
#'
#'   autoplot(fit)
#'
#'   library(yardstick)
#'   predict(fit, te_data) |>
#'    dplyr::bind_cols(te_data) |>
#'    rmse(outcome, .pred)
#'
#' }
#' }
#' @export
brulee_auto_int <- function(x, ...) {
  UseMethod("brulee_auto_int")
}

#' @export
#' @rdname brulee_auto_int
brulee_auto_int.default <- function(x, ...) {
  stop(
    "`brulee_auto_int()` is not defined for a '",
    class(x)[1],
    "'.",
    call. = FALSE
  )
}

# XY method - data frame

#' @export
#' @rdname brulee_auto_int
brulee_auto_int.data.frame <- function(
  x,
  y,
  epochs = 100L,
  num_embedding = 16L,
  num_attn_feat = 16L,
  num_attn_heads = 2L,
  num_attn_blocks = 3L,
  activation = "relu",
  hidden_units = NULL,
  hidden_activations = NULL,
  penalty = 0.001,
  mixture = 0,
  dropout = 0,
  dropout_attn = 0,
  dropout_embedding = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.01,

  rate_schedule = "none",
  momentum = 0.0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
  device = NULL,
  ...
) {
  processed <- hardhat::mold(x, y)

  brulee_auto_int_bridge(
    processed,
    epochs = epochs,
    num_embedding = num_embedding,
    num_attn_feat = num_attn_feat,
    num_attn_heads = num_attn_heads,
    num_attn_blocks = num_attn_blocks,
    activation = activation,
    hidden_units = hidden_units,
    hidden_activations = hidden_activations,
    dropout = dropout,
    learn_rate = learn_rate,
    rate_schedule = rate_schedule,
    penalty = penalty,
    mixture = mixture,
    dropout_attn = dropout_attn,
    dropout_embedding = dropout_embedding,
    validation = validation,
    optimizer = optimizer,
    momentum = momentum,
    batch_size = batch_size,
    class_weights = class_weights,
    stop_iter = stop_iter,
    grad_value_clip = grad_value_clip,
    grad_norm_clip = grad_norm_clip,
    verbose = verbose,
    device = device,
    ...
  )
}

# XY method - matrix

#' @export
#' @rdname brulee_auto_int
brulee_auto_int.matrix <- function(
  x,
  y,
  epochs = 100L,
  num_embedding = 16L,
  num_attn_feat = 16L,
  num_attn_heads = 2L,
  num_attn_blocks = 3L,
  activation = "relu",
  hidden_units = NULL,
  hidden_activations = NULL,
  dropout = 0,
  penalty = 0.001,
  mixture = 0,
  dropout_attn = 0,
  dropout_embedding = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.01,
  rate_schedule = "none",
  momentum = 0.0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
  device = NULL,
  ...
) {
  processed <- hardhat::mold(x, y)

  brulee_auto_int_bridge(
    processed,
    epochs = epochs,
    num_embedding = num_embedding,
    num_attn_feat = num_attn_feat,
    num_attn_heads = num_attn_heads,
    num_attn_blocks = num_attn_blocks,
    activation = activation,
    hidden_units = hidden_units,
    hidden_activations = hidden_activations,
    dropout = dropout,
    learn_rate = learn_rate,
    rate_schedule = rate_schedule,
    momentum = momentum,
    penalty = penalty,
    mixture = mixture,
    dropout_attn = dropout_attn,
    dropout_embedding = dropout_embedding,
    validation = validation,
    optimizer = optimizer,
    batch_size = batch_size,
    class_weights = class_weights,
    stop_iter = stop_iter,
    grad_value_clip = grad_value_clip,
    grad_norm_clip = grad_norm_clip,
    verbose = verbose,
    device = device,
    ...
  )
}

# Formula method

#' @export
#' @rdname brulee_auto_int
brulee_auto_int.formula <- function(
  formula,
  data,
  epochs = 100L,
  num_embedding = 16L,
  num_attn_feat = 16L,
  num_attn_heads = 2L,
  num_attn_blocks = 3L,
  activation = "relu",
  hidden_units = NULL,
  hidden_activations = NULL,
  dropout = 0,
  penalty = 0.001,
  mixture = 0,
  dropout_attn = 0,
  dropout_embedding = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.01,
  rate_schedule = "none",
  momentum = 0.0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
  device = NULL,
  ...
) {
  processed <- hardhat::mold(
    formula,
    data,
    blueprint = hardhat::default_formula_blueprint(indicators = "none")
  )

  brulee_auto_int_bridge(
    processed,
    epochs = epochs,
    num_embedding = num_embedding,
    num_attn_feat = num_attn_feat,
    num_attn_heads = num_attn_heads,
    num_attn_blocks = num_attn_blocks,
    activation = activation,
    hidden_units = hidden_units,
    hidden_activations = hidden_activations,
    dropout = dropout,
    learn_rate = learn_rate,
    rate_schedule = rate_schedule,
    momentum = momentum,
    penalty = penalty,
    mixture = mixture,
    dropout_attn = dropout_attn,
    dropout_embedding = dropout_embedding,
    validation = validation,
    optimizer = optimizer,
    batch_size = batch_size,
    class_weights = class_weights,
    stop_iter = stop_iter,
    grad_value_clip = grad_value_clip,
    grad_norm_clip = grad_norm_clip,
    verbose = verbose,
    device = device,
    ...
  )
}

# Recipe method

#' @export
#' @rdname brulee_auto_int
brulee_auto_int.recipe <- function(
  x,
  data,
  epochs = 100L,
  num_embedding = 16L,
  num_attn_feat = 16L,
  num_attn_heads = 2L,
  num_attn_blocks = 3L,
  activation = "relu",
  hidden_units = NULL,
  hidden_activations = NULL,
  dropout = 0,
  penalty = 0.001,
  mixture = 0,
  dropout_attn = 0,
  dropout_embedding = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.01,
  rate_schedule = "none",
  momentum = 0.0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
  device = NULL,
  ...
) {
  processed <- hardhat::mold(x, data)

  brulee_auto_int_bridge(
    processed,
    epochs = epochs,
    num_embedding = num_embedding,
    num_attn_feat = num_attn_feat,
    num_attn_heads = num_attn_heads,
    num_attn_blocks = num_attn_blocks,
    activation = activation,
    hidden_units = hidden_units,
    hidden_activations = hidden_activations,
    dropout = dropout,
    learn_rate = learn_rate,
    rate_schedule = rate_schedule,
    momentum = momentum,
    penalty = penalty,
    mixture = mixture,
    dropout_attn = dropout_attn,
    dropout_embedding = dropout_embedding,
    validation = validation,
    optimizer = optimizer,
    batch_size = batch_size,
    class_weights = class_weights,
    stop_iter = stop_iter,
    grad_value_clip = grad_value_clip,
    grad_norm_clip = grad_norm_clip,
    verbose = verbose,
    device = device,
    ...
  )
}

# ------------------------------------------------------------------------------
# Bridge

brulee_auto_int_bridge <- function(
  processed,
  epochs,
  num_embedding,
  num_attn_feat,
  num_attn_heads,
  num_attn_blocks,
  activation,
  hidden_units,
  hidden_activations,
  dropout,
  learn_rate,
  rate_schedule,
  momentum,
  penalty,
  mixture,
  dropout_attn,
  dropout_embedding,
  class_weights,
  validation,
  optimizer,
  batch_size,
  stop_iter,
  grad_value_clip,
  grad_norm_clip,
  verbose,
  device,
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

  # Validate AutoInt-specific arguments
  auto_int_validated <- validate_auto_int_args(
    num_embedding = num_embedding,
    num_attn_feat = num_attn_feat,
    num_attn_heads = num_attn_heads,
    num_attn_blocks = num_attn_blocks,
    activation = activation,
    dropout_attn = dropout_attn,
    dropout_embedding = dropout_embedding,
    grad_value_clip = grad_value_clip,
    grad_norm_clip = grad_norm_clip,
    call = call
  )

  # Validate hidden layer arguments
  hidden_validated <- validate_hidden_args(
    hidden_units = hidden_units,
    hidden_activations = hidden_activations,
    call = call
  )

  check_double(dropout, single = TRUE, 0, 1, incl = c(TRUE, FALSE), call = call)

  # Handle batch_size
  if (!is.null(batch_size) && optimizer != "LBFGS") {
    if (is.numeric(batch_size) && !is.integer(batch_size)) {
      batch_size <- as.integer(batch_size)
    }
    check_integer(batch_size, single = TRUE, 1, call = call)
  }
  if (is.null(batch_size) && optimizer != "LBFGS") {
    batch_size <- 256L
  }

  # Validate common arguments
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
  # Split predictors into categorical and continuous

  predictors <- processed$predictors
  split <- split_predictors_auto_int(predictors, call = call)

  ## ---------------------------------------------------------------------------
  # Validate outcome (accepts both numeric and factor)

  outcome <- validate_mlp_outcome(processed$outcomes[[1]], call = call)

  # ----------------------------------------------------------------------------

  lvls <- levels(outcome)
  xtab <- table(outcome)
  class_weights <- check_class_weights(class_weights, lvls, xtab, call = call)

  ## ---------------------------------------------------------------------------

  fit <- auto_int_fit_imp(
    x_cat = split$x_cat,
    x_cont = split$x_cont,
    pred_lvls = split$pred_lvls,
    cat_names = split$cat_names,
    cont_names = split$cont_names,
    y = outcome,
    epochs = epochs,
    num_embedding = auto_int_validated$num_embedding,
    num_attn_feat = auto_int_validated$num_attn_feat,
    num_attn_heads = auto_int_validated$num_attn_heads,
    num_attn_blocks = auto_int_validated$num_attn_blocks,
    activation = auto_int_validated$activation,
    hidden_units = hidden_validated$hidden_units,
    hidden_activations = hidden_validated$hidden_activations,
    dropout = dropout,
    learn_rate = learn_rate,
    rate_schedule = rate_schedule,
    momentum = momentum,
    penalty = penalty,
    mixture = mixture,
    dropout_attn = auto_int_validated$dropout_attn,
    dropout_embedding = auto_int_validated$dropout_embedding,
    validation = validation,
    optimizer = optimizer,
    batch_size = batch_size,
    class_weights = class_weights,
    stop_iter = stop_iter,
    grad_value_clip = grad_value_clip,
    grad_norm_clip = grad_norm_clip,
    verbose = verbose,
    device = device,
    ...
  )

  new_brulee_auto_int(
    model_obj = fit$model_obj,
    estimates = fit$estimates,
    best_epoch = fit$best_epoch,
    loss = fit$loss,
    dims = fit$dims,
    top_interactions = fit$top_interactions,
    y_stats = fit$y_stats,
    output_type = fit$output_type,
    parameters = fit$parameters,
    device = fit$device,
    blueprint = processed$blueprint
  )
}

# ------------------------------------------------------------------------------
# Predictor splitting

split_predictors_auto_int <- function(predictors, call = rlang::caller_env()) {
  cat_idx <- purrr::map_lgl(
    predictors,
    \(col) is.factor(col) || is.character(col)
  )
  cont_idx <- purrr::map_lgl(predictors, \(col) is.numeric(col))

  cat_names <- names(predictors)[cat_idx]
  cont_names <- names(predictors)[cont_idx]

  # Check for unsupported types
  other_idx <- !cat_idx & !cont_idx
  if (any(other_idx)) {
    bad <- names(predictors)[other_idx]
    cli::cli_abort(
      "Column{?s} {.val {bad}} {?is/are} neither factor/character nor numeric.
       {.fn brulee_auto_int} requires all predictors to be factor or numeric.",
      call = call
    )
  }

  # Process categorical
  if (length(cat_names) > 0) {
    for (nm in cat_names) {
      if (!is.factor(predictors[[nm]])) {
        predictors[[nm]] <- as.factor(predictors[[nm]])
      }
    }
    pred_lvls <- purrr::map_int(cat_names, \(nm) nlevels(predictors[[nm]]))
    names(pred_lvls) <- cat_names
    x_cat <- do.call(
      cbind,
      lapply(cat_names, function(nm) {
        as.integer(predictors[[nm]])
      })
    )
    colnames(x_cat) <- cat_names
  } else {
    pred_lvls <- integer(0)
    x_cat <- NULL
  }

  # Process continuous
  if (length(cont_names) > 0) {
    x_cont <- as.matrix(predictors[, cont_names, drop = FALSE])
  } else {
    x_cont <- NULL
  }

  list(
    x_cat = x_cat,
    x_cont = x_cont,
    pred_lvls = pred_lvls,
    cat_names = cat_names,
    cont_names = cont_names
  )
}

# ------------------------------------------------------------------------------
# Validation

validate_auto_int_args <- function(
  num_embedding,
  num_attn_feat,
  num_attn_heads,
  num_attn_blocks,
  activation,
  dropout_attn,
  dropout_embedding,
  grad_value_clip,
  grad_norm_clip,
  call = rlang::caller_env()
) {
  if (is.numeric(num_embedding) && !is.integer(num_embedding)) {
    num_embedding <- as.integer(num_embedding)
  }
  check_integer(num_embedding, single = TRUE, 1, call = call)

  if (is.numeric(num_attn_feat) && !is.integer(num_attn_feat)) {
    num_attn_feat <- as.integer(num_attn_feat)
  }
  check_integer(num_attn_feat, single = TRUE, 1, call = call)

  if (is.numeric(num_attn_heads) && !is.integer(num_attn_heads)) {
    num_attn_heads <- as.integer(num_attn_heads)
  }
  check_integer(num_attn_heads, single = TRUE, 1, call = call)

  if (is.numeric(num_attn_blocks) && !is.integer(num_attn_blocks)) {
    num_attn_blocks <- as.integer(num_attn_blocks)
  }
  check_integer(num_attn_blocks, single = TRUE, 1, call = call)

  check_double(dropout_attn, single = TRUE, 0, call = call)
  if (dropout_attn >= 1) {
    cli::cli_abort("{.arg dropout_attn} must be less than 1.", call = call)
  }

  check_double(dropout_embedding, single = TRUE, 0, call = call)
  if (dropout_embedding >= 1) {
    cli::cli_abort("{.arg dropout_embedding} must be less than 1.", call = call)
  }

  check_string(activation, call = call)
  act_choices <- brulee_activations()
  if (!(activation %in% act_choices)) {
    cli::cli_abort(
      "{.arg activation} should be one of: {.val {act_choices}}.",
      call = call
    )
  }

  check_double(
    grad_value_clip,
    single = TRUE,
    0,
    Inf,
    incl = c(FALSE, TRUE),
    call = call
  )
  check_double(
    grad_norm_clip,
    single = TRUE,
    0,
    Inf,
    incl = c(FALSE, TRUE),
    call = call
  )

  list(
    num_embedding = num_embedding,
    num_attn_feat = num_attn_feat,
    num_attn_heads = num_attn_heads,
    num_attn_blocks = num_attn_blocks,
    activation = activation,
    dropout_attn = dropout_attn,
    dropout_embedding = dropout_embedding,
    grad_value_clip = grad_value_clip,
    grad_norm_clip = grad_norm_clip
  )
}

validate_hidden_args <- function(
  hidden_units,
  hidden_activations,
  call = rlang::caller_env()
) {
  if (is.null(hidden_units) && is.null(hidden_activations)) {
    return(list(hidden_units = NULL, hidden_activations = NULL))
  }

  if (is.null(hidden_units) != is.null(hidden_activations)) {
    cli::cli_abort(
      "Both {.arg hidden_units} and {.arg hidden_activations} must be provided together or both be {.code NULL}.",
      call = call
    )
  }

  if (is.numeric(hidden_units) && !is.integer(hidden_units)) {
    hidden_units <- as.integer(hidden_units)
  }
  check_integer(hidden_units, single = FALSE, 1, call = call)

  if (length(hidden_units) > 1 && length(hidden_activations) == 1) {
    hidden_activations <- rep(hidden_activations, length(hidden_units))
  }

  if (length(hidden_units) != length(hidden_activations)) {
    cli::cli_abort(
      "{.arg hidden_activations} must be a single value or a vector with the same length as {.arg hidden_units}.",
      call = call
    )
  }

  allowed_activation <- brulee_activations()
  good_activation <- hidden_activations %in% allowed_activation
  if (!all(good_activation)) {
    cli::cli_abort(
      "{.arg hidden_activations} should be one of: {.val {allowed_activation}}, not {.val {hidden_activations[!good_activation]}}.",
      call = call
    )
  }

  list(
    hidden_units = hidden_units,
    hidden_activations = hidden_activations
  )
}

# ------------------------------------------------------------------------------
# Constructor

new_brulee_auto_int <- function(
  model_obj,
  estimates,
  best_epoch,
  loss,
  dims,
  top_interactions,
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

  if (!inherits(top_interactions, "tbl_df")) {
    cli::cli_abort("{.arg top_interactions} should be a tibble", call = NULL)
  }
  int_nms <- c("attention_weight", "feature_1", "feature_2")
  if (!identical(sort(names(top_interactions)), int_nms)) {
    cli::cli_abort(
      "{.arg top_interactions} should be a tibble with names {.val {int_nms}}.",
      call = NULL
    )
  }

  num_items <- lengths(estimates)
  estimates <- estimates[num_items > 0]

  hardhat::new_model(
    model_obj = model_obj,
    estimates = estimates,
    best_epoch = best_epoch,
    loss = loss,
    dims = dims,
    top_interactions = top_interactions,
    y_stats = y_stats,
    output_type = output_type,
    parameters = parameters,
    device = device,
    blueprint = blueprint,
    class = "brulee_auto_int"
  )
}

## -----------------------------------------------------------------------------
# Fit implementation

auto_int_fit_imp <- function(
  x_cat,
  x_cont,
  pred_lvls,
  cat_names,
  cont_names,
  y,
  epochs = 100L,
  batch_size = 256L,
  num_embedding = 16L,
  num_attn_feat = 16L,
  num_attn_heads = 2L,
  num_attn_blocks = 3L,
  penalty = 0.001,
  mixture = 0,
  dropout_attn = 0,
  dropout_embedding = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.01,
  rate_schedule = "none",
  momentum = 0.0,
  activation = "relu",
  hidden_units = NULL,
  hidden_activations = NULL,
  dropout = 0,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
  device = "cpu",
  ...
) {
  start_seed <- sample.int(10^5, 1)
  torch::torch_manual_seed(start_seed)

  ## ---------------------------------------------------------------------------

  n <- length(y)
  p_cat <- length(pred_lvls)
  if (is.null(x_cont)) {
    p_cont <- 0L
  } else {
    p_cont <- ncol(x_cont)
  }
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
  # Split validation

  if (validation > 0) {
    in_val <- sample(seq_len(n), floor(n * validation))
    if (!is.null(x_cat)) {
      x_cat_val <- x_cat[in_val, , drop = FALSE]
    } else {
      x_cat_val <- NULL
    }
    if (!is.null(x_cont)) {
      x_cont_val <- x_cont[in_val, , drop = FALSE]
    } else {
      x_cont_val <- NULL
    }
    y_val <- y[in_val]
    if (!is.null(x_cat)) {
      x_cat <- x_cat[-in_val, , drop = FALSE]
    } else {
      x_cat <- NULL
    }
    if (!is.null(x_cont)) {
      x_cont <- x_cont[-in_val, , drop = FALSE]
    } else {
      x_cont <- NULL
    }
    y <- y[-in_val]
  } else {
    x_cat_val <- NULL
    x_cont_val <- NULL
    y_val <- NULL
  }

  ## ---------------------------------------------------------------------------
  # Scale numeric outcomes

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
  # Batch size

  if (optimizer == "LBFGS") {
    batch_size <- length(y)
  }
  batch_size <- min(batch_size, length(y))

  ## ---------------------------------------------------------------------------
  # Setup torch data

  or_dtype <- torch::torch_get_default_dtype()
  on.exit(torch::torch_set_default_dtype(or_dtype))
  torch::torch_set_default_dtype(torch::torch_float32())

  # Build the module on the CPU, then move it to `device`. See the
  # "Device-handling notes" comment block at the top of R/0_utils.R for the
  # full rationale. AutoInt's embedding layer holds a learnable
  # `cont_weights` parameter (registered via `nn_parameter`) plus categorical
  # `nn_embedding` modules; inside `with_device(mps, ...)` these would be
  # allocated and initialized on MPS, drawing from the MPS RNG, which
  # `torch_manual_seed()` does NOT reliably reset. CPU init drives those
  # draws from the properly-seeded CPU RNG, then `model$to(device)` moves
  # everything to the target backend, giving reproducible initial weights
  # from the same seed on every backend.
  torch::torch_manual_seed(start_seed + 1)
  model <- auto_int_module(
    pred_lvls = pred_lvls,
    n_continuous = p_cont,
    num_embedding = num_embedding,
    num_attn_feat = num_attn_feat,
    num_attn_heads = num_attn_heads,
    num_attn_blocks = num_attn_blocks,
    dropout_attn = dropout_attn,
    dropout_embedding = dropout_embedding,
    activation = activation,
    hidden_units = hidden_units,
    hidden_activations = hidden_activations,
    dropout = dropout,
    y_dim = y_dim
  )
  model$to(device = device)

  training_output <- torch::with_device(device = device, {
    # NOTE: every torch_tensor() / float_32() call below passes `device`
    # explicitly. `with_device(...)` looks like it should set a default
    # device for these calls, but it does NOT propagate to torch_tensor() --
    # see the "Device-handling notes" at the top of R/0_utils.R. Without
    # the explicit `device =` arg these tensors would land on the CPU and
    # later trigger device-mismatch errors when fed to the MPS/CUDA model.
    make_auto_int_tensors <- function(xc, xn, yv) {
      if (!is.null(xc)) {
        t_cat <- torch::torch_tensor(
          xc,
          dtype = torch::torch_long(),
          device = device
        )
      } else {
        t_cat <- NULL
      }
      if (!is.null(xn)) {
        t_cont <- float_32(xn, device = device)
      } else {
        t_cont <- NULL
      }
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

    train_tensors <- make_auto_int_tensors(x_cat, x_cont, y)

    # Build dataset using tensor_dataset with named tensors
    # We combine cat and cont into the dataset; the model forward will split them
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
    dl <- torch::dataloader(ds, batch_size = batch_size)

    dl_val <- NULL
    if (validation > 0) {
      val_tensors <- make_auto_int_tensors(x_cat_val, x_cont_val, y_val)
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

    ## ---------------------------------------------------------------------------
    # Return value scaffold

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
        activation = activation,
        hidden_units = hidden_units,
        hidden_activations = hidden_activations,
        dropout = dropout,
        num_embedding = num_embedding,
        num_attn_feat = num_attn_feat,
        num_attn_heads = num_attn_heads,
        num_attn_blocks = num_attn_blocks,
        learn_rate = learn_rate,
        class_weights = as.numeric(class_weights),
        penalty = penalty,
        mixture = mixture,
        dropout_attn = dropout_attn,
        dropout_embedding = dropout_embedding,
        validation = validation,
        optimizer = optimizer,
        batch_size = batch_size,
        momentum = momentum,
        stop_iter = stop_iter,
        grad_value_clip = grad_value_clip,
        grad_norm_clip = grad_norm_clip,
        sched = rate_schedule,
        sched_opt = list(...)
      )
    )

    ## ---------------------------------------------------------------------------
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

    ## ---------------------------------------------------------------------------
    # Initial evaluation (epoch 0)

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
      purrr::map(model$state_dict(), \(x) torch::as_array(x$cpu()))

    res$model_obj <- model_to_raw(model)
    res$estimates <- param_per_epoch[[1]]
    res$loss <- loss_vec[1]
    res$best_epoch <- best_epoch

    ## ---------------------------------------------------------------------------
    # Training loop

    training_result <- run_auto_int_training_loop(
      model = model,
      dl = dl,
      dl_val = dl_val,
      loss_fn = loss_fn,
      optimizer_obj = optimizer_obj,
      epochs = epochs,
      learn_rate = learn_rate,
      stop_iter = stop_iter,
      grad_value_clip = grad_value_clip,
      grad_norm_clip = grad_norm_clip,
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

    ## ---------------------------------------------------------------------------
    # Compute top feature interactions

    model$eval()
    torch::with_no_grad({
      if (validation > 0) {
        bt <- get_batch_tensors(dl_val$dataset)
      } else {
        bt <- get_batch_tensors(dl$dataset)
      }
      model(bt$x_cat, bt$x_cont)
    })

    res$top_interactions <- compute_top_interactions(
      model$backbone$last_attention_weights,
      all_features
    )

    ## ---------------------------------------------------------------------------

    res$parameters$class_weights <- as.numeric(class_weights)
    names(res$parameters$class_weights) <- lvls

    res
  })

  training_output$device <- device

  training_output
}


# ------------------------------------------------------------------------------
# AutoInt training loop (handles dual-input x_cat/x_cont)

run_auto_int_training_loop <- function(
  model,
  dl,
  dl_val,
  loss_fn,
  optimizer_obj,
  epochs,
  learn_rate,
  stop_iter,
  grad_value_clip = Inf,
  grad_norm_clip = Inf,
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

    # Training batches
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

          if (is.finite(grad_value_clip)) {
            try(
              torch::nn_utils_clip_grad_value_(
                model$parameters,
                grad_value_clip
              ),
              silent = TRUE
            )
          }
          if (is.finite(grad_norm_clip)) {
            try(
              torch::nn_utils_clip_grad_norm_(
                model$parameters,
                grad_norm_clip
              ),
              silent = TRUE
            )
          }

          loss
        }
        optimizer_obj$step(cl)
      }
    )

    # Evaluate on validation or training set
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

    # Save parameters for this epoch before any early-stopping check so the
    # returned loss curve (including a terminal NaN from numerical overflow)
    # stays aligned with `param_per_epoch`.
    param_per_epoch[[epoch]] <-
      purrr::map(model$state_dict(), \(x) torch::as_array(x$cpu()))

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
      best_epoch <- epoch
    }

    loss_prev <- loss_curr

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
    loss_vec = loss_vec[seq_along(param_per_epoch)],
    best_epoch = best_epoch
  )
}

# ------------------------------------------------------------------------------
# Compute top feature interactions from attention weights

compute_top_interactions <- function(
  attention_weights,
  feature_names,
  n_top = NULL
) {
  n_blocks <- length(attention_weights)
  n_feat <- length(feature_names)

  if (is.null(n_top)) {
    n_top <- n_feat * (n_feat - 1)
  }

  # Average attention across all blocks and batch
  avg_mats <- lapply(attention_weights, function(wt) {
    as.matrix(wt$mean(dim = 1)$squeeze()$detach()$cpu())
  })

  # Average across blocks
  avg_mat <- Reduce("+", avg_mats) / n_blocks

  # Zero diagonal
  diag(avg_mat) <- 0

  # Find top interactions
  n_top <- min(n_top, n_feat * (n_feat - 1))
  top_idx <- order(as.vector(avg_mat), decreasing = TRUE)[seq_len(n_top)]

  row_idx <- ((top_idx - 1) %% n_feat) + 1
  col_idx <- ((top_idx - 1) %/% n_feat) + 1

  dplyr::tibble(
    feature_1 = feature_names[row_idx],
    feature_2 = feature_names[col_idx],
    attention_weight = avg_mat[cbind(row_idx, col_idx)]
  )
}

# ------------------------------------------------------------------------------
# Torch modules

auto_int_embedding_module <- torch::nn_module(
  "auto_int_embedding_module",
  initialize = function(pred_lvls, n_continuous, num_embedding) {
    self$n_cat <- length(pred_lvls)
    self$n_cont <- n_continuous

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
      self$cont_weights <- torch::nn_parameter(
        torch::torch_randn(n_continuous, num_embedding)
      )
    }
  },
  forward = function(x_cat = NULL, x_cont = NULL) {
    parts <- list()

    if (self$n_cat > 0 && !is.null(x_cat)) {
      cat_embeds <- lapply(seq_len(self$n_cat), function(i) {
        self$cat_embeddings[[i]](x_cat[, i])
      })
      parts <- c(parts, list(torch::torch_stack(cat_embeds, dim = 2)))
    }

    if (self$n_cont > 0 && !is.null(x_cont)) {
      cont_tensor <- x_cont$unsqueeze(3) * self$cont_weights$unsqueeze(1)
      parts <- c(parts, list(cont_tensor))
    }

    torch::torch_cat(parts, dim = 2)
  }
)

auto_int_backbone_module <- torch::nn_module(
  "auto_int_backbone_module",
  initialize = function(
    num_embedding,
    num_attn_feat,
    num_attn_heads,
    num_attn_blocks,
    dropout_attn,
    activation
  ) {
    self$num_attn_blocks <- num_attn_blocks
    num_attn <- num_attn_feat * num_attn_heads
    self$num_attn <- num_attn

    self$input_proj <- torch::nn_linear(num_embedding, num_attn)

    self$attention_layers <- torch::nn_module_list(lapply(
      seq_len(num_attn_blocks),
      function(i) {
        torch::nn_multihead_attention(
          embed_dim = num_attn,
          num_heads = num_attn_heads,
          dropout = dropout_attn,
          batch_first = TRUE
        )
      }
    ))

    self$V_res <- torch::nn_linear(num_embedding, num_attn)
    self$act <- get_activation_fn(activation)

    self$last_attention_weights <- NULL
  },
  forward = function(x) {
    h <- self$input_proj(x)
    attn_weights_list <- list()

    for (i in seq_len(self$num_attn_blocks)) {
      attn_result <- self$attention_layers[[i]](h, h, h)
      h <- attn_result[[1]]
      attn_weights_list[[i]] <- attn_result[[2]]
    }

    # Single residual from original embeddings + activation
    h <- h + self$V_res(x)
    h <- self$act(h)

    self$last_attention_weights <- attn_weights_list
    h
  }
)

auto_int_module <- torch::nn_module(
  "auto_int_module",
  initialize = function(
    pred_lvls,
    n_continuous,
    num_embedding,
    num_attn_feat,
    num_attn_heads,
    num_attn_blocks,
    dropout_attn,
    dropout_embedding,
    activation,
    hidden_units,
    hidden_activations,
    dropout,
    y_dim
  ) {
    num_features <- length(pred_lvls) + n_continuous
    num_attn <- num_attn_feat * num_attn_heads
    flattened_dim <- num_features * num_attn

    self$embedding <- auto_int_embedding_module(
      pred_lvls = pred_lvls,
      n_continuous = n_continuous,
      num_embedding = num_embedding
    )

    self$embedding_drop <- torch::nn_dropout(p = dropout_embedding)

    self$backbone <- auto_int_backbone_module(
      num_embedding = num_embedding,
      num_attn_feat = num_attn_feat,
      num_attn_heads = num_attn_heads,
      num_attn_blocks = num_attn_blocks,
      dropout_attn = dropout_attn,
      activation = activation
    )

    if (!is.null(hidden_units)) {
      hidden_layers <- list()
      input_dim <- flattened_dim
      for (i in seq_along(hidden_units)) {
        hidden_layers[[length(hidden_layers) + 1]] <-
          torch::nn_linear(input_dim, hidden_units[i])
        hidden_layers[[length(hidden_layers) + 1]] <-
          get_activation_fn(hidden_activations[i])
        input_dim <- hidden_units[i]
      }
      self$hidden <- torch::nn_sequential(!!!hidden_layers)
      if (dropout > 0) {
        self$hidden_drop <- torch::nn_dropout(p = dropout)
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
      self$output_head <- torch::nn_linear(flattened_dim, y_dim)
    }

    self$y_dim <- y_dim
  },
  forward = function(x_cat = NULL, x_cont = NULL) {
    embeds <- self$embedding(x_cat, x_cont)
    embeds <- self$embedding_drop(embeds)
    h <- self$backbone(embeds)
    h_flat <- h$reshape(c(h$shape[1], -1))

    if (!is.null(self$hidden)) {
      h_flat <- self$hidden(h_flat)
      if (!is.null(self$hidden_drop)) {
        h_flat <- self$hidden_drop(h_flat)
      }
    }
    # Classification returns raw logits; softmax is applied at predict time
    # so the loss can use nnf_cross_entropy (numerically stable).
    self$output_head(h_flat)
  }
)

## -----------------------------------------------------------------------------

get_num_auto_int_coef <- function(x) {
  length(unlist(x$estimates[[1]]))
}


#' @export
print.brulee_auto_int <- function(x, ...) {
  cat(cli::style_bold("AutoInt network"), "\n\n", sep = "")

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
    " " = "Activation: {.val {x$parameters$activation}}",
    " " = "Embedding dim: {x$parameters$num_embedding}",
    " " = "Attention: {x$parameters$num_attn_heads} heads x {x$parameters$num_attn_feat} dim, {x$parameters$num_attn_blocks} block{?s}"
  )
  if (!is.null(x$parameters$hidden_units)) {
    units_str <- paste(x$parameters$hidden_units, collapse = ", ")
    hidden_info <- "Hidden layers: {units_str} units, {.val {unique(x$parameters$hidden_activations)}} activation"
    if (x$parameters$dropout > 0) {
      hidden_info <- paste0(
        hidden_info,
        ", dropout={signif(x$parameters$dropout, 3)}"
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
  if (x$parameters$dropout_attn > 0 || x$parameters$dropout_embedding > 0) {
    param_lst <- c(
      param_lst,
      " " = "Dropout: attention={signif(x$parameters$dropout_attn, 3)}, embedding={signif(x$parameters$dropout_embedding, 3)}"
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

  n_params <- format(get_num_auto_int_coef(x), big.mark = ",")
  res_list <- c(" " = "# Parameters: {n_params}")

  if (!is.null(x$loss)) {
    it <- x$best_epoch
    # `loss` includes epoch zero at position 1, so the best epoch's loss is at
    # `it + 1` (matching the predict()/coef() indexing).
    loss_val <- signif(x$loss[it + 1], 3)
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
