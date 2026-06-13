#' Fit neural networks
#'
#' `brulee_mlp()` fits neural network models. Multiple layers can be used. For
#' working with two-layer networks in tidymodels, `brulee_mlp_two_layer()` can
#' be helpful for specifying tuning parameters as scalars.
#'
#' @param x Depending on the context:
#'
#'   * A __data frame__ of predictors.
#'   * A __matrix__ of predictors.
#'   * A __recipe__ specifying a set of preprocessing steps
#'     created from [recipes::recipe()].
#'
#'  The predictor data should be standardized (e.g. centered or scaled).
#'
#' @param y When `x` is a __data frame__ or __matrix__, `y` is the outcome
#' specified as:
#'
#'   * A __data frame__ with 1 column (numeric or factor).
#'   * A __matrix__ with numeric column  (numeric or factor).
#'   * A  __vector__  (numeric or factor).
#'
#' @param data When a __recipe__ or __formula__ is used, `data` is specified as:
#'
#'   * A __data frame__ containing both the predictors and the outcome.
#'
#' @param formula A formula specifying the outcome term(s) on the left-hand side,
#' and the predictor term(s) on the right-hand side.
#' @param epochs An integer for the number of epochs of training.
#' @param penalty The amount of weight decay (i.e., L2 regularization).
#' @param mixture Proportion of Lasso Penalty (type: double, default: 0.0). A
#'   value of mixture = 1 corresponds to a pure lasso model, while mixture = 0
#'   indicates ridge regression (a.k.a weight decay). Must be zero for
#'   optimizers `"ADAMw"`, `"RMSprop"`, `"Adadelta"`.
#' @param hidden_units An integer for the number of hidden units, or a vector
#'   of integers. If a vector of integers, the model will have `length(hidden_units)`
#'   layers each with `hidden_units[i]` hidden units.
#' @param hidden_units_2 An integer for the number of hidden units for a second layer.
#' @param activation A character vector for the activation function (such as
#'  "relu", "tanh", "sigmoid", and so on). See [brulee_activations()] for
#'  a list of possible values. If `hidden_units` is a vector, `activation`
#'  can be a character vector with length equals to `length(hidden_units)`
#'  specifying the activation for each hidden layer.
#' @param activation_2  A character vector for the activation function for a second layer.
#' @param optimizer The method used in the optimization procedure. Possible choices
#'   are `"SGD"`,  `"ADAMw"`, `"Adadelta"`, `"Adagrad"`, `"RMSprop"`, and
#'   `"LBFGS"`. `"LBFGS"` is the only second-order method and does not use
#'   batches.
#' @param learn_rate A positive number that controls the initial rapidity that
#' the model moves along the descent path. Values around 0.1 or less are
#' typical.
#' @param rate_schedule A single character value for how the learning rate
#' should change as the optimization proceeds. Possible values are
#' `"none"` (the default), `"decay_time"`, `"decay_expo"`, `"cyclic"` and
#' `"step"`. See [schedule_decay_time()] for more details.
#' @param momentum A positive number usually on `[0.50, 0.99]` for the momentum
#' parameter in gradient descent.  (optimizers `"SGD"`,  and `"RMSprop"` only,
#' ignored otherwise).
#' @param dropout The proportion of parameters set to zero.
#' @param class_weights Numeric class weights (classification only). The value
#' can be:
#'
#'  * A named numeric vector (in any order) where the names are the outcome
#'    factor levels.
#'  * An unnamed numeric vector assumed to be in the same order as the outcome
#'    factor levels.
#'  * A single numeric value for the least frequent class in the training data
#'    and all other classes receive a weight of one.
#' @param validation The proportion of the data randomly assigned to a
#'  validation set.
#' @param batch_size An integer for the number of training set points in each
#'  batch. (`optimizer != "LBFGS"` only, ignored otherwise)
#' @param stop_iter A non-negative integer for how many iterations with no
#' improvement before stopping.
#' @param grad_norm_clip,grad_value_clip Two numeric values, possibly `Inf`,
#' that prevents the gradient's values or norm(s) from exceeding the specified
#' value. This can be helpful if training stops early with the message that
#' `"Loss is NaN at epoch x Training is stopped."`
#' @param verbose A logical that prints out the iteration history.
#' @param device A single character string for the device to train on (e.g.,
#'   `"cpu"` or `"cuda"` for GPU). If `NULL`, the function will use the GPU if
#'   available, otherwise CPU. See [training_efficiency].
#' @param ... Options to pass to the learning rate schedulers via
#' [set_learn_rate()]. For example, the `reduction` or `steps` arguments to
#' [schedule_step()] could be passed here.
#'
#' @details
#'
#' This function fits feed-forward neural network models for regression (when
#'  the outcome is a number) or classification (a factor). For regression, the
#'  mean squared error is optimized and cross-entropy is the loss function for
#'  classification.
#'
#' When the outcome is a number, the function internally standardizes the
#' outcome data to have mean zero and a standard deviation of one. The prediction
#' function creates predictions on the original scale.
#'
#' By default, training halts when the validation loss increases for at least
#' `step_iter` iterations. If `validation = 0` the training set loss is used.
#'
#' The _predictors_ data should all be numeric and encoded in the same units (e.g.
#' standardized to the same range or distribution). If there are factor
#' predictors, use a recipe or formula to create indicator variables (or some
#' other method) to make them numeric. Predictors should be in the same units
#' before training.
#'
#' The model objects are saved for each epoch so that the number of epochs can
#' be efficiently tuned. Both the [coef()] and [predict()] methods for this
#' model have an `epoch` argument (which defaults to the epoch with the best
#' loss value).
#'
#' The use of the L1 penalty (a.k.a. the lasso penalty) does _not_ force
#' parameters to be strictly zero (as it does in packages such as \pkg{glmnet}).
#' The zeroing out of parameters is a specific feature the optimization method
#' used in those packages.
#'
#' ## Learning Rates
#'
#' The learning rate can be set to constant (the default) or dynamically set
#' via a learning rate scheduler (via the `rate_schedule`). Using
#' `rate_schedule = 'none'` uses the `learn_rate` argument. Otherwise, any
#' arguments to the schedulers can be passed via `...`.
#'
#' @references
#'
#' adagrad (adaptive gradient algorithm): Duchi, J., Hazan, E., & Singer, Y.
#' (2011). Adaptive subgradient methods for online learning and stochastic
#' optimization. _Journal of machine learning research_, 12(7).
#'
#' adadelta: Zeiler, M. D. (2012). Adadelta: an adaptive learning rate method.
#' arXiv preprint arXiv:1212.5701.
#'
#' ADAMw: Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay
#' regularization. arXiv preprint arXiv:1711.05101.
#'
#'
#' @seealso [predict.brulee_mlp()], [coef.brulee_mlp()], [autoplot.brulee_mlp()]
#' @return
#'
#' A `brulee_mlp` object with elements:
#'  * `models_obj`: a serialized raw vector for the torch module.
#'  * `estimates`: a list of matrices with the model parameter estimates per
#'                 epoch.
#'  * `best_epoch`: an integer for the epoch with the smallest loss.
#'  * `loss`: A vector of loss values (MSE for regression, negative log-
#'            likelihood for classification) at each epoch.
#'  * `dim`: A list of data dimensions.
#'  * `y_stats`: A list of summary statistics for numeric outcomes.
#'  * `parameters`: A list of some tuning parameter values.
#'  * `blueprint`: The `hardhat` blueprint data.
#'
#' @examplesIf !brulee:::is_cran_check()
#' \donttest{
#' if (torch::torch_is_installed() & rlang::is_installed(c("recipes", "yardstick", "modeldata"))) {
#'
#'  ## -----------------------------------------------------------------------------
#'  # regression examples (increase # epochs to get better results)
#'
#'  data(ames, package = "modeldata")
#'
#'  ames$Sale_Price <- log10(ames$Sale_Price)
#'
#'  set.seed(122)
#'  in_train <- sample(1:nrow(ames), 2000)
#'  ames_train <- ames[ in_train,]
#'  ames_test  <- ames[-in_train,]
#'
#'
#'  # Using matrices
#'  set.seed(1)
#'  fit <-
#'    brulee_mlp(x = as.matrix(ames_train[, c("Longitude", "Latitude")]),
#'                y = ames_train$Sale_Price, penalty = 0.10)
#'
#'  # Using recipe
#'  library(recipes)
#'
#'  ames_rec <-
#'   recipe(Sale_Price ~ Bldg_Type + Neighborhood + Year_Built + Gr_Liv_Area +
#'          Full_Bath + Year_Sold + Lot_Area + Central_Air + Longitude + Latitude,
#'          data = ames_train) |>
#'    # Transform some highly skewed predictors
#'    step_BoxCox(Lot_Area, Gr_Liv_Area) |>
#'    # Lump some rarely occurring categories into "other"
#'    step_other(Neighborhood, threshold = 0.05)  |>
#'    # Encode categorical predictors as binary.
#'    step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
#'    # Add an interaction effect:
#'    step_interact(~ starts_with("Central_Air"):Year_Built) |>
#'    step_zv(all_predictors()) |>
#'    step_normalize(all_numeric_predictors())
#'
#'  set.seed(2)
#'  fit <- brulee_mlp(ames_rec, data = ames_train, hidden_units = 20,
#'                     dropout = 0.05, rate_schedule = "cyclic", step_size = 4)
#'  fit
#'
#'  autoplot(fit)
#'
#'  library(ggplot2)
#'
#'  predict(fit, ames_test) |>
#'    bind_cols(ames_test) |>
#'    ggplot(aes(x = .pred, y = Sale_Price)) +
#'    geom_abline(col = "green") +
#'    geom_point(alpha = .3) +
#'    lims(x = c(4, 6), y = c(4, 6)) +
#'    coord_fixed(ratio = 1)
#'
#'  library(yardstick)
#'  predict(fit, ames_test) |>
#'    bind_cols(ames_test) |>
#'    rmse(Sale_Price, .pred)
#'
#'  # Using multiple hidden layers and activation functions
#'  set.seed(2)
#'  hidden_fit <- brulee_mlp(ames_rec, data = ames_train,
#'                     hidden_units = c(15L, 17L), activation = c("relu", "elu"),
#'                     dropout = 0.05, rate_schedule = "cyclic", step_size = 4)
#'
#'  predict(hidden_fit, ames_test) |>
#'    bind_cols(ames_test) |>
#'    rmse(Sale_Price, .pred)
#'
#'  # ------------------------------------------------------------------------------
#'  # classification
#'
#'  library(dplyr)
#'  library(ggplot2)
#'
#'  data("parabolic", package = "modeldata")
#'
#'  set.seed(1)
#'  in_train <- sample(1:nrow(parabolic), 300)
#'  parabolic_tr <- parabolic[ in_train,]
#'  parabolic_te <- parabolic[-in_train,]
#'
#'  set.seed(2)
#'  cls_fit <- brulee_mlp(class ~ ., data = parabolic_tr, hidden_units = 2,
#'                         epochs = 200L, learn_rate = 0.1, activation = "elu",
#'                         penalty = 0.1, batch_size = 2^8, optimizer = "SGD")
#'
#'  summary(cls_fit)
#'
#'  autoplot(cls_fit)
#'
#'  grid_points <- seq(-4, 4, length.out = 100)
#'
#'  grid <- expand.grid(X1 = grid_points, X2 = grid_points)
#'
#'  predict(cls_fit, grid, type = "prob") |>
#'   bind_cols(grid) |>
#'   ggplot(aes(X1, X2)) +
#'   geom_contour(aes(z = .pred_Class1), breaks = 1/2, col = "black") +
#'   geom_point(data = parabolic_te, aes(col = class))
#'
#'  }
#' }
#' @export
brulee_mlp <- function(x, ...) {
  UseMethod("brulee_mlp")
}

#' @export
#' @rdname brulee_mlp
brulee_mlp.default <- function(x, ...) {
  stop(
    "`brulee_mlp()` is not defined for a '",
    class(x)[1],
    "'.",
    call. = FALSE
  )
}

# XY method - data frame

#' @export
#' @rdname brulee_mlp
brulee_mlp.data.frame <-
  function(
    x,
    y,
    epochs = 100L,
    hidden_units = 3L,
    activation = "relu",
    penalty = 0.001,
    mixture = 0,
    dropout = 0,
    validation = 0.1,
    optimizer = "LBFGS",
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

    brulee_mlp_bridge(
      processed,
      epochs = epochs,
      hidden_units = hidden_units,
      activation = activation,
      learn_rate = learn_rate,
      rate_schedule = rate_schedule,
      penalty = penalty,
      mixture = mixture,
      dropout = dropout,
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
#' @rdname brulee_mlp
brulee_mlp.matrix <- function(
  x,
  y,
  epochs = 100L,
  hidden_units = 3L,
  activation = "relu",
  penalty = 0.001,
  mixture = 0,
  dropout = 0,
  validation = 0.1,
  optimizer = "LBFGS",
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

  brulee_mlp_bridge(
    processed,
    epochs = epochs,
    hidden_units = hidden_units,
    activation = activation,
    learn_rate = learn_rate,
    rate_schedule = rate_schedule,
    momentum = momentum,
    penalty = penalty,
    mixture = mixture,
    dropout = dropout,
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
#' @rdname brulee_mlp
brulee_mlp.formula <-
  function(
    formula,
    data,
    epochs = 100L,
    hidden_units = 3L,
    activation = "relu",
    penalty = 0.001,
    mixture = 0,
    dropout = 0,
    validation = 0.1,
    optimizer = "LBFGS",
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
    processed <- hardhat::mold(formula, data)

    brulee_mlp_bridge(
      processed,
      epochs = epochs,
      hidden_units = hidden_units,
      activation = activation,
      learn_rate = learn_rate,
      rate_schedule = rate_schedule,
      momentum = momentum,
      penalty = penalty,
      mixture = mixture,
      dropout = dropout,
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
#' @rdname brulee_mlp
brulee_mlp.recipe <-
  function(
    x,
    data,
    epochs = 100L,
    hidden_units = 3L,
    activation = "relu",
    penalty = 0.001,
    mixture = 0,
    dropout = 0,
    validation = 0.1,
    optimizer = "LBFGS",
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

    brulee_mlp_bridge(
      processed,
      epochs = epochs,
      hidden_units = hidden_units,
      activation = activation,
      learn_rate = learn_rate,
      rate_schedule = rate_schedule,
      momentum = momentum,
      penalty = penalty,
      mixture = mixture,
      dropout = dropout,
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

brulee_mlp_bridge <- function(
  processed,
  epochs,
  hidden_units,
  activation,
  learn_rate,
  rate_schedule,
  momentum,
  penalty,
  mixture,
  dropout,
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

  # Guess device if not specified
  device <- guess_brulee_device(device)

  # Validate MLP-specific arguments
  mlp_validated <- validate_mlp_args(
    hidden_units = hidden_units,
    activation = activation,
    dropout = dropout,
    grad_value_clip = grad_value_clip,
    grad_norm_clip = grad_norm_clip,
    call = call
  )

  # Extract validated/coerced values
  hidden_units <- mlp_validated$hidden_units
  activation <- mlp_validated$activation

  # Handle batch_size special logic for MLP (optimizer-dependent)
  if (!is.null(batch_size) & optimizer != "LBFGS") {
    if (is.numeric(batch_size) & !is.integer(batch_size)) {
      batch_size <- as.integer(batch_size)
    }
    check_integer(batch_size, single = TRUE, 1, call = call)
  }
  if (is.null(batch_size) & optimizer != "LBFGS") {
    batch_size <- 32L
    if (batch_size >= nrow(processed)) {
      batch_size <- max(2, ceiling(nrow(processed) / 10))
      batch_size <- as.integer(batch_size)
    }
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

  # Extract validated/coerced values
  epochs <- validated$epochs
  batch_size <- validated$batch_size

  ## -----------------------------------------------------------------------------

  # Process predictors
  predictors <- process_predictors(processed$predictors, call = call)

  ## -----------------------------------------------------------------------------

  # Validate outcome (MLP accepts both numeric and factor)
  outcome <- validate_mlp_outcome(processed$outcomes[[1]], call = call)

  # ------------------------------------------------------------------------------

  lvls <- levels(outcome)
  xtab <- table(outcome)
  class_weights <- check_class_weights(class_weights, lvls, xtab, call = call)

  ## -----------------------------------------------------------------------------

  fit <-
    mlp_fit_imp(
      x = predictors,
      y = outcome,
      epochs = epochs,
      hidden_units = hidden_units,
      activation = activation,
      learn_rate = learn_rate,
      rate_schedule = rate_schedule,
      momentum = momentum,
      penalty = penalty,
      mixture = mixture,
      dropout = dropout,
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

  new_brulee_mlp(
    model_obj = fit$model_obj,
    estimates = fit$estimates,
    best_epoch = fit$best_epoch,
    loss = fit$loss,
    dims = fit$dims,
    y_stats = fit$y_stats,
    parameters = fit$parameters,
    device = fit$device,
    blueprint = processed$blueprint
  )
}

new_brulee_mlp <- function(
  model_obj,
  estimates,
  best_epoch,
  loss,
  dims,
  y_stats,
  parameters,
  device,
  blueprint
) {
  if (!inherits(model_obj, "raw")) {
    cli::cli_abort("{.arg model_obj} should be a raw vector.", call = NULL)
  }
  if (!is.list(estimates)) {
    cli::cli_abort("{.arg estimates} should be a list.", call = NULL)
  }
  if (!is.vector(best_epoch) || !is.integer(best_epoch)) {
    cli::cli_abort("{.arg best_epoch} should be an integer.", call = NULL)
  }
  if (!is.vector(loss) || !is.numeric(loss)) {
    cli::cli_abort("{.arg loss} should be a numeric vector.", call = NULL)
  }
  if (!is.list(dims)) {
    cli::cli_abort("{.arg dims} should be a list.", call = NULL)
  }
  if (!is.list(y_stats)) {
    cli::cli_abort("{.arg y_stats} should be a list.", call = NULL)
  }
  if (!is.list(parameters)) {
    cli::cli_abort("{.arg parameters} should be a list.", call = NULL)
  }
  if (!inherits(blueprint, "hardhat_blueprint")) {
    cli::cli_abort(
      "{.arg blueprint} should be a hardhat blueprint.",
      call = NULL
    )
  }

  # Save the estimates that have values
  num_items <- purrr::map_int(estimates, length)
  estimates <- estimates[num_items > 0]

  hardhat::new_model(
    model_obj = model_obj,
    estimates = estimates,
    best_epoch = best_epoch,
    loss = loss,
    dims = dims,
    y_stats = y_stats,
    parameters = parameters,
    device = device,
    blueprint = blueprint,
    class = "brulee_mlp"
  )
}

## -----------------------------------------------------------------------------
# Fit code

mlp_fit_imp <-
  function(
    x,
    y,
    epochs = 100L,
    batch_size = 32,
    hidden_units = 3L,
    penalty = 0.001,
    mixture = 0,
    dropout = 0,
    validation = 0.1,
    optimizer = "LBFGS",
    learn_rate = 0.01,
    rate_schedule = "none",
    momentum = 0.0,
    activation = "relu",
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
    # General data checks:

    check_data_att(x, y)

    # Check missing values
    compl_data <- check_missing_data(x, y, "brulee_mlp", verbose)
    x <- compl_data$x
    y <- compl_data$y
    n <- length(y)
    p <- ncol(x)

    if (is.factor(y)) {
      lvls <- levels(y)
      y_dim <- length(lvls)
      # the model will output softmax values.
      # so we need to use negative likelihood loss and
      # pass the log of softmax.
      loss_fn <- function(input, target, wts = NULL) {
        nnf_nll_loss(
          weight = weights_to_tensor(wts, device = input$device),
          input = torch::torch_log(input),
          target = target,
        )
      }
    } else {
      y_dim <- 1
      lvls <- NULL
      loss_fn <- function(input, target, wts = NULL) {
        nnf_mse_loss(input, target$view(c(-1, 1)))
      }
    }

    # Split validation set
    val_split <- split_validation(x, y, validation)
    x <- val_split$x_train
    y <- val_split$y_train
    x_val <- val_split$x_val
    y_val <- val_split$y_val

    # Scale outcomes for regression
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

    # Determine batch size (MLP-specific logic for LBFGS)
    if (optimizer == "LBFGS") {
      batch_size <- nrow(x)
    }
    batch_size <- min(batch_size, nrow(x))

    ## ---------------------------------------------------------------------------
    # Convert to index sampler and data loader

    or_dtype <- torch::torch_get_default_dtype()
    on.exit(torch::torch_set_default_dtype(or_dtype))
    torch::torch_set_default_dtype(torch::torch_float32())

    ## ---------------------------------------------------------------------------
    # Re-seed and build the module on the CPU, then move it to `device`. See
    # the "Device-handling notes" comment block at the top of R/0_utils.R for
    # the full rationale. The re-seed `start_seed + 1` lets different
    # optimizers start from identical initial weights for the same input
    # seed. Building on the CPU is necessary because `nn_linear()` inside
    # `with_device(mps, ...)` allocates parameters on MPS, and `nn_init_*`
    # then draws from the MPS RNG, which `torch_manual_seed()` does NOT
    # reliably reset. CPU init lets the properly-seeded CPU RNG drive
    # initialization, so MPS/CUDA/CPU runs all produce reproducible initial
    # weights from the same seed.
    torch::torch_manual_seed(start_seed + 1)
    model <- mlp_module(ncol(x), hidden_units, activation, dropout, y_dim)
    model$to(device = device)

    # Set device context for training
    training_output <- torch::with_device(device = device, {
      torch_data <- setup_torch_data(
        x,
        y,
        x_val,
        y_val,
        batch_size,
        validation,
        device = device
      )
      dl <- torch_data$dl
      dl_val <- torch_data$dl_val

      # ------------------------------------------------------------------------------
      # Return value

      res <-
        list(
          dims = list(
            p = p,
            n = n,
            h = hidden_units,
            y = y_dim,
            levels = lvls,
            features = colnames(x)
          ),
          y_stats = y_stats,
          parameters = list(
            activation = activation,
            hidden_units = hidden_units,
            learn_rate = learn_rate,
            class_weights = as.numeric(class_weights),
            penalty = penalty,
            mixture = mixture,
            dropout = dropout,
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

      # Note that if a penalty is used, it might affect the `loss_fn` _or_ the
      # optimizer depending on whether it's pure L2 (mixture = 0) or has L1 component.
      loss_fn <- make_penalized_loss(
        loss_fn,
        model,
        penalty,
        mixture,
        optimizer
      )

      optimizer_obj <- set_optimizer(
        optimizer,
        model,
        learn_rate,
        momentum,
        penalty,
        mixture
      )

      ## ---------------------------------------------------------------------------

      best_epoch <- 0L
      poor_epoch <- 0L
      loss_vec <- rep(NA_real_, epochs + 1)

      if (validation > 0) {
        pred <- model(dl_val$dataset$tensors$x)
        loss <- loss_fn(pred, dl_val$dataset$tensors$y, class_weights)
      } else {
        pred <- model(dl$dataset$tensors$x)
        loss <- loss_fn(pred, dl$dataset$tensors$y, class_weights)
      }

      loss_vec[1] <- loss$item()
      loss_prev <- loss_curr <- loss_vec[1]
      loss_min <- loss_prev

      if (verbose) {
        epoch_chr <- format_epoch_labels(0:epochs)
        cli::cli_inform(
          "epoch: {epoch_chr[1]}, learn rate: {signif(learn_rate, 3)}, {loss_label} {signif(loss_curr, 3)}"
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

      ## -----------------------------------------------------------------------------

      # Run training loop
      training_result <- run_training_loop(
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
        grad_value_clip = grad_value_clip,
        grad_norm_clip = grad_norm_clip,
        rate_schedule = rate_schedule,
        ...
      )

      # Prepend initial parameters and loss to match MLP's original behavior
      param_per_epoch <- c(
        list(param_per_epoch[[1]]),
        training_result$param_per_epoch
      )
      loss_vec <- c(loss_vec[1], training_result$loss_vec)
      best_epoch <- training_result$best_epoch

      # Update result object
      res$model_obj <- model_to_raw(model)
      res$estimates <- param_per_epoch
      res$loss <- loss_vec
      res$best_epoch <- best_epoch

      res
    })

    ## ---------------------------------------------------------------------------

    # Add device to result
    training_output$device <- device

    training_output
  }


mlp_module <-
  torch::nn_module(
    "mlp_module",
    initialize = function(num_pred, hidden_units, act_type, dropout, y_dim) {
      layers <- list()

      # input layer
      layers[[1]] <- torch::nn_linear(num_pred, hidden_units[1])
      layers[[1]] <- init_layer(layers[[1]], "linear")

      layers[[2]] <- get_activation_fn(act_type[1])

      # if hidden units is a vector then we add those layers
      if (length(hidden_units) > 1) {
        for (i in 2:length(hidden_units)) {
          layers[[length(layers) + 1]] <- torch::nn_linear(
            hidden_units[i - 1],
            hidden_units[i]
          )
          layers[[length(layers)]] <- init_layer(
            layers[[length(layers)]],
            "linear"
          )

          layers[[length(layers) + 1]] <- get_activation_fn(act_type[i])
        }
      }

      # we only add dropout between the last layer and the output layer
      if (dropout > 0) {
        layers[[length(layers) + 1]] <- torch::nn_dropout(p = dropout)
      }

      # output layer
      layers[[length(layers) + 1]] <- torch::nn_linear(
        hidden_units[length(hidden_units)],
        y_dim
      )
      layers[[length(layers)]] <- init_layer(layers[[length(layers)]], "linear")

      # conditionally add the softmax layer
      if (y_dim > 1) {
        layers[[length(layers) + 1]] <- torch::nn_softmax(dim = 2)
      }

      # create a sequential module that calls the layers in the same order.
      self$model <- torch::nn_sequential(!!!layers)
    },
    forward = function(x) {
      self$model(x)
    }
  )

## -----------------------------------------------------------------------------

get_num_mlp_coef <- function(x) {
  length(unlist(x$estimates[[1]]))
}

get_units <- function(x) {
  if (length(x$dims$h) > 1) {
    res <- paste0("c(", paste(x$dims$h, collapse = ","), ") hidden units,")
  } else {
    res <- paste(format(x$dims$h, big.mark = ","), "hidden units,")
  }
  res
}

get_acts <- function(x) {
  if (length(x$dims$h) > 1) {
    res <- paste0(
      "c(",
      paste(x$parameters$activation, collapse = ","),
      ") activation,"
    )
  } else {
    res <- paste(x$parameters$activation, "activation,")
  }
  res
}


#' @export
print.brulee_mlp <- function(x, ...) {
  cat(cli::style_bold("Multilayer perceptron"), "\n\n", sep = "")

  brulee_print(x, ...)
}

#' @rdname summary.brulee
#' @export
summary.brulee_mlp <- function(object, ...) {
  module <- revive_model(object$model_obj)
  num_pred <- length(object$dims$features)
  y_dim <- as.integer(object$dims$y)
  seq_module <- module$model
  child_names <- names(seq_module$children)

  total <- 0L
  cat(cli::style_bold("Multilayer perceptron architecture"), "\n", sep = "")
  cat(
    "inputs: ",
    num_pred,
    " | output dim: ",
    y_dim,
    " | components: ",
    length(child_names),
    "\n\n",
    sep = ""
  )

  for (nm in child_names) {
    mod <- seq_module[[nm]]
    if (arch_is_noop(mod)) {
      next
    }
    n_par <- arch_param_count(mod)
    total <- total + n_par
    cat(arch_fmt_row(arch_fmt_module(mod), n_par, indent = "  "))
  }

  cat(
    "\n",
    cli::style_bold("Total parameters: "),
    format(total, big.mark = ","),
    "\n",
    sep = ""
  )
  invisible(object)
}

## -----------------------------------------------------------------------------

check_mixture <- function(mix, opt, call = rlang::caller_env()) {
  if (identical(mix, 1.0)) {
    return(mix)
  }
  # ADAMw requires pure L2 penalty (mixture = 0)
  if (opt == "ADAMw" & !identical(mix, 0.0)) {
    cli::cli_warn(
      "For the {opt} optimizer, the penalty needs to be a pure L2 penalty (i.e.,
   {.code mixture} is 0.0). The value is changed from {signif(mix, 2)} to 0.0.",
      call = call
    )
    mix <- 0.0
  }
  mix
}

set_optimizer <- function(
  optimizer,
  model,
  learn_rate,
  momentum,
  penalty,
  mixture = 0
) {
  # Determine if weight_decay should be used:
  # - ADAMw always uses weight_decay (and check_mixture enforces mixture = 0)
  # - Other optimizers (except LBFGS) use weight_decay only for pure L2 (mixture = 0)
  # - LBFGS doesn't support weight_decay, penalty is always in loss

  weight_decay <- 0.0
  if (optimizer != "LBFGS") {
    if (optimizer == "ADAMw" || identical(mixture, 0.0)) {
      weight_decay <- penalty
    }
  }

  if (optimizer == "LBFGS") {
    res <- torch::optim_lbfgs(
      model$parameters,
      lr = learn_rate,
      history_size = 1
    )
  } else if (optimizer == "SGD") {
    res <- torch::optim_sgd(
      model$parameters,
      lr = learn_rate,
      momentum = momentum,
      nesterov = momentum > 0.0,
      weight_decay = weight_decay
    )
  } else if (optimizer == "RMSprop") {
    res <- torch::optim_rmsprop(
      model$parameters,
      lr = learn_rate,
      momentum = momentum,
      weight_decay = weight_decay
    )
  } else if (optimizer == "ADAMw") {
    res <- torch::optim_adamw(
      model$parameters,
      lr = learn_rate,
      weight_decay = weight_decay
    )
  } else if (optimizer == "Adadelta") {
    res <- torch::optim_adadelta(
      model$parameters,
      lr = learn_rate,
      weight_decay = weight_decay
    )
  } else if (optimizer == "Adagrad") {
    res <- torch::optim_adagrad(
      model$parameters,
      lr = learn_rate,
      weight_decay = weight_decay
    )
  } else {
    cli::cli_abort("Unsupported optimizer {.val {optimizer}}.", call = NULL)
  }
  res
}

init_layer <- function(layer, act) {
  gain_for_rng <- torch::nn_init_calculate_gain(act)
  offset <- sqrt(prod(dim(layer$bias)) + prod(dim(layer$weight)))
  layer$bias <- nn_init_normal_(layer$bias, std = gain_for_rng / offset)
  layer$weight <- nn_init_normal_(layer$weight, std = gain_for_rng / offset)
  layer
}

# ------------------------------------------------------------------------------

#' @export
#' @rdname brulee_mlp
brulee_mlp_two_layer <- function(x, ...) {
  UseMethod("brulee_mlp_two_layer")
}

#' @export
#' @rdname brulee_mlp
brulee_mlp_two_layer.default <- function(x, ...) {
  stop(
    "`brulee_mlp_two_layer()` is not defined for a '",
    class(x)[1],
    "'.",
    call. = FALSE
  )
}

# XY method - data frame

#' @export
#' @rdname brulee_mlp
brulee_mlp_two_layer.data.frame <-
  function(
    x,
    y,
    epochs = 100L,
    hidden_units = 3L,
    hidden_units_2 = 3L,
    activation = "relu",
    activation_2 = "relu",
    penalty = 0.001,
    mixture = 0,
    dropout = 0,
    validation = 0.1,
    optimizer = "LBFGS",
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

    hidden_units_all <- c(hidden_units, hidden_units_2)
    activation_all <- c(activation, activation_2)

    res <-
      brulee_mlp_bridge(
        processed,
        epochs = epochs,
        hidden_units = hidden_units_all,
        activation = activation_all,
        learn_rate = learn_rate,
        rate_schedule = rate_schedule,
        penalty = penalty,
        mixture = mixture,
        dropout = dropout,
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
    class(res) <- c("brulee_mlp_two_layer", class(res))
    res
  }

# XY method - matrix

#' @export
#' @rdname brulee_mlp
brulee_mlp_two_layer.matrix <- function(
  x,
  y,
  epochs = 100L,
  hidden_units = 3L,
  hidden_units_2 = 3L,
  activation = "relu",
  activation_2 = "relu",
  penalty = 0.001,
  mixture = 0,
  dropout = 0,
  validation = 0.1,
  optimizer = "LBFGS",
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

  hidden_units_all <- c(hidden_units, hidden_units_2)
  activation_all <- c(activation, activation_2)

  res <-
    brulee_mlp_bridge(
      processed,
      epochs = epochs,
      hidden_units = hidden_units_all,
      activation = activation_all,
      learn_rate = learn_rate,
      rate_schedule = rate_schedule,
      momentum = momentum,
      penalty = penalty,
      mixture = mixture,
      dropout = dropout,
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
  class(res) <- c("brulee_mlp_two_layer", class(res))
  res
}

# Formula method

#' @export
#' @rdname brulee_mlp
brulee_mlp_two_layer.formula <-
  function(
    formula,
    data,
    epochs = 100L,
    hidden_units = 3L,
    hidden_units_2 = 3L,
    activation = "relu",
    activation_2 = "relu",
    penalty = 0.001,
    mixture = 0,
    dropout = 0,
    validation = 0.1,
    optimizer = "LBFGS",
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
    processed <- hardhat::mold(formula, data)

    hidden_units_all <- c(hidden_units, hidden_units_2)
    activation_all <- c(activation, activation_2)

    res <-
      brulee_mlp_bridge(
        processed,
        epochs = epochs,
        hidden_units = hidden_units_all,
        activation = activation_all,
        learn_rate = learn_rate,
        rate_schedule = rate_schedule,
        momentum = momentum,
        penalty = penalty,
        mixture = mixture,
        dropout = dropout,
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
    class(res) <- c("brulee_mlp_two_layer", class(res))
    res
  }

# Recipe method

#' @export
#' @rdname brulee_mlp
brulee_mlp_two_layer.recipe <-
  function(
    x,
    data,
    epochs = 100L,
    hidden_units = 3L,
    hidden_units_2 = 3L,
    activation = "relu",
    activation_2 = "relu",
    penalty = 0.001,
    mixture = 0,
    dropout = 0,
    validation = 0.1,
    optimizer = "LBFGS",
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

    hidden_units_all <- c(hidden_units, hidden_units_2)
    activation_all <- c(activation, activation_2)

    res <-
      brulee_mlp_bridge(
        processed,
        epochs = epochs,
        hidden_units = hidden_units_all,
        activation = activation_all,
        learn_rate = learn_rate,
        rate_schedule = rate_schedule,
        momentum = momentum,
        penalty = penalty,
        mixture = mixture,
        dropout = dropout,
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
    class(res) <- c("brulee_mlp_two_layer", class(res))
    res
  }
