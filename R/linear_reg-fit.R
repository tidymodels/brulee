#' Fit a linear regression model
#'
#' `brulee_linear_reg()` fits a linear regression model.
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
#'   * A __data frame__ with 1 numeric column.
#'   * A __matrix__ with 1 numeric column.
#'   * A numeric __vector__.
#'
#' @param data When a __recipe__ or __formula__ is used, `data` is specified as:
#'
#'   * A __data frame__ containing both the predictors and the outcome.
#'
#' @inheritParams brulee_mlp
#' @param optimizer The method used in the optimization procedure. Possible choices
#'   are `"SGD"`,  `"ADAMw"`, `"Adadelta"`, `"Adagrad"`, `"RMSprop"`, and
#'   `"LBFGS"`. `"LBFGS"` is the only second-order method, does not use
#'   batches, and is the default.
#' @details
#'
#' This function fits a linear combination of coefficients and predictors to
#' model the numeric outcome. The training process optimizes the
#' mean squared error loss function.
#'
#' The function internally standardizes the outcome data to have mean zero and
#'  a standard deviation of one. The prediction function creates predictions on
#'  the original scale.
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
#' @seealso [predict.brulee_linear_reg()], [coef.brulee_linear_reg()],
#' [autoplot.brulee_linear_reg()]
#'
#' @return
#'
#' A `brulee_linear_reg` object with elements:
#'  * `models_obj`: a serialized raw vector for the torch module.
#'  * `estimates`: a list of matrices with the model parameter estimates per
#'                 epoch.
#'  * `best_epoch`: an integer for the epoch with the smallest loss.
#'  * `loss`: A vector of loss values (MSE) at each epoch.
#'  * `dim`: A list of data dimensions.
#'  * `y_stats`: A list of summary statistics for numeric outcomes.
#'  * `parameters`: A list of some tuning parameter values.
#'  * `blueprint`: The `hardhat` blueprint data.
#'
#' @examplesIf !brulee:::is_cran_check()
#' \donttest{
#' if (torch::torch_is_installed()  & rlang::is_installed(c("recipes", "yardstick", "modeldata"))) {
#'
#'  ## -----------------------------------------------------------------------------
#'
#'  library(recipes)
#'  library(yardstick)
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
#'  brulee_linear_reg(x = as.matrix(ames_train[, c("Longitude", "Latitude")]),
#'                     y = ames_train$Sale_Price,
#'                     penalty = 0.10, epochs = 1, batch_size = 64)
#'
#'  # Using recipe
#'  library(recipes)
#'
#'  ames_rec <-
#'   recipe(Sale_Price ~ Bldg_Type + Neighborhood + Year_Built + Gr_Liv_Area +
#'          Full_Bath + Year_Sold + Lot_Area + Central_Air + Longitude + Latitude,
#'          data = ames_train) |>
#'     # Transform some highly skewed predictors
#'     step_BoxCox(Lot_Area, Gr_Liv_Area) |>
#'     # Lump some rarely occurring categories into "other"
#'     step_other(Neighborhood, threshold = 0.05)  |>
#'     # Encode categorical predictors as binary.
#'     step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
#'     # Add an interaction effect:
#'     step_interact(~ starts_with("Central_Air"):Year_Built) |>
#'     step_zv(all_predictors()) |>
#'     step_normalize(all_numeric_predictors())
#'
#'  set.seed(2)
#'  fit <- brulee_linear_reg(ames_rec, data = ames_train, epochs = 5)
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
#'  }
#'
#' }
#' @export
brulee_linear_reg <- function(x, ...) {
  UseMethod("brulee_linear_reg")
}

#' @export
#' @rdname brulee_linear_reg
brulee_linear_reg.default <- function(x, ...) {
  stop(
    "`brulee_linear_reg()` is not defined for a '",
    class(x)[1],
    "'.",
    call. = FALSE
  )
}

# XY method - data frame

#' @export
#' @rdname brulee_linear_reg
brulee_linear_reg.data.frame <-
  function(
    x,
    y,
    epochs = 20L,
    penalty = 0.001,
    mixture = 0,
    validation = 0.1,
    optimizer = "LBFGS",
    learn_rate = 1.0,
    momentum = 0.0,
    batch_size = NULL,
    stop_iter = 5,
    verbose = FALSE,
    device = NULL,
    ...
  ) {
    processed <- hardhat::mold(x, y)

    brulee_linear_reg_bridge(
      processed,
      epochs = epochs,
      optimizer = optimizer,
      learn_rate = learn_rate,
      penalty = penalty,
      mixture = mixture,
      validation = validation,
      momentum = momentum,
      batch_size = batch_size,
      stop_iter = stop_iter,
      verbose = verbose,
      device = device,
      ...
    )
  }

# XY method - matrix

#' @export
#' @rdname brulee_linear_reg
brulee_linear_reg.matrix <- function(
  x,
  y,
  epochs = 20L,
  penalty = 0.001,
  mixture = 0,
  validation = 0.1,
  optimizer = "LBFGS",
  learn_rate = 1,
  momentum = 0.0,
  batch_size = NULL,
  stop_iter = 5,
  verbose = FALSE,
  device = NULL,
  ...
) {
  processed <- hardhat::mold(x, y)

  brulee_linear_reg_bridge(
    processed,
    epochs = epochs,
    optimizer = optimizer,
    learn_rate = learn_rate,
    momentum = momentum,
    penalty = penalty,
    mixture = mixture,
    validation = validation,
    batch_size = batch_size,
    stop_iter = stop_iter,
    verbose = verbose,
    device = device,
    ...
  )
}

# Formula method

#' @export
#' @rdname brulee_linear_reg
brulee_linear_reg.formula <-
  function(
    formula,
    data,
    epochs = 20L,
    penalty = 0.001,
    mixture = 0,
    validation = 0.1,
    optimizer = "LBFGS",
    learn_rate = 1,
    momentum = 0.0,
    batch_size = NULL,
    stop_iter = 5,
    verbose = FALSE,
    device = NULL,
    ...
  ) {
    processed <- hardhat::mold(formula, data)

    brulee_linear_reg_bridge(
      processed,
      epochs = epochs,
      optimizer = optimizer,
      learn_rate = learn_rate,
      momentum = momentum,
      penalty = penalty,
      mixture = mixture,
      validation = validation,
      batch_size = batch_size,
      stop_iter = stop_iter,
      verbose = verbose,
      device = device,
      ...
    )
  }

# Recipe method

#' @export
#' @rdname brulee_linear_reg
brulee_linear_reg.recipe <-
  function(
    x,
    data,
    epochs = 20L,
    penalty = 0.001,
    mixture = 0,
    validation = 0.1,
    optimizer = "LBFGS",
    learn_rate = 1,
    momentum = 0.0,
    batch_size = NULL,
    stop_iter = 5,
    verbose = FALSE,
    device = NULL,
    ...
  ) {
    processed <- hardhat::mold(x, data)

    brulee_linear_reg_bridge(
      processed,
      epochs = epochs,
      optimizer = optimizer,
      learn_rate = learn_rate,
      momentum = momentum,
      penalty = penalty,
      mixture = mixture,
      validation = validation,
      batch_size = batch_size,
      stop_iter = stop_iter,
      verbose = verbose,
      device = device,
      ...
    )
  }

# ------------------------------------------------------------------------------
# Bridge

brulee_linear_reg_bridge <- function(
  processed,
  epochs,
  optimizer,
  learn_rate,
  momentum,
  penalty,
  mixture,
  dropout,
  validation,
  batch_size,
  stop_iter,
  verbose,
  device,
  ...
) {
  if (!torch::torch_is_installed()) {
    cli::cli_abort(
      "The torch backend has not been installed; use `torch::install_torch()`."
    )
  }

  # Guess device if not specified
  device <- guess_brulee_device(device)

  f_nm <- "brulee_linear_reg"

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
    fn = f_nm
  )

  # Extract validated/coerced values
  epochs <- validated$epochs
  batch_size <- validated$batch_size

  ## -----------------------------------------------------------------------------

  # Process predictors
  predictors <- process_predictors(processed$predictors, fn = f_nm)

  ## -----------------------------------------------------------------------------

  # Validate outcome
  outcome <- validate_numeric_outcome(processed$outcomes[[1]], fn = f_nm)

  ## -----------------------------------------------------------------------------

  fit <-
    linear_reg_fit_imp(
      x = predictors,
      y = outcome,
      epochs = epochs,
      optimizer = optimizer,
      learn_rate = learn_rate,
      momentum = momentum,
      penalty = penalty,
      mixture = mixture,
      validation = validation,
      batch_size = batch_size,
      stop_iter = stop_iter,
      verbose = verbose,
      device = device
    )

  new_brulee_linear_reg(
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

new_brulee_linear_reg <- function(
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
    cli::cli_abort("'model_obj' should be a raw vector.")
  }
  if (!is.list(estimates)) {
    cli::cli_abort("'parameters' should be a list")
  }
  if (!is.vector(loss) || !is.numeric(loss)) {
    cli::cli_abort("'loss' should be a numeric vector")
  }
  if (!is.list(dims)) {
    cli::cli_abort("'dims' should be a list")
  }
  if (!is.list(parameters)) {
    cli::cli_abort("'parameters' should be a list")
  }
  if (!inherits(blueprint, "hardhat_blueprint")) {
    cli::cli_abort("'blueprint' should be a hardhat blueprint")
  }
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
    class = "brulee_linear_reg"
  )
}

## -----------------------------------------------------------------------------
# Fit code

linear_reg_fit_imp <-
  function(
    x,
    y,
    epochs = 20L,
    batch_size = NULL,
    penalty = 0.001,
    mixture = 0,
    validation = 0.1,
    optimizer = "LBFGS",
    learn_rate = 1,
    momentum = 0.0,
    stop_iter = 5,
    verbose = FALSE,
    device = "cpu",
    ...
  ) {
    torch::torch_manual_seed(sample.int(10^5, 1)) # TODO doesn't give reproducible results

    ## ---------------------------------------------------------------------------
    # General data checks:

    check_data_att(x, y)

    # Check missing values
    compl_data <- check_missing_data(x, y, "brulee_linear_reg", verbose)
    x <- compl_data$x
    y <- compl_data$y
    n <- length(y)
    p <- ncol(x)

    y_dim <- 1
    loss_fn <- function(input, target) {
      nnf_mse_loss(input, target$view(c(-1, 1)))
    }

    # Split validation set
    val_split <- split_validation(x, y, validation)
    x <- val_split$x_train
    y <- val_split$y_train
    x_val <- val_split$x_val
    y_val <- val_split$y_val

    # Scale outcomes for regression
    y_stats <- scale_stats(y)
    y_stats <- list(mean = 0, sd = 1)
    y <- scale_y(y, y_stats)

    if (validation > 0) {
      y_val <- scale_y(y_val, y_stats)
    }
    loss_label <- "\tLoss (scaled):"

    # Determine batch size
    batch_size <- determine_batch_size(batch_size, optimizer, nrow(x))

    ## ---------------------------------------------------------------------------
    # Set torch dtype for both data and model

    or_dtype <- torch::torch_get_default_dtype()
    on.exit(torch::torch_set_default_dtype(or_dtype))
    torch::torch_set_default_dtype(torch::torch_float64())

    # Set device context for training
    training_output <- torch::with_device(device = device, {
      # Convert to index sampler and data loader
      torch_data <- setup_torch_data(x, y, x_val, y_val, batch_size, validation, device = device)
      dl <- torch_data$dl
      dl_val <- torch_data$dl_val

      ## -------------------------------------------------------------------------
      # Initialize model and optimizer
      model <- linear_reg_module(ncol(x))
      model$to(device = device)  # Move model to the correct device
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
        class_weights = NULL,
        loss_label = loss_label,
        verbose = verbose,
        ...
      )

      list(
        model = model,
        training_result = training_result
      )
    })

    ## ---------------------------------------------------------------------------

    list(
      model_obj = model_to_raw(training_output$model),
      estimates = training_output$training_result$param_per_epoch,
      loss = training_output$training_result$loss_vec,
      best_epoch = training_output$training_result$best_epoch,
      dims = list(p = p, n = n, h = 0, y = y_dim, features = colnames(x)),
      y_stats = y_stats,
      parameters = list(
        learn_rate = learn_rate,
        penalty = penalty,
        mixture = mixture,
        validation = validation,
        batch_size = batch_size,
        momentum = momentum
      ),
      device = device
    )
  }


linear_reg_module <-
  torch::nn_module(
    "linear_reg_module",
    initialize = function(num_pred) {
      self$fc1 <- torch::nn_linear(num_pred, 1)
    },
    forward = function(x) {
      x |> self$fc1()
    }
  )

## -----------------------------------------------------------------------------

#' @export
print.brulee_linear_reg <- function(x, ...) {
  cat(cli::style_bold("Linear regression"), "\n\n", sep = "")
  brulee_print(x)
}
