#' Fit Regularization Learning Networks (RLN)
#'
#' `brulee_rln()` fits a single-hidden-layer neural network where each weight
#' learns its own adaptive regularization coefficient.
#'
#' @inheritParams brulee_mlp
#' @param hidden_units An integer for the number of units in the single hidden
#'   layer. Must be >= 1.
#' @param norm An integer for the regularization norm: `1L` for L1 (default)
#'   or `2L` for L2. L1 is recommended by the original paper.
#' @param avg_reg A numeric value for the target mean of the log-scale
#'   per-weight regularization coefficients (Theta in the paper). Controls
#'   average regularization strength: `exp(avg_reg)` is approximately the
#'   geometric mean of the coefficients. Default is `-10`, corresponding to
#'   `exp(-10) ≈ 0.000045`.
#' @param rln_learn_rate A numeric value for the step size used to update the
#'   per-weight regularization coefficients (nu in the paper). Typically large
#'   (default `6e5`) because updates operate in log-scale.
#'
#' @details
#'
#' This function fits Regularization Learning Network (RLN) models for
#' regression (numeric outcomes only). Unlike standard regularization, which
#' applies a single global penalty, RLN learns a separate regularization
#' coefficient for each weight in the hidden layer. After each gradient step,
#' the per-weight coefficients (lambdas) are updated and projected to keep
#' their mean at `avg_reg`.
#'
#' ## Architecture
#'
#' The network is a single-hidden-layer MLP:
#'
#' - Linear transformation (predictors -> `hidden_units`)
#' - Activation function
#' - Linear transformation (`hidden_units` -> 1 output)
#'
#' Weights are initialized with Xavier normal initialization.
#'
#' ## RLN Update
#'
#' After each optimizer step, the per-weight regularization coefficients are
#' updated using the gradient of the Counterfactual Loss with respect to the
#' coefficients, then projected onto a simplex so that `mean(lambda) == avg_reg`.
#' The RMSprop optimizer is the default, as used in the original paper.
#'
#' ## Other Notes
#'
#' The outcome is internally standardized to have mean zero and standard
#' deviation one. Predictions are returned on the original scale.
#'
#' By default, training halts when the validation loss increases for at least
#' `stop_iter` consecutive iterations. If `validation = 0` the training set
#' loss is used.
#'
#' Predictors should all be numeric and on comparable scales. Use a recipe
#' with `step_normalize()` to standardize them before training.
#'
#' Model parameters are saved each epoch so that `epoch` can be tuned
#' efficiently via the `epoch` argument of [predict.brulee_rln()] and
#' [coef.brulee_rln()].
#'
#' @references
#'
#' Shavitt, I., & Segal, E. (2018). Regularization learning networks: Deep
#' learning for tabular datasets. In _Advances in neural information processing
#' systems_ (pp. 1379-1389).
#'
#' @seealso [predict.brulee_rln()], [coef.brulee_rln()],
#' [autoplot.brulee_rln()]
#'
#' @return
#'
#' A `brulee_rln` object with elements:
#'  * `model_obj`: a serialized raw vector for the torch module.
#'  * `estimates`: a list of model parameter matrices per epoch.
#'  * `best_epoch`: an integer for the epoch with the smallest loss.
#'  * `loss`: a numeric vector of loss values (scaled MSE) at each epoch.
#'  * `dims`: a list of data dimensions.
#'  * `y_stats`: a list of mean and standard deviation for the outcome.
#'  * `parameters`: a list of tuning parameter values.
#'  * `blueprint`: the `hardhat` blueprint data.
#'
#' @examplesIf !brulee:::is_cran_check()
#' \donttest{
#' if (torch::torch_is_installed() & rlang::is_installed(c("recipes", "yardstick", "modeldata"))) {
#'
#'  data(ames, package = "modeldata")
#'  ames$Sale_Price <- log10(ames$Sale_Price)
#'
#'  set.seed(122)
#'  in_train <- sample(1:nrow(ames), 2000)
#'  ames_train <- ames[ in_train,]
#'  ames_test  <- ames[-in_train,]
#'
#'  library(recipes)
#'
#'  ames_rec <-
#'   recipe(Sale_Price ~ Longitude + Latitude, data = ames_train) |>
#'     step_normalize(all_numeric_predictors())
#'
#'  set.seed(2)
#'  fit <- brulee_rln(ames_rec, data = ames_train, hidden_units = 20L, epochs = 50L)
#'  fit
#'
#'  autoplot(fit)
#'
#'  library(yardstick)
#'  predict(fit, ames_test) |>
#'    bind_cols(ames_test) |>
#'    rmse(Sale_Price, .pred)
#'
#' }
#' }
#' @export
brulee_rln <- function(x, ...) {
  UseMethod("brulee_rln")
}

#' @export
#' @rdname brulee_rln
brulee_rln.default <- function(x, ...) {
  stop(
    "`brulee_rln()` is not defined for a '",
    class(x)[1],
    "'.",
    call. = FALSE
  )
}

# XY method - data frame

#' @export
#' @rdname brulee_rln
brulee_rln.data.frame <- function(
  x,
  y,
  epochs = 100L,
  hidden_units = 5L,
  norm = 1L,
  avg_reg = -10,
  rln_learn_rate = 6e5,
  activation = "relu",
  validation = 0.1,
  optimizer = "RMSprop",
  learn_rate = 0.001,
  rate_schedule = "none",
  momentum = 0.0,
  batch_size = NULL,
  stop_iter = 5,
  verbose = FALSE,
  ...
) {
  processed <- hardhat::mold(x, y)
  brulee_rln_bridge(
    processed,
    epochs = epochs,
    hidden_units = hidden_units,
    norm = norm,
    avg_reg = avg_reg,
    rln_learn_rate = rln_learn_rate,
    activation = activation,
    validation = validation,
    optimizer = optimizer,
    learn_rate = learn_rate,
    rate_schedule = rate_schedule,
    momentum = momentum,
    batch_size = batch_size,
    stop_iter = stop_iter,
    verbose = verbose,
    ...
  )
}

# XY method - matrix

#' @export
#' @rdname brulee_rln
brulee_rln.matrix <- function(
  x,
  y,
  epochs = 100L,
  hidden_units = 5L,
  norm = 1L,
  avg_reg = -10,
  rln_learn_rate = 6e5,
  activation = "relu",
  validation = 0.1,
  optimizer = "RMSprop",
  learn_rate = 0.001,
  rate_schedule = "none",
  momentum = 0.0,
  batch_size = NULL,
  stop_iter = 5,
  verbose = FALSE,
  ...
) {
  processed <- hardhat::mold(x, y)
  brulee_rln_bridge(
    processed,
    epochs = epochs,
    hidden_units = hidden_units,
    norm = norm,
    avg_reg = avg_reg,
    rln_learn_rate = rln_learn_rate,
    activation = activation,
    validation = validation,
    optimizer = optimizer,
    learn_rate = learn_rate,
    rate_schedule = rate_schedule,
    momentum = momentum,
    batch_size = batch_size,
    stop_iter = stop_iter,
    verbose = verbose,
    ...
  )
}

# Formula method

#' @export
#' @rdname brulee_rln
brulee_rln.formula <- function(
  formula,
  data,
  epochs = 100L,
  hidden_units = 5L,
  norm = 1L,
  avg_reg = -10,
  rln_learn_rate = 6e5,
  activation = "relu",
  validation = 0.1,
  optimizer = "RMSprop",
  learn_rate = 0.001,
  rate_schedule = "none",
  momentum = 0.0,
  batch_size = NULL,
  stop_iter = 5,
  verbose = FALSE,
  ...
) {
  processed <- hardhat::mold(formula, data)
  brulee_rln_bridge(
    processed,
    epochs = epochs,
    hidden_units = hidden_units,
    norm = norm,
    avg_reg = avg_reg,
    rln_learn_rate = rln_learn_rate,
    activation = activation,
    validation = validation,
    optimizer = optimizer,
    learn_rate = learn_rate,
    rate_schedule = rate_schedule,
    momentum = momentum,
    batch_size = batch_size,
    stop_iter = stop_iter,
    verbose = verbose,
    ...
  )
}

# Recipe method

#' @export
#' @rdname brulee_rln
brulee_rln.recipe <- function(
  x,
  data,
  epochs = 100L,
  hidden_units = 5L,
  norm = 1L,
  avg_reg = -10,
  rln_learn_rate = 6e5,
  activation = "relu",
  validation = 0.1,
  optimizer = "RMSprop",
  learn_rate = 0.001,
  rate_schedule = "none",
  momentum = 0.0,
  batch_size = NULL,
  stop_iter = 5,
  verbose = FALSE,
  ...
) {
  processed <- hardhat::mold(x, data)
  brulee_rln_bridge(
    processed,
    epochs = epochs,
    hidden_units = hidden_units,
    norm = norm,
    avg_reg = avg_reg,
    rln_learn_rate = rln_learn_rate,
    activation = activation,
    validation = validation,
    optimizer = optimizer,
    learn_rate = learn_rate,
    rate_schedule = rate_schedule,
    momentum = momentum,
    batch_size = batch_size,
    stop_iter = stop_iter,
    verbose = verbose,
    ...
  )
}

# ------------------------------------------------------------------------------
# Bridge

brulee_rln_bridge <- function(
  processed,
  epochs,
  hidden_units,
  norm,
  avg_reg,
  rln_learn_rate,
  activation,
  validation,
  optimizer,
  learn_rate,
  rate_schedule,
  momentum,
  batch_size,
  stop_iter,
  verbose,
  ...
) {
  if (!torch::torch_is_installed()) {
    cli::cli_abort(
      "The torch backend has not been installed; use `torch::install_torch()`."
    )
  }

  f_nm <- "brulee_rln"

  rln_validated <- validate_rln_args(
    hidden_units = hidden_units,
    norm = norm,
    avg_reg = avg_reg,
    rln_learn_rate = rln_learn_rate,
    activation = activation,
    fn = f_nm
  )

  hidden_units <- rln_validated$hidden_units
  norm <- rln_validated$norm
  activation <- rln_validated$activation

  if (!is.null(batch_size) & optimizer != "LBFGS") {
    if (is.numeric(batch_size) & !is.integer(batch_size)) {
      batch_size <- as.integer(batch_size)
    }
    check_integer(batch_size, single = TRUE, 1, fn = f_nm)
  }

  # penalty and mixture are not used by RLN; pass 0 as placeholders
  validated <- validate_common_args(
    epochs = epochs,
    batch_size = batch_size,
    penalty = 0,
    mixture = 0,
    validation = validation,
    momentum = momentum,
    learn_rate = learn_rate,
    verbose = verbose,
    fn = f_nm
  )

  epochs <- validated$epochs
  batch_size <- validated$batch_size

  predictors <- process_predictors(processed$predictors, fn = f_nm)

  if (is.null(batch_size) & optimizer != "LBFGS") {
    batch_size <- 32L
    if (batch_size >= nrow(predictors)) {
      batch_size <- max(2L, as.integer(ceiling(nrow(predictors) / 10)))
    }
  }

  outcome <- validate_mlp_outcome(processed$outcomes[[1]], fn = f_nm)
  if (is.factor(outcome)) {
    cli::cli_abort(
      paste0(
        f_nm,
        "() only supports numeric outcomes. ",
        "For classification use `brulee_mlp()` or `brulee_resnet()`."
      )
    )
  }

  fit <- rln_fit_imp(
    x = predictors,
    y = outcome,
    epochs = epochs,
    hidden_units = hidden_units,
    norm = norm,
    avg_reg = avg_reg,
    rln_learn_rate = rln_learn_rate,
    activation = activation,
    validation = validation,
    optimizer = optimizer,
    learn_rate = learn_rate,
    rate_schedule = rate_schedule,
    momentum = momentum,
    batch_size = batch_size,
    stop_iter = stop_iter,
    verbose = verbose,
    ...
  )

  new_brulee_rln(
    model_obj = fit$model_obj,
    estimates = fit$estimates,
    best_epoch = fit$best_epoch,
    loss = fit$loss,
    dims = fit$dims,
    y_stats = fit$y_stats,
    parameters = fit$parameters,
    blueprint = processed$blueprint
  )
}

new_brulee_rln <- function(
  model_obj,
  estimates,
  best_epoch,
  loss,
  dims,
  y_stats,
  parameters,
  blueprint
) {
  if (!inherits(model_obj, "raw")) {
    cli::cli_abort("'model_obj' should be a raw vector.")
  }
  if (!is.list(estimates)) {
    cli::cli_abort("'estimates' should be a list.")
  }
  if (!is.vector(best_epoch) || !is.integer(best_epoch)) {
    cli::cli_abort("'best_epoch' should be an integer.")
  }
  if (!is.vector(loss) || !is.numeric(loss)) {
    cli::cli_abort("'loss' should be a numeric vector.")
  }
  if (!is.list(dims)) {
    cli::cli_abort("'dims' should be a list.")
  }
  if (!is.list(y_stats)) {
    cli::cli_abort("'y_stats' should be a list.")
  }
  if (!is.list(parameters)) {
    cli::cli_abort("'parameters' should be a list.")
  }
  if (!inherits(blueprint, "hardhat_blueprint")) {
    cli::cli_abort("'blueprint' should be a hardhat blueprint.")
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
    parameters = parameters,
    blueprint = blueprint,
    class = "brulee_rln"
  )
}

# ------------------------------------------------------------------------------
# Fit implementation

rln_fit_imp <- function(
  x,
  y,
  epochs = 100L,
  batch_size = 32L,
  hidden_units = 5L,
  norm = 1L,
  avg_reg = -10,
  rln_learn_rate = 6e5,
  validation = 0.1,
  optimizer = "RMSprop",
  learn_rate = 0.001,
  rate_schedule = "none",
  momentum = 0.0,
  activation = "relu",
  stop_iter = 5,
  verbose = FALSE,
  ...
) {
  start_seed <- sample.int(10^5, 1)
  torch::torch_manual_seed(start_seed)

  check_data_att(x, y)

  compl_data <- check_missing_data(x, y, "brulee_rln", verbose)
  x <- compl_data$x
  y <- compl_data$y
  n <- length(y)
  p <- ncol(x)

  loss_fn <- function(input, target, wts = NULL) {
    nnf_mse_loss(input, target$view(c(-1, 1)))
  }

  val_split <- split_validation(x, y, validation)
  x <- val_split$x_train
  y <- val_split$y_train
  x_val <- val_split$x_val
  y_val <- val_split$y_val

  y_stats <- scale_stats(y)
  y <- scale_y(y, y_stats)
  if (validation > 0) {
    y_val <- scale_y(y_val, y_stats)
  }
  loss_label <- "\tLoss (scaled):"

  if (optimizer == "LBFGS") {
    batch_size <- nrow(x)
  }
  batch_size <- min(batch_size, nrow(x))

  or_dtype <- torch::torch_get_default_dtype()
  on.exit(torch::torch_set_default_dtype(or_dtype))
  torch::torch_set_default_dtype(torch::torch_float64())

  torch::torch_manual_seed(start_seed + 1)

  torch_data <- setup_torch_data(x, y, x_val, y_val, batch_size, validation)
  dl <- torch_data$dl
  dl_val <- torch_data$dl_val

  res <- list(
    dims = list(
      p = p,
      n = n,
      h = hidden_units,
      y = 1L,
      features = colnames(x)
    ),
    y_stats = y_stats,
    parameters = list(
      activation = activation,
      hidden_units = hidden_units,
      norm = norm,
      avg_reg = avg_reg,
      rln_learn_rate = rln_learn_rate,
      learn_rate = learn_rate,
      validation = validation,
      optimizer = optimizer,
      batch_size = batch_size,
      momentum = momentum,
      stop_iter = stop_iter,
      sched = rate_schedule,
      sched_opt = list(...)
    )
  )

  model <- rln_module(
    num_pred = ncol(x),
    hidden_units = hidden_units,
    activation = activation
  )

  optimizer_obj <- set_optimizer(
    optimizer,
    model,
    learn_rate,
    momentum,
    penalty = 0,
    mixture = 0
  )

  rln_state <- make_rln_state(
    first_linear = model$linear1,
    norm = norm,
    avg_reg = avg_reg,
    rln_learn_rate = rln_learn_rate
  )

  best_epoch <- 0L
  loss_vec <- rep(NA_real_, epochs + 1)

  if (validation > 0) {
    pred <- model(dl_val$dataset$tensors$x)
    loss <- loss_fn(pred, dl_val$dataset$tensors$y)
  } else {
    pred <- model(dl$dataset$tensors$x)
    loss <- loss_fn(pred, dl$dataset$tensors$y)
  }

  loss_vec[1] <- loss$item()

  if (verbose) {
    epoch_chr <- gsub(" ", "0", format(0:epochs))
    cli::cli_inform(
      "epoch: {epoch_chr[1]}, learn rate: {signif(learn_rate, 3)}, {loss_label} {signif(loss_vec[1], 3)}"
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

  rln_state$on_train_begin()

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
    rate_schedule = rate_schedule,
    batch_callback = rln_state$on_batch_end,
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

  res
}

# ------------------------------------------------------------------------------
# RLN state closure

make_rln_state <- function(first_linear, norm, avg_reg, rln_learn_rate) {
  weights <- NULL
  lambdas <- NULL
  prev_regularization <- NULL

  on_train_begin <- function() {
    weights <<- as.array(first_linear$weight$detach())
    lambdas <<- matrix(avg_reg, nrow = nrow(weights), ncol = ncol(weights))
    prev_regularization <<- NULL
  }

  on_batch_end <- function() {
    prev_weights <- weights
    weights <<- as.array(first_linear$weight$detach())
    gradients <- weights - prev_weights

    norms_derivative <- if (norm == 1L) sign(weights) else weights * 2

    if (!is.null(prev_regularization)) {
      lambda_gradients <- gradients * prev_regularization
      lambdas <<- lambdas - rln_learn_rate * lambda_gradients
      lambdas <<- lambdas + (avg_reg - mean(lambdas))
    }

    max_lambdas <- log(abs(weights / norms_derivative))
    max_lambdas[!is.finite(max_lambdas)] <- Inf
    lambdas <<- pmin(lambdas, max_lambdas)

    regularization <- norms_derivative * exp(lambdas)
    regularization[!is.finite(regularization)] <- 0
    new_weights <- weights - regularization

    torch::with_no_grad({
      first_linear$weight$copy_(
        torch::torch_tensor(new_weights, dtype = torch::torch_float64())
      )
    })

    prev_regularization <<- regularization
  }

  list(
    on_train_begin = on_train_begin,
    on_batch_end = on_batch_end
  )
}

# ------------------------------------------------------------------------------
# Torch module

rln_module <-
  torch::nn_module(
    "rln_module",
    initialize = function(num_pred, hidden_units, activation) {
      self$linear1 <- torch::nn_linear(num_pred, hidden_units)
      self$act <- get_activation_fn(activation)
      self$linear2 <- torch::nn_linear(hidden_units, 1L)

      torch::nn_init_xavier_normal_(self$linear1$weight)
      torch::nn_init_zeros_(self$linear1$bias)
      torch::nn_init_xavier_normal_(self$linear2$weight)
      torch::nn_init_zeros_(self$linear2$bias)
    },
    forward = function(x) {
      x |>
        self$linear1() |>
        self$act() |>
        self$linear2()
    }
  )

# ------------------------------------------------------------------------------

get_num_rln_coef <- function(x) {
  length(unlist(x$estimates[[1]]))
}

#' @export
print.brulee_rln <- function(x, ...) {
  cat(cli::style_bold("Regularization Learning Network (RLN)"), "\n\n", sep = "")

  n <- format(x$dims$n, big.mark = ",")
  p <- format(x$dims$p, big.mark = ",")

  cli::cli_bullets(c(
    " " = "Samples: {n}",
    " " = "Predictors: {p}"
  ))

  cat("\n")

  norm_label <- if (x$parameters$norm == 1L) "L1" else "L2"

  cli::cli_bullets(c(
    " " = "Activation: {.val {x$parameters$activation}}",
    " " = "# Hidden Units: {x$parameters$hidden_units}",
    " " = "Norm: {norm_label}",
    " " = "avg_reg (Theta): {signif(x$parameters$avg_reg, 3)}",
    " " = "RLN Learning Rate (nu): {signif(x$parameters$rln_learn_rate, 3)}",
    " " = "Optimizer LR: {signif(x$parameters$learn_rate, 3)}, Schedule: {.val {x$parameters$sched}}",
    " " = "Stopping iterations: {x$parameters$stop_iter}"
  ))

  if (x$parameters$validation > 0) {
    cli::cli_bullets(c(" " = "% Validation: {signif(x$parameters$validation, 3)}"))
  }

  cli::cli_bullets(c(" " = "Optimizer: {.val {x$parameters$optimizer}}"))

  if (x$parameters$optimizer != "LBFGS") {
    cli::cli_bullets(c(" " = "Batch Size: {x$parameters$batch_size}"))
  }

  cat("\n")

  n_params <- format(get_num_rln_coef(x), big.mark = ",")
  res_list <- c(" " = "# Parameters: {n_params}")

  if (!is.null(x$loss)) {
    it <- x$best_epoch
    loss_val <- signif(x$loss[it], 3)
    epoch_str <- cli::pluralize("{it} epoch{?s}")
    if (x$parameters$validation > 0) {
      res_list <- c(res_list, " " = "scaled validation loss after {epoch_str}: {loss_val}")
    } else {
      res_list <- c(res_list, " " = "scaled training loss after {epoch_str}: {loss_val}")
    }
  }

  cli::cli_bullets(res_list)
  invisible(x)
}
