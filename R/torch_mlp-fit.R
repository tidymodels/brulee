#' Fit a single layer neural network using torch
#'
#' `torch_mlp()` fits a model.
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
#' @param formula A formula specifying the outcome terms on the left-hand side,
#' and the predictor terms on the right-hand side.
#'
#' @param epochs An integer for the number of epochs of training.
#' @param hidden_units An integer for the number of hidden units.
#' @param activation A string for the activation function. Possible values are
#'  "relu", and "elu".
#' @param penalty The amount of weight decay (i.e., L2 regularization).
#' @param dropout The proportion of parameters set to zero.
#' @param learning_rate A positive number (usually less than 0.1).
#' @param validation The proportion of the data randomly assigned to a
#'  validation set.
#' @param conv_crit A non-negative number for convergence.
#' @param verbose A logical that prints out the iteration history.
#'
#' @param ... Not currently used, but required for extensibility.
#'
#' @return
#'
#' A `torch_mlp` object.
#'
#'
#' @export
torch_mlp <- function(x, ...) {
 UseMethod("torch_mlp")
}

#' @export
#' @rdname torch_mlp
torch_mlp.default <- function(x, ...) {
 stop("`torch_mlp()` is not defined for a '", class(x)[1], "'.", call. = FALSE)
}

# XY method - data frame

#' @export
#' @rdname torch_mlp
torch_mlp.data.frame <-
 function(x,
          y,
          epochs = 100L,
          hidden_units = 3L,
          activation = "relu",
          penalty = 0,
          dropout = 0,
          validation = 0.1,
          learning_rate = 0.01,
          conv_crit = -Inf,
          verbose = FALSE,
          ...) {
  processed <- hardhat::mold(x, y)

  torch_mlp_bridge(
   processed,
   epochs = epochs,
   hidden_units = hidden_units,
   activation = activation,
   learning_rate = learning_rate,
   penalty = penalty,
   dropout = dropout,
   validation = validation,
   conv_crit = conv_crit,
   verbose = verbose,
   ...
  )
 }

# XY method - matrix

#' @export
#' @rdname torch_mlp
torch_mlp.matrix <- function(x,
                             y,
                             epochs = 100L,
                             hidden_units = 3L,
                             activation = "relu",
                             penalty = 0,
                             dropout = 0,
                             validation = 0.1,
                             learning_rate = 0.01,
                             conv_crit = -Inf,
                             verbose = FALSE,
                             ...) {
 processed <- hardhat::mold(x, y)

 torch_mlp_bridge(
  processed,
  epochs = epochs,
  hidden_units = hidden_units,
  activation = activation,
  learning_rate = learning_rate,
  penalty = penalty,
  dropout = dropout,
  validation = validation,
  conv_crit = conv_crit,
  verbose = verbose,
  ...
 )
}

# Formula method

#' @export
#' @rdname torch_mlp
torch_mlp.formula <-
 function(formula,
          data,
          epochs = 100L,
          hidden_units = 3L,
          activation = "relu",
          penalty = 0,
          dropout = 0,
          validation = 0.1,
          learning_rate = 0.01,
          conv_crit = -Inf,
          verbose = FALSE,
          ...) {
  processed <- hardhat::mold(formula, data)

  torch_mlp_bridge(
   processed,
   epochs = epochs,
   hidden_units = hidden_units,
   activation = activation,
   learning_rate = learning_rate,
   penalty = penalty,
   dropout = dropout,
   validation = validation,
   conv_crit = conv_crit,
   verbose = verbose,
   ...
  )
 }

# Recipe method

#' @export
#' @rdname torch_mlp
torch_mlp.recipe <-
 function(x,
          data,
          epochs = 100L,
          hidden_units = 3L,
          activation = "relu",
          penalty = 0,
          dropout = 0,
          validation = 0.1,
          learning_rate = 0.01,
          conv_crit = -Inf,
          verbose = FALSE,
          ...) {
  processed <- hardhat::mold(x, data)

  torch_mlp_bridge(
   processed,
   epochs = epochs,
   hidden_units = hidden_units,
   activation = activation,
   learning_rate = learning_rate,
   penalty = penalty,
   dropout = dropout,
   validation = validation,
   conv_crit = conv_crit,
   verbose = verbose,
   ...
  )
 }

# ------------------------------------------------------------------------------
# Bridge

torch_mlp_bridge <- function(processed, epochs, hidden_units, activation,
                             learning_rate, penalty, dropout, validation,
                             conv_crit, verbose, ...) {

 f_nm <- "torch_mlp"
 # check values of various argument values
 if (is.numeric(epochs) & !is.integer(epochs)) {
  epochs <- as.integer(epochs)
 }
 check_integer(epochs, single = TRUE, 2, fn = f_nm)
 check_double(learning_rate, single = TRUE, 0, incl = c(FALSE, TRUE), fn = f_nm)
 check_logical(verbose, single = TRUE, fn = f_nm)

 ## -----------------------------------------------------------------------------

 predictors <- processed$predictors

 if (is.data.frame(predictors)) {
  trms <- stats::terms(~ ., data = processed$predictors)
  predictors <- stats::model.matrix(trms, processed$predictors)
  predictors <- predictors[, -1]
 } else {
  trms <- NULL
 }

 ## -----------------------------------------------------------------------------

 outcome <- processed$outcomes[[1]]

 ## -----------------------------------------------------------------------------

 fit <- torch_mlp_reg_fit_imp(x = predictors, y = outcome, epochs = epochs,
                              hidden_units = hidden_units,
                              activation = activation,
                              learning_rate = learning_rate,
                              penalty = penalty, dropout = dropout,
                              validation = validation, conv_crit = conv_crit,
                              verbose = verbose)

 new_torch_mlp(
  coefs = fit$coefficients,
  loss = fit$loss,
  blueprint = processed$blueprint,
  terms = trms,
  param = fit$param
 )
}

new_torch_mlp <- function(coefs, loss, blueprint, terms, param) {
 hardhat::new_model(coefs = coefs,
                    loss = loss,
                    blueprint = blueprint,
                    terms = terms,
                    param = param,
                    class = "torch_mlp")
}

## -----------------------------------------------------------------------------
# Fit code

torch_mlp_reg_fit_imp <-
 function(x, y,
          epochs = 100L,
          hidden_units = 3L,
          penalty = 0,
          dropout = 0,
          validation = 0.1,
          learning_rate = 0.01,
          activation = "relu",
          conv_crit = -Inf,
          verbose = FALSE,
          ...) {

  torch::torch_manual_seed(sample.int(10^5, 1)) # TODO doesn't give reproducible results

  ## ---------------------------------------------------------------------------
  # General data checks:

  check_data_att(x, y)

  # Check missing values
  compl_data <- check_missing_data(x, y, "torch_mlp", verbose)
  x <- compl_data$x
  y <- compl_data$y
  n <- length(y)

  if (validation > 0) {
   in_val <- sample(seq_along(y), floor(n * validation))
   x_val <- x[in_val,, drop = FALSE]
   y_val <- y[in_val]
   x <- x[-in_val,, drop = FALSE]
   y <- y[-in_val]
  }

  ## ---------------------------------------------------------------------------
  # Convert to index sampler and data loader
  ds <- lantern::matrix_to_dataset(x, y)
  dl <- torch::dataloader(ds)

  if (validation > 0) {
   ds_val <- lantern::matrix_to_dataset(x_val, y_val)
   dl_val <- torch::dataloader(ds_val)
  }

  ## ---------------------------------------------------------------------------
  # Initialize model and optimizer
  model <- mlp_reg_module(ncol(x), hidden_units, activation, dropout)

  # Write a optim wrapper
  optimizer <-
   torch::optim_sgd(model$parameters, lr = learning_rate, weight_decay = penalty)

  ## ---------------------------------------------------------------------------

  loss_prev <- 10^38
  loss_vec <- rep(NA_real_, epochs)
  if (verbose) {
   epoch_chr <- format(1:epochs)
  }

  # Optimize parameters
  for (epoch in 1:epochs) {

   if (validation > 0) {
    pred <- model(dl_val$dataset$data$x)[,1]
    loss <- torch::nnf_mse_loss(pred, dl_val$dataset$data$y)
   } else {
    pred <- model(dl$dataset$data$x)[,1]
    loss <- torch::nnf_mse_loss(pred, dl$dataset$data$y)
   }

   loss_curr <- as.array(loss)
   loss_vec[epoch] <- loss_curr
   loss_diff <- (loss_prev - loss_curr)/loss_prev
   loss_prev <- loss_curr

   if (verbose) {
    message("epoch:", epoch_chr[epoch], "\tRMSE:", signif(sqrt(loss_curr), 5))
   }

   if (epoch > 1 & loss_diff <= conv_crit) {
    break()
   }

   optimizer$zero_grad()
   loss$backward()
   optimizer$step()
  }

  ## ---------------------------------------------------------------------------
  # convert results to R objects

  # Convert each element to an array and convert back for prediction
  beta <- lapply(model$parameters, as.array)

  # TODO save, validation, activation, penalty, dropout, learning_rate
  list(
   coefficients = beta,
   loss = sqrt(loss_vec[!is.na(loss_vec)]),
   activation = activation,
   param = list(activation = activation, learning_rate = learning_rate,
                penalty = penalty, dropout = dropout, validation = validation)
  )
 }


mlp_reg_module <-
 torch::nn_module(
  "mlp_reg",
  initialize = function(num_pred, hidden_units, act_type, dropout) {
   self$x_to_h <- torch::nn_linear(num_pred, hidden_units)
   self$h_to_y <- torch::nn_linear(hidden_units, 1)
   if (dropout > 0) {
    self$dropout <- nn_dropout(p = dropout)
   } else {
    self$dropout = identity
   }
   if (act_type == "relu")
    self$activation <- torch::nn_relu()
   else if (act_type == "elu")
    self$activation <- torch::nn_elu()
   else if (act_type == "tanh")
    self$activation <- torch::nn_tanh()
  },
  forward = function(x) {
   x %>%
    self$x_to_h() %>%
    self$activation() %>%
    self$dropout() %>%
    self$h_to_y()
  }
 )

## -----------------------------------------------------------------------------

#' @export
print.torch_mlp <- function(x, ...) {
 cat("Multilayer perceptron via torch\n\n")
 num_coef <- sum(purrr::map_dbl(x$coefs, ~ prod(dim(.x))))
 cat(x$param$activation, "activation\n")
 cat(
  nrow(x$coefs$x_to_h.weight), "hidden units,",
  format(num_coef, big.mark = ","), "model coefficients\n"
  )
 if (x$param$penalty > 0) {
  cat("weight decay:", x$param$penalty, "\n")
 }
 if (x$param$dropout > 0) {
  cat("dropout proportion:", x$param$dropout, "\n")
 }

 if (!is.null(x$loss)) {
  if (x$param$validation > 0) {
   cat("final validation RMSE after", length(x$loss), "epochs:",
       signif(x$loss[length(x$loss)]), "\n")
  } else {
   cat("final training set RMSE after", length(x$loss), "epochs:",
       signif(x$loss[length(x$loss)]), "\n")
  }

 }
 invisible(x)
}

# coef.torch_mlp <- function(object, ...) {
#  object$coef
# }
#
#
# tidy.torch_mlp <- function(x, ...) {
#  tibble::tibble(term = names(object$coef), estimate = unname(object$coef))
# }

