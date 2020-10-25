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
#' @details
#'
#' This function fits single layer, feed-forward neural network models for
#' regression (when the outcome is a number) or classification (a factor). For
#' regression, the mean squared error is optimized and cross-entropy is the loss
#' function for classification.
#'
#' The predictors data should all be numeric and encoded in the same units (e.g.
#' standardized to the same range or distribution). If there are factor
#' predictors, use a recipe or formula to create indicator variables (or some
#' other method) to make them numeric.
#'
#' If `conv_crit` is used, it stops training when the difference in the loss
#' function is below `conv_crit` or if it gets worse. The default trains the
#' model over the specified number of epochs.
#'
#' @return
#'
#' A `torch_mlp` object with elements:
#'
#'  * `coefs`: A matrix of all model parameters for each epoch.
#'  * `loss`: A vector of loss values (MSE for regression, negative log-
#'            likelihood for classification) at each epoch.
#'  * `dim`: A list of data dimensions.
#'  * `parameters`: A list of some tuning parameter values.
#'  * `blueprint`: The `hardhat` blueprint data.
#'
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
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
#'  torch_mlp(x = as.matrix(ames_train[, c("Longitude", "Latitude")]),
#'            y = ames_train$Sale_Price, penalty = 0.10, epochs = 600)
#'
#'  # Using recipe
#'  library(recipes)
#'
#'  ames_rec <-
#'   recipe(Sale_Price ~ Longitude + Latitude + Alley, data = ames_train) %>%
#'   step_dummy(Alley) %>%
#'   step_normalize(all_predictors())
#'
#'  set.seed(1)
#'  fit <- torch_mlp(ames_rec, data = ames_train, dropout = 0.25, epochs = 600)
#'  fit
#'
#'  autoplot(fit)
#'
#'  predict(fit, ames_test)
#'  }
#'
#' }
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
  if(!torch::torch_is_installed()) {
    rlang::abort("The torch backend has not been installed; use `torch::install_torch()`.")
  }

 f_nm <- "torch_mlp"
 # check values of various argument values
 if (is.numeric(epochs) & !is.integer(epochs)) {
  epochs <- as.integer(epochs)
 }
 if (is.numeric(hidden_units) & !is.integer(hidden_units)) {
  hidden_units <- as.integer(hidden_units)
 }
 check_integer(epochs, single = TRUE, 2, fn = f_nm)
 check_integer(hidden_units, single = TRUE, 1, fn = f_nm)
 check_double(penalty, single = TRUE, 0, incl = c(TRUE, TRUE), fn = f_nm)
 check_double(dropout, single = TRUE, 0, 1, incl = c(TRUE, FALSE), fn = f_nm)
 check_double(validation, single = TRUE, 0, 1, incl = c(TRUE, FALSE), fn = f_nm)
 check_double(learning_rate, single = TRUE, 0, incl = c(FALSE, TRUE), fn = f_nm)
 check_logical(verbose, single = TRUE, fn = f_nm)
 check_character(activation, single = TRUE, fn = f_nm)

 ## -----------------------------------------------------------------------------

 predictors <- processed$predictors

 if (!is.matrix(predictors)) {
  predictors <- as.matrix(predictors)
  if (is.character(predictors)) {
   rlang::abort(
    paste(
    "There were some non-numeric columns in the predictors.",
    "Please use a formula or recipe to encode all of the predictors as numeric."
    )
   )
  }
 }

 ## -----------------------------------------------------------------------------

 outcome <- processed$outcomes[[1]]

 ## -----------------------------------------------------------------------------

 fit <-
  torch_mlp_reg_fit_imp(
   x = predictors,
   y = outcome,
   epochs = epochs,
   hidden_units = hidden_units,
   activation = activation,
   learning_rate = learning_rate,
   penalty = penalty,
   dropout = dropout,
   validation = validation,
   conv_crit = conv_crit,
   verbose = verbose
  )

 new_torch_mlp(
  coefs = fit$coefficients,
  models = fit$models,
  loss = fit$loss,
  dims = fit$dims,
  parameters = fit$parameters,
  blueprint = processed$blueprint
 )
}

new_torch_mlp <- function(coefs, models, loss, dims, parameters, blueprint) {

  if (!is.array(coefs) || !is.numeric(coefs)) {
    rlang::abort("'coefs' should be a numeric array.")
  }
  if (!is.list(models)) {
    rlang::abort("'models' should be a list.")
  }
  if (!is.vector(loss) || !is.numeric(loss)) {
    rlang::abort("'loss' should be a numeric vector")
  }
  if (!is.list(dims)) {
    rlang::abort("'dims' should be a list")
  }
  if (!is.list(parameters)) {
    rlang::abort("'parameters' should be a list")
  }
  if (!inherits(blueprint, "hardhat_blueprint")) {
    rlang::abort("'blueprint' should be a hardhat blueprint")
  }
 hardhat::new_model(coefs = coefs,
                    models = models,
                    loss = loss,
                    dims = dims,
                    parameters = parameters,
                    blueprint = blueprint,
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
  p <- ncol(x)

  if (is.factor(y)) {
   y_dim <- length(levels(y))
   # the model will output softmax values.
   # so we need to use negative likelihood loss and
   # pass the log of softmax.
   loss_fn <- function(input, target) {
     nnf_nll_loss(
       input = torch::torch_log(input),
       target = target
     )
   }
  } else {
   y_dim <- 1
   loss_fn <- function(input, target) {
     nnf_mse_loss(input, target$view(c(-1,1)))
   }
  }

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
  model <- mlp_module(ncol(x), hidden_units, activation, dropout, y_dim)

  # Write a optim wrapper
  optimizer <-
   torch::optim_sgd(model$parameters, lr = learning_rate, weight_decay = penalty)

  ## ---------------------------------------------------------------------------

  loss_prev <- 10^38
  loss_vec <- rep(NA_real_, epochs)
  if (verbose) {
   epoch_chr <- format(1:epochs)
  }

  ## -----------------------------------------------------------------------------

  model_per_epoch <- list()
  param_values <- init_param_matrix(epochs, p, hidden_units, y_dim)
  # Optimize parameters
  for (epoch in 1:epochs) {

   if (validation > 0) {
    pred <- model(dl_val$dataset$data$x)
    loss <- loss_fn(pred, dl_val$dataset$data$y)
   } else {
    pred <- model(dl$dataset$data$x)
    loss <- loss_fn(pred, dl$dataset$data$y)
   }

   loss_curr <- as.array(loss)
   loss_vec[epoch] <- loss_curr
   loss_diff <- (loss_prev - loss_curr)/loss_prev
   loss_prev <- loss_curr

   if (verbose) {
    message("epoch:", epoch_chr[epoch], "\tLoss:", signif(loss_curr, 5))
   }

   if (epoch > 1 & loss_diff <= conv_crit) {
    break()
   }

   optimizer$zero_grad()
   loss$backward()
   optimizer$step()

   param_values[epoch,] <- flatten_param(model$parameters)
   model_per_epoch[[epoch]] <- model_to_raw(model)

  }

  ## ---------------------------------------------------------------------------

  list(
   coefficients = param_values[complete.cases(param_values),, drop = FALSE],
   models = model_per_epoch,
   loss = loss_vec[!is.na(loss_vec)],
   dims = list(p = p, n = n, h = hidden_units, y = y_dim),
   parameters = list(activation = activation, learning_rate = learning_rate,
                     penalty = penalty, dropout = dropout, validation = validation)
  )
 }


mlp_module <-
 torch::nn_module(
  "mlp_module",
  initialize = function(num_pred, hidden_units, act_type, dropout, y_dim) {
   self$x_to_h <- torch::nn_linear(num_pred, hidden_units)
   self$h_to_y <- torch::nn_linear(hidden_units, y_dim)

   if (dropout > 0) {
    self$dropout <- torch::nn_dropout(p = dropout)
   } else {
    self$dropout <- identity
   }

   self$activation <- get_activation_fn(act_type)

   if (y_dim > 1) {
     self$transform <- torch::nn_softmax(dim = 2)
   } else {
     self$transform <- identity
   }

  },
  forward = function(x) {
    x %>%
      self$x_to_h() %>%
      self$activation() %>%
      self$dropout() %>%
      self$h_to_y() %>%
      self$transform()
  }
 )

## -----------------------------------------------------------------------------

#' @export
print.torch_mlp <- function(x, ...) {
  cat("Multilayer perceptron via torch\n\n")
  cat(x$param$activation, "activation\n")
  cat(
    format(x$dims$n, big.mark = ","), "samples,",
    format(x$dims$p, big.mark = ","), "features,",
    x$dims$h, "hidden units,",
    format(ncol(x$coefs), big.mark = ","), "model coefficients\n"
  )
  if (x$parameters$penalty > 0) {
    cat("weight decay:", x$parameters$penalty, "\n")
  }
  if (x$parameters$dropout > 0) {
    cat("dropout proportion:", x$parameters$dropout, "\n")
  }

  if (!is.null(x$loss)) {
    if (x$parameters$validation > 0) {
      cat("final validation loss after", length(x$loss), "epochs:",
          signif(x$loss[length(x$loss)]), "\n")
    } else {
      cat("final training set loss after", length(x$loss), "epochs:",
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


flatten_param <- function(x) {
  param <- lapply(x, as.array)
  param <- lapply(param, as.vector)
  unlist(param)
}

init_param_matrix <- function(epochs, p, h, y_dim) {
  x_to_h <- (h * p) + h
  x_to_y <- (y_dim * h) + y_dim
  num_param <- x_to_h + x_to_y
  matrix(NA, nrow = epochs, ncol = num_param)
}


unflatten_param <- function(x, epoch) {
  epoch <- min(epoch, nrow(x$coefs))
  p <- x$dims$p
  h <- x$dims$h
  y_dim <- x$dims$y
  param <- x$coefs[epoch,]

  x_to_h_slopes <- 1:(h * p)
  ind <- max(x_to_h_slopes) + 1
  x_to_h_int <- ind:(ind + h - 1)
  ind <- max(x_to_h_int) + 1

  h_to_y_slopes <- ind:(ind + h - 1)
  ind <- max(h_to_y_slopes) + 1
  h_to_y_int <- ind:length(param)

  x_to_h <-
    cbind(
      matrix(param[x_to_h_int],    ncol = 1, nrow = h),
      matrix(param[x_to_h_slopes], ncol = p, nrow = h)
    )
  h_to_y <-
    cbind(
      matrix(param[h_to_y_int],    ncol = 1, nrow = y_dim),
      matrix(param[h_to_y_slopes], ncol = h, nrow = y_dim)
    )
  list(x_to_h = x_to_h, h_to_y = h_to_y)
}

## -----------------------------------------------------------------------------

get_activation_fn <- function(arg, ...) {
  if (arg == "relu") {
    res <- torch::nn_relu(...)
  } else if (arg == "elu") {
    res <- torch::nn_elu(...)
  } else if (arg == "tanh") {
    res <- torch::nn_tanh(...)
  } else {
    res <- identity
  }
  res
}

## -----------------------------------------------------------------------------

#' Plot model loss over epochs
#'
#' @param object A `torch_mlp` object.
#' @param ... Not currently used
#' @return A `ggplot` object.
#' @details This function plots the loss function across the available epochs.
#' @export
autoplot.torch_mlp <- function(object, ...) {
  x <- tibble::tibble(iteration = seq(along = object$loss), loss = object$loss)

  if(object$parameters$validation > 0) {
    lab <- "loss (validation set)"
  } else {
    lab <- "loss (training set)"
  }

  ggplot2::ggplot(x, ggplot2::aes(x = iteration, y = loss)) +
    ggplot2::geom_line() +
    ggplot2::labs(y = lab)
}

model_to_raw <- function(model) {
  con <- rawConnection(raw(), open = "wr")
  torch::torch_save(model, con)
  on.exit({close(con)}, add = TRUE)
  r <- rawConnectionValue(con)
  r
}
