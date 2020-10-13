#' Fit a linear regression using torch
#'
#' `torch_linear_reg()` fits a model.
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
#' @param learning_rate A positive number (usually less than 0.1).
#' @param conv_crit A non-negative number for convergence.
#' @param verbose A logical that prints out the iteration history.
#'
#' @param ... Not currently used, but required for extensibility.
#'
#' @return
#'
#' A `torch_linear_reg` object.
#'
#' @examples
#' predictors <- mtcars[, -1]
#' outcome <- mtcars[, 1]
#'
#' # XY interface
#' mod <- torch_linear_reg(predictors, outcome)
#'
#' # Formula interface
#' mod2 <- torch_linear_reg(mpg ~ ., mtcars)
#'
#' # Recipes interface
#' library(recipes)
#' rec <- recipe(mpg ~ ., mtcars)
#' rec <- step_log(rec, disp)
#' mod3 <- torch_linear_reg(rec, mtcars)
#'
#' @export
torch_linear_reg <- function(x, ...) {
  UseMethod("torch_linear_reg")
}

#' @export
#' @rdname torch_linear_reg
torch_linear_reg.default <- function(x, ...) {
  stop("`torch_linear_reg()` is not defined for a '", class(x)[1], "'.", call. = FALSE)
}

# XY method - data frame

#' @export
#' @rdname torch_linear_reg
torch_linear_reg.data.frame <- function(x, y, epochs = 100L,
                                        learning_rate = 0.01, conv_crit = 0,
                                        verbose = FALSE, ...) {
  processed <- hardhat::mold(x, y)
  torch_linear_reg_bridge(processed, epochs = epochs, learning_rate = learning_rate,
                          conv_crit = conv_crit, verbose = verbose, ...)
}

# XY method - matrix

#' @export
#' @rdname torch_linear_reg
torch_linear_reg.matrix <- function(x, y, epochs = 100L,
                                    learning_rate = 0.01, conv_crit = 0,
                                    verbose = FALSE, ...) {
  processed <- hardhat::mold(x, y)
  torch_linear_reg_bridge(processed, epochs = epochs,
                          learning_rate = learning_rate, conv_crit = conv_crit,
                          verbose = verbose, ...)
}

# Formula method

#' @export
#' @rdname torch_linear_reg
torch_linear_reg.formula <- function(formula, data, epochs = 100L,
                                     learning_rate = 0.01, conv_crit = 0,
                                     verbose = FALSE, ...) {
  processed <- hardhat::mold(formula, data)
  torch_linear_reg_bridge(processed, epochs = epochs,
                          learning_rate = learning_rate, conv_crit = conv_crit,
                          verbose = verbose, ...)
}

# Recipe method

#' @export
#' @rdname torch_linear_reg
torch_linear_reg.recipe <- function(x, data, epochs = 100L,
                                    learning_rate = 0.01, conv_crit = 0,
                                    verbose = FALSE, ...) {
  processed <- hardhat::mold(x, data)
  torch_linear_reg_bridge(processed, epochs = epochs,
                          learning_rate = learning_rate, conv_crit = conv_crit,
                          verbose = verbose, ...)
}

# ------------------------------------------------------------------------------
# Bridge

torch_linear_reg_bridge <- function(processed, epochs, learning_rate,
                                    conv_crit, verbose, ...) {

  f_nm <- "torch_linear_reg"
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

  fit <- torch_linear_reg_fit_imp(x = predictors, y = outcome, epochs = epochs,
                                  learning_rate = learning_rate,
                                  conv_crit = conv_crit, verbose = verbose)

  new_torch_linear_reg(
    coefs = fit$coefficients,
    loss = fit$loss,
    blueprint = processed$blueprint,
    terms = trms
  )
}

new_torch_linear_reg <- function(coefs, loss, blueprint, terms) {
  hardhat::new_model(coefs = coefs,
                     loss = loss,
                     blueprint = blueprint,
                     terms = terms,
                     class = "torch_linear_reg")
}

## -----------------------------------------------------------------------------
# Fit code

torch_linear_reg_fit_imp <-
  function(x, y,
           epochs = 100L,
           learning_rate = 0.01,
           conv_crit = 0,
           verbose = FALSE,
           ...) {

    ## ---------------------------------------------------------------------------

    # check matrices/vectors, matrix type, matrix column names
    if (!is.matrix(x) || !is.numeric(x)) {
      rlang::abort("'x' should be a numeric matrix.")
    }
    nms <- colnames(x)
    if (length(nms) != ncol(x)) {
      rlang::abort("Every column of 'x' should have a name.")
    }
    if (!is.vector(y)) {
      rlang::abort("'y' should be a vector.")
    }

    ## ---------------------------------------------------------------------------
    # Check missing values
    compl_data <- check_missing_data(x, y, "torch_linear_reg", verbose)
    x <- compl_data$x
    y <- compl_data$y

    ## ---------------------------------------------------------------------------
    # Convert to index sampler and data loader
    ds <- matrix_to_dataset(x, y)
    dl <- torch::dataloader(ds)

    ## ---------------------------------------------------------------------------
    # Initialize model and optimizer
    model <- linear_reg_module(ncol(x))
    model$parameters$fc1.bias$set_data(torch::torch_tensor(mean(y)))

    # Write a optim wrapper
    optimizer <- torch::optim_sgd(model$parameters, lr = learning_rate)

    ## ---------------------------------------------------------------------------

    loss_prev <- 10^38
    loss_vec <- rep(NA_real_, epochs)
    epoch_chr <- format(1:epochs)

    # Optimize parameters
    for (epoch in 1:epochs) {

      pred <- model(dl$dataset$data$x)[,1]
      loss <- torch::nnf_mse_loss(pred, dl$dataset$data$y)

      loss_curr <- as.array(loss)
      loss_vec[epoch] <- loss_curr
      loss_diff <- (loss_prev - loss_curr)/loss_prev
      loss_prev <- loss_curr

      if (loss_diff <= conv_crit) {
        break()
      }

      if (verbose) {
        message("epoch:", epoch_chr[epoch], "\tRMSE:", signif(sqrt(loss_curr), 5))
      }

      optimizer$zero_grad()
      loss$backward()
      optimizer$step()
    }

    ## ---------------------------------------------------------------------------
    # convert results to R objects

    beta <-
      c(
        as.array(model$parameters$fc1.bias),
        as.array(model$parameters$fc1.weight)[1,]
      )
    names(beta) <- c("(Intercept)", colnames(x))

    list(coefficients = beta, loss = sqrt(loss_vec[!is.na(loss_vec)]))
  }

## -----------------------------------------------------------------------------

linear_reg_module <-
  torch::nn_module(
    "linear_reg",
    initialize = function(num_pred) {
      self$fc1 <- torch::nn_linear(num_pred, 1)
    },
    forward = function(x) {
      x %>% self$fc1()
    }
  )

## -----------------------------------------------------------------------------

print.torch_linear_reg <- function(x, ...) {
  cat("Linear regression via torch\n")
  cat(length(x$coefs), "model coefficients\n")
  if (!is.null(x$loss)) {
    cat("Final RMSE after", length(x$loss), "epochs:",
        signif(x$loss[length(x$loss)]), "\n")
  }
  invisible(x)
}

coef.torch_linear_reg <- function(object, ...) {
  object$coef
}


tidy.torch_linear_reg <- function(x, ...) {
  tibble::tibble(term = names(object$coef), estimate = unname(object$coef))
}

