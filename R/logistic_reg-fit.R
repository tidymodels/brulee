#' Fit a logistic regression model
#'
#' `brulee_logistic_reg()` fits a model.
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
#'   * A __data frame__ with 1 factor column (with two levels).
#'   * A __matrix__ with 1 factor column (with two levels).
#'   * A factor __vector__ (with two levels).
#'
#' @param data When a __recipe__ or __formula__ is used, `data` is specified as:
#'
#'   * A __data frame__ containing both the predictors and the outcome.
#'
#' @inheritParams brulee_mlp
#'
#' @param optimizer The method used in the optimization procedure. Possible choices
#'   are 'LBFGS' and 'SGD'. Default is 'LBFGS'.
#' @param learn_rate A positive number that controls the rapidity that the model
#' moves along the descent path. Values around 0.1 or less are typical.
#' (`optimizer = "SGD"` only)
#' @param momentum A positive number usually on `[0.50, 0.99]` for the momentum
#' parameter in gradient descent.  (`optimizer = "SGD"` only)
#'
#' @details
#'
#' This function fits a linear combination of coefficients and predictors to
#' model the log odds of the classes. The training process optimizes the
#' cross-entropy loss function (a.k.a Bernoulli loss).
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
#' @seealso [predict.brulee_logistic_reg()], [coef.brulee_logistic_reg()],
#' [autoplot.brulee_logistic_reg()]
#'
#' @return
#'
#' A `brulee_logistic_reg` object with elements:
#'  * `models_obj`: a serialized raw vector for the torch module.
#'  * `estimates`: a list of matrices with the model parameter estimates per
#'                 epoch.
#'  * `best_epoch`: an integer for the epoch with the smallest loss.
#'  * `loss`: A vector of loss values (MSE for regression, negative log-
#'            likelihood for classification) at each epoch.
#'  * `dim`: A list of data dimensions.
#'  * `parameters`: A list of some tuning parameter values.
#'  * `blueprint`: The `hardhat` blueprint data.
#'
#' @examples
#' if (torch::torch_is_installed()) {
#'
#'  library(recipes)
#'  library(yardstick)
#'
#'  ## -----------------------------------------------------------------------------
#'  # increase # epochs to get better results
#'
#'  data(cells, package = "modeldata")
#'
#'  cells$case <- NULL
#'
#'  set.seed(122)
#'  in_train <- sample(1:nrow(cells), 1000)
#'  cells_train <- cells[ in_train,]
#'  cells_test  <- cells[-in_train,]
#'
#'  # Using matrices
#'  set.seed(1)
#'  brulee_logistic_reg(x = as.matrix(cells_train[, c("fiber_width_ch_1", "width_ch_1")]),
#'                       y = cells_train$class,
#'                       penalty = 0.10, epochs = 3)
#'
#'  # Using recipe
#'  library(recipes)
#'
#'  cells_rec <-
#'   recipe(class ~ ., data = cells_train) %>%
#'   # Transform some highly skewed predictors
#'   step_YeoJohnson(all_numeric_predictors()) %>%
#'   step_normalize(all_numeric_predictors()) %>%
#'   step_pca(all_numeric_predictors(), num_comp = 10)
#'
#'  set.seed(2)
#'  fit <- brulee_logistic_reg(cells_rec, data = cells_train,
#'                              penalty = .01, epochs = 5)
#'  fit
#'
#'  autoplot(fit)
#'
#'  library(yardstick)
#'  predict(fit, cells_test, type = "prob") %>%
#'   bind_cols(cells_test) %>%
#'   roc_auc(class, .pred_PS)
#' }
#'
#' @export
brulee_logistic_reg <- function(x, ...) {
  UseMethod("brulee_logistic_reg")
}

#' @export
#' @rdname brulee_logistic_reg
brulee_logistic_reg.default <- function(x, ...) {
  stop("`brulee_logistic_reg()` is not defined for a '", class(x)[1], "'.", call. = FALSE)
}

# XY method - data frame

#' @export
#' @rdname brulee_logistic_reg
brulee_logistic_reg.data.frame <-
  function(x,
           y,
           epochs = 20L,
           penalty = 0.001,
           mixture = 0,
           validation = 0.1,
           optimizer = "LBFGS",
           learn_rate = 1.0,
           momentum = 0.0,
           batch_size = NULL,
           class_weights = NULL,
           stop_iter = 5,
           verbose = FALSE,
           ...) {
    processed <- hardhat::mold(x, y)

    brulee_logistic_reg_bridge(
      processed,
      epochs = epochs,
      optimizer = optimizer,
      learn_rate = learn_rate,
      penalty = penalty,
      mixture = mixture,
      validation = validation,
      momentum = momentum,
      batch_size = batch_size,
      class_weights = class_weights,
      stop_iter = stop_iter,
      verbose = verbose,
      ...
    )
  }

# XY method - matrix

#' @export
#' @rdname brulee_logistic_reg
brulee_logistic_reg.matrix <- function(x,
                                       y,
                                       epochs = 20L,
                                       penalty = 0.001,
                                       mixture = 0,
                                       validation = 0.1,
                                       optimizer = "LBFGS",
                                       learn_rate = 1,
                                       momentum = 0.0,
                                       batch_size = NULL,
                                       class_weights = NULL,
                                       stop_iter = 5,
                                       verbose = FALSE,
                                       ...) {
  processed <- hardhat::mold(x, y)

  brulee_logistic_reg_bridge(
    processed,
    epochs = epochs,
    optimizer = optimizer,
    learn_rate = learn_rate,
    momentum = momentum,
    penalty = penalty,
    mixture = mixture,
    validation = validation,
    batch_size = batch_size,
    class_weights = class_weights,
    stop_iter = stop_iter,
    verbose = verbose,
    ...
  )
}

# Formula method

#' @export
#' @rdname brulee_logistic_reg
brulee_logistic_reg.formula <-
  function(formula,
           data,
           epochs = 20L,
           penalty = 0.001,
           mixture = 0,
           validation = 0.1,
           optimizer = "LBFGS",
           learn_rate = 1,
           momentum = 0.0,
           batch_size = NULL,
           class_weights = NULL,
           stop_iter = 5,
           verbose = FALSE,
           ...) {
    processed <- hardhat::mold(formula, data)

    brulee_logistic_reg_bridge(
      processed,
      epochs = epochs,
      optimizer = optimizer,
      learn_rate = learn_rate,
      momentum = momentum,
      penalty = penalty,
      mixture = mixture,
      validation = validation,
      batch_size = batch_size,
      class_weights = class_weights,
      stop_iter = stop_iter,
      verbose = verbose,
      ...
    )
  }

# Recipe method

#' @export
#' @rdname brulee_logistic_reg
brulee_logistic_reg.recipe <-
  function(x,
           data,
           epochs = 20L,
           penalty = 0.001,
           mixture = 0,
           validation = 0.1,
           optimizer = "LBFGS",
           learn_rate = 1,
           momentum = 0.0,
           batch_size = NULL,
           class_weights = NULL,
           stop_iter = 5,
           verbose = FALSE,
           ...) {
    processed <- hardhat::mold(x, data)

    brulee_logistic_reg_bridge(
      processed,
      epochs = epochs,
      optimizer = optimizer,
      learn_rate = learn_rate,
      momentum = momentum,
      penalty = penalty,
      mixture = mixture,
      validation = validation,
      batch_size = batch_size,
      class_weights = class_weights,
      stop_iter = stop_iter,
      verbose = verbose,
      ...
    )
  }

# ------------------------------------------------------------------------------
# Bridge

brulee_logistic_reg_bridge <- function(processed, epochs, optimizer,
                                       learn_rate, momentum, penalty, mixture, class_weights,
                                       validation, batch_size, stop_iter, verbose, ...) {
  if(!torch::torch_is_installed()) {
    rlang::abort("The torch backend has not been installed; use `torch::install_torch()`.")
  }

  f_nm <- "brulee_logistic_reg"
  # check values of various argument values
  if (is.numeric(epochs) & !is.integer(epochs)) {
    epochs <- as.integer(epochs)
  }
  check_integer(epochs, single = TRUE, 1, fn = f_nm)
  if (!is.null(batch_size)) {
    if (is.numeric(batch_size) & !is.integer(batch_size)) {
      batch_size <- as.integer(batch_size)
    }
    check_integer(batch_size, single = TRUE, 1, fn = f_nm)
  }
  check_double(penalty, single = TRUE, 0, incl = c(TRUE, TRUE), fn = f_nm)
  check_double(mixture, single = TRUE, 0, 1, incl = c(TRUE, TRUE), fn = f_nm)
  check_double(validation, single = TRUE, 0, 1, incl = c(TRUE, FALSE), fn = f_nm)
  check_double(momentum, single = TRUE, 0, 1, incl = c(TRUE, TRUE), fn = f_nm)
  check_double(learn_rate, single = TRUE, 0, incl = c(FALSE, TRUE), fn = f_nm)
  check_logical(verbose, single = TRUE, fn = f_nm)

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
  check_double(penalty, single = TRUE, 0, incl = c(TRUE, TRUE), fn = f_nm)
  check_double(validation, single = TRUE, 0, 1, incl = c(TRUE, FALSE), fn = f_nm)
  check_double(momentum, single = TRUE, 0, 1, incl = c(TRUE, TRUE), fn = f_nm)
  check_double(learn_rate, single = TRUE, 0, incl = c(FALSE, TRUE), fn = f_nm)
  check_logical(verbose, single = TRUE, fn = f_nm)

  ## -----------------------------------------------------------------------------

  outcome <- processed$outcomes[[1]]
  if (length(levels(outcome)) > 2) {
    rlang::abort("logistic regression is for outcomes with two classes.")
  }

  # ------------------------------------------------------------------------------

  lvls <- levels(outcome)
  xtab <- table(outcome)
  class_weights <- check_class_weights(class_weights, lvls, xtab, f_nm)

  ## -----------------------------------------------------------------------------

  fit <-
    logistic_reg_fit_imp(
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
      class_weights = class_weights,
      stop_iter = stop_iter,
      verbose = verbose
    )

  new_brulee_logistic_reg(
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

new_brulee_logistic_reg <- function( model_obj, estimates, best_epoch, loss,
                                     dims, y_stats, parameters, blueprint) {
  if (!inherits(model_obj, "raw")) {
    rlang::abort("'model_obj' should be a raw vector.")
  }
  if (!is.list(estimates)) {
    rlang::abort("'parameters' should be a list")
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
  hardhat::new_model(model_obj = model_obj,
                     estimates = estimates,
                     best_epoch = best_epoch,
                     loss = loss,
                     dims = dims,
                     y_stats = y_stats,
                     parameters = parameters,
                     blueprint = blueprint,
                     class = "brulee_logistic_reg")
}

## -----------------------------------------------------------------------------
# Fit code

logistic_reg_fit_imp <-
  function(x, y,
           epochs = 20L,
           batch_size = 32,
           penalty = 0.001,
           mixture = 0,
           validation = 0.1,
           optimizer = "LBFGS",
           learn_rate = 1,
           momentum = 0.0,
           class_weights = NULL,
           stop_iter = 5,
           verbose = FALSE,
           ...) {

    torch::torch_manual_seed(sample.int(10^5, 1))

    ## ---------------------------------------------------------------------------
    # General data checks:

    check_data_att(x, y)

    # Check missing values
    compl_data <- check_missing_data(x, y, "brulee_logistic_reg", verbose)
    x <- compl_data$x
    y <- compl_data$y
    n <- length(y)
    p <- ncol(x)

    lvls <- levels(y)
    y_dim <- length(lvls)
    # the model will output softmax values.
    # so we need to use negative likelihood loss and
    # pass the log of softmax.
    loss_fn <- function(input, target, wts = NULL) {
      nnf_nll_loss(
        weight = wts,
        input = torch::torch_log(input),
        target = target
      )
    }

    if (validation > 0) {
      in_val <- sample(seq_along(y), floor(n * validation))
      x_val <- x[in_val,, drop = FALSE]
      y_val <- y[in_val]
      x <- x[-in_val,, drop = FALSE]
      y <- y[-in_val]
    }
    y_stats <- list(mean = NA_real_, sd = NA_real_)
    loss_label <- "\tLoss:"

    if (is.null(batch_size)) {
      batch_size <- nrow(x)
    } else {
      batch_size <- min(batch_size, nrow(x))
    }

    ## ---------------------------------------------------------------------------
    # Convert to index sampler and data loader
    ds <- brulee::matrix_to_dataset(x, y)
    dl <- torch::dataloader(ds, batch_size = batch_size)

    if (validation > 0) {
      ds_val <- brulee::matrix_to_dataset(x_val, y_val)
      dl_val <- torch::dataloader(ds_val)
    }

    ## ---------------------------------------------------------------------------
    # Initialize model and optimizer
    model <- logistic_module(ncol(x), y_dim)
    loss_fn <- make_penalized_loss(loss_fn, model, penalty, mixture)

    # Write a optim wrapper
    if (optimizer == "LBFGS") {
      optimizer <- torch::optim_lbfgs(model$parameters, lr = learn_rate,
                                      history_size = 5)
    } else if (optimizer == "SGD") {
      optimizer <-
        torch::optim_sgd(model$parameters, lr = learn_rate, momentum = momentum)
    } else {
      rlang::abort(paste0("Unknown optimizer '", optimizer, "'"))
    }

    ## ---------------------------------------------------------------------------

    loss_prev <- 10^38
    loss_min <- loss_prev
    poor_epoch <- 0
    best_epoch <- 1
    loss_vec <- rep(NA_real_, epochs)
    if (verbose) {
      epoch_chr <- format(1:epochs)
    }

    ## -----------------------------------------------------------------------------

    param_per_epoch <- list()

    # Optimize parameters
    for (epoch in 1:epochs) {

      # training loop
      coro::loop(
        for (batch in dl) {
          cl <- function() {
            optimizer$zero_grad()
            pred <- model(batch$x)
            loss <- loss_fn(pred, batch$y, class_weights)
            loss$backward()
            loss
          }
          optimizer$step(cl)
        }
      )

      # calculate loss on the full datasets
      if (validation > 0) {
        pred <- model(dl_val$dataset$tensors$x)
        loss <- loss_fn(pred, dl_val$dataset$tensors$y, class_weights)
      } else {
        pred <- model(dl$dataset$tensors$x)
        loss <- loss_fn(pred, dl$dataset$tensors$y, class_weights)
      }

      # calculate losses
      loss_curr <- loss$item()
      loss_vec[epoch] <- loss_curr

      if (is.nan(loss_curr)) {
        rlang::warn("Current loss in NaN. Training wil be stopped.")
        break()
      }

      if (loss_curr >= loss_min) {
        poor_epoch <- poor_epoch + 1
        loss_note <- paste0(" ", cli::symbol$cross, " ")
      } else {
        loss_min <- loss_curr
        loss_note <- NULL
        poor_epoch <- 0
        best_epoch <- epoch
      }
      loss_prev <- loss_curr

      # persists models and coefficients
      param_per_epoch[[epoch]] <-
        lapply(model$state_dict(), function(x) torch::as_array(x$cpu()))

      if (verbose) {
        msg <- paste("epoch:", epoch_chr[epoch], loss_label,
                     signif(loss_curr, 5), loss_note)

        rlang::inform(msg)
      }

      if (poor_epoch == stop_iter) {
        break()
      }

    }

    # ------------------------------------------------------------------------------

    class_weights <- as.numeric(class_weights)
    names(class_weights) <- lvls

    ## ---------------------------------------------------------------------------

    list(
      model_obj = model_to_raw(model),
      estimates = param_per_epoch,
      loss = loss_vec[1:length(param_per_epoch)],
      best_epoch = best_epoch,
      dims = list(p = p, n = n, h = 0, y = y_dim, levels = lvls, features = colnames(x)),
      y_stats = y_stats,
      parameters = list(learn_rate = learn_rate,
                        penalty = penalty, mixture = mixture,
                        validation = validation,
                        class_weights = class_weights,
                        batch_size = batch_size, momentum = momentum)
    )
  }

logistic_module <-
  torch::nn_module(
    "logistic_reg_module",
    initialize = function(num_pred, num_classes) {
      self$fc1 <- torch::nn_linear(num_pred, num_classes)
      self$transform <- torch::nn_softmax(dim = 2)
    },
    forward = function(x) {
      x %>%
        self$fc1() %>%
        self$transform()
    }
  )

## -----------------------------------------------------------------------------

get_num_logistic_reg_coef <- function(x) {
  model <- x$estimates[[1]]
  param <- vapply(model, function(.x) prod(dim(.x)), double(1))
  sum(unlist(param))
}

#' @export
print.brulee_logistic_reg <- function(x, ...) {
  cat("Logistic regression\n\n")
  brulee_print(x)
}

