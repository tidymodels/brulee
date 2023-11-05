#' Fit neural networks
#'
#' `brulee_mlp()` fits neural network models using stochastic gradient
#' descent. Multiple layers can be used.
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
#'   indicates ridge regression (a.k.a weight decay).
#' @param hidden_units An integer for the number of hidden units, or a vector
#'   of integers. If a vector of integers, the model will have `length(hidden_units)`
#'   layers each with `hidden_units[i]` hidden units.
#' @param activation A character vector for the activation function )such as
#'  "relu", "tanh", "sigmoid", and so on). See [brulee_activations()] for
#'  a list of possible values. If `hidden_units` is a vector, `activation`
#'  can be a character vector with length equals to `length(hidden_units)`
#'  specifying the activation for each hidden layer.
#' @param optimizer The method used in the optimization procedure. Possible choices
#'   are 'LBFGS' and 'SGD'. Default is 'LBFGS'.
#' @param learn_rate A positive number that controls the initial rapidity that
#' the model moves along the descent path. Values around 0.1 or less are
#' typical.
#' @param rate_schedule A single character value for how the learning rate
#' should change as the optimization proceeds. Possible values are
#' `"none"` (the default), `"decay_time"`, `"decay_expo"`, `"cyclic"` and
#' `"step"`. See [schedule_decay_time()] for more details.
#' @param momentum A positive number usually on `[0.50, 0.99]` for the momentum
#' parameter in gradient descent.  (`optimizer = "SGD"` only)
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
#'  batch. (`optimizer = "SGD"` only)
#' @param stop_iter A non-negative integer for how many iterations with no
#' improvement before stopping.
#' @param verbose A logical that prints out the iteration history.
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
#'          data = ames_train) %>%
#'    # Transform some highly skewed predictors
#'    step_BoxCox(Lot_Area, Gr_Liv_Area) %>%
#'    # Lump some rarely occurring categories into "other"
#'    step_other(Neighborhood, threshold = 0.05)  %>%
#'    # Encode categorical predictors as binary.
#'    step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
#'    # Add an interaction effect:
#'    step_interact(~ starts_with("Central_Air"):Year_Built) %>%
#'    step_zv(all_predictors()) %>%
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
#'  predict(fit, ames_test) %>%
#'    bind_cols(ames_test) %>%
#'    ggplot(aes(x = .pred, y = Sale_Price)) +
#'    geom_abline(col = "green") +
#'    geom_point(alpha = .3) +
#'    lims(x = c(4, 6), y = c(4, 6)) +
#'    coord_fixed(ratio = 1)
#'
#'  library(yardstick)
#'  predict(fit, ames_test) %>%
#'    bind_cols(ames_test) %>%
#'    rmse(Sale_Price, .pred)
#'
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
#'  autoplot(cls_fit)
#'
#'  grid_points <- seq(-4, 4, length.out = 100)
#'
#'  grid <- expand.grid(X1 = grid_points, X2 = grid_points)
#'
#'  predict(cls_fit, grid, type = "prob") %>%
#'   bind_cols(grid) %>%
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
  stop("`brulee_mlp()` is not defined for a '", class(x)[1], "'.", call. = FALSE)
}

# XY method - data frame

#' @export
#' @rdname brulee_mlp
brulee_mlp.data.frame <-
  function(x,
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
           verbose = FALSE,
           ...) {
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
      verbose = verbose,
      ...
    )
  }

# XY method - matrix

#' @export
#' @rdname brulee_mlp
brulee_mlp.matrix <- function(x,
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
                              verbose = FALSE,
                              ...) {
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
    verbose = verbose,
    ...
  )
}

# Formula method

#' @export
#' @rdname brulee_mlp
brulee_mlp.formula <-
  function(formula,
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
           verbose = FALSE,
           ...) {
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
      verbose = verbose,
      ...
    )
  }

# Recipe method

#' @export
#' @rdname brulee_mlp
brulee_mlp.recipe <-
  function(x,
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
           verbose = FALSE,
           ...) {
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
      verbose = verbose,
      ...
    )
  }

# ------------------------------------------------------------------------------
# Bridge

brulee_mlp_bridge <- function(processed, epochs, hidden_units, activation,
                              learn_rate, rate_schedule, momentum, penalty,
                              mixture, dropout, class_weights, validation, optimizer,
                              batch_size, stop_iter, verbose, ...) {
  if(!torch::torch_is_installed()) {
    cli::cli_abort("The torch backend has not been installed; use `torch::install_torch()`.")
  }

  f_nm <- "brulee_mlp"
  # check values of various argument values
  if (is.numeric(epochs) & !is.integer(epochs)) {
    epochs <- as.integer(epochs)
  }
  if (is.numeric(hidden_units) & !is.integer(hidden_units)) {
    hidden_units <- as.integer(hidden_units)
  }
  if (length(hidden_units) > 1 && length(activation) == 1) {
    activation <- rep(activation, length(hidden_units))
  }
  if (length(hidden_units) != length(activation)) {
    cli::cli_abort("'activation' must be a single value or a vector with the same length as 'hidden_units'")
  }

  allowed_activation <- brulee_activations()
  good_activation <- activation %in% allowed_activation
  if (!all(good_activation)) {
   cli::cli_abort(paste("'activation' should be one of: ", paste0(allowed_activation, collapse = ", ")))
  }

  if (optimizer == "LBFGS" & !is.null(batch_size)) {
   cli::cli_warn("'batch_size' is only used for the SGD optimizer.")
   batch_size <- NULL
  }

  if (!is.null(batch_size) & optimizer == "SGD") {
    if (is.numeric(batch_size) & !is.integer(batch_size)) {
      batch_size <- as.integer(batch_size)
    }
    check_integer(batch_size, single = TRUE, 1, fn = f_nm)
  }

  check_integer(epochs, single = TRUE, 1, fn = f_nm)
  check_integer(hidden_units, single = FALSE, 1, fn = f_nm)
  check_double(penalty, single = TRUE, 0, incl = c(TRUE, TRUE), fn = f_nm)
  check_double(mixture, single = TRUE, 0, 1, incl = c(TRUE, TRUE), fn = f_nm)
  check_double(dropout, single = TRUE, 0, 1, incl = c(TRUE, FALSE), fn = f_nm)
  check_double(validation, single = TRUE, 0, 1, incl = c(TRUE, FALSE), fn = f_nm)
  check_double(momentum, single = TRUE, 0, 1, incl = c(TRUE, TRUE), fn = f_nm)
  check_double(learn_rate, single = TRUE, 0, incl = c(FALSE, TRUE), fn = f_nm)
  check_logical(verbose, single = TRUE, fn = f_nm)
  check_character(activation, single = FALSE, fn = f_nm)

  ## -----------------------------------------------------------------------------

  predictors <- processed$predictors

  if (!is.matrix(predictors)) {
    predictors <- as.matrix(predictors)
    if (is.character(predictors)) {
      cli::cli_abort(
        paste(
          "There were some non-numeric columns in the predictors.",
          "Please use a formula or recipe to encode all of the predictors as numeric."
        )
      )
    }
  }

  ## -----------------------------------------------------------------------------

  outcome <- processed$outcomes[[1]]

  # ------------------------------------------------------------------------------

  lvls <- levels(outcome)
  xtab <- table(outcome)
  class_weights <- check_class_weights(class_weights, lvls, xtab, f_nm)

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
      verbose = verbose,
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
    blueprint = processed$blueprint
  )
}

new_brulee_mlp <- function( model_obj, estimates, best_epoch, loss, dims,
                            y_stats, parameters, blueprint) {
  if (!inherits(model_obj, "raw")) {
    cli::cli_abort("'model_obj' should be a raw vector.")
  }
  if (!is.list(estimates)) {
    cli::cli_abort("'parameters' should be a list")
  }
  if (!is.vector(best_epoch) || !is.integer(best_epoch)) {
    cli::cli_abort("'best_epoch' should be an integer")
  }
  if (!is.vector(loss) || !is.numeric(loss)) {
    cli::cli_abort("'loss' should be a numeric vector")
  }
  if (!is.list(dims)) {
    cli::cli_abort("'dims' should be a list")
  }
  if (!is.list(y_stats)) {
    cli::cli_abort("'y_stats' should be a list")
  }
  if (!is.list(parameters)) {
    cli::cli_abort("'parameters' should be a list")
  }
  if (!inherits(blueprint, "hardhat_blueprint")) {
    cli::cli_abort("'blueprint' should be a hardhat blueprint")
  }
  hardhat::new_model(model_obj = model_obj,
                     estimates = estimates,
                     best_epoch = best_epoch,
                     loss = loss,
                     dims = dims,
                     y_stats = y_stats,
                     parameters = parameters,
                     blueprint = blueprint,
                     class = "brulee_mlp")
}

## -----------------------------------------------------------------------------
# Fit code

mlp_fit_imp <-
  function(x, y,
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
           verbose = FALSE,
           ...) {

    torch::torch_manual_seed(sample.int(10^5, 1))

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
          weight = wts,
          input = torch::torch_log(input),
          target = target
        )
      }
    } else {
      y_dim <- 1
      lvls <- NULL
      loss_fn <- function(input, target, wts = NULL) {
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

    if (is.null(batch_size) & optimizer == "SGD") {
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
    model <- mlp_module(ncol(x), hidden_units, activation, dropout, y_dim)
    loss_fn <- make_penalized_loss(loss_fn, model, penalty, mixture)

    # Set the optimizer (will be set again below)
    optimizer_obj <- set_optimizer(optimizer, model, learn_rate, momentum)

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

     # For future work with other optimizers, see
     # https://github.com/tidymodels/brulee/pull/56#discussion_r972049108
     # "Creating a new optimizer every epoch will reset the optimizer state.
     # For example, SGD with momentum keeps track of the latest update for each
     # parameter, so it might be OK to just restart.
     # But other optimizers like Adam, will keep a moving average of updates and
     # resetting them can interfere in training."

     learn_rate <- set_learn_rate(epoch - 1, learn_rate, type = rate_schedule, ...)
     optimizer_obj <- set_optimizer(optimizer, model, learn_rate, momentum)

      # training loop
      coro::loop(
       for (batch in dl) {
        cl <- function() {
         optimizer_obj$zero_grad()
         pred <- model(batch$x)
         loss <- loss_fn(pred, batch$y, class_weights)
         loss$backward()
         loss
        }
        optimizer_obj$step(cl)
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
        cli::cli_warn("Current loss in NaN. Training wil be stopped.")
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
        msg <- paste("epoch:", epoch_chr[epoch], "learn rate", signif(learn_rate, 3),
                     loss_label, signif(loss_curr, 3), loss_note)

        cli::cli_inform(msg)
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
      dims = list(p = p, n = n, h = hidden_units, y = y_dim, levels = lvls, features = colnames(x)),
      y_stats = y_stats,
      parameters = list(
       activation = activation,
       hidden_units = hidden_units,
       learn_rate = learn_rate,
       class_weights = class_weights,
       penalty = penalty,
       mixture = mixture,
       dropout = dropout,
       validation = validation,
       optimizer = optimizer,
       batch_size = batch_size,
       momentum = momentum,
       sched = rate_schedule,
       sched_opt = list(...)
      )
    )
  }


mlp_module <-
  torch::nn_module(
    "mlp_module",
    initialize = function(num_pred, hidden_units, act_type, dropout, y_dim) {

      layers <- list()

      # input layer
      layers[[1]] <- torch::nn_linear(num_pred, hidden_units[1])
      layers[[2]] <- get_activation_fn(act_type[1])

      # if hidden units is a vector then we add those layers
      if (length(hidden_units) > 1) {
        for (i in 2:length(hidden_units)) {
          layers[[length(layers) + 1]] <-
            torch::nn_linear(hidden_units[i-1], hidden_units[i])

          layers[[length(layers) + 1]] <- get_activation_fn(act_type[i])
        }
      }

      # we only add dropout between the last layer and the output layer
      if (dropout > 0) {
        layers[[length(layers) + 1]] <- torch::nn_dropout(p = dropout)
      }

      # output layer
      layers[[length(layers) + 1]] <-
        torch::nn_linear(hidden_units[length(hidden_units)], y_dim)

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

get_units<- function(x) {
  if (length(x$dims$h) > 1) {
    res <- paste0("c(", paste(x$dims$h, collapse = ","), ") hidden units,")
  } else {
    res <- paste(format(x$dims$h, big.mark = ","), "hidden units,")
  }
  res
}


#' @export
print.brulee_mlp <- function(x, ...) {
  cat("Multilayer perceptron\n\n")
  cat(x$param$activation, "activation\n")
  cat(
    get_units(x), "",
    format(get_num_mlp_coef(x), big.mark = ","), "model parameters\n"
  )
  brulee_print(x, ...)
}

## -----------------------------------------------------------------------------

set_optimizer <- function(optimizer, model, learn_rate, momentum) {
 if (optimizer == "LBFGS") {
  res <- torch::optim_lbfgs(model$parameters, lr = learn_rate, history_size = 5)
 } else if (optimizer == "SGD") {
  res <- torch::optim_sgd(model$parameters, lr = learn_rate, momentum = momentum)
 } else {
  cli::cli_abort(paste0("Unknown optimizer '", optimizer, "'"))
 }
 res
}
