#' Fit a linear regression model
#'
#' `lantern_linear_reg()` fits a model.
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
#' @param penalty The amount of weight decay (i.e., L2 regularization).
#' @param optimizer The method used in the optimization procedure. Possible choices
#'   are 'LBFGS' and 'SGD'. Default is 'LBFGS'.
#' @param learn_rate A positive number. Default is 1 for LBFGS; smaller values
#' are normally chosen for other optimizers. (`optimizer = "SGD"` only)
#' @param momentum A positive number on `[0, 1]` for the momentum parameter in
#'  gradient descent. (`optimizer = "SGD"` only)
#' @param validation The proportion of the data randomly assigned to a
#'  validation set.
#' @param batch_size An integer for the number of training set points in each
#'  batch.
#' @param stop_iter A non-negative integer for how many iterations with no
#' improvement before stopping.
#' @param verbose A logical that prints out the iteration history.
#'
#' @param ... Not currently used, but required for extensibility.
#'
#' @details
#'
#'
#' The _predictors_ data should all be numeric and encoded in the same units (e.g.
#' standardized to the same range or distribution). If there are factor
#' predictors, use a recipe or formula to create indicator variables (or some
#' other method) to make them numeric.
#'
#' The function internally standardizes the
#' outcome data to have mean zero and a standard deviation of one. The prediction
#' function creates predictions on the original scale.
#'
#' @return
#'
#' A `lantern_linear_reg` object with elements:
#'  * `models`: a list object of serialized models for each epoch before stopping.
#'  * `best_epoch`: an integer for the epoch with the smallest loss.
#'  * `loss`: A vector of loss values (MSE) at each epoch.
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
#'  lantern_linear_reg(x = as.matrix(ames_train[, c("Longitude", "Latitude")]),
#'                     y = ames_train$Sale_Price,
#'                     penalty = 0.10, epochs = 1, batch_size = 64)
#'
#'  # Using recipe
#'  library(recipes)
#'
#'  ames_rec <-
#'   recipe(Sale_Price ~ Bldg_Type + Neighborhood + Year_Built + Gr_Liv_Area +
#'          Full_Bath + Year_Sold + Lot_Area + Central_Air + Longitude + Latitude,
#'          data = ames_train) %>%
#'     # Transform some highly skewed predictors
#'     step_BoxCox(Lot_Area, Gr_Liv_Area) %>%
#'     # Lump some rarely occuring categories into "other"
#'     step_other(Neighborhood, threshold = 0.05)  %>%
#'     # Encode categorical predictors as binary.
#'     step_dummy(all_nominal(), one_hot = TRUE) %>%
#'     # Add an interaction effect:
#'     step_interact(~ starts_with("Central_Air"):Year_Built) %>%
#'     step_zv(all_predictors()) %>%
#'     step_normalize(all_predictors())
#'
#'  set.seed(2)
#'  fit <- lantern_linear_reg(ames_rec, data = ames_train,
#'                            epochs = 5, batch_size = 32)
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
#'  }
#'
#' }
#' @export
lantern_linear_reg <- function(x, ...) {
  UseMethod("lantern_linear_reg")
}

#' @export
#' @rdname lantern_linear_reg
lantern_linear_reg.default <- function(x, ...) {
  stop("`lantern_linear_reg()` is not defined for a '", class(x)[1], "'.", call. = FALSE)
}

# XY method - data frame

#' @export
#' @rdname lantern_linear_reg
lantern_linear_reg.data.frame <-
  function(x,
           y,
           epochs = 20L,
           penalty = 0.001,
           validation = 0.1,
           optimizer = "LBFGS",
           learn_rate = 1.0,
           momentum = 0.0,
           batch_size = NULL,
           stop_iter = 5,
           verbose = FALSE,
           ...) {
    processed <- hardhat::mold(x, y)

    lantern_linear_reg_bridge(
      processed,
      epochs = epochs,
      optimizer = optimizer,
      learn_rate = learn_rate,
      penalty = penalty,
      validation = validation,
      momentum = momentum,
      batch_size = batch_size,
      stop_iter = stop_iter,
      verbose = verbose,
      ...
    )
  }

# XY method - matrix

#' @export
#' @rdname lantern_linear_reg
lantern_linear_reg.matrix <- function(x,
                                      y,
                                      epochs = 20L,
                                      penalty = 0.001,
                                      validation = 0.1,
                                      optimizer = "LBFGS",
                                      learn_rate = 1,
                                      momentum = 0.0,
                                      batch_size = NULL,
                                      stop_iter = 5,
                                      verbose = FALSE,
                                      ...) {
  processed <- hardhat::mold(x, y)

  lantern_linear_reg_bridge(
    processed,
    epochs = epochs,
    optimizer = optimizer,
    learn_rate = learn_rate,
    momentum = momentum,
    penalty = penalty,
    validation = validation,
    batch_size = batch_size,
    stop_iter = stop_iter,
    verbose = verbose,
    ...
  )
}

# Formula method

#' @export
#' @rdname lantern_linear_reg
lantern_linear_reg.formula <-
  function(formula,
           data,
           epochs = 20L,
           penalty = 0.001,
           validation = 0.1,
           optimizer = "LBFGS",
           learn_rate = 1,
           momentum = 0.0,
           batch_size = NULL,
           stop_iter = 5,
           verbose = FALSE,
           ...) {
    processed <- hardhat::mold(formula, data)

    lantern_linear_reg_bridge(
      processed,
      epochs = epochs,
      optimizer = optimizer,
      learn_rate = learn_rate,
      momentum = momentum,
      penalty = penalty,
      validation = validation,
      batch_size = batch_size,
      stop_iter = stop_iter,
      verbose = verbose,
      ...
    )
  }

# Recipe method

#' @export
#' @rdname lantern_linear_reg
lantern_linear_reg.recipe <-
  function(x,
           data,
           epochs = 20L,
           penalty = 0.001,
           validation = 0.1,
           optimizer = "LBFGS",
           learn_rate = 1,
           momentum = 0.0,
           batch_size = NULL,
           stop_iter = 5,
           verbose = FALSE,
           ...) {
    processed <- hardhat::mold(x, data)

    lantern_linear_reg_bridge(
      processed,
      epochs = epochs,
      optimizer = optimizer,
      learn_rate = learn_rate,
      momentum = momentum,
      penalty = penalty,
      validation = validation,
      batch_size = batch_size,
      stop_iter = stop_iter,
      verbose = verbose,
      ...
    )
  }

# ------------------------------------------------------------------------------
# Bridge

lantern_linear_reg_bridge <- function(processed, epochs, optimizer,
                                      learn_rate, momentum, penalty, dropout,
                                      validation, batch_size, stop_iter, verbose, ...) {
  if(!torch::torch_is_installed()) {
    rlang::abort("The torch backend has not been installed; use `torch::install_torch()`.")
  }

  f_nm <- "lantern_linear_reg"
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

  ## -----------------------------------------------------------------------------

  outcome <- processed$outcomes[[1]]

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
      validation = validation,
      batch_size = batch_size,
      stop_iter = stop_iter,
      verbose = verbose
    )

  new_lantern_linear_reg(
    models = fit$models,
    best_epoch = fit$best_epoch,
    loss = fit$loss,
    dims = fit$dims,
    y_stats = fit$y_stats,
    parameters = fit$parameters,
    blueprint = processed$blueprint
  )
}

new_lantern_linear_reg <- function( models, best_epoch, loss, dims, y_stats, parameters, blueprint) {
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
  hardhat::new_model(models = models,
                     best_epoch = best_epoch,
                     loss = loss,
                     dims = dims,
                     y_stats = y_stats,
                     parameters = parameters,
                     blueprint = blueprint,
                     class = "lantern_linear_reg")
}

## -----------------------------------------------------------------------------
# Fit code

linear_reg_fit_imp <-
  function(x, y,
           epochs = 20L,
           batch_size = 32,
           penalty = 0.001,
           validation = 0.1,
           optimizer = "LBFGS",
           learn_rate = 1,
           momentum = 0.0,
           stop_iter = 5,
           verbose = FALSE,
           ...) {

    torch::torch_manual_seed(sample.int(10^5, 1)) # TODO doesn't give reproducible results

    ## ---------------------------------------------------------------------------
    # General data checks:

    check_data_att(x, y)

    # Check missing values
    compl_data <- check_missing_data(x, y, "lantern_linear_reg", verbose)
    x <- compl_data$x
    y <- compl_data$y
    n <- length(y)
    p <- ncol(x)

    y_dim <- 1
    loss_fn <- function(input, target) {
      nnf_mse_loss(input, target$view(c(-1,1)))
    }

    if (validation > 0) {
      in_val <- sample(seq_along(y), floor(n * validation))
      x_val <- x[in_val,, drop = FALSE]
      y_val <- y[in_val]
      x <- x[-in_val,, drop = FALSE]
      y <- y[-in_val]
    }

    y_stats <- scale_stats(y)
    y_stats <- list(mean = 0, sd = 1)
    y <- scale_y(y, y_stats)

    if (validation > 0) {
      y_val <- scale_y(y_val, y_stats)
    }
    loss_label <- "\tLoss (scaled):"

    if (is.null(batch_size)) {
      batch_size <- nrow(x)
    } else {
      batch_size <- min(batch_size, nrow(x))
    }

    ## ---------------------------------------------------------------------------
    # Convert to index sampler and data loader
    ds <- lantern::matrix_to_dataset(x, y)
    dl <- torch::dataloader(ds, batch_size = batch_size)

    if (validation > 0) {
      ds_val <- lantern::matrix_to_dataset(x_val, y_val)
      dl_val <- torch::dataloader(ds_val)
    }

    ## ---------------------------------------------------------------------------
    # Initialize model and optimizer
    model <- linear_reg_module(ncol(x))

    # Write a optim wrapper
    if (optimizer == "LBFGS") {
      optimizer <- torch::optim_lbfgs(model$parameters, lr = learn_rate,
                                      history_size = 5)
    } else if (optimizer == "SGD") {
      optimizer <-
        torch::optim_sgd(model$parameters, lr = learn_rate,
                         weight_decay = penalty, momentum = momentum)
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

    model_per_epoch <- list()

    # Optimize parameters
    for (epoch in 1:epochs) {
      # training loop
      coro::loop(
        for (batch in dl) {
          cl <- function() {
            optimizer$zero_grad()
            pred <- model(batch$x)
            loss <- loss_fn(pred, batch$y)
            loss$backward()
            loss
          }
          optimizer$step(cl)
        }
      )

      # calculate loss on the full datasets
      if (validation > 0) {
        pred <- model(dl_val$dataset$tensors$x)
        loss <- loss_fn(pred, dl_val$dataset$tensors$y)
      } else {
        pred <- model(dl$dataset$tensors$x)
        loss <- loss_fn(pred, dl$dataset$tensors$y)
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
      model_per_epoch[[epoch]] <- model_to_raw(model)

      if (verbose) {
        msg <- paste("epoch:", epoch_chr[epoch], loss_label,
                     signif(loss_curr, 3), loss_note)

        rlang::inform(msg)
      }

      if (poor_epoch == stop_iter) {
        break()
      }

    }

    ## ---------------------------------------------------------------------------

    list(
      models = model_per_epoch,
      loss = loss_vec[1:length(model_per_epoch)],
      best_epoch = best_epoch,
      dims = list(p = p, n = n, h = 0, y = y_dim),

      y_stats = y_stats,
      stats = y_stats,
      parameters = list(learn_rate = learn_rate,
                        penalty = penalty, validation = validation,
                        batch_size = batch_size, momentum = momentum)
    )
  }


linear_reg_module <-
  torch::nn_module(
    "linear_reg_module",
    initialize = function(num_pred) {
      self$fc1 <- torch::nn_linear(num_pred, 1)
    },
    forward = function(x) {
      x %>% self$fc1()
    }
  )

## -----------------------------------------------------------------------------

#' @export
print.lantern_linear_reg <- function(x, ...) {
  cat("Linear regression\n\n")
  lantern_print(x)
}

#' @export
coef.lantern_linear_reg <- lantern_coefs

## -----------------------------------------------------------------------------

#' Plot model loss over epochs
#'
#' @param object A `lantern_linear_reg` object.
#' @param ... Not currently used
#' @return A `ggplot` object.
#' @details This function plots the loss function across the available epochs.
#' @export
autoplot.lantern_linear_reg <- lantern_plot
