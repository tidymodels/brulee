#' Fit residual neural networks (ResNet)
#'
#' `brulee_resnet()` fits residual network models with skip connections.
#'
#' @inheritParams brulee_mlp
#' @param hidden_units An integer vector specifying the number of hidden units
#'   in each layer. The length of this vector determines the number of layers.
#'   Each value must be >= 1.
#' @param bottleneck_units An integer vector specifying the intermediate dimension
#'   within each layer. Must have the same length as `hidden_units`. Each value
#'   must be >= 2.
#' @param residual_at An integer vector specifying which layer indices should
#'   have residual (skip) connections. For example, `residual_at = c(2, 4)`
#'   creates residual connections after layers 2 and 4, forming two residual
#'   blocks (layers 1-2 and 3-4). If `NULL` (default), every layer gets its own
#'   skip connection. Use `integer(0)` for no residual connections (i.e., a
#'   purely feed-forward model only).
#'
#' @details
#'
#' This function fits residual network (ResNet) models for regression (when
#'  the outcome is a number) or classification (a factor). ResNets use skip
#'  connections that add the input of a block to its output, allowing gradients
#'  to flow more easily through deep networks. For regression, the mean squared
#'  error is optimized and cross-entropy is the loss function for classification.
#'
#' ## Architecture
#'
#' The network consists of a sequence of layers, each with batch normalization,
#' two linear transformations (with an intermediate bottleneck dimension), and
#' activation functions. Residual (skip) connections can be placed at specified
#' layers via the `residual_at` parameter.
#'
#' Each layer follows this pattern:
#' - Batch normalization (input dimension)
#' - Linear transformation (input dimension -> `bottleneck_units[i]`)
#' - Activation function (ReLU by default)
#' - Dropout (if specified)
#' - Linear transformation (`bottleneck_units[i]` -> `hidden_units[i]`)
#' - Dropout (if specified)
#'
#' When a residual connection is specified at layer `i` via `residual_at`, the
#' output of layer `i` is added to the input from the start of that residual
#' block. If dimensions don't match, a linear projection is automatically added.
#'
#' ## Residual Blocks
#'
#' The `residual_at` parameter defines where skip connections occur:
#' - `residual_at = 3` creates one block spanning layers 1-3
#' - `residual_at = c(2, 4)` creates two blocks: layers 1-2 and layers 3-4
#' - `residual_at = NULL` (default) places a skip connection at every layer
#' - `residual_at = integer(0)` creates no residual connections (a purely
#'    feed-forward model)
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
#' @references
#'
#' He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for
#' image recognition. In _Proceedings of the IEEE conference on computer vision
#' and pattern recognition_ (pp. 770-778).
#'
#' He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity mappings in deep
#' residual networks. In _European conference on computer vision_ (pp. 630-645).
#' Springer, Cham.
#'
#' Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021).
#' Revisiting deep learning models for tabular data. _Advances in neural
#' information processing systems_, 34, 18932-18943.
#'
#' Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018).
#' Mobilenetv2: Inverted residuals and linear bottlenecks. In _Proceedings of
#' the IEEE conference on computer vision and pattern recognition_ (pp.
#' 4510-4520).
#'
#' @seealso [predict.brulee_resnet()], [coef.brulee_resnet()],
#' [autoplot.brulee_resnet()]
#'
#' @return
#'
#' A `brulee_resnet` object with elements:
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
#'  # Using recipe
#'  library(recipes)
#'
#'  ames_rec <-
#'   recipe(Sale_Price ~ Longitude + Latitude, data = ames_train) |>
#'     step_normalize(all_numeric_predictors())
#'
#'  set.seed(2)
#'  fit <- brulee_resnet(ames_rec, data = ames_train,
#'                       hidden_units = c(20, 10), bottleneck_units = c(15, 8),
#'                       residual_at = 2,
#'                       epochs = 50, batch_size = 32)
#'  fit
#'
#'  summary(fit)
#'
#'  autoplot(fit)
#'
#'  library(yardstick)
#'  predict(fit, ames_test) |>
#'    bind_cols(ames_test) |>
#'    rmse(Sale_Price, .pred)
#'
#'  # ------------------------------------------------------------------------------
#'  # classification
#'
#'  library(dplyr)
#'
#'  data("parabolic", package = "modeldata")
#'
#'  set.seed(1)
#'  in_train <- sample(1:nrow(parabolic), 300)
#'  parabolic_tr <- parabolic[ in_train,]
#'  parabolic_te <- parabolic[-in_train,]
#'
#'  set.seed(2)
#'  cls_fit <- brulee_resnet(class ~ ., data = parabolic_tr,
#'                           hidden_units = c(8, 5), bottleneck_units = c(6, 4),
#'                           residual_at = 1:2,
#'                           epochs = 200L, learn_rate = 0.1, activation = "elu",
#'                           penalty = 0.1, batch_size = 2^8)
#'  autoplot(cls_fit)
#'
#'  predict(cls_fit, parabolic_te, type = "prob") |>
#'    bind_cols(parabolic_te) |>
#'    roc_auc(class, .pred_Class1)
#'
#'  }
#' }
#' @export
brulee_resnet <- function(x, ...) {
  UseMethod("brulee_resnet")
}

#' @export
#' @rdname brulee_resnet
brulee_resnet.default <- function(x, ...) {
  stop(
    "`brulee_resnet()` is not defined for a '",
    class(x)[1],
    "'.",
    call. = FALSE
  )
}

# XY method - data frameste

#' @export
#' @rdname brulee_resnet
brulee_resnet.data.frame <-
  function(
    x,
    y,
    epochs = 100L,
    hidden_units = 3L,
    bottleneck_units = hidden_units,
    residual_at = NULL,
    activation = "relu",
    penalty = 0.001,
    mixture = 0,
    dropout = 0,
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

    brulee_resnet_bridge(
      processed,
      epochs = epochs,
      hidden_units = hidden_units,
      bottleneck_units = bottleneck_units,
      residual_at = residual_at,
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
#' @rdname brulee_resnet
brulee_resnet.matrix <- function(
  x,
  y,
  epochs = 100L,
  hidden_units = 3L,
  bottleneck_units = hidden_units,
  residual_at = NULL,
  activation = "relu",
  penalty = 0.001,
  mixture = 0,
  dropout = 0,
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

  brulee_resnet_bridge(
    processed,
    epochs = epochs,
    hidden_units = hidden_units,
    bottleneck_units = bottleneck_units,
    residual_at = residual_at,
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
#' @rdname brulee_resnet
brulee_resnet.formula <-
  function(
    formula,
    data,
    epochs = 100L,
    hidden_units = 3L,
    bottleneck_units = hidden_units,
    residual_at = NULL,
    activation = "relu",
    penalty = 0.001,
    mixture = 0,
    dropout = 0,
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
    processed <- hardhat::mold(formula, data)

    brulee_resnet_bridge(
      processed,
      epochs = epochs,
      hidden_units = hidden_units,
      bottleneck_units = bottleneck_units,
      residual_at = residual_at,
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
#' @rdname brulee_resnet
brulee_resnet.recipe <-
  function(
    x,
    data,
    epochs = 100L,
    hidden_units = 3L,
    bottleneck_units = hidden_units,
    residual_at = NULL,
    activation = "relu",
    penalty = 0.001,
    mixture = 0,
    dropout = 0,
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

    brulee_resnet_bridge(
      processed,
      epochs = epochs,
      hidden_units = hidden_units,
      bottleneck_units = bottleneck_units,
      residual_at = residual_at,
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

brulee_resnet_bridge <- function(
  processed,
  epochs,
  hidden_units,
  bottleneck_units,
  residual_at,
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

  # Validate ResNet-specific arguments
  resnet_validated <- validate_resnet_args(
    hidden_units = hidden_units,
    bottleneck_units = bottleneck_units,
    residual_at = residual_at,
    activation = activation,
    dropout = dropout,
    grad_value_clip = grad_value_clip,
    grad_norm_clip = grad_norm_clip,
    call = call
  )

  # Extract validated/coerced values
  hidden_units <- resnet_validated$hidden_units
  bottleneck_units <- resnet_validated$bottleneck_units
  residual_at <- resnet_validated$residual_at
  activation <- resnet_validated$activation

  # Handle batch_size special logic (same as MLP)
  if (!is.null(batch_size) & optimizer != "LBFGS") {
    if (is.numeric(batch_size) & !is.integer(batch_size)) {
      batch_size <- as.integer(batch_size)
    }
    check_integer(batch_size, single = TRUE, 1, call = call)
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

  if (is.null(batch_size) & optimizer != "LBFGS") {
    batch_size <- 32L
    if (batch_size >= nrow(predictors)) {
      batch_size <- max(2, ceiling(nrow(predictors) / 10))
      batch_size <- as.integer(batch_size)
    }
  }

  ## -----------------------------------------------------------------------------

  # Validate outcome (ResNet accepts both numeric and factor)
  outcome <- validate_mlp_outcome(processed$outcomes[[1]], call = call)

  # ------------------------------------------------------------------------------

  lvls <- levels(outcome)
  xtab <- table(outcome)
  class_weights <- check_class_weights(class_weights, lvls, xtab, call = call)

  ## -----------------------------------------------------------------------------

  fit <-
    resnet_fit_imp(
      x = predictors,
      y = outcome,
      epochs = epochs,
      hidden_units = hidden_units,
      bottleneck_units = bottleneck_units,
      residual_at = residual_at,
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

  new_brulee_resnet(
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

new_brulee_resnet <- function(
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
    class = "brulee_resnet"
  )
}

## -----------------------------------------------------------------------------
# Fit code

resnet_fit_imp <-
  function(
    x,
    y,
    epochs = 100L,
    batch_size = 32,
    hidden_units = 3L,
    bottleneck_units = hidden_units,
    residual_at = NULL,
    penalty = 0.001,
    mixture = 0,
    dropout = 0,
    validation = 0.1,
    optimizer = "ADAMw",
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
    compl_data <- check_missing_data(x, y, "brulee_resnet", verbose)
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
          weight = weights_to_tensor(wts),
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

    # Determine batch size (same logic as MLP for LBFGS)
    if (optimizer == "LBFGS") {
      batch_size <- nrow(x)
    }
    batch_size <- min(batch_size, nrow(x))

    ## ---------------------------------------------------------------------------
    # Set torch dtype for both data and model

    or_dtype <- torch::torch_get_default_dtype()
    on.exit(torch::torch_set_default_dtype(or_dtype))
    torch::torch_set_default_dtype(torch::torch_float64())

    # Set device context for training
    training_output <- torch::with_device(device = device, {
      # Reset the seed so that different optimizers start from the same values
      torch::torch_manual_seed(start_seed + 1)

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
            num_layers = length(hidden_units),
            bottleneck_units = bottleneck_units,
            y = y_dim,
            levels = lvls,
            features = colnames(x)
          ),
          y_stats = y_stats,
          parameters = list(
            activation = activation,
            hidden_units = hidden_units,
            bottleneck_units = bottleneck_units,
            residual_at = residual_at,
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
      # Initialize model and optimizer

      d_type <- torch::torch_get_default_dtype()
      on.exit(torch::torch_set_default_dtype(d_type))
      torch::torch_set_default_dtype(torch::torch_float64())

      model <- resnet_module(
        num_pred = ncol(x),
        bottleneck_units = bottleneck_units,
        hidden_units = hidden_units,
        residual_at = residual_at,
        activation = activation,
        dropout = dropout,
        y_dim = y_dim
      )
      model$to(device = device) # Move model to the correct device

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

    # Add names back to class_weights
    training_output$parameters$class_weights <- as.numeric(class_weights)
    names(training_output$parameters$class_weights) <- lvls

    # Add device to result
    training_output$device <- device

    ## ---------------------------------------------------------------------------

    training_output
  }


resnet_layer_module <-
  torch::nn_module(
    "resnet_layer_module",
    initialize = function(
      input_dim,
      bottleneck_units,
      hidden_units,
      activation,
      dropout
    ) {
      self$bn <- torch::nn_batch_norm1d(input_dim)
      self$linear1 <- torch::nn_linear(input_dim, bottleneck_units)
      self$act <- get_activation_fn(activation)
      self$dropout1 <- torch::nn_dropout(dropout)
      self$linear2 <- torch::nn_linear(bottleneck_units, hidden_units)
      self$dropout2 <- torch::nn_dropout(dropout)
    },
    forward = function(x) {
      x |>
        self$bn() |>
        self$linear1() |>
        self$act() |>
        self$dropout1() |>
        self$linear2() |>
        self$dropout2()
    }
  )

resnet_module <-
  torch::nn_module(
    "resnet_module",
    initialize = function(
      num_pred,
      bottleneck_units,
      hidden_units,
      residual_at,
      activation,
      dropout,
      y_dim
    ) {
      num_layers <- length(hidden_units)

      # Validate lengths match
      if (length(bottleneck_units) != num_layers) {
        stop("bottleneck_units and hidden_units must have the same length")
      }

      # Ensure activation is a vector
      if (length(activation) == 1 && num_layers > 1) {
        activation <- rep(activation, num_layers)
      }

      if (length(activation) != num_layers) {
        stop("activation must be length 1 or match the number of layers")
      }

      # Store configuration
      self$num_layers <- num_layers
      self$residual_at <- residual_at
      self$hidden_units <- hidden_units

      # Build layers
      self$layers <- torch::nn_module_list()

      current_dim <- num_pred
      for (i in seq_len(num_layers)) {
        layer <- resnet_layer_module(
          input_dim = current_dim,
          bottleneck_units = bottleneck_units[i],
          hidden_units = hidden_units[i],
          activation = activation[i],
          dropout = dropout
        )
        self$layers$append(layer)
        current_dim <- hidden_units[i]
      }

      # Build projection layers for residual connections
      # Store which layers need projections and their dimensions
      self$projection_layers <- list()

      if (length(residual_at) > 0) {
        # Determine residual block boundaries
        block_starts <- c(1, residual_at[seq_len(length(residual_at) - 1)] + 1)
        block_ends <- residual_at

        for (i in seq_along(block_starts)) {
          start_idx <- block_starts[i]
          end_idx <- block_ends[i]

          # Determine input dimension to this block
          if (start_idx == 1) {
            identity_dim <- num_pred
          } else {
            identity_dim <- hidden_units[start_idx - 1]
          }

          output_dim <- hidden_units[end_idx]

          # Create projection if dimensions don't match
          if (identity_dim != output_dim) {
            proj_name <- paste0("proj_", end_idx)
            self[[proj_name]] <- torch::nn_linear(identity_dim, output_dim)
            self$projection_layers[[as.character(end_idx)]] <- proj_name
          }
        }
      }

      # Output layers
      self$bn_out <- torch::nn_batch_norm1d(current_dim)
      self$linear_out <- torch::nn_linear(current_dim, y_dim)
      self$y_dim <- y_dim
    },

    forward = function(x) {
      identity <- NULL
      block_start_idx <- 1
      residual_idx <- 1

      for (i in seq_len(self$num_layers)) {
        # Check if this is the start of a residual block
        if (residual_idx <= length(self$residual_at)) {
          if (i == block_start_idx) {
            identity <- x # Save identity at block start
          }
        }

        # Apply layer
        x <- self$layers[[i]](x)

        # Check if this is the end of a residual block
        if (
          residual_idx <= length(self$residual_at) &&
            i == self$residual_at[residual_idx]
        ) {
          # Apply projection if needed
          proj_key <- as.character(i)
          if (proj_key %in% names(self$projection_layers)) {
            proj_name <- self$projection_layers[[proj_key]]
            identity <- self[[proj_name]](identity)
          }

          # Residual addition
          x <- x + identity

          # Update for next block
          residual_idx <- residual_idx + 1
          block_start_idx <- i + 1
          identity <- NULL
        }
      }

      # Output head
      x <- x |>
        self$bn_out() |>
        self$linear_out()

      if (self$y_dim > 1L) {
        x <- torch::nn_softmax(dim = 2)(x)
      }

      x
    }
  )

## -----------------------------------------------------------------------------

get_num_resnet_coef <- function(x) {
  length(unlist(x$estimates[[1]]))
}


#' @export
print.brulee_resnet <- function(x, ...) {
  cat(cli::style_bold("Residual network (ResNet)"), "\n\n", sep = "")
  brulee_print(x, ...)
}
