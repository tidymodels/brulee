#' Fit residual neural networks (ResNet)
#'
#' `brulee_resnet()` fits residual network models with skip connections.
#'
#' @inheritParams brulee_mlp
#' @param hidden_units An integer for the number of hidden units, or a vector
#'   of integers. If a vector of integers, the model will have
#'   `length(hidden_units)` layers each with `hidden_units[i]` hidden units. If
#'   a single integer is passed and `num_layers > 1`, the value is used for all
#'   layers.
#' @param num_layers An integer for the number of layers within each residual block.
#'   Must be >= 1.
#' @param block_units An integer for the number of hidden units in each layer
#'   within a residual block. Must be >= 2.
#'
#' @details
#'
#' This function fits residual network (ResNet) models for regression (when
#'  the outcome is a number) or classification (a factor). ResNets use skip
#'  connections that add the input of a block to its output, allowing gradients
#'  to flow more easily through deep networks. For regression, the mean squared
#'  error is optimized and cross-entropy is the loss function for classification.
#'
#' The network consists of `hidden_units` residual blocks. Each block contains
#' `num_layers` layers, with each layer having `block_units` hidden units.
#' Skip connections add the block input to the block output.
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
#' He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for
#' image recognition. In _Proceedings of the IEEE conference on computer vision
#' and pattern recognition_ (pp. 770-778).
#'
#' He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity mappings in deep
#' residual networks. In _European conference on computer vision_ (pp. 630-645).
#' Springer, Cham.
#'
#'
#' @seealso [predict.brulee_resnet()], [coef.brulee_resnet()], [autoplot.brulee_resnet()]
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
#'                       hidden_units = 3, num_layers = 2, block_units = 10,
#'                       epochs = 50, batch_size = 32)
#'  fit
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
#'                           hidden_units = 2, num_layers = 2, block_units = 5,
#'                           epochs = 200L, learn_rate = 0.1, activation = "elu",
#'                           penalty = 0.1, batch_size = 2^8, optimizer = "SGD")
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

# XY method - data frame

#' @export
#' @rdname brulee_resnet
brulee_resnet.data.frame <-
  function(
    x,
    y,
    epochs = 100L,
    hidden_units = 3L,
    num_layers = 2L,
    block_units = 10L,
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
    ...
  ) {
    processed <- hardhat::mold(x, y)

    brulee_resnet_bridge(
      processed,
      epochs = epochs,
      hidden_units = hidden_units,
      num_layers = num_layers,
      block_units = block_units,
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
  num_layers = 2L,
  block_units = 10L,
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
  ...
) {
  processed <- hardhat::mold(x, y)

  brulee_resnet_bridge(
    processed,
    epochs = epochs,
    hidden_units = hidden_units,
    num_layers = num_layers,
    block_units = block_units,
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
    num_layers = 2L,
    block_units = 10L,
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
    ...
  ) {
    processed <- hardhat::mold(formula, data)

    brulee_resnet_bridge(
      processed,
      epochs = epochs,
      hidden_units = hidden_units,
      num_layers = num_layers,
      block_units = block_units,
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
    num_layers = 2L,
    block_units = 10L,
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
    ...
  ) {
    processed <- hardhat::mold(x, data)

    brulee_resnet_bridge(
      processed,
      epochs = epochs,
      hidden_units = hidden_units,
      num_layers = num_layers,
      block_units = block_units,
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
      ...
    )
  }

# ------------------------------------------------------------------------------
# Bridge

brulee_resnet_bridge <- function(
  processed,
  epochs,
  hidden_units,
  num_layers,
  block_units,
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
  ...
) {
  if (!torch::torch_is_installed()) {
    cli::cli_abort(
      "The torch backend has not been installed; use `torch::install_torch()`."
    )
  }

  f_nm <- "brulee_resnet"

  if (length(hidden_units) == 1 & num_layers > 1) {
    hidden_units <- rep(hidden_units, num_layers)
  }

  if (length(block_units) == 1 & num_layers > 1) {
    block_units <- rep(block_units, num_layers)
  }

  if (length(hidden_units) != length(block_units)) {
    cli::cli_abort(
      "The lengths of {.arg hidden_units} ({length(hidden_units)} and
    {.arg block_units}  ({length(block_units)} should be the same."
    )
  }

  # Validate ResNet-specific arguments
  resnet_validated <- validate_resnet_args(
    num_layers = num_layers,
    block_units = block_units,
    fn = f_nm
  )

  # Extract validated/coerced values
  num_layers <- resnet_validated$num_layers
  block_units <- resnet_validated$block_units

  # Validate hidden_units, activation, dropout, gradient clipping
  resnet_mlp_validated <- validate_mlp_args(
    hidden_units = hidden_units,
    activation = activation,
    dropout = dropout,
    grad_value_clip = grad_value_clip,
    grad_norm_clip = grad_norm_clip,
    fn = f_nm
  )

  # Extract validated/coerced values
  hidden_units <- resnet_mlp_validated$hidden_units
  activation <- resnet_mlp_validated$activation

  # Handle batch_size special logic (same as MLP)
  if (!is.null(batch_size) & optimizer != "LBFGS") {
    if (is.numeric(batch_size) & !is.integer(batch_size)) {
      batch_size <- as.integer(batch_size)
    }
    check_integer(batch_size, single = TRUE, 1, fn = f_nm)
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
    fn = f_nm
  )

  # Extract validated/coerced values
  epochs <- validated$epochs
  batch_size <- validated$batch_size

  ## -----------------------------------------------------------------------------

  # Process predictors
  predictors <- process_predictors(processed$predictors, fn = f_nm)

  ## -----------------------------------------------------------------------------

  # Validate outcome (ResNet accepts both numeric and factor)
  outcome <- validate_mlp_outcome(processed$outcomes[[1]], fn = f_nm)

  # ------------------------------------------------------------------------------

  lvls <- levels(outcome)
  xtab <- table(outcome)
  class_weights <- check_class_weights(class_weights, lvls, xtab, f_nm)

  ## -----------------------------------------------------------------------------

  fit <-
    resnet_fit_imp(
      x = predictors,
      y = outcome,
      epochs = epochs,
      hidden_units = hidden_units,
      num_layers = num_layers,
      block_units = block_units,
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
  blueprint
) {
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
    num_layers = 2L,
    block_units = 10L,
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
          weight = wts,
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
    # Convert to index sampler and data loader

    or_dtype <- torch::torch_get_default_dtype()
    on.exit(torch::torch_set_default_dtype(or_dtype))
    torch::torch_set_default_dtype(torch::torch_float64())

    # Reset the seed so that different optimizers start from the same values
    torch::torch_manual_seed(start_seed + 1)

    torch_data <- setup_torch_data(x, y, x_val, y_val, batch_size, validation)
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
          num_blocks = hidden_units,
          num_layers = num_layers,
          block_units = block_units,
          y = y_dim,
          levels = lvls,
          features = colnames(x)
        ),
        y_stats = y_stats,
        parameters = list(
          activation = activation,
          hidden_units = hidden_units,
          num_layers = num_layers,
          block_units = block_units,
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
      blk_width = block_units,
      d_hidden = hidden_units,
      num_blocks = num_layers,
      act_type = activation,
      dropout = dropout,
      y_dim = y_dim
    )

    mixture <- check_mixture(mixture, optimizer)

    # Note that if a penalty is used, it might affect the `loss_fn` _or_ the
    # optimizer. See `opt_uses_penalty()` where the determination is made.
    loss_fn <- make_penalized_loss(loss_fn, model, penalty, mixture, optimizer)

    optimizer_obj <- set_optimizer(
      optimizer,
      model,
      learn_rate,
      momentum,
      penalty
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
      epoch_chr <- gsub(" ", "0", format(0:epochs))
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

    ## ---------------------------------------------------------------------------

    # Add names back to class_weights
    res$parameters$class_weights <- as.numeric(class_weights)
    names(res$parameters$class_weights) <- lvls

    ## ---------------------------------------------------------------------------

    res
  }


resnet_block_module <-
  torch::nn_module(
    "resnet_block_module",
    initialize = function(blk_width, d_hidden, dropout, act_type) {
      self$bn1 <- torch::nn_batch_norm1d(blk_width)
      self$linear1 <- torch::nn_linear(blk_width, d_hidden)
      self$act1 <- get_activation_fn(act_type)
      self$bn2 <- torch::nn_batch_norm1d(d_hidden)
      self$linear2 <- torch::nn_linear(d_hidden, blk_width)
      self$act2 <- get_activation_fn(act_type)
      self$dropout <- torch::nn_dropout(dropout)
    },
    forward = function(x) {
      z <- x |>
        self$bn1() |>
        self$act1() |>
        self$linear1() |>
        self$dropout()
      z <- z |>
        self$bn2() |>
        self$act2() |>
        self$linear2() |>
        self$dropout()
      x + z
    }
  )

resnet_module <-
  torch::nn_module(
    "resnet_module",
    initialize = function(
      num_pred,
      blk_width,
      d_hidden,
      num_blocks,
      act_type,
      dropout,
      y_dim
    ) {
      # Input projection: num_pred -> blk_width
      self$linear_in <- torch::nn_linear(num_pred, blk_width)

      # Residual blocks
      self$blocks <- torch::nn_module_list(
        lapply(seq_len(num_blocks), function(.) {
          resnet_block_module(blk_width, d_hidden, dropout, act_type)
        })
      )

      # Prediction head
      self$bn_out <- torch::nn_batch_norm1d(blk_width)
      self$act_out <- get_activation_fn(act_type)
      self$linear_out <- torch::nn_linear(blk_width, y_dim)

      # For classification (y_dim > 1), add softmax
      self$y_dim <- y_dim
    },
    forward = function(x) {
      x <- self$linear_in(x)

      for (i in seq_along(self$blocks)) {
        x <- self$blocks[[i]](x)
      }

      x <- x |>
        self$bn_out() |>
        self$act_out() |>
        self$linear_out()

      if (self$y_dim == 1L) {
        x <- torch::torch_squeeze(x, dim = 2L)
      } else {
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
  cat("Residual network (ResNet)\n\n")
  cat(
    x$parameters$activation,
    " activation,\n",
    x$dims$num_blocks,
    " residual blocks,\n",
    x$dims$num_layers,
    " layers per block,\n",
    x$dims$block_units,
    " units per layer,\n",
    format(get_num_resnet_coef(x), big.mark = ","),
    " model parameters\n",
    sep = ""
  )
  brulee_print(x, ...)
}
