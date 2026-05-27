#' Split data into training and validation sets
#'
#' @param x Predictor matrix
#' @param y Outcome vector
#' @param validation Proportion to use for validation (0 to 1)
#'
#' @return List with x_train, y_train, x_val, y_val
#' @keywords internal
#' @noRd
split_validation <- function(x, y, validation) {
  if (validation <= 0) {
    return(list(
      x_train = x,
      y_train = y,
      x_val = NULL,
      y_val = NULL
    ))
  }

  n <- length(y)
  in_val <- sample(seq_along(y), floor(n * validation))

  list(
    x_train = x[-in_val, , drop = FALSE],
    y_train = y[-in_val],
    x_val = x[in_val, , drop = FALSE],
    y_val = y[in_val]
  )
}

#' Determine appropriate batch size based on optimizer and data size
#'
#' @param batch_size User-specified batch size (can be NULL)
#' @param optimizer Optimizer name
#' @param n_rows Number of rows in training data
#'
#' @return Integer batch size
#' @keywords internal
#' @noRd
determine_batch_size <- function(
  batch_size,
  optimizer,
  n_rows,
  call = rlang::caller_env()
) {
  # LBFGS doesn't use batches - set batch size to full dataset
  if (optimizer == "LBFGS") {
    if (!is.null(batch_size)) {
      cli::cli_warn(
        "{.arg batch_size} is only used for the SGD optimizer.",
        call = call
      )
    }
    return(n_rows)
  }

  # For other optimizers
  if (is.null(batch_size)) {
    return(n_rows)
  }

  min(batch_size, n_rows)
}

#' Setup torch datasets and dataloaders
#'
#' @param x_train Training predictors
#' @param y_train Training outcome
#' @param x_val Validation predictors (can be NULL)
#' @param y_val Validation outcome (can be NULL)
#' @param batch_size Batch size
#' @param validation Validation proportion (> 0 if using validation)
#'
#' @return List with dl (training dataloader) and dl_val (validation dataloader)
#' @keywords internal
#' @noRd
setup_torch_data <- function(
  x_train,
  y_train,
  x_val,
  y_val,
  batch_size,
  validation,
  device = NULL
) {
  # Create training dataloader
  ds <- matrix_to_dataset(x_train, y_train, device = device)
  dl <- torch::dataloader(ds, batch_size = batch_size)

  # Create validation dataloader if needed
  dl_val <- NULL
  if (validation > 0) {
    ds_val <- matrix_to_dataset(x_val, y_val, device = device)
    dl_val <- torch::dataloader(ds_val)
  }

  list(
    dl = dl,
    dl_val = dl_val
  )
}

#' Run the training loop over epochs
#'
#' @param model Torch module
#' @param dl Training dataloader
#' @param dl_val Validation dataloader (can be NULL)
#' @param loss_fn Loss function
#' @param optimizer_obj Torch optimizer
#' @param epochs Number of epochs
#' @param learn_rate Initial learning rate
#' @param stop_iter Number of epochs to wait before early stopping
#' @param validation Validation proportion
#' @param class_weights Class weights for classification (NULL for regression)
#' @param loss_label Label for loss output
#' @param verbose Whether to print progress
#' @param grad_value_clip Gradient value clipping threshold (Inf = no clipping)
#' @param grad_norm_clip Gradient norm clipping threshold (Inf = no clipping)
#' @param rate_schedule Learning rate schedule type
#' @param batch_callback Optional function called after each optimizer step.
#'   Used by `brulee_rln()` to apply the per-weight regularization update.
#' @param ... Additional arguments passed to set_learn_rate
#'
#' @return List with param_per_epoch, loss_vec, and best_epoch
#' @keywords internal
#' @noRd
run_training_loop <- function(
  model,
  dl,
  dl_val,
  loss_fn,
  optimizer_obj,
  epochs,
  learn_rate,
  stop_iter,
  validation,
  class_weights = NULL,
  loss_label = "\tLoss:",
  verbose = FALSE,
  grad_value_clip = Inf,
  grad_norm_clip = Inf,
  rate_schedule = "none",
  batch_callback = NULL,
  ...
) {
  loss_prev <- 10^38
  loss_min <- loss_prev
  poor_epoch <- 0
  best_epoch <- 1L
  loss_vec <- rep(NA_real_, epochs)
  param_per_epoch <- list()

  if (verbose) {
    epoch_chr <- format(1:epochs)
  }

  # Main training loop
  for (epoch in 1:epochs) {
    # Update learning rate
    learn_rate <- set_learn_rate(
      epoch - 1,
      learn_rate,
      type = rate_schedule,
      ...
    )

    for (i in seq_along(optimizer_obj$param_groups)) {
      optimizer_obj$param_groups[[i]]$lr <- learn_rate
    }

    # Training loop over batches
    coro::loop(
      for (batch in dl) {
        cl <- function() {
          optimizer_obj$zero_grad()
          pred <- model(batch$x)

          # Call loss function with or without class_weights
          if (is.null(class_weights)) {
            loss <- loss_fn(pred, batch$y)
          } else {
            loss <- loss_fn(pred, batch$y, class_weights)
          }

          loss$backward()

          # Gradient clipping (MLP only, by default Inf = no clipping)
          if (is.finite(grad_value_clip)) {
            try(
              torch::nn_utils_clip_grad_value_(
                model$parameters,
                grad_value_clip
              ),
              silent = TRUE
            )
          }
          if (is.finite(grad_norm_clip)) {
            try(
              torch::nn_utils_clip_grad_norm_(
                model$parameters,
                grad_norm_clip
              ),
              silent = TRUE
            )
          }

          loss
        }
        optimizer_obj$step(cl)
        if (!is.null(batch_callback)) {
          batch_callback()
        }
      }
    )

    # Calculate loss on validation or training set
    if (validation > 0) {
      pred <- model(dl_val$dataset$tensors$x)
      if (is.null(class_weights)) {
        loss <- loss_fn(pred, dl_val$dataset$tensors$y)
      } else {
        loss <- loss_fn(pred, dl_val$dataset$tensors$y, class_weights)
      }
    } else {
      pred <- model(dl$dataset$tensors$x)
      if (is.null(class_weights)) {
        loss <- loss_fn(pred, dl$dataset$tensors$y)
      } else {
        loss <- loss_fn(pred, dl$dataset$tensors$y, class_weights)
      }
    }

    # Calculate and store loss
    loss_curr <- loss$item()
    loss_vec[epoch] <- loss_curr

    # Check for NaN
    if (is.nan(loss_curr)) {
      cli::cli_warn(
        "Early stopping occurred at epoch {epoch} due to numerical overflow of the loss function."
      )
      break()
    }

    # Track best epoch
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

    # Save parameters for this epoch
    param_per_epoch[[epoch]] <-
      lapply(model$state_dict(), function(x) torch::as_array(x$cpu()))

    # Verbose output
    if (verbose) {
      cli::cli_inform(
        "epoch: {epoch_chr[epoch]}, learn rate: {signif(learn_rate, 3)}, {loss_label} {signif(loss_curr, 3)}"
      )
    }

    # Early stopping
    if (poor_epoch == stop_iter) {
      break()
    }
  }

  # Return results
  list(
    param_per_epoch = param_per_epoch,
    loss_vec = loss_vec[1:length(param_per_epoch)],
    best_epoch = best_epoch
  )
}
