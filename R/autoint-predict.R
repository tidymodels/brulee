#' Predict from a `brulee_auto_int`
#'
#' @param object A `brulee_auto_int` object.
#'
#' @param new_data A data frame or matrix of new predictors.
#' @param epoch An integer for the epoch to make predictions. If this value
#' is larger than the maximum number that was fit, a warning is issued and the
#' parameters from the last epoch are used. If left `NULL`, the epoch
#' associated with the smallest loss is used.
#' @param type A single character. The type of predictions to generate.
#' Valid options are:
#'
#' - `"numeric"` for numeric predictions.
#' - `"class"` for hard class predictions
#' - `"prob"` for soft class predictions (i.e., class probabilities)
#'
#' @param ... Not used, but required for extensibility.
#'
#' @return
#'
#' A tibble of predictions. The number of rows in the tibble is guaranteed
#' to be the same as the number of rows in `new_data`.
#'
#' @examplesIf !brulee:::is_cran_check()
#' \donttest{
#' if (torch::torch_is_installed() && rlang::is_installed(c("recipes", "modeldata"))) {
#'   set.seed(87261)
#'   tr_data <- modeldata::sim_classification(500)
#'   te_data <- modeldata::sim_classification(50)
#'
#'   set.seed(2)
#'   fit <- brulee_auto_int(class ~ ., data = tr_data,
#'                          epochs = 50L, batch_size = 64L, stop_iter = 10L,
#'                          hidden_units = 5, hidden_activations = "relu",
#'                          learn_rate = 0.01, penalty = 0.01)
#'   fit
#'
#'   autoplot(fit)
#'
#'  predict(fit, te_data)
#'  predict(fit, te_data, type = "prob")
#' }
#' }
#' @export
predict.brulee_auto_int <- function(
  object,
  new_data,
  type = NULL,
  epoch = NULL,
  ...
) {
  call <- rlang::current_env()
  forged <- hardhat::forge(new_data, object$blueprint)
  type <- check_type(object, type, call = call)
  if (is.null(epoch)) {
    epoch <- object$best_epoch
  }
  predict_brulee_auto_int_bridge(
    type,
    object,
    forged$predictors,
    epoch = epoch,
    call = call
  )
}

# ------------------------------------------------------------------------------
# Bridge

predict_brulee_auto_int_bridge <- function(
  type,
  model,
  predictors,
  epoch,
  call = rlang::caller_env()
) {
  predict_function <- get_auto_int_predict_function(type)

  max_epoch <- length(model$estimates)
  last_epoch_note(epoch, max_epoch, call = call)

  predictions <- predict_function(model, predictors, epoch)
  hardhat::validate_prediction_size(predictions, predictors)
  predictions
}

get_auto_int_predict_function <- function(type) {
  switch(
    type,
    numeric = predict_brulee_auto_int_numeric,
    prob = predict_brulee_auto_int_prob,
    class = predict_brulee_auto_int_class
  )
}

# ------------------------------------------------------------------------------
# Implementation

predict_brulee_auto_int_raw <- function(model, predictors, epoch) {
  device <- get_safe_device(model$device)

  # Split predictors into categorical and continuous using stored metadata
  cat_names <- model$dims$cat_names
  cont_names <- model$dims$cont_names

  # Build categorical tensor
  x_cat <- NULL
  if (length(cat_names) > 0) {
    cat_cols <- intersect(cat_names, names(predictors))
    if (length(cat_cols) > 0) {
      cat_mat <- do.call(
        cbind,
        lapply(cat_cols, function(nm) {
          as.integer(predictors[[nm]])
        })
      )
      x_cat <- torch::torch_tensor(
        cat_mat,
        dtype = torch::torch_long(),
        device = device
      )
    }
  }

  # Build continuous tensor
  x_cont <- NULL
  if (length(cont_names) > 0) {
    cont_cols <- intersect(cont_names, names(predictors))
    if (length(cont_cols) > 0) {
      x_cont <- float_32(
        as.matrix(predictors[, cont_cols, drop = FALSE]),
        device = device
      )
    }
  }

  # Revive model from raw format
  module <- revive_model(model$model_obj, device)

  # Load parameters for the requested epoch
  estimates <- model$estimates[[epoch + 1]]
  estimates <- lapply(estimates, float_32, device = device)
  module$load_state_dict(estimates)
  module$eval()

  predictions <- module(x_cat, x_cont)
  predictions <- to_probs(predictions, model)
  predictions <- as.array(predictions)
  predictions[is.nan(predictions)] <- NA
  predictions
}

predict_brulee_auto_int_numeric <- function(model, predictors, epoch) {
  predictions <- predict_brulee_auto_int_raw(model, predictors, epoch)
  predictions <- predictions * model$y_stats$sd + model$y_stats$mean
  hardhat::spruce_numeric(predictions[, 1])
}

predict_brulee_auto_int_prob <- function(model, predictors, epoch) {
  predictions <- predict_brulee_auto_int_raw(model, predictors, epoch)
  lvs <- get_levels(model)
  hardhat::spruce_prob(pred_levels = lvs, predictions)
}

predict_brulee_auto_int_class <- function(model, predictors, epoch) {
  predictions <- predict_brulee_auto_int_raw(model, predictors, epoch)
  predictions <- apply(predictions, 1, which.max2)
  lvs <- get_levels(model)
  hardhat::spruce_class(factor(lvs[predictions], levels = lvs))
}
