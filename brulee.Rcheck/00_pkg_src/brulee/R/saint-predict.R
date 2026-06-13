#' Predict from a `brulee_saint`
#'
#' @param object A `brulee_saint` object.
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
#' if (torch::torch_is_installed() & rlang::is_installed(c("recipes", "modeldata"))) {
#'   set.seed(87261)
#'   tr_data <- modeldata::sim_classification(500)
#'   te_data <- modeldata::sim_classification(50)
#'
#'   set.seed(2)
#'   fit <- brulee_saint(class ~ ., data = tr_data,
#'                       epochs = 50L, batch_size = 64L, stop_iter = 10L,
#'                       learn_rate = 0.001)
#'   fit
#'
#'   autoplot(fit)
#'
#'  predict(fit, te_data)
#'  predict(fit, te_data, type = "prob")
#' }
#' }
#' @export
predict.brulee_saint <- function(
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
  predict_brulee_saint_bridge(
    type,
    object,
    forged$predictors,
    epoch = epoch,
    call = call
  )
}

# ------------------------------------------------------------------------------
# Bridge

predict_brulee_saint_bridge <- function(
  type,
  model,
  predictors,
  epoch,
  call = rlang::caller_env()
) {
  predict_function <- get_saint_predict_function(type)

  max_epoch <- length(model$estimates)
  last_epoch_note(epoch, max_epoch, call = call)

  predictions <- predict_function(model, predictors, epoch)
  hardhat::validate_prediction_size(predictions, predictors)
  predictions
}

get_saint_predict_function <- function(type) {
  switch(
    type,
    numeric = predict_brulee_saint_numeric,
    prob = predict_brulee_saint_prob,
    class = predict_brulee_saint_class
  )
}

# ------------------------------------------------------------------------------
# Implementation

predict_brulee_saint_raw <- function(model, predictors, epoch) {
  device <- get_safe_device(model$device)

  cat_names <- model$dims$cat_names
  cont_names <- model$dims$cont_names

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

  module <- revive_model(model$model_obj, device)

  estimates <- model$estimates[[epoch + 1]]
  estimates <- lapply(estimates, float_32, device = device)
  module$load_state_dict(estimates)
  module$eval()

  row_attn <- model$parameters$row_attention_on_predict %||% FALSE
  if (!is.null(module$backbone$use_row_attention)) {
    module$backbone$use_row_attention <- row_attn
  }

  predictions <- module(x_cat, x_cont)
  predictions <- as.array(predictions)
  predictions[is.nan(predictions)] <- NA
  predictions
}

predict_brulee_saint_numeric <- function(model, predictors, epoch) {
  predictions <- predict_brulee_saint_raw(model, predictors, epoch)
  predictions <- predictions * model$y_stats$sd + model$y_stats$mean
  hardhat::spruce_numeric(predictions[, 1])
}

predict_brulee_saint_prob <- function(model, predictors, epoch) {
  predictions <- predict_brulee_saint_raw(model, predictors, epoch)
  lvs <- get_levels(model)
  hardhat::spruce_prob(pred_levels = lvs, predictions)
}

predict_brulee_saint_class <- function(model, predictors, epoch) {
  predictions <- predict_brulee_saint_raw(model, predictors, epoch)
  predictions <- apply(predictions, 1, which.max2)
  lvs <- get_levels(model)
  hardhat::spruce_class(factor(lvs[predictions], levels = lvs))
}
