#' Predict from a `lantern_logistic_reg`
#'
#' @param object A `lantern_logistic_reg` object.
#'
#' @param new_data A data frame or matrix of new predictors.
#' @param epoch An integer for the epoch to make predictions from. If this value
#' is larger than the maximum number that was fit, a warning is issued and the
#' parameters from the last epoch are used.
#' @param type A single character. The type of predictions to generate.
#' Valid options are:
#'
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
#'
#' @export
predict.lantern_logistic_reg <- function(object, new_data, type = NULL, epoch = NULL, ...) {
 forged <- hardhat::forge(new_data, object$blueprint)
 type <- check_type(object, type)
 if (is.null(epoch)) {
   epoch <- object$best_epoch
 }
 predict_lantern_logistic_reg_bridge(type, object, forged$predictors, epoch = epoch)
}

# ------------------------------------------------------------------------------
# Bridge

predict_lantern_logistic_reg_bridge <- function(type, model, predictors, epoch) {

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

 predict_function <- get_logistic_reg_predict_function(type)

 max_epoch <- length(model$models)
 if (epoch > max_epoch) {
  msg <- paste("The model fit only", max_epoch, "epochs; predictions cannot",
               "be made at epoch", epoch, "so last epoch is used.")
  rlang::warn(msg)
 }

 predictions <- predict_function(model, predictors, epoch)
 hardhat::validate_prediction_size(predictions, predictors)
 predictions
}

get_logistic_reg_predict_function <- function(type) {
 switch(
  type,
  prob    = predict_lantern_logistic_reg_prob,
  class   = predict_lantern_logistic_reg_class
 )
}

# ------------------------------------------------------------------------------
# Implementation

predict_lantern_logistic_reg_raw <- function(model, predictors, epoch) {
 module <- revive_model(model, epoch)
 module$eval() # put the model in evaluation mode
 predictions <- module(torch::torch_tensor(predictors))
 predictions <- as.array(predictions)
 # torch doesn't have a NA type so it returns NaN
 predictions[is.nan(predictions)] <- NA
 predictions
}

predict_lantern_logistic_reg_prob <- function(model, predictors, epoch) {
 predictions <- predict_lantern_logistic_reg_raw(model, predictors, epoch)
 lvs <- get_levels(model)
 hardhat::spruce_prob(pred_levels = lvs, predictions)
}

predict_lantern_logistic_reg_class <- function(model, predictors, epoch) {
 predictions <- predict_lantern_logistic_reg_raw(model, predictors, epoch)
 predictions <- apply(predictions, 1, which.max2) # take the maximum value
 lvs <- get_levels(model)
 hardhat::spruce_class(factor(lvs[predictions], levels = lvs))
}
