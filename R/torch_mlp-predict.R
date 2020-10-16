#' Predict from a `torch_mlp`
#'
#' @param object A `torch_mlp` object.
#'
#' @param new_data A data frame or matrix of new predictors.
#' @param epoch An integer for the epoch to make predictions from. If this value
#' is larger than the maximum number that was fit, a warning is issued and the
#' parameters from the last epoch are used.
#' @param type A single character. The type of predictions to generate.
#' Valid options are:
#'
#' - `"numeric"` for numeric predictions.
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
predict.torch_mlp <- function(object, new_data, type = "numeric", epoch = NULL, ...) {
 forged <- hardhat::forge(new_data, object$blueprint)
 rlang::arg_match(type, mlp_valid_predict_types())
 if (is.null(epoch)) {
  epoch <- length(object$models)
 }
 predict_torch_mlp_bridge(type, object, forged$predictors, epoch = epoch)
}

mlp_valid_predict_types <- function() {
 c("numeric")
}

# ------------------------------------------------------------------------------
# Bridge

predict_torch_mlp_bridge <- function(type, model, predictors, epoch) {

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

 predict_function <- get_mlp_predict_function(type)

 max_epoch <- nrow(model$coefs)
 if (epoch > max_epoch) {
  msg <- paste("The model fit only", max_epoch, "epochs; predictions cannot",
               "be made at epoch", epoch, "so last epoch is used.")
  rlang::warn(msg)
 }

 predictions <- predict_function(model, predictors, epoch)

 hardhat::validate_prediction_size(predictions, predictors)

 predictions
}

get_mlp_predict_function <- function(type) {
 switch(
  type,
  numeric = predict_torch_mlp_numeric
 )
}

# ------------------------------------------------------------------------------
# Implementation

add_intercept <- function(x) {
 if (!is.array(x)) {
  x <- as.array(x)
 }
 cbind(rep(1, nrow(x)), x)
}

predict_torch_mlp_numeric <- function(model, predictors, epoch) {
  con <- rawConnection(model$models[[epoch]])
  on.exit({close(con)}, add = TRUE)
  module <- torch::torch_load(con)
  module$eval() # put the model in evaluation mode
  predictions <- as.array(module(torch::torch_tensor(predictors)))
  hardhat::spruce_numeric(unname(predictions[,1]))
}
