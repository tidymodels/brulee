#' Predict from a `torch_linear_reg`
#'
#' @param object A `torch_linear_reg` object.
#'
#' @param new_data A data frame or matrix of new predictors.
#'
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
#' @examples
#' train <- mtcars[1:20,]
#' test <- mtcars[21:32, -1]
#'
#' # Fit
#' mod <- torch_linear_reg(mpg ~ cyl + log(drat), train)
#'
#' # Predict, with preprocessing
#' predict(mod, test)
#'
#' @export
predict.torch_linear_reg <- function(object, new_data, type = "numeric", ...) {
  forged <- hardhat::forge(new_data, object$blueprint)
  rlang::arg_match(type, valid_predict_types())
  predict_torch_linear_reg_bridge(type, object, forged$predictors)
}

valid_predict_types <- function() {
  c("numeric")
}

# ------------------------------------------------------------------------------
# Bridge

predict_torch_linear_reg_bridge <- function(type, model, predictors) {

  if (is.data.frame(predictors)) {
    predictors <- stats::model.matrix(model$terms, predictors)
  }

  predict_function <- get_predict_function(type)
  predictions <- predict_function(model, predictors)

  hardhat::validate_prediction_size(predictions, predictors)

  predictions
}

get_predict_function <- function(type) {
  switch(
    type,
    numeric = predict_torch_linear_reg_numeric
  )
}

# ------------------------------------------------------------------------------
# Implementation

predict_torch_linear_reg_numeric <- function(model, predictors) {
  predictions <- predictors %*% model$coefs
  hardhat::spruce_numeric(unname(predictions[,1]))
}
