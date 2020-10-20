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
predict.torch_mlp <- function(object, new_data, type = NULL, epoch = NULL, ...) {
 forged <- hardhat::forge(new_data, object$blueprint)
 type <- check_type(object, type)
 if (is.null(epoch)) {
  epoch <- length(object$models)
 }
 predict_torch_mlp_bridge(type, object, forged$predictors, epoch = epoch)
}

mlp_valid_predict_types <- function() {
 c("numeric", "prob", "class")
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
  numeric = predict_torch_mlp_numeric,
  prob    = predict_torch_mlp_prob,
  class   = predict_torch_mlp_class
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

predict_torch_mlp_raw <- function(model, predictors, epoch) {
  con <- rawConnection(model$models[[epoch]])
  on.exit({close(con)}, add = TRUE)
  module <- torch::torch_load(con)
  module$eval() # put the model in evaluation mode
  predictions <- module(torch::torch_tensor(predictors))
}

predict_torch_mlp_numeric <- function(model, predictors, epoch) {
  predictions <- predict_torch_mlp_raw(model, predictors, epoch)
  hardhat::spruce_numeric(unname(as.array(predictions)[,1]))
}

predict_torch_mlp_prob <- function(model, predictors, epoch) {
  predictions <- predict_torch_mlp_raw(model, predictors, epoch)
  lvs <- levels(fit_df$blueprint$ptypes$outcomes$.outcome) # is this the correct way?
  hardhat::spruce_prob(pred_levels = lvs, as.array(predictions))
}

predict_torch_mlp_class <- function(model, predictors, epoch) {
  predictions <- predict_torch_mlp_raw(model, predictors, epoch)
  predictions <- torch_max(predictions, dim = 2)
  predictions <- as.integer(predictions[[2]]) # ids of higher values
  lvs <- levels(fit_df$blueprint$ptypes$outcomes$.outcome)
  hardhat::spruce_class(factor(lvs[predictions], levels = lvs))
}

check_type <- function(model, type) {

  outcome_ptype <- fit_df$blueprint$ptypes$outcomes$.outcome

  if (is.null(type)) {
    if (is.factor(outcome_ptype))
      type <- "class"
    else if (is.numeric(outcome_ptype))
      type <- "numeric"
    else
      rlang::abort(glue::glue("Unknown outcome type '{class(outcome_ptype)}'"))
  }

  type <- rlang::arg_match(type, mlp_valid_predict_types())

  if (is.factor(outcome_ptype)) {
    if (!type %in% c("prob", "class"))
      rlang::abort(glue::glue("Outcome is factor and the prediction type is '{type}'."))
  } else if (is.numeric(outcome_ptype)) {
    if (type != "numeric")
      rlang::abort(glue::glue("Outcome is numeric and the prediction type is '{type}'."))
  }

  type
}
