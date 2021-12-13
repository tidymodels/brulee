#' Predict from a `lantern_mlp`
#'
#' @param object A `lantern_mlp` object.
#'
#' @param new_data A data frame or matrix of new predictors.
#' @param epoch An integer for the epoch to make predictions from. If this value
#' is larger than the maximum number that was fit, a warning is issued and the
#' parameters from the last epoch are used.
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
#'
#' @export
predict.lantern_mlp <- function(object, new_data, type = NULL, epoch = NULL, ...) {
  forged <- hardhat::forge(new_data, object$blueprint)
  type <- check_type(object, type)
  if (is.null(epoch)) {
    epoch <- object$best_epoch
  }
  predict_lantern_mlp_bridge(type, object, forged$predictors, epoch = epoch)
}

# ------------------------------------------------------------------------------
# Bridge

predict_lantern_mlp_bridge <- function(type, model, predictors, epoch) {

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

get_mlp_predict_function <- function(type) {
  switch(
    type,
    numeric = predict_lantern_mlp_numeric,
    prob    = predict_lantern_mlp_prob,
    class   = predict_lantern_mlp_class
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

revive_model <- function(model, epoch) {
  con <- rawConnection(model$models[[epoch]])
  on.exit({close(con)}, add = TRUE)
  module <- torch::torch_load(con)
  module
}

predict_lantern_mlp_raw <- function(model, predictors, epoch) {
  module <- revive_model(model, epoch)
  module$eval() # put the model in evaluation mode
  predictions <- module(torch::torch_tensor(predictors))
  predictions <- as.array(predictions)
  # torch doesn't have a NA type so it returns NaN
  predictions[is.nan(predictions)] <- NA
  predictions
}

predict_lantern_mlp_numeric <- function(model, predictors, epoch) {
  predictions <- predict_lantern_mlp_raw(model, predictors, epoch)
  predictions <- predict_lantern_mlp_raw(model, predictors, epoch)
  predictions <- predictions * model$y_stats$sd + model$y_stats$mean
  hardhat::spruce_numeric(predictions[,1])
}

predict_lantern_mlp_prob <- function(model, predictors, epoch) {
  predictions <- predict_lantern_mlp_raw(model, predictors, epoch)
  lvs <- get_levels(model)
  hardhat::spruce_prob(pred_levels = lvs, predictions)
}

predict_lantern_mlp_class <- function(model, predictors, epoch) {
  predictions <- predict_lantern_mlp_raw(model, predictors, epoch)
  predictions <- apply(predictions, 1, which.max2) # take the maximum value
  lvs <- get_levels(model)
  hardhat::spruce_class(factor(lvs[predictions], levels = lvs))
}

# a which max alternative that returns NA if any
# value is NA
which.max2 <- function(x) {
  if (any(is.na(x)))
    NA
  else
    which.max(x)
}

# get levels from a model object
get_levels <- function(model) {
  # Assumes univariate models
  levels(model$blueprint$ptypes$outcomes[[1]])
}


valid_predict_types <- function() {
  c("numeric", "prob", "class")
}

check_type <- function(model, type) {

  outcome_ptype <- model$blueprint$ptypes$outcomes[[1]]

  if (is.null(type)) {
    if (is.factor(outcome_ptype))
      type <- "class"
    else if (is.numeric(outcome_ptype))
      type <- "numeric"
    else
      rlang::abort(glue::glue("Unknown outcome type '{class(outcome_ptype)}'"))
  }

  type <- rlang::arg_match(type, valid_predict_types())

  if (is.factor(outcome_ptype)) {
    if (!type %in% c("prob", "class"))
      rlang::abort(glue::glue("Outcome is factor and the prediction type is '{type}'."))
  } else if (is.numeric(outcome_ptype)) {
    if (type != "numeric")
      rlang::abort(glue::glue("Outcome is numeric and the prediction type is '{type}'."))
  }

  type
}
