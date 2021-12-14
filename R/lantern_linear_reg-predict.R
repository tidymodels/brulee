#' Predict from a `lantern_linear_reg`
#'
#' @param object A `lantern_linear_reg` object.
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
predict.lantern_linear_reg <- function(object, new_data, type = NULL, epoch = NULL, ...) {
  forged <- hardhat::forge(new_data, object$blueprint)
  type <- check_type(object, type)
  if (is.null(epoch)) {
    epoch <- object$best_epoch
  }
  predict_lantern_linear_reg_bridge(type, object, forged$predictors, epoch = epoch)
}

# ------------------------------------------------------------------------------
# Bridge

predict_lantern_linear_reg_bridge <- function(type, model, predictors, epoch) {

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

  predict_function <- get_linear_reg_predict_function(type)

  max_epoch <- length(model$estimates)
  if (epoch > max_epoch) {
    msg <- paste("The model fit only", max_epoch, "epochs; predictions cannot",
                 "be made at epoch", epoch, "so last epoch is used.")
    rlang::warn(msg)
  }

  predictions <- predict_function(model, predictors, epoch)
  hardhat::validate_prediction_size(predictions, predictors)
  predictions
}

get_linear_reg_predict_function <- function(type) {
  predict_lantern_linear_reg_numeric
}

# ------------------------------------------------------------------------------
# Implementation


predict_lantern_linear_reg_raw <- function(model, predictors, epoch) {
  # convert from raw format
  module <- revive_model(model$model_obj)
  # get current model parameters
  estimates <- model$estimates[[epoch]]
  # convert to torch representation
  estimates <- lapply(estimates, torch::torch_tensor)
  # stuff back into the model
  module$load_state_dict(estimates)
  # put the model in evaluation mode
  module$eval()
  predictions <- module(torch::torch_tensor(predictors))
  predictions <- as.array(predictions)
  # torch doesn't have a NA type so it returns NaN
  predictions[is.nan(predictions)] <- NA
  predictions
}

predict_lantern_linear_reg_numeric <- function(model, predictors, epoch) {
  predictions <- predict_lantern_linear_reg_raw(model, predictors, epoch)
  predictions <- predictions * model$y_stats$sd + model$y_stats$mean
  hardhat::spruce_numeric(predictions[,1])
}
