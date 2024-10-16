#' Predict from a `brulee_linear_reg`
#'
#' @inheritParams predict.brulee_mlp
#' @param object A `brulee_linear_reg` object.
#' @param type A single character. The type of predictions to generate.
#' Valid options are:
#'
#' - `"numeric"` for numeric predictions.
#'
#' @return
#'
#' A tibble of predictions. The number of rows in the tibble is guaranteed
#' to be the same as the number of rows in `new_data`.
#'
#' @examples
#' \donttest{
#' if (torch::torch_is_installed() & rlang::is_installed("recipes")) {
#'
#'  data(ames, package = "modeldata")
#'
#'  ames$Sale_Price <- log10(ames$Sale_Price)
#'
#'  set.seed(1)
#'  in_train <- sample(1:nrow(ames), 2000)
#'  ames_train <- ames[ in_train,]
#'  ames_test  <- ames[-in_train,]
#'
#'  # Using recipe
#'  library(recipes)
#'
#'  ames_rec <-
#'   recipe(Sale_Price ~ Longitude + Latitude, data = ames_train) %>%
#'     step_normalize(all_numeric_predictors())
#'
#'  set.seed(2)
#'  fit <- brulee_linear_reg(ames_rec, data = ames_train,
#'                            epochs = 50, batch_size = 32)
#'
#'  predict(fit, ames_test)
#' }
#' }
#' @export
predict.brulee_linear_reg <- function(object, new_data, type = NULL, epoch = NULL, ...) {
  forged <- hardhat::forge(new_data, object$blueprint)
  type <- check_type(object, type)
  if (is.null(epoch)) {
    epoch <- object$best_epoch
  }
  predict_brulee_linear_reg_bridge(type, object, forged$predictors, epoch = epoch)
}

# ------------------------------------------------------------------------------
# Bridge

predict_brulee_linear_reg_bridge <- function(type, model, predictors, epoch) {

  if (!is.matrix(predictors)) {
    predictors <- as.matrix(predictors)
    if (is.character(predictors)) {
      cli::cli_abort(
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
    cli::cli_warn(msg)
  }

  predictions <- predict_function(model, predictors, epoch)
  hardhat::validate_prediction_size(predictions, predictors)
  predictions
}

get_linear_reg_predict_function <- function(type) {
  predict_brulee_linear_reg_numeric
}

# ------------------------------------------------------------------------------
# Implementation


predict_brulee_linear_reg_raw <- function(model, predictors, epoch) {
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

predict_brulee_linear_reg_numeric <- function(model, predictors, epoch) {
  predictions <- predict_brulee_linear_reg_raw(model, predictors, epoch)
  predictions <- predictions * model$y_stats$sd + model$y_stats$mean
  hardhat::spruce_numeric(predictions[,1])
}
