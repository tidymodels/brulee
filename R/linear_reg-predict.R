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
#' @examplesIf !brulee:::is_cran_check()
#' \donttest{
#' if (torch::torch_is_installed() && rlang::is_installed("recipes")) {
#'
#'  data(ames, package = "modeldata")
#'
#'  ames$Sale_Price <- log10(ames$Sale_Price)
#'
#'  set.seed(1)
#'  in_train <- sample(seq_len(nrow(ames)), 2000)
#'  ames_train <- ames[ in_train,]
#'  ames_test  <- ames[-in_train,]
#'
#'  # Using recipe
#'  library(recipes)
#'
#'  ames_rec <-
#'   recipe(Sale_Price ~ Longitude + Latitude, data = ames_train) |>
#'     step_normalize(all_numeric_predictors())
#'
#'  set.seed(2)
#'  fit <- brulee_linear_reg(ames_rec, data = ames_train, epochs = 50)
#'
#'  predict(fit, ames_test)
#' }
#' }
#' @export
predict.brulee_linear_reg <- function(
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
  predict_brulee_linear_reg_bridge(
    type,
    object,
    forged$predictors,
    epoch = epoch,
    call = call
  )
}

# ------------------------------------------------------------------------------
# Bridge

predict_brulee_linear_reg_bridge <- function(
  type,
  model,
  predictors,
  epoch,
  call = rlang::caller_env()
) {
  if (!is.matrix(predictors)) {
    predictors <- as.matrix(predictors)
    check_character_matrix(predictors, call = call)
  }

  predict_function <- get_linear_reg_predict_function(type)

  max_epoch <- length(model$estimates)
  last_epoch_note(epoch, max_epoch, call = call)

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
  # Get safe device (fallback to CPU if trained device unavailable)
  device <- get_safe_device(model$device)

  # convert from raw format
  module <- revive_model(model$model_obj, device)
  # get current model parameters
  estimates <- model$estimates[[epoch + 1]]
  # convert to torch representation on correct device
  estimates <- purrr::map(estimates, float_32, device = device)
  # stuff back into the model
  module$load_state_dict(estimates)
  # put the model in evaluation mode
  module$eval()
  predictions <- module(float_32(predictors, device))
  predictions <- as.array(predictions)
  # torch doesn't have a NA type so it returns NaN
  predictions[is.nan(predictions)] <- NA
  predictions
}

predict_brulee_linear_reg_numeric <- function(model, predictors, epoch) {
  predictions <- predict_brulee_linear_reg_raw(model, predictors, epoch)
  predictions <- predictions * model$y_stats$sd + model$y_stats$mean
  hardhat::spruce_numeric(predictions[, 1])
}
