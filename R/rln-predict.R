#' Predict from a `brulee_rln`
#'
#' @param object A `brulee_rln` object.
#' @param new_data A data frame or matrix of new predictors.
#' @param epoch An integer for the epoch to make predictions. If larger than
#'   the number of epochs fit, a warning is issued and the last epoch is used.
#'   If `NULL` (default), the epoch with the smallest loss is used.
#' @param type A single character. The only valid option is `"numeric"` for
#'   numeric predictions.
#' @param ... Not used, but required for extensibility.
#'
#' @return
#'
#' A tibble of predictions with the same number of rows as `new_data`.
#'
#' @examplesIf !brulee:::is_cran_check()
#' \donttest{
#' if (torch::torch_is_installed() && rlang::is_installed(c("recipes", "modeldata"))) {
#'
#'  data(ames, package = "modeldata")
#'  ames$Sale_Price <- log10(ames$Sale_Price)
#'
#'  set.seed(1)
#'  in_train <- sample(seq_len(nrow(ames)), 2000)
#'  ames_train <- ames[ in_train,]
#'  ames_test  <- ames[-in_train,]
#'
#'  library(recipes)
#'
#'  ames_rec <-
#'   recipe(Sale_Price ~ Longitude + Latitude, data = ames_train) |>
#'     step_normalize(all_numeric_predictors())
#'
#'  set.seed(2)
#'  fit <- brulee_rln(ames_rec, data = ames_train, hidden_units = 20L, epochs = 30L)
#'
#'  predict(fit, ames_test)
#' }
#' }
#' @export
predict.brulee_rln <- function(
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
  predict_brulee_rln_bridge(
    type,
    object,
    forged$predictors,
    epoch = epoch,
    call = call
  )
}

# ------------------------------------------------------------------------------
# Bridge

predict_brulee_rln_bridge <- function(
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

  max_epoch <- length(model$estimates)
  if (epoch > max_epoch) {
    last_epoch_note(epoch, max_epoch, call = call)
    epoch <- max_epoch
  }

  predictions <- predict_brulee_rln_numeric(model, predictors, epoch)
  hardhat::validate_prediction_size(predictions, predictors)
  predictions
}

# ------------------------------------------------------------------------------
# Implementation

predict_brulee_rln_raw <- function(model, predictors, epoch) {
  device <- get_safe_device(model$device)
  module <- revive_model(model$model_obj, device)
  estimates <- model$estimates[[epoch + 1]]
  estimates <- lapply(estimates, float_32, device = device)
  module$load_state_dict(estimates)
  module$eval()
  predictions <- module(float_32(predictors, device))
  predictions <- as.array(predictions)
  predictions[is.nan(predictions)] <- NA
  predictions
}

predict_brulee_rln_numeric <- function(model, predictors, epoch) {
  predictions <- predict_brulee_rln_raw(model, predictors, epoch)
  predictions <- predictions * model$y_stats$sd + model$y_stats$mean
  hardhat::spruce_numeric(predictions[, 1])
}
