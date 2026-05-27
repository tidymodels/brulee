#' Predict from a `brulee_multinomial_reg`
#'
#' @inheritParams predict.brulee_mlp
#' @param object A `brulee_multinomial_reg` object.
#' @param type A single character. The type of predictions to generate.
#' Valid options are:
#'
#' - `"class"` for hard class predictions
#' - `"prob"` for soft class predictions (i.e., class probabilities)
#'
#' @return
#'
#' A tibble of predictions. The number of rows in the tibble is guaranteed
#' to be the same as the number of rows in `new_data`.
#'
#' @examplesIf !brulee:::is_cran_check()
#' \donttest{
#' if (torch::torch_is_installed() & rlang::is_installed(c("recipes", "yardstick", "modeldata"))) {
#'
#'   library(recipes)
#'   library(yardstick)
#'
#'   data(penguins, package = "modeldata")
#'
#'   penguins <- penguins |> na.omit()
#'
#'   set.seed(122)
#'   in_train <- sample(1:nrow(penguins), 200)
#'   penguins_train <- penguins[ in_train,]
#'   penguins_test  <- penguins[-in_train,]
#'
#'   rec <- recipe(island ~ ., data = penguins_train) |>
#'     step_dummy(species, sex) |>
#'     step_normalize(all_numeric_predictors())
#'
#'   set.seed(3)
#'   fit <- brulee_multinomial_reg(rec, data = penguins_train, epochs = 5)
#'   fit
#'
#'   predict(fit, penguins_test) |>
#'     bind_cols(penguins_test) |>
#'     conf_mat(island, .pred_class)
#' }
#' }
#' @export
predict.brulee_multinomial_reg <- function(
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
  predict_brulee_multinomial_reg_bridge(
    type,
    object,
    forged$predictors,
    epoch = epoch,
    call = call
  )
}

# ------------------------------------------------------------------------------
# Bridge

predict_brulee_multinomial_reg_bridge <- function(
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

  predict_function <- get_multinomial_reg_predict_function(type)

  max_epoch <- length(model$estimates)
  last_epoch_note(epoch, max_epoch, call = call)

  predictions <- predict_function(model, predictors, epoch)
  hardhat::validate_prediction_size(predictions, predictors)
  predictions
}

get_multinomial_reg_predict_function <- function(type) {
  switch(
    type,
    prob = predict_brulee_multinomial_reg_prob,
    class = predict_brulee_multinomial_reg_class
  )
}

# ------------------------------------------------------------------------------
# Implementation

predict_brulee_multinomial_reg_raw <- function(model, predictors, epoch) {
  # Get safe device (fallback to CPU if trained device unavailable)
  device <- get_safe_device(model$device)

  # convert from raw format
  module <- revive_model(model$model_obj, device)
  # get current model parameters
  estimates <- model$estimates[[epoch]]
  # convert to torch representation on correct device
  estimates <- lapply(estimates, float_64, device = device)
  # stuff back into the model
  module$load_state_dict(estimates)
  # put the model in evaluation mode
  module$eval()
  predictions <- module(float_64(predictors, device))
  predictions <- as.array(predictions)
  # torch doesn't have a NA type so it returns NaN
  predictions[is.nan(predictions)] <- NA
  predictions
}

predict_brulee_multinomial_reg_prob <- function(model, predictors, epoch) {
  predictions <- predict_brulee_multinomial_reg_raw(model, predictors, epoch)
  lvs <- get_levels(model)
  hardhat::spruce_prob(pred_levels = lvs, predictions)
}

predict_brulee_multinomial_reg_class <- function(model, predictors, epoch) {
  predictions <- predict_brulee_multinomial_reg_raw(model, predictors, epoch)
  predictions <- apply(predictions, 1, which.max2) # take the maximum value
  lvs <- get_levels(model)
  hardhat::spruce_class(factor(lvs[predictions], levels = lvs))
}
