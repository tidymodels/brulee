#' Predict from a `brulee_logistic_reg`
#'
#' @inheritParams predict.brulee_mlp
#' @param object A `brulee_logistic_reg` object.
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
#' @examples
#' \donttest{
#' if (torch::torch_is_installed() & rlang::is_installed(c("recipes", "yardstick", "modeldata"))) {
#'
#'   library(recipes)
#'   library(yardstick)
#'
#'   data(penguins, package = "modeldata")
#'
#'   penguins <- penguins %>% na.omit()
#'
#'   set.seed(122)
#'   in_train <- sample(1:nrow(penguins), 200)
#'   penguins_train <- penguins[ in_train,]
#'   penguins_test  <- penguins[-in_train,]
#'
#'   rec <- recipe(sex ~ ., data = penguins_train) %>%
#'     step_dummy(all_nominal_predictors()) %>%
#'     step_normalize(all_numeric_predictors())
#'
#'   set.seed(3)
#'   fit <- brulee_logistic_reg(rec, data = penguins_train, epochs = 5)
#'   fit
#'
#'   predict(fit, penguins_test)
#'
#'   predict(fit, penguins_test, type = "prob") %>%
#'     bind_cols(penguins_test) %>%
#'     roc_curve(sex, .pred_female) %>%
#'     autoplot()
#'
#' }
#' }
#' @export
predict.brulee_logistic_reg <- function(object, new_data, type = NULL, epoch = NULL, ...) {
  forged <- hardhat::forge(new_data, object$blueprint)
  type <- check_type(object, type)
  if (is.null(epoch)) {
    epoch <- object$best_epoch
  }
  predict_brulee_logistic_reg_bridge(type, object, forged$predictors, epoch = epoch)
}

# ------------------------------------------------------------------------------
# Bridge

predict_brulee_logistic_reg_bridge <- function(type, model, predictors, epoch) {

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

  predict_function <- get_logistic_reg_predict_function(type)

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

get_logistic_reg_predict_function <- function(type) {
  switch(
    type,
    prob    = predict_brulee_logistic_reg_prob,
    class   = predict_brulee_logistic_reg_class
  )
}

# ------------------------------------------------------------------------------
# Implementation

predict_brulee_logistic_reg_raw <- function(model, predictors, epoch) {
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

predict_brulee_logistic_reg_prob <- function(model, predictors, epoch) {
  predictions <- predict_brulee_logistic_reg_raw(model, predictors, epoch)
  lvs <- get_levels(model)
  hardhat::spruce_prob(pred_levels = lvs, predictions)
}

predict_brulee_logistic_reg_class <- function(model, predictors, epoch) {
  predictions <- predict_brulee_logistic_reg_raw(model, predictors, epoch)
  predictions <- apply(predictions, 1, which.max2) # take the maximum value
  lvs <- get_levels(model)
  hardhat::spruce_class(factor(lvs[predictions], levels = lvs))
}
