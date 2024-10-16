#' Predict from a `brulee_mlp`
#'
#' @param object A `brulee_mlp` object.
#'
#' @param new_data A data frame or matrix of new predictors.
#' @param epoch An integer for the epoch to make predictions. If this value
#' is larger than the maximum number that was fit, a warning is issued and the
#' parameters from the last epoch are used. If left `NULL`, the epoch
#' associated with the smallest loss is used.
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
#' @examples
#' \donttest{
#' if (torch::torch_is_installed() & rlang::is_installed(c("recipes", "modeldata"))) {
#'  # regression example:
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
#'  fit <- brulee_mlp(ames_rec, data = ames_train, epochs = 50, batch_size = 32)
#'
#'  predict(fit, ames_test)
#' }
#' }
#' @export
predict.brulee_mlp <- function(object, new_data, type = NULL, epoch = NULL, ...) {
  forged <- hardhat::forge(new_data, object$blueprint)
  type <- check_type(object, type)
  if (is.null(epoch)) {
    epoch <- object$best_epoch
  }
  predict_brulee_mlp_bridge(type, object, forged$predictors, epoch = epoch)
}

# ------------------------------------------------------------------------------
# Bridge

predict_brulee_mlp_bridge <- function(type, model, predictors, epoch) {

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

  predict_function <- get_mlp_predict_function(type)

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

get_mlp_predict_function <- function(type) {
  switch(
    type,
    numeric = predict_brulee_mlp_numeric,
    prob    = predict_brulee_mlp_prob,
    class   = predict_brulee_mlp_class
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

revive_model <- function(model) {
  con <- rawConnection(model)
  on.exit({close(con)}, add = TRUE)
  module <- torch::torch_load(con)
  module
}

predict_brulee_mlp_raw <- function(model, predictors, epoch) {
  # convert from raw format
  module <- revive_model(model$model_obj)
  # get current model parameters
  estimates <- model$estimates[[epoch]]
  # convert to torch representation
  estimates <- lapply(estimates, torch::torch_tensor)

  # stuff back into the model
  module$load_state_dict(estimates)
  module$eval() # put the model in evaluation mode
  predictions <- module(torch::torch_tensor(predictors))
  predictions <- as.array(predictions)
  # torch doesn't have a NA type so it returns NaN
  predictions[is.nan(predictions)] <- NA
  predictions
}

predict_brulee_mlp_numeric <- function(model, predictors, epoch) {
  predictions <- predict_brulee_mlp_raw(model, predictors, epoch)
  predictions <- predictions * model$y_stats$sd + model$y_stats$mean
  hardhat::spruce_numeric(predictions[,1])
}

predict_brulee_mlp_prob <- function(model, predictors, epoch) {
  predictions <- predict_brulee_mlp_raw(model, predictors, epoch)
  lvs <- get_levels(model)
  hardhat::spruce_prob(pred_levels = lvs, predictions)
}

predict_brulee_mlp_class <- function(model, predictors, epoch) {
  predictions <- predict_brulee_mlp_raw(model, predictors, epoch)
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
      cli::cli_abort(glue::glue("Unknown outcome type '{class(outcome_ptype)}'"))
  }

  type <- rlang::arg_match(type, valid_predict_types())

  if (is.factor(outcome_ptype)) {
    if (!type %in% c("prob", "class"))
      cli::cli_abort(glue::glue("Outcome is factor and the prediction type is '{type}'."))
  } else if (is.numeric(outcome_ptype)) {
    if (type != "numeric")
      cli::cli_abort(glue::glue("Outcome is numeric and the prediction type is '{type}'."))
  }

  type
}
