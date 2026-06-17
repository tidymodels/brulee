# Predict from a fitted TabICL model. The pretrained checkpoint is reloaded from
# `object$path`, the stored training rows provide the in-context examples, and
# the ensemble engine produces class probabilities / labels (classification) or
# numeric predictions (regression).

#' Predict from a `brulee_tab_icl`
#'
#' @param object A `brulee_tab_icl` object from [brulee_tab_icl()].
#' @param new_data A data frame or matrix of new predictors.
#' @param type A single character string for the type of prediction. Valid
#'   options are:
#'
#'   - `"class"` for hard class predictions (classification).
#'   - `"prob"` for class probabilities (classification).
#'   - `"numeric"` for numeric predictions (regression).
#'
#'   If `NULL` (the default), the natural type for the outcome is used:
#'   `"class"` for a factor outcome and `"numeric"` for a numeric one.
#' @param ... Not used, but required for extensibility.
#'
#' @details
#'
#' Because TabICL is an in-context learner, prediction reloads the pretrained
#' weights from the checkpoint directory stored on `object` and conditions on the
#' training rows captured at fit time. The same preprocessing and ensembling used
#' for `object` are applied to `new_data`; see [brulee_tab_icl()] for details.
#' For classification, `"prob"` returns one column per class (named
#' `.pred_<level>`) and `"class"` returns the highest-probability class.
#'
#' @return
#'
#' A tibble of predictions. The number of rows is guaranteed to match
#' `new_data`. For `type = "prob"` there is one column per outcome class; for
#' `"class"` and `"numeric"` there is a single prediction column.
#'
#' @seealso [brulee_tab_icl()]
#'
#' @examples
#' \dontrun{
#' if (torch::torch_is_installed() & rlang::is_installed("modeldata")) {
#'   data(penguins, package = "modeldata")
#'   penguins <- na.omit(penguins)
#'
#'   fit <- brulee_tab_icl(
#'     species ~ .,
#'     data = penguins,
#'     path = "path/to/tabicl-classifier"
#'   )
#'   predict(fit, penguins)
#'   predict(fit, penguins, type = "prob")
#' }
#' }
#' @rdname predict.brulee_tab_icl
#' @export
predict.brulee_tab_icl <- function(object, new_data, type = NULL, ...) {
  call <- rlang::current_env()
  forged <- hardhat::forge(new_data, object$blueprint)
  type <- check_type(object, type, call = call)

  loaded <- tabicl_load_model(object$path, device = object$device)
  x_test <- tabicl_encode_transform(object$encoders, forged$predictors)

  if (object$task == "classification") {
    n_classes <- length(object$levels)
    members <- tabicl_make_members(
      object$n_estimators,
      ncol(object$train_x),
      n_classes,
      object$norm_methods,
      classification = TRUE
    )
    proba <- tabicl_classifier_proba(
      loaded,
      object$train_x,
      object$train_y,
      x_test,
      members,
      temperature = object$softmax_temperature,
      device = object$device
    )
    predictions <- switch(
      type,
      prob = hardhat::spruce_prob(object$levels, proba),
      class = hardhat::spruce_class(factor(
        object$levels[apply(proba, 1, which.max2)],
        levels = object$levels
      ))
    )
  } else {
    members <- tabicl_make_members(
      object$n_estimators,
      ncol(object$train_x),
      n_classes = 0L,
      object$norm_methods,
      classification = FALSE
    )
    est <- tabicl_regressor_mean(
      loaded,
      object$train_x,
      object$train_y,
      x_test,
      members,
      device = object$device
    )
    predictions <- hardhat::spruce_numeric(est)
  }

  hardhat::validate_prediction_size(predictions, new_data)
  predictions
}
