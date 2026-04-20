#' Validate common arguments across brulee modeling functions
#'
#' @param epochs Number of epochs (will be coerced to integer if numeric)
#' @param batch_size Batch size (can be NULL)
#' @param penalty Penalty parameter
#' @param mixture Mixture parameter (0 = ridge, 1 = lasso)
#' @param validation Validation split proportion
#' @param momentum Momentum for optimizer
#' @param learn_rate Learning rate
#' @param verbose Logical for verbose output
#' @param fn Function name for error messages
#'
#' @return A list with validated/coerced arguments
#' @keywords internal
#' @noRd
validate_common_args <- function(
  epochs,
  batch_size,
  penalty,
  mixture,
  validation,
  momentum,
  learn_rate,
  verbose,
  fn = NULL
) {
  # Coerce epochs to integer if needed
  if (is.numeric(epochs) & !is.integer(epochs)) {
    epochs <- as.integer(epochs)
  }
  check_integer(epochs, single = TRUE, 1, fn = fn)

  # Validate and coerce batch_size if provided
  if (!is.null(batch_size)) {
    if (is.numeric(batch_size) & !is.integer(batch_size)) {
      batch_size <- as.integer(batch_size)
    }
    check_integer(batch_size, single = TRUE, 1, fn = fn)
  }

  # Validate penalty, mixture, validation, momentum, learn_rate, verbose
  check_double(penalty, single = TRUE, 0, incl = c(TRUE, TRUE), fn = fn)
  check_double(mixture, single = TRUE, 0, 1, incl = c(TRUE, TRUE), fn = fn)
  check_double(
    validation,
    single = TRUE,
    0,
    1,
    incl = c(TRUE, FALSE),
    fn = fn
  )
  check_double(momentum, single = TRUE, 0, 1, incl = c(TRUE, TRUE), fn = fn)
  check_double(learn_rate, single = TRUE, 0, incl = c(FALSE, TRUE), fn = fn)
  check_logical(verbose, single = TRUE, fn = fn)

  # Return validated values (some may have been coerced)
  list(
    epochs = epochs,
    batch_size = batch_size,
    penalty = penalty,
    mixture = mixture,
    validation = validation,
    momentum = momentum,
    learn_rate = learn_rate,
    verbose = verbose
  )
}

#' Process predictors to ensure they are a numeric matrix
#'
#' @param predictors Predictor data (data.frame or matrix)
#' @param fn Function name for error messages
#'
#' @return A numeric matrix
#' @keywords internal
#' @noRd
process_predictors <- function(predictors, fn = NULL) {
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

  predictors
}

#' Validate MLP-specific arguments
#'
#' @param hidden_units Vector of hidden units per layer
#' @param activation Vector of activation functions
#' @param dropout Dropout proportion
#' @param grad_value_clip Value for gradient clipping
#' @param grad_norm_clip Norm for gradient clipping
#' @param fn Function name for error messages
#'
#' @return List of validated arguments
#' @keywords internal
#' @noRd
validate_mlp_args <- function(
  hidden_units,
  activation,
  dropout,
  grad_value_clip,
  grad_norm_clip,
  fn = NULL
) {
  # Coerce hidden_units to integer if needed
  if (is.numeric(hidden_units) & !is.integer(hidden_units)) {
    hidden_units <- as.integer(hidden_units)
  }

  # Expand activation if single value
  if (length(hidden_units) > 1 && length(activation) == 1) {
    activation <- rep(activation, length(hidden_units))
  }

  # Check lengths match
  if (length(hidden_units) != length(activation)) {
    cli::cli_abort(
      "'activation' must be a single value or a vector with the same length as 'hidden_units'"
    )
  }

  # Validate activation values
  allowed_activation <- brulee_activations()
  good_activation <- activation %in% allowed_activation
  if (!all(good_activation)) {
    cli::cli_abort(
      "{.arg activation} should be one of: {allowed_activation}, not
    {.val {activation}}."
    )
  }

  # Validate argument types and ranges
  check_integer(hidden_units, single = FALSE, 1, fn = fn)
  check_double(dropout, single = TRUE, 0, 1, incl = c(TRUE, FALSE), fn = fn)
  check_double(
    grad_norm_clip,
    single = TRUE,
    0,
    Inf,
    incl = c(FALSE, TRUE),
    fn = fn
  )
  check_double(
    grad_value_clip,
    single = TRUE,
    0,
    Inf,
    incl = c(FALSE, TRUE),
    fn = fn
  )
  check_character(activation, single = FALSE, fn = fn)

  list(
    hidden_units = hidden_units,
    activation = activation,
    dropout = dropout,
    grad_value_clip = grad_value_clip,
    grad_norm_clip = grad_norm_clip
  )
}
