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

#' Validate ResNet-specific arguments
#'
#' @param hidden_units Vector of hidden units per layer
#' @param block_units Vector of block units per layer
#' @param residual_at Vector of layer indices where residual connections occur
#' @param activation Vector of activation functions
#' @param dropout Dropout proportion
#' @param grad_value_clip Value for gradient clipping
#' @param grad_norm_clip Norm for gradient clipping
#' @param fn Function name for error messages
#'
#' @return List of validated arguments
#' @keywords internal
#' @noRd
validate_resnet_args <- function(
  hidden_units,
  block_units,
  residual_at = NULL,
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

  # Coerce block_units to integer if needed
  if (is.numeric(block_units) & !is.integer(block_units)) {
    block_units <- as.integer(block_units)
  }

  # Validate basic types
  check_integer(hidden_units, single = FALSE, 1, fn = fn)
  check_integer(block_units, single = FALSE, 2, fn = fn)

  num_layers <- length(hidden_units)

  # Validate lengths match
  if (length(block_units) != num_layers) {
    cli::cli_abort(
      "The length of {.arg block_units} ({length(block_units)}) must match the length of {.arg hidden_units} ({num_layers})."
    )
  }

  # Expand activation if single value
  if (length(activation) == 1 && num_layers > 1) {
    activation <- rep(activation, num_layers)
  }

  if (length(activation) != num_layers) {
    cli::cli_abort(
      "The length of {.arg activation} ({length(activation)}) must match the length of {.arg hidden_units} ({num_layers})."
    )
  }

  # Validate activation values
  allowed_activation <- brulee_activations()
  good_activation <- activation %in% allowed_activation
  if (!all(good_activation)) {
    cli::cli_abort(
      "{.arg activation} should be one of: {allowed_activation}, not {.val {activation}}."
    )
  }

  # Validate residual_at
  if (!is.null(residual_at)) {
    # Coerce to integer
    if (is.numeric(residual_at) && !is.integer(residual_at)) {
      residual_at <- as.integer(residual_at)
    }

    # Check type and range
    check_integer(residual_at, single = FALSE, 1, fn = fn)

    if (any(residual_at > num_layers)) {
      cli::cli_abort(
        "All values in {.arg residual_at} must be between 1 and {num_layers} (the number of layers)."
      )
    }

    # Check for duplicates
    if (any(duplicated(residual_at))) {
      cli::cli_warn(
        "{.arg residual_at} contains duplicate values. Removing duplicates."
      )
      residual_at <- unique(residual_at)
    }

    # Sort for consistency
    if (is.unsorted(residual_at)) {
      residual_at <- sort(residual_at)
    }
  } else {
    # Default: make entire network one residual block
    residual_at <- num_layers
  }

  # Validate dropout and gradient clipping
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
    block_units = block_units,
    residual_at = residual_at,
    activation = activation,
    dropout = dropout,
    grad_value_clip = grad_value_clip,
    grad_norm_clip = grad_norm_clip
  )
}
