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
    check_character_matrix(predictors)
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
      "{.arg activation} must be a single value or a vector with the same length as {.arg hidden_units}."
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
#' @param bottleneck_units Vector of BatchNorm output dimensions per layer
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
  bottleneck_units,
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

  # Coerce bottleneck_units to integer if needed
  if (is.numeric(bottleneck_units) & !is.integer(bottleneck_units)) {
    bottleneck_units <- as.integer(bottleneck_units)
  }

  # Validate basic types
  check_integer(hidden_units, single = FALSE, 1, fn = fn)
  check_integer(bottleneck_units, single = FALSE, 2, fn = fn)

  num_layers <- length(hidden_units)

  # Validate lengths match
  if (length(bottleneck_units) != num_layers) {
    cli::cli_abort(
      "The length of {.arg bottleneck_units} ({length(bottleneck_units)}) must match the length of {.arg hidden_units} ({num_layers})."
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
    # Default: place a skip connection at every layer
    residual_at <- seq_len(num_layers)
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
    bottleneck_units = bottleneck_units,
    residual_at = residual_at,
    activation = activation,
    dropout = dropout,
    grad_value_clip = grad_value_clip,
    grad_norm_clip = grad_norm_clip
  )
}

#' Guess the appropriate device for brulee models
#'
#' @param device Device specification ("cpu", "cuda", or "mps"), or NULL
#'   for automatic detection. See [training_efficiency].
#'
#' @return Character string specifying device ("cpu", "cuda", or "mps")
#' @keywords internal
#' @noRd
#'
#' @details
#' Note: MPS (Apple Metal Performance Shaders) is not automatically selected because
#' it doesn't support float64 dtype, which is required by brulee. Users can explicitly
#' specify device="mps" if they modify their code to use float32.
guess_brulee_device <- function(device = NULL) {
  if (!is.null(device)) {
    return(tolower(device))
  }
  if (torch::cuda_is_available()) {
    return("cuda")
  }
  # Skip MPS auto-detection as it doesn't support float64
  # if (torch::backends_mps_is_available()) {
  #   return("mps")
  # }
  "cpu"
}

#' Validate device specification
#'
#' @param device Character string specifying device
#' @param fn Function name for error messages
#'
#' @return Validated device string (lowercase)
#' @keywords internal
#' @noRd
validate_device <- function(device, fn = NULL) {
  device <- tolower(device)
  valid_devices <- c("cpu", "cuda", "mps")
  if (!device %in% valid_devices) {
    cli::cli_abort(
      "{.arg device} must be one of {.val {valid_devices}}, not {.val {device}}.",
      call = fn
    )
  }
  device
}

#' Get a safe device for prediction
#'
#' Checks if the specified device is available. If not, falls back to CPU
#' with a warning.
#'
#' @param device Character string specifying device
#'
#' @return Character string specifying an available device
#' @keywords internal
#' @noRd
get_safe_device <- function(device) {
  if (device == "cpu") {
    return("cpu")
  }

  available <- if (device == "cuda") {
    torch::cuda_is_available()
  } else if (device == "mps") {
    torch::backends_mps_is_available()
  } else {
    FALSE
  }

  if (!available) {
    cli::cli_warn(
      "Model was trained on {.val {device}} but {device} is not available. Using CPU for prediction."
    )
    return("cpu")
  }
  device
}


#' Validate RLN-specific arguments
#'
#' @param hidden_units Number of units in the single hidden layer
#' @param penalty_type Regularization norm ("L1" or "L2")
#' @param penalty_average Target log10-scale mean of regularization coefficients
#' @param step_rate Step size for lambda updates
#' @param activation Activation function name
#' @param fn Function name for error messages
#'
#' @return List of validated/coerced arguments
#' @keywords internal
#' @noRd
validate_rln_args <- function(
  hidden_units,
  penalty_type,
  penalty_average,
  step_rate,
  activation,
  fn = NULL
) {
  if (is.numeric(hidden_units) & !is.integer(hidden_units)) {
    hidden_units <- as.integer(hidden_units)
  }
  check_integer(hidden_units, single = TRUE, 1, fn = fn)

  penalty_type <- toupper(as.character(penalty_type))
  if (!penalty_type %in% c("L1", "L2")) {
    cli::cli_abort(
      '{.arg penalty_type} must be "L1" or "L2", not {.val {penalty_type}}.'
    )
  }

  check_double(
    penalty_average,
    single = TRUE,
    0,
    incl = c(FALSE, TRUE),
    fn = fn
  )

  check_double(step_rate, single = TRUE, 0, incl = c(FALSE, TRUE), fn = fn)

  allowed_activation <- brulee_activations()
  if (!activation %in% allowed_activation) {
    cli::cli_abort(
      "{.arg activation} should be one of: {allowed_activation}, not {.val {activation}}."
    )
  }
  check_character(activation, single = TRUE, fn = fn)

  list(
    hidden_units = hidden_units,
    penalty_type = penalty_type,
    penalty_average = penalty_average,
    step_rate = step_rate,
    activation = activation
  )
}
