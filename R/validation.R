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
#' @param call Caller environment for error reporting
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
  call = rlang::caller_env()
) {
  # Coerce epochs to integer if needed
  if (is.numeric(epochs) & !is.integer(epochs)) {
    epochs <- as.integer(epochs)
  }

  check_integer(epochs, single = TRUE, 1, call = call)

  # Validate and coerce batch_size if provided
  if (!is.null(batch_size)) {
    if (is.numeric(batch_size) & !is.integer(batch_size)) {
      batch_size <- as.integer(batch_size)
    }
    check_integer(batch_size, single = TRUE, 1, call = call)
  }

  # Validate penalty, mixture, validation, momentum, learn_rate, verbose
  check_double(penalty, single = TRUE, 0, incl = c(TRUE, TRUE), call = call)
  check_double(mixture, single = TRUE, 0, 1, incl = c(TRUE, TRUE), call = call)
  check_double(
    validation,
    single = TRUE,
    0,
    1,
    incl = c(TRUE, FALSE),
    call = call
  )
  check_double(momentum, single = TRUE, 0, 1, incl = c(TRUE, TRUE), call = call)
  check_double(learn_rate, single = TRUE, 0, incl = c(FALSE, TRUE), call = call)
  check_bool(verbose, call = call)

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
#' @param call Caller environment for error reporting
#'
#' @return A numeric matrix
#' @keywords internal
#' @noRd
process_predictors <- function(predictors, call = rlang::caller_env()) {
  if (!is.matrix(predictors)) {
    predictors <- as.matrix(predictors)
    check_character_matrix(predictors, call = call)
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
#' @param call Caller environment for error reporting
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
  call = rlang::caller_env()
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
      "{.arg activation} must be a single value or a vector with the same length as {.arg hidden_units}.",
      call = call
    )
  }

  # Validate activation values
  allowed_activation <- brulee_activations()
  good_activation <- activation %in% allowed_activation
  if (!all(good_activation)) {
    cli::cli_abort(
      "{.arg activation} should be one of: {allowed_activation}, not
    {.val {activation}}.",
      call = call
    )
  }

  # Validate argument types and ranges
  check_integer(hidden_units, single = FALSE, 1, call = call)
  check_double(dropout, single = TRUE, 0, 1, incl = c(TRUE, FALSE), call = call)
  check_double(
    grad_norm_clip,
    single = TRUE,
    0,
    Inf,
    incl = c(FALSE, TRUE),
    call = call
  )
  check_double(
    grad_value_clip,
    single = TRUE,
    0,
    Inf,
    incl = c(FALSE, TRUE),
    call = call
  )
  check_character(activation, call = call)

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
#' @param call Caller environment for error reporting
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
  call = rlang::caller_env()
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
  check_integer(hidden_units, single = FALSE, 1, call = call)
  check_integer(bottleneck_units, single = FALSE, 2, call = call)

  num_layers <- length(hidden_units)

  # Validate lengths match
  if (length(bottleneck_units) != num_layers) {
    cli::cli_abort(
      "The length of {.arg bottleneck_units} ({length(bottleneck_units)}) must match the length of {.arg hidden_units} ({num_layers}).",
      call = call
    )
  }

  # Expand activation if single value
  if (length(activation) == 1 && num_layers > 1) {
    activation <- rep(activation, num_layers)
  }

  if (length(activation) != num_layers) {
    cli::cli_abort(
      "The length of {.arg activation} ({length(activation)}) must match the length of {.arg hidden_units} ({num_layers}).",
      call = call
    )
  }

  # Validate activation values
  allowed_activation <- brulee_activations()
  good_activation <- activation %in% allowed_activation
  if (!all(good_activation)) {
    cli::cli_abort(
      "{.arg activation} should be one of: {allowed_activation}, not {.val {activation}}.",
      call = call
    )
  }

  # Validate residual_at
  if (!is.null(residual_at)) {
    # Coerce to integer
    if (is.numeric(residual_at) && !is.integer(residual_at)) {
      residual_at <- as.integer(residual_at)
    }

    # Check type and range
    check_integer(residual_at, single = FALSE, 1, call = call)

    if (any(residual_at > num_layers)) {
      cli::cli_abort(
        "All values in {.arg residual_at} must be between 1 and {num_layers} (the number of layers).",
        call = call
      )
    }

    # Check for duplicates
    if (any(duplicated(residual_at))) {
      cli::cli_warn(
        "{.arg residual_at} contains duplicate values. Removing duplicates.",
        call = call
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
  check_double(dropout, single = TRUE, 0, 1, incl = c(TRUE, FALSE), call = call)
  check_double(
    grad_norm_clip,
    single = TRUE,
    0,
    Inf,
    incl = c(FALSE, TRUE),
    call = call
  )
  check_double(
    grad_value_clip,
    single = TRUE,
    0,
    Inf,
    incl = c(FALSE, TRUE),
    call = call
  )
  check_character(activation, call = call)

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
#' Auto-detection is platform-driven: on macOS, MPS (Apple Metal Performance
#' Shaders) is preferred when available; on other platforms, CUDA is preferred
#' when available. Falls back to CPU otherwise. Users on an Intel Mac with a
#' CUDA-capable eGPU must pass `device = "cuda"` explicitly.
guess_brulee_device <- function(device = NULL) {
  if (!is.null(device)) {
    return(tolower(device))
  }
  is_mac <- Sys.info()[["sysname"]] == "Darwin"
  if (is_mac && torch::backends_mps_is_available()) {
    return("mps")
  }
  if (!is_mac && torch::cuda_is_available()) {
    return("cuda")
  }
  "cpu"
}

#' Validate device specification
#'
#' @param device Character string specifying device
#' @param call Caller environment for error reporting
#'
#' @return Validated device string (lowercase)
#' @keywords internal
#' @noRd
validate_device <- function(device, call = rlang::caller_env()) {
  device <- tolower(device)
  valid_devices <- c("cpu", "cuda", "mps")
  if (!device %in% valid_devices) {
    cli::cli_abort(
      "{.arg device} must be one of {.val {valid_devices}}, not {.val {device}}.",
      call = call
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

  if (device == "cuda") {
    available <- torch::cuda_is_available()
  } else if (device == "mps") {
    available <- torch::backends_mps_is_available()
  } else {
    available <- FALSE
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
#' @param call Caller environment for error reporting
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
  call = rlang::caller_env()
) {
  if (is.numeric(hidden_units) & !is.integer(hidden_units)) {
    hidden_units <- as.integer(hidden_units)
  }
  check_integer(hidden_units, single = TRUE, 1, call = call)

  penalty_type <- toupper(as.character(penalty_type))
  if (!penalty_type %in% c("L1", "L2")) {
    cli::cli_abort(
      '{.arg penalty_type} must be "L1" or "L2", not {.val {penalty_type}}.',
      call = call
    )
  }

  check_double(
    penalty_average,
    single = TRUE,
    0,
    incl = c(FALSE, TRUE),
    call = call
  )

  check_double(step_rate, single = TRUE, 0, incl = c(FALSE, TRUE), call = call)

  allowed_activation <- brulee_activations()
  if (!activation %in% allowed_activation) {
    cli::cli_abort(
      "{.arg activation} should be one of: {allowed_activation}, not {.val {activation}}.",
      call = call
    )
  }
  check_string(activation, call = call)

  list(
    hidden_units = hidden_units,
    penalty_type = penalty_type,
    penalty_average = penalty_average,
    step_rate = step_rate,
    activation = activation
  )
}
