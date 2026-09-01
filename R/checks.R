# Additional type checkers designed for testing argument values.

check_number_whole_vec <- function(
  x,
  call = rlang::caller_env(),
  arg = rlang::caller_arg(x),
  ...
) {
  arg = rlang::caller_arg(x)
  for (i in x) {
    check_number_whole(i, arg = arg, call = call, ...)
  }
  x <- as.integer(x)
  invisible(x)
}

check_number_decimal_vec <- function(
  x,
  arg = rlang::caller_arg(x),
  allow_na = FALSE,
  call = rlang::caller_env(),
  ...
) {
  if (!is.double(x)) {
    cli::cli_abort("{.arg {arg}} should be a double vector.", call = call)
  }

  if (!allow_na && anyNA(x)) {
    cli::cli_abort(
      "{.arg {arg}} should not contain missing values.",
      call = call
    )
  }

  invisible(x)
}

# ------------------------------------------------------------------------------

check_missing_data <- function(x, y, fn = "some function", verbose = FALSE) {
  compl_data <- complete.cases(x, y)
  if (!all(compl_data)) {
    x <- x[compl_data, , drop = FALSE]
    y <- y[compl_data]
    if (verbose) {
      cli::cli_warn(
        "{.fn {fn}} removed {sum(!compl_data)} rows of data due to missing values."
      )
    }
  }
  list(x = x, y = y)
}

check_data_att <- function(x, y, call = rlang::caller_env()) {
  hardhat::validate_outcomes_are_univariate(y)

  # check matrices/vectors, matrix type, matrix column names
  if (!is.matrix(x) || !is.numeric(x)) {
    cli::cli_abort("{.arg x} should be a numeric matrix.", call = call)
  }
  nms <- colnames(x)
  if (length(nms) != ncol(x)) {
    cli::cli_abort("Every column of {.arg x} should have a name.", call = call)
  }
  if (!is.vector(y) && !is.factor(y)) {
    cli::cli_abort("{.arg y} should be a vector.", call = call)
  }
  invisible(NULL)
}

# ------------------------------------------------------------------------------
# Scalar type checkers (thin wrappers around rlang standalone)

check_integer <- function(
  x,
  single = TRUE,
  x_min = -Inf,
  x_max = Inf,
  incl = c(TRUE, TRUE),
  arg = rlang::caller_arg(x),
  call = rlang::caller_env()
) {
  if (single) {
    check_number_whole(x, min = NULL, max = NULL, arg = arg, call = call)
  } else {
    if (!is.numeric(x) || !all(x == trunc(x), na.rm = TRUE)) {
      cli::cli_abort("{.arg {arg}} must be integer-valued.", call = call)
    }
  }

  check_range(
    x,
    x_min = x_min,
    x_max = x_max,
    incl = incl,
    arg = arg,
    call = call
  )
  invisible(x)
}

check_double <- function(
  x,
  single = TRUE,
  x_min = -Inf,
  x_max = Inf,
  incl = c(TRUE, TRUE),
  arg = rlang::caller_arg(x),
  call = rlang::caller_env()
) {
  if (single) {
    check_number_decimal(x, min = NULL, max = NULL, arg = arg, call = call)
  } else {
    if (!is.numeric(x)) {
      cli::cli_abort("{.arg {arg}} must be numeric.", call = call)
    }
  }

  check_range(
    x,
    x_min = x_min,
    x_max = x_max,
    incl = incl,
    arg = arg,
    call = call
  )
  invisible(x)
}

check_range <- function(
  x,
  x_min = -Inf,
  x_max = Inf,

  incl = c(TRUE, TRUE),
  arg = rlang::caller_arg(x),
  call = rlang::caller_env()
) {
  if (incl[[1]]) {
    low_fail <- any(x < x_min)
  } else {
    low_fail <- any(x <= x_min)
  }

  if (incl[[2]]) {
    high_fail <- any(x > x_max)
  } else {
    high_fail <- any(x >= x_max)
  }

  if (low_fail || high_fail) {
    if (incl[[1]]) {
      lbr <- "["
    } else {
      lbr <- "("
    }
    if (incl[[2]]) {
      rbr <- "]"
    } else {
      rbr <- ")"
    }
    cli::cli_abort(
      "{.arg {arg}} must be in the range {lbr}{x_min}, {x_max}{rbr}.",
      call = call
    )
  }
  invisible(x)
}

# ------------------------------------------------------------------------------
# Domain-specific validators

optimizer_values <- c("SGD", "ADAMw", "Adadelta", "Adagrad", "RMSprop", "LBFGS")

check_optimizer <- function(
  x,
  arg = rlang::caller_arg(x),
  call = rlang::caller_env()
) {
  check_string(x, arg = arg, call = call)
  if (!x %in% optimizer_values) {
    cli::cli_abort(
      "{.arg {arg}} must be one of {.val {optimizer_values}}, not {.val {x}}.",
      call = call
    )
  }
  invisible(x)
}

classification_loss_values <- c("nll", "focal")

check_classification_loss <- function(
  x,
  arg = rlang::caller_arg(x),
  call = rlang::caller_env()
) {
  check_string(x, arg = arg, call = call)
  if (!x %in% classification_loss_values) {
    cli::cli_abort(
      "{.arg {arg}} must be one of {.val {classification_loss_values}}, not {.val {x}}.",
      call = call
    )
  }
  invisible(x)
}

# ------------------------------------------------------------------------------

check_class_weights <- function(wts, lvls, xtab, call = rlang::caller_env()) {
  if (length(lvls) == 0) {
    return(NULL)
  }

  if (is.null(wts)) {
    wts <- rep(1, length(lvls))
    names(wts) <- lvls
    return(wts)
  }
  if (!is.numeric(wts)) {
    cli::cli_abort(
      "{.arg class_weights} must be a numeric vector.",
      call = call
    )
  }

  if (length(wts) == 1) {
    val <- wts
    wts <- rep(1, length(lvls))
    minority <- names(xtab)[which.min(xtab)]
    wts[lvls == minority] <- val
    names(wts) <- lvls
  }

  if (length(lvls) != length(wts)) {
    cli::cli_abort(
      "There were {length(wts)} class weights given but {length(lvls)} were expected.",
      call = call
    )
  }

  nms <- names(wts)
  if (is.null(nms)) {
    names(wts) <- lvls
  } else {
    if (!identical(sort(nms), sort(lvls))) {
      cli::cli_abort(
        "Names for {.arg class_weights} should be: {.val {lvls}}.",
        call = call
      )
    }
    wts <- wts[lvls]
  }

  wts
}

# Quantile levels must be strictly inside (0, 1). `hardhat::quantile_pred()`
# permits 0 and 1 but the TabICL tail model evaluates `log(clamp(alpha, tol))`
# there, which returns a finite but meaningless extreme rather than an infinity.
# Sortedness and uniqueness are left to `hardhat::quantile_pred()`.
check_quantile_levels <- function(
  x,
  arg = rlang::caller_arg(x),
  call = rlang::caller_env()
) {
  if (!is.numeric(x) || length(x) < 1L) {
    cli::cli_abort(
      "{.arg {arg}} must be a non-empty numeric vector.",
      call = call
    )
  }
  if (anyNA(x)) {
    cli::cli_abort(
      "{.arg {arg}} cannot contain missing values.",
      call = call
    )
  }
  if (any(!is.finite(x) | x <= 0 | x >= 1)) {
    cli::cli_abort(
      "{.arg {arg}} must be in the open interval (0, 1).",
      call = call
    )
  }
  # Uniqueness and sortedness are re-checked by `hardhat::quantile_pred()`, but
  # only after the ensemble has run and reported against its own call. Checking
  # here fails fast and attributes the error to the caller.
  if (anyDuplicated(x)) {
    cli::cli_abort("{.arg {arg}} must be unique.", call = call)
  }
  if (is.unsorted(x)) {
    cli::cli_abort(
      "{.arg {arg}} must be sorted in increasing order.",
      call = call
    )
  }
  invisible(x)
}

check_character_matrix <- function(x, call = rlang::caller_env()) {
  if (is.character(x)) {
    cli::cli_abort(
      "There were some non-numeric columns in the predictors.
        Please use a formula or recipe to encode all of the predictors as numeric.",
      call = call
    )
  }
  invisible(NULL)
}
