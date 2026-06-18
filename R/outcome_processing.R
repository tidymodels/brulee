#' Validate outcome for linear regression
#'
#' @param outcome The outcome variable
#' @param call Caller environment for error reporting
#'
#' @return The validated outcome
#' @keywords internal
#' @noRd
validate_numeric_outcome <- function(outcome, call = rlang::caller_env()) {
  if (!is.numeric(outcome)) {
    cli::cli_abort("{.arg outcome} should be numeric.", call = call)
  }

  outcome
}

#' Validate outcome for binary logistic regression
#'
#' @param outcome The outcome variable (should be a factor)
#' @param call Caller environment for error reporting
#'
#' @return The validated outcome
#' @keywords internal
#' @noRd
validate_binary_outcome <- function(outcome, call = rlang::caller_env()) {
  if (!is.factor(outcome)) {
    cli::cli_abort("{.arg outcome} should be a factor.", call = call)
  }

  if (nlevels(outcome) > 2) {
    cli::cli_abort(
      "Logistic regression is for outcomes with two classes.",
      call = call
    )
  }

  if (nlevels(outcome) < 2) {
    cli::cli_abort(
      "Logistic regression requires outcomes with two classes.",
      call = call
    )
  }

  outcome
}

#' Validate outcome for multinomial regression
#'
#' @param outcome The outcome variable (should be a factor)
#' @param call Caller environment for error reporting
#'
#' @return The validated outcome
#' @keywords internal
#' @noRd
validate_multiclass_outcome <- function(outcome, call = rlang::caller_env()) {
  if (!is.factor(outcome)) {
    cli::cli_abort("{.arg outcome} should be a factor.", call = call)
  }

  if (nlevels(outcome) < 3) {
    cli::cli_abort(
      "Multinomial regression is for outcomes with 3+ classes.",
      call = call
    )
  }

  outcome
}

#' Validate outcome for MLP (can be numeric or factor)
#'
#' @param outcome The outcome variable
#' @param call Caller environment for error reporting
#'
#' @return The validated outcome
#' @keywords internal
#' @noRd
validate_mlp_outcome <- function(outcome, call = rlang::caller_env()) {
  if (!is.numeric(outcome) && !is.factor(outcome)) {
    cli::cli_abort("{.arg outcome} should be numeric or a factor.", call = call)
  }
  outcome
}
