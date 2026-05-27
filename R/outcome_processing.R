#' Validate outcome for linear regression
#'
#' @param outcome The outcome variable
#' @param fn Function name for error messages
#'
#' @return The validated outcome
#' @keywords internal
#' @noRd
validate_numeric_outcome <- function(outcome, fn = NULL) {
  if (!is.numeric(outcome)) {
    cli::cli_abort(
      paste0(format_msg(fn, "outcome"), " to be numeric.")
    )
  }
  outcome
}

#' Validate outcome for binary logistic regression
#'
#' @param outcome The outcome variable (should be a factor)
#' @param fn Function name for error messages
#'
#' @return The validated outcome
#' @keywords internal
#' @noRd
validate_binary_outcome <- function(outcome, fn = NULL) {
  if (!is.factor(outcome)) {
    cli::cli_abort(
      paste0(format_msg(fn, "outcome"), " to be a factor.")
    )
  }

  if (length(levels(outcome)) > 2) {
    cli::cli_abort("Logistic regression is for outcomes with two classes.")
  }

  if (length(levels(outcome)) < 2) {
    cli::cli_abort("Logistic regression requires outcomes with two classes.")
  }

  outcome
}

#' Validate outcome for multinomial regression
#'
#' @param outcome The outcome variable (should be a factor)
#' @param fn Function name for error messages
#'
#' @return The validated outcome
#' @keywords internal
#' @noRd
validate_multiclass_outcome <- function(outcome, fn = NULL) {
  if (!is.factor(outcome)) {
    cli::cli_abort(
      paste0(format_msg(fn, "outcome"), " to be a factor.")
    )
  }

  if (length(levels(outcome)) < 3) {
    cli::cli_abort("Multinomial regression is for outcomes with 3+ classes.")
  }

  outcome
}

#' Validate outcome for MLP (can be numeric or factor)
#'
#' @param outcome The outcome variable
#' @param fn Function name for error messages
#'
#' @return The validated outcome
#' @keywords internal
#' @noRd
validate_mlp_outcome <- function(outcome, fn = NULL) {
  if (!is.numeric(outcome) && !is.factor(outcome)) {
    cli::cli_abort(
      paste0(format_msg(fn, "outcome"), " to be numeric or a factor.")
    )
  }
  outcome
}
