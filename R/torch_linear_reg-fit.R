#' Fit a `torch_linear_reg`
#'
#' `torch_linear_reg()` fits a model.
#'
#' @param x Depending on the context:
#'
#'   * A __data frame__ of predictors.
#'   * A __matrix__ of predictors.
#'   * A __recipe__ specifying a set of preprocessing steps
#'     created from [recipes::recipe()].
#'
#' @param y When `x` is a __data frame__ or __matrix__, `y` is the outcome
#' specified as:
#'
#'   * A __data frame__ with 1 numeric column.
#'   * A __matrix__ with 1 numeric column.
#'   * A numeric __vector__.
#'
#' @param data When a __recipe__ or __formula__ is used, `data` is specified as:
#'
#'   * A __data frame__ containing both the predictors and the outcome.
#'
#' @param formula A formula specifying the outcome terms on the left-hand side,
#' and the predictor terms on the right-hand side.
#'
#' @param ... Not currently used, but required for extensibility.
#'
#' @return
#'
#' A `torch_linear_reg` object.
#'
#' @examples
#' predictors <- mtcars[, -1]
#' outcome <- mtcars[, 1]
#'
#' # XY interface
#' mod <- torch_linear_reg(predictors, outcome)
#'
#' # Formula interface
#' mod2 <- torch_linear_reg(mpg ~ ., mtcars)
#'
#' # Recipes interface
#' library(recipes)
#' rec <- recipe(mpg ~ ., mtcars)
#' rec <- step_log(rec, disp)
#' mod3 <- torch_linear_reg(rec, mtcars)
#'
#' @export
torch_linear_reg <- function(x, ...) {
  UseMethod("torch_linear_reg")
}

#' @export
#' @rdname torch_linear_reg
torch_linear_reg.default <- function(x, ...) {
  stop("`torch_linear_reg()` is not defined for a '", class(x)[1], "'.", call. = FALSE)
}

# XY method - data frame

#' @export
#' @rdname torch_linear_reg
torch_linear_reg.data.frame <- function(x, y, ...) {
  processed <- hardhat::mold(x, y)
  torch_linear_reg_bridge(processed, ...)
}

# XY method - matrix

#' @export
#' @rdname torch_linear_reg
torch_linear_reg.matrix <- function(x, y, ...) {
  processed <- hardhat::mold(x, y)
  torch_linear_reg_bridge(processed, ...)
}

# Formula method

#' @export
#' @rdname torch_linear_reg
torch_linear_reg.formula <- function(formula, data, ...) {
  processed <- hardhat::mold(formula, data)
  torch_linear_reg_bridge(processed, ...)
}

# Recipe method

#' @export
#' @rdname torch_linear_reg
torch_linear_reg.recipe <- function(x, data, ...) {
  processed <- hardhat::mold(x, data)
  torch_linear_reg_bridge(processed, ...)
}

# ------------------------------------------------------------------------------
# Bridge

torch_linear_reg_bridge <- function(processed, ...) {
  predictors <- processed$predictors
  outcome <- processed$outcomes[[1]]

  fit <- torch_linear_reg_impl(predictors, outcome)

  new_torch_linear_reg(
    coefs = fit$coefs,
    blueprint = processed$blueprint
  )
}


# ------------------------------------------------------------------------------
# Implementation

torch_linear_reg_impl <- function(predictors, outcome) {
  list(coefs = 1)
}
