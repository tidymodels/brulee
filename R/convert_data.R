#' Convert data to torch format
#'
#' For an x/y interface, `matrix_to_dataset()` converts the data to proper
#'  encodings then formats the results for consumption by `torch`.
#'
#' @param x A numeric matrix of predictors.
#' @param y A vector. If regression than `y` is numeric. For classification, it
#'  is a factor.
#' @return An R6 index sampler object with classes "training_set",
#'  "dataset", and "R6".
#' @details Missing values should be removed before passing data to this function.
#' @examples
#' if (torch::torch_is_installed()) {
#'   matrix_to_dataset(as.matrix(mtcars[, -1]), mtcars$mpg)
#' }
#' @export
matrix_to_dataset <- function(x, y) {
  x <- torch::torch_tensor(x)
  if (is.factor(y)) {
    y <- as.numeric(y)
    y <- torch::torch_tensor(y, dtype = torch_long())
  } else {
    y <- torch::torch_tensor(y)
  }
  torch::tensor_dataset(x = x, y = y)
}

# ------------------------------------------------------------------------------

scale_stats <- function(x) {
  res <- list(mean = mean(x, na.rm = TRUE), sd = stats::sd(x, na.rm = TRUE))
  if (res$sd == 0) {
    cli::cli_abort("There is no variation in `y`.")
  }
  res
}

scale_y <- function(y, stats) {
  (y - stats$mean)/stats$sd
}
