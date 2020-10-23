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
#' matrix_to_dataset(as.matrix(mtcars[, -1]), mtcars$mpg)
#' @export
matrix_to_dataset <- torch::dataset(
 name = "torch_training_set",

 initialize = function(x, y) {
  self$data <- self$prepare_training_set(x, y)
 },

 .getitem = function(index) {
  x <- self$data$x[index,]
  y <- self$data$y[index]
  list(x = x, y = y)
 },

 .length = function() {
  self$data$x$size()[[1]]
 },

 prepare_training_set = function(x, y) {
  if (is.factor(y)) {
   y <- as.numeric(y)
   y <- torch::torch_tensor(y)$to(torch_long())
  } else {
    y <- torch::torch_tensor(y)
  }

  list(x = torch::torch_tensor(x), y = y)
 }
)
