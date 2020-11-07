## -----------------------------------------------------------------------------

#' Plot model loss over epochs
#'
#' @param object A `torch_mlp` object.
#' @param ... Not currently used
#' @return A `ggplot` object.
#' @details This function plots the loss function across the available epochs.
#' @export
autoplot.torch_mlp <- function(object, ...) {
 x <- tibble::tibble(iteration = seq(along = object$loss), loss = object$loss)

 if(object$parameters$validation > 0) {
  lab <- "loss (validation set)"
 } else {
  lab <- "loss (training set)"
 }

 ggplot2::ggplot(x, ggplot2::aes(x = iteration, y = loss)) +
  ggplot2::geom_line() +
  ggplot2::labs(y = lab)
}

model_to_raw <- function(model) {
 con <- rawConnection(raw(), open = "wr")
 torch::torch_save(model, con)
 on.exit({close(con)}, add = TRUE)
 r <- rawConnectionValue(con)
 r
}

## -----------------------------------------------------------------------------

#' Plot model loss over epochs
#'
#' @param object A `torch_linear_reg` object.
#' @param ... Not currently used
#' @return A `ggplot` object.
#' @details This function plots the loss function across the available epochs.
#' @export
autoplot.torch_linear_reg <- autoplot.torch_mlp

