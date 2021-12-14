
# used for autoplots
brulee_plot <- function(object, ...) {
 x <- tibble::tibble(iteration = seq(along = object$loss), loss = object$loss)

 if(object$parameters$validation > 0) {
  if (is.na(object$y_stats$mean)) {
   lab <- "loss (validation set)"
  } else {
   lab <- "loss (validation set, scaled)"
  }
 } else {
  if (is.na(object$y_stats$mean)) {
   lab <- "loss (training set)"
  } else {
   lab <- "loss (training set, scaled)"
  }
 }

 ggplot2::ggplot(x, ggplot2::aes(x = iteration, y = loss)) +
  ggplot2::geom_line() +
  ggplot2::labs(y = lab)+
  ggplot2::geom_vline(xintercept = object$best_epoch, lty = 2, col = "green")
}


## -----------------------------------------------------------------------------

#' Plot model loss over epochs
#'
#' @param object A `brulee_mlp`, `brulee_logistic_reg`,
#' `brulee_multinomial_reg`, or `brulee_linear_reg` object.
#' @param ... Not currently used
#' @return A `ggplot` object.
#' @details This function plots the loss function across the available epochs.
#' @name brulee-autoplot
#' @export
autoplot.brulee_mlp <- brulee_plot

#' @rdname brulee-autoplot
#' @export
autoplot.brulee_logistic_reg <- brulee_plot

#' @rdname brulee-autoplot
#' @export
autoplot.brulee_multinomial_reg <- brulee_plot

#' @rdname brulee-autoplot
#' @export
autoplot.brulee_linear_reg <- brulee_plot

