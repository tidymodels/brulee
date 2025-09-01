# used for autoplots
brulee_plot <- function(object, ...) {
  x <- tibble::tibble(
    iteration = seq_len(length(object$loss)) - 1,
    loss = object$loss
  )

  if (object$parameters$validation > 0) {
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
    ggplot2::labs(y = lab) +
    ggplot2::geom_vline(xintercept = object$best_epoch, lty = 2, col = "green")
}


## -----------------------------------------------------------------------------

#' Plot model loss over epochs
#'
#' @param object A `brulee_mlp`, `brulee_logistic_reg`,
#' `brulee_multinomial_reg`, or `brulee_linear_reg` object.
#' @param ... Not currently used
#' @return A `ggplot` object.
#' @details This function plots the loss function across the available epochs. A
#' vertical line shows the epoch with the best loss value.
#' @examplesIf !brulee:::is_cran_check()
#' \donttest{
#' if (torch::torch_is_installed() & rlang::is_installed(c("recipes", "yardstick", "modeldata"))) {
#'  library(ggplot2)
#'  library(recipes)
#'  theme_set(theme_bw())
#'
#'  data(ames, package = "modeldata")
#'
#'  ames$Sale_Price <- log10(ames$Sale_Price)
#'
#'  set.seed(1)
#'  in_train <- sample(1:nrow(ames), 2000)
#'  ames_train <- ames[ in_train,]
#'  ames_test  <- ames[-in_train,]
#'
#'  ames_rec <-
#'   recipe(Sale_Price ~ Longitude + Latitude, data = ames_train) |>
#'     step_normalize(all_numeric_predictors())
#'
#'  set.seed(2)
#'  fit <- brulee_mlp(ames_rec, data = ames_train, epochs = 50, batch_size = 32)
#'
#'  autoplot(fit)
#' }
#' }
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
