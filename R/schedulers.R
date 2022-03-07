#' Decrease the learning rate over time
#'
#' Learning rate schedulers alter the learning rate to decrease as the number of
#' training epochs increases. The `learn_rate_*()` functions are individual
#' schedulers and [set_learn_rate()] is a general interface.
#' @param epoch An integer for the number of training epochs (zero being the
#' initial value),
#' @param initial A positive numeric value for the starting learning rate.
#' @param decay A positive numeric constant for decreasing the rate (see
#' Details below).
#' @param reduction A positive numeric constant stating the proportional decrease
#' in the learning rate occurring at every `steps` epochs.
#' @param steps The number of epochs before the learning rate changes.
#' @param type A single character value for the type of scheduler. Possible
#' values are: "decay_time", "decay_expo", "constant", and "step".
#' @param ... Arguments to pass to the individual scheduler functions (e.g.
#' `reduction`).
#' @return A numeric value for the updated learning rate.
#' @details
#' The details for how the schedulers change the rates:
#'
#' * `learn_rate_decay_time()`: \eqn{rate(epoch) = initial/(1 + decay \times epoch)}
#' * `learn_rate_decay_expo()`: \eqn{initial\exp(-decay \times epoch)}
#' * `learn_rate_step()`: \eqn{initial \times reduction^{floor(epoch / steps)}}
#'
#' @seealso [brulee_mlp()]
#' @examples
#' library(ggplot2)
#' library(dplyr)
#' library(purrr)
#'
#' iters <- seq(0, 50, length.out = 200)
#'
#' bind_rows(
#'  tibble(epoch = iters, rate = map_dbl(iters, learn_rate_constant), type = "constant"),
#'  tibble(epoch = iters, rate = map_dbl(iters, learn_rate_decay_time), type = "decay_time"),
#'  tibble(epoch = iters, rate = map_dbl(iters, learn_rate_decay_expo), type = "decay_expo"),
#'  tibble(epoch = iters, rate = map_dbl(iters, learn_rate_step), type = "step")
#' ) %>%
#'  ggplot(aes(epoch, rate)) +
#'  geom_line() +
#'  facet_wrap(~ type)
#' @export

learn_rate_decay_time <- function(epoch, initial = 0.1, decay = 1) {
 check_rate_arg_value(initial)
 check_rate_arg_value(decay)
 initial / (1 + decay * epoch)
}

#' @export
#' @rdname learn_rate_decay_time
learn_rate_constant <- function(epoch, initial = 0.1) {
 check_rate_arg_value(initial)
 initial
}

#' @export
#' @rdname learn_rate_decay_time
learn_rate_decay_expo <- function(epoch, initial = 0.1, decay = 1) {
 check_rate_arg_value(initial)
 check_rate_arg_value(decay)
 initial * exp(-decay * epoch)
}

#' @export
#' @rdname learn_rate_decay_time
learn_rate_step <- function(epoch, initial = 0.1, reduction = 1/2, steps = 5) {
 check_rate_arg_value(initial)
 check_rate_arg_value(reduction)
 check_rate_arg_value(steps)
 initial * reduction^floor(epoch / steps)
}

#' @export
#' @rdname learn_rate_decay_time
set_learn_rate <- function(epoch, type = "constant", ...) {
 types <- c("decay_time", "decay_expo", "constant", "step")
 types <- rlang::arg_match0(type, types, arg_nm = "type")
 fn <- paste0("learn_rate_", type)
 args <- list(...)
 cl <- rlang::call2(fn, epoch = epoch, !!!args)
 rlang::eval_tidy(cl)
}

check_rate_arg_value <- function(x) {
 nm <- as.character(match.call()$x)
 if (is.null(x) || !is.numeric(x) || length(x) != 1 || any(x <= 0)) {
  msg <- paste0("Argument '", nm, "' should be a single positive value.")
  rlang::abort(msg)
 }
 invisible(NULL)
}
