#' Change the learning rate over time
#'
#' Learning rate schedulers alter the learning rate to adjust as training
#' proceeds. In most cases, the learning rate decreases as epochs increase.
#' The `schedule_*()` functions are individual schedulers and
#' [set_learn_rate()] is a general interface.
#' @param epoch An integer for the number of training epochs (zero being the
#' initial value),
#' @param initial A positive numeric value for the starting learning rate.
#' @param decay A positive numeric constant for decreasing the rate (see
#' Details below).
#' @param reduction A positive numeric constant stating the proportional decrease
#' in the learning rate occurring at every `steps` epochs.
#' @param steps The number of epochs before the learning rate changes.
#' @param largest The maximum learning rate in the cycle.
#' @param step_size The half-length of a cycle.
#' @param learn_rate A constant learning rate (when no scheduler is used),
#' @param type A single character value for the type of scheduler. Possible
#' values are: "decay_time", "decay_expo", "none", "cyclic", and "step".
#' @param ... Arguments to pass to the individual scheduler functions (e.g.
#' `reduction`).
#' @return A numeric value for the updated learning rate.
#' @details
#' The details for how the schedulers change the rates:
#'
#' * `schedule_decay_time()`: \eqn{rate(epoch) = initial/(1 + decay \times epoch)}
#' * `schedule_decay_expo()`: \eqn{rate(epoch) = initial\exp(-decay \times epoch)}
#' * `schedule_step()`: \eqn{rate(epoch) = initial \times reduction^{floor(epoch / steps)}}
#' * `schedule_cyclic()`: \eqn{cycle = floor( 1 + (epoch / 2 / step size) )},
#'  \eqn{x = abs( ( epoch / step size ) - ( 2 * cycle) + 1 )}, and
#'  \eqn{rate(epoch) = initial + ( largest - initial ) * \max( 0, 1 - x)}
#'
#'
#' @seealso [brulee_mlp()]
#' @examples
#' if (rlang::is_installed("purrr")) {
#'  library(ggplot2)
#'  library(dplyr)
#'  library(purrr)
#'
#'  iters <- 0:50
#'
#'  bind_rows(
#'   tibble(epoch = iters, rate = map_dbl(iters, schedule_decay_time), type = "decay_time"),
#'   tibble(epoch = iters, rate = map_dbl(iters, schedule_decay_expo), type = "decay_expo"),
#'   tibble(epoch = iters, rate = map_dbl(iters, schedule_step), type = "step"),
#'   tibble(epoch = iters, rate = map_dbl(iters, schedule_cyclic), type = "cyclic")
#'  ) %>%
#'   ggplot(aes(epoch, rate)) +
#'   geom_line() +
#'   facet_wrap(~ type)
#'
#' }
#'
#' @export

schedule_decay_time <- function(epoch, initial = 0.1, decay = 1) {
 check_rate_arg_value(initial)
 check_rate_arg_value(decay)
 initial / (1 + decay * epoch)
}

#' @export
#' @rdname schedule_decay_time
schedule_decay_expo <- function(epoch, initial = 0.1, decay = 1) {
 check_rate_arg_value(initial)
 check_rate_arg_value(decay)
 initial * exp(-decay * epoch)
}

#' @export
#' @rdname schedule_decay_time
schedule_step <- function(epoch, initial = 0.1, reduction = 1/2, steps = 5) {
 check_rate_arg_value(initial)
 check_rate_arg_value(reduction)
 check_rate_arg_value(steps)
 initial * reduction^floor(epoch / steps)
}

#' @export
#' @rdname schedule_decay_time
schedule_cyclic <- function(epoch, initial = 0.001, largest = 0.1, step_size = 5) {
 check_rate_arg_value(initial)
 check_rate_arg_value(largest)
 check_rate_arg_value(step_size)

 if (largest < initial) {
  tmp <- initial
  largest <- initial
  initial <- tmp
 } else if (largest == initial) {
  initial <- initial / 10
 }

 cycle <- floor( 1 + (epoch / 2 / step_size) )
 x <- abs( ( epoch / step_size ) - ( 2 * cycle) + 1 )
 initial + ( largest - initial ) * max( 0, 1 - x)
}

# Learning rate can be either static (via rate_schedule == "none") or dynamic.
# Either way, set_learn_rate() figures this out and sets it accordingly.

#' @export
#' @rdname schedule_decay_time
set_learn_rate <- function(epoch, learn_rate, type = "none", ...) {
 types <- c("decay_time", "decay_expo", "none", "step", "cyclic")
 types <- rlang::arg_match0(type, types, arg_nm = "type")
 if (type == "none") {
  return(learn_rate)
 }

 fn <- paste0("schedule_", type)
 args <- list(...)

 cl <- rlang::call2(fn, epoch = epoch, !!!args)
 rlang::eval_tidy(cl)
}

# ------------------------------------------------------------------------------

check_rate_arg_value <- function(x) {
 nm <- as.character(match.call()$x)
 if (is.null(x) || !is.numeric(x) || length(x) != 1 || any(x <= 0)) {
  msg <- paste0("Argument '", nm, "' should be a single positive value.")
  cli::cli_abort(msg)
 }
 invisible(NULL)
}
