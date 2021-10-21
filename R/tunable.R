#' Internal functions and methods
#' @export
#' @keywords internal
#' @name lantern-internal
tunable.lantern_mlp <- function(x, ...) {
 tibble::tibble(
  name = c('epochs', 'hidden_units', 'activation', 'penalty', 'dropout',
           'learn_rate', 'momentum', 'batch_size', 'class_weights', 'stop_iter'),
  call_info = list(
   list(pkg = "dials", fun = "epochs", range = c(5L, 100L)),
   list(pkg = "dials", fun = "hidden_units"),
   list(pkg = "dials", fun = "activation"),
   list(pkg = "dials", fun = "penalty"),
   list(pkg = "dials", fun = "dropout"),
   list(pkg = "dials", fun = "learn_rate", range = c(-3, -1/5)),
   list(pkg = "dials", fun = "momentum"),
   list(pkg = "dials", fun = "batch_size"),
   list(pkg = "dials", fun = "stop_iter"),
   list(pkg = "dials", fun = "class_weights")
  ),
  source = "model_spec",
  component = class(x)[class(x) != "model_spec"][1],
  component_id =  "main"
 )
}

#' @export
#' @keywords internal
#' @rdname lantern-internal
tunable.lantern_logistic_reg <- function(x, ...) {
 tibble::tibble(
  name = c('epochs', 'penalty', 'learn_rate', 'momentum', 'batch_size',
           'class_weights', 'stop_iter'),
  call_info = list(
   list(pkg = "dials", fun = "epochs", range = c(5L, 100L)),
   list(pkg = "dials", fun = "penalty"),
   list(pkg = "dials", fun = "learn_rate", range = c(-3, -1/5)),
   list(pkg = "dials", fun = "momentum"),
   list(pkg = "dials", fun = "batch_size"),
   list(pkg = "dials", fun = "stop_iter"),
   list(pkg = "dials", fun = "class_weights")
  ),
  source = "model_spec",
  component = class(x)[class(x) != "model_spec"][1],
  component_id =  "main"
 )
}

#' @export
#' @keywords internal
#' @rdname lantern-internal
tunable.lantern_linear_reg <- function(x, ...) {
 tibble::tibble(
  name = c('epochs', 'penalty', 'learn_rate', 'momentum', 'batch_size',
           'stop_iter'),
  call_info = list(
   list(pkg = "dials", fun = "epochs", range = c(5L, 100L)),
   list(pkg = "dials", fun = "penalty"),
   list(pkg = "dials", fun = "learn_rate", range = c(-3, -1/5)),
   list(pkg = "dials", fun = "momentum"),
   list(pkg = "dials", fun = "batch_size"),
   list(pkg = "dials", fun = "stop_iter")
  ),
  source = "model_spec",
  component = class(x)[class(x) != "model_spec"][1],
  component_id =  "main"
 )
}