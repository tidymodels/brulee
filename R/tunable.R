tune_activations <- c("relu", "tanh", "elu", "log_sigmoid", "tanhshrink")

#' Internal functions and methods
#' @export
#' @keywords internal
#' @name brulee-internal
tunable.brulee_mlp <- function(x, ...) {
   tibble::tibble(
      name = c('epochs', 'hidden_units', 'activation', 'penalty', 'dropout',
               'learn_rate', 'momentum', 'batch_size', 'class_weights', 'stop_iter'),
      call_info = list(
         list(pkg = "dials", fun = "epochs", range = c(5L, 500L)),
         list(pkg = "dials", fun = "hidden_units", range = c(2L, 50L)),
         list(pkg = "dials", fun = "activation", values = tune_activations),
         list(pkg = "dials", fun = "penalty"),
         list(pkg = "dials", fun = "dropout"),
         list(pkg = "dials", fun = "learn_rate", range = c(-3, -1/5)),
         list(pkg = "dials", fun = "momentum", range = c(0.50, 0.95)),
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
#' @name brulee-internal
tunable.brulee_mlp_two_layer <- function(x, ...) {
 tibble::tibble(
  name = c('epochs', 'hidden_units', 'hidden_units_2', 'activation', 'activation_2',
           'penalty', 'dropout', 'learn_rate', 'momentum', 'batch_size',
           'class_weights', 'stop_iter'),
  call_info = list(
   list(pkg = "dials", fun = "epochs", range = c(5L, 500L)),
   list(pkg = "dials", fun = "hidden_units", range = c(2L, 50L)),
   list(pkg = "dials", fun = "hidden_units_2", range = c(2L, 50L)),
   list(pkg = "dials", fun = "activation", values = tune_activations),
   list(pkg = "dials", fun = "activation_2", values = tune_activations),
   list(pkg = "dials", fun = "penalty"),
   list(pkg = "dials", fun = "dropout"),
   list(pkg = "dials", fun = "learn_rate", range = c(-3, -1/5)),
   list(pkg = "dials", fun = "momentum", range = c(0.50, 0.95)),
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
#' @rdname brulee-internal
tunable.brulee_logistic_reg <- function(x, ...) {
   tibble::tibble(
      name = c('epochs', 'penalty', 'learn_rate', 'momentum', 'batch_size',
               'class_weights', 'stop_iter'),
      call_info = list(
         list(pkg = "dials", fun = "epochs", range = c(5L, 100L)),
         list(pkg = "dials", fun = "penalty"),
         list(pkg = "dials", fun = "learn_rate", range = c(-3, -1/5)),
         list(pkg = "dials", fun = "momentum", range = c(0.50, 0.95)),
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
#' @rdname brulee-internal
tunable.brulee_multinomial_reg <- function(x, ...) {
   tibble::tibble(
      name = c('epochs', 'penalty', 'learn_rate', 'momentum', 'batch_size',
               'class_weights', 'stop_iter'),
      call_info = list(
         list(pkg = "dials", fun = "epochs", range = c(5L, 100L)),
         list(pkg = "dials", fun = "penalty"),
         list(pkg = "dials", fun = "learn_rate", range = c(-3, -1/5)),
         list(pkg = "dials", fun = "momentum", range = c(0.50, 0.95)),
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
#' @rdname brulee-internal
tunable.brulee_linear_reg <- function(x, ...) {
   tibble::tibble(
      name = c('epochs', 'penalty', 'learn_rate', 'momentum', 'batch_size',
               'stop_iter'),
      call_info = list(
         list(pkg = "dials", fun = "epochs", range = c(5L, 100L)),
         list(pkg = "dials", fun = "penalty"),
         list(pkg = "dials", fun = "learn_rate", range = c(-3, -1/5)),
         list(pkg = "dials", fun = "momentum", range = c(0.50, 0.95)),
         list(pkg = "dials", fun = "batch_size"),
         list(pkg = "dials", fun = "stop_iter")
      ),
      source = "model_spec",
      component = class(x)[class(x) != "model_spec"][1],
      component_id =  "main"
   )
}
