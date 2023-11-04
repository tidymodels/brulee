allowed_activation <-
 c("celu", "elu", "gelu", "hardshrink", "hardsigmoid",
   "hardtanh", "leaky_relu", "linear", "log_sigmoid", "relu", "relu6",
   "rrelu", "selu", "sigmoid", "silu", "softplus", "softshrink",
   "softsign", "tanh", "tanhshrink")

#' Activation functions for neural networks in brulee
#'
#' @return A character vector of values.
#' @export
brulee_activations <- function() {
 allowed_activation
}

get_activation_fn <- function(arg, ...) {

 if (arg == "linear") {
  res <- identity
 } else {
  cl <- rlang::call2(paste0("nn_", arg), .ns = "torch")
  res <- rlang::eval_bare(cl)
 }

 res
}
