allowed_activation <-
 c("celu", "elu", "gelu", "hardshrink", "hardsigmoid",
   "hardtanh", "leaky_relu", "linear", "log_sigmoid", "relu", "relu6",
   "rrelu", "selu", "sigmoid", "silu", "softplus", "softshrink",
   "softsign", "tanh", "tanhshrink")

#' Activation functions for neural networks in brulee
#'
#' @return A character vector of values.
#' @seealso [torch::nn_celu()], [torch::nn_elu()], [torch::nn_gelu()],
#' [torch::nn_hardshrink()], [torch::nn_hardsigmoid()], [torch::nn_hardtanh()],
#' [torch::nn_leaky_relu()], [torch::nn_identity()], [torch::nn_log_sigmoid()],
#' [torch::nn_relu()], [torch::nn_relu6()], [torch::nn_rrelu()], [torch::nn_selu()],
#' [torch::nn_sigmoid()], [torch::nn_silu()], [torch::nn_softplus()],
#' [torch::nn_softshrink()], [torch::nn_softsign()], [torch::nn_tanh()],
#' [torch::nn_tanhshrink()]
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
