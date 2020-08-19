new_torch_linear_reg <- function(coefs, blueprint) {
  hardhat::new_model(coefs = coefs, blueprint = blueprint, class = "torch_linear_reg")
}
