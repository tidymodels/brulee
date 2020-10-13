new_torch_linear_reg <- function(coefs, loss, blueprint, terms) {
 hardhat::new_model(coefs = coefs,
                    loss = loss,
                    blueprint = blueprint,
                    terms = terms,
                    class = "torch_linear_reg")
}
