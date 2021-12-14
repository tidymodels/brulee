library(testthat)
library(brulee)
library(tibble)

RNGkind("Mersenne-Twister")

if (torch::torch_is_installed()) {
 test_check("brulee")
}
