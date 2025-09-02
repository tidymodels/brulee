library(testthat)
library(brulee)
library(tibble)
library

RNGkind("Mersenne-Twister")

# CRAN is currently showing false positives for this
# if (torch::torch_is_installed()) {

# Temporarily avoid the issue
if (identical(Sys.getenv("NOT_CRAN"), "true")) {
 test_check("brulee")
}
