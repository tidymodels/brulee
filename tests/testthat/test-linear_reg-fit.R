
test_that("linear regression test", {
 skip_if(!torch::torch_is_installed())
 skip_on_os("mac", arch = "aarch64")

 # ------------------------------------------------------------------------------

 set.seed(1)
 df <- tibble::tibble(
  x1 = runif(100),
  x2 = runif(100),
  y = 3 + 2*x1 + 3*x2
 )

 set.seed(1)
 expect_error(
  lin_reg_fit_lbfgs <- brulee_linear_reg(y ~ ., df, epochs = 2),
  regexp = NA
 )

 # regression tests
 save_coef(lin_reg_fit_lbfgs)
 expect_equal(
  last_param(lin_reg_fit_lbfgs),
  load_coef(lin_reg_fit_lbfgs),
  tolerance = 0.1
 )

 expect_equal(
  as.numeric(coef(lin_reg_fit_lbfgs)),
  as.numeric(coef(lm(y ~ ., df))),
  tolerance = 0.1
 )

 expect_error(
  lin_reg_fit_sgd <-
   brulee_linear_reg(y ~ ., df, epochs = 10, learn_rate = 0.1, optimizer = "SGD"),
  regexp = NA
 )

 # regression tests
 save_coef(lin_reg_fit_sgd)
 expect_equal(
  last_param(lin_reg_fit_sgd),
  load_coef(lin_reg_fit_sgd),
  tolerance = 0.1
 )

})

