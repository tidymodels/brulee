test_that("logistic regression", {

 df <- tibble::tibble(
  x1 = rnorm(100),
  x2 = rnorm(100),
  logit = x1 + x2 + rnorm(100, sd = 0.25),
  y = as.factor(exp(logit)/(1 + exp(logit)) > 0.5)
 )
 df$logit <- NULL

 expect_error(
  fit <- lantern_logistic_reg(y ~ ., df, epochs = 2, verbose = TRUE),
  regexp = NA
 )

 expect_error(
  fit <- lantern_logistic_reg(y ~ ., df, epochs = 10, learn_rate = 0.1, optimizer = "SGD"),
  regexp = NA
 )


})
