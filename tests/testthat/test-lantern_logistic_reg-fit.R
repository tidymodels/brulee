
set.seed(1)
df <- tibble::tibble(
  x1 = rnorm(100),
  x2 = rnorm(100),
  logit = x1 + x2 + rnorm(100, sd = 0.25),
  y = as.factor(ifelse(exp(logit)/(1 + exp(logit)) > 0.6, "a", "b"))
)
df$logit <- NULL

# ------------------------------------------------------------------------------

test_that("logistic regression", {
  skip_if_not(torch::torch_is_installed())

  expect_snapshot({
    set.seed(1)
    fit <- lantern_logistic_reg(y ~ ., df, epochs = 2, verbose = TRUE)
  })

  expect_error(
    fit <- lantern_logistic_reg(y ~ ., df, epochs = 10, learn_rate = 0.1,
                                optimizer = "SGD"),
    regexp = NA
  )
})

# ------------------------------------------------------------------------------

test_that("class weights - logistic regression", {
  skip_if_not(torch::torch_is_installed())

  expect_snapshot({
    set.seed(1)
    fit <- lantern_logistic_reg(y ~ ., df, epochs = 2, verbose = TRUE,
                                class_weights = 12)
  })


  expect_snapshot({
    set.seed(1)
    fit <- lantern_logistic_reg(y ~ ., df, epochs = 2, verbose = TRUE,
                                class_weights = c(a = 12, b = 1))
  })

  expect_error(
    fit <- lantern_logistic_reg(y ~ ., df, epochs = 10, learn_rate = 0.1,
                                optimizer = "SGD"),
    regexp = NA
  )
})
