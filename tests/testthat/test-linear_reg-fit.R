
set.seed(1)
df <- tibble::tibble(
  x1 = runif(100),
  x2 = runif(100),
  y = 3 + 2*x1 + 3*x2
)

test_that("linear regression test", {
  skip_if(!torch::torch_is_installed())

  set.seed(1)
  expect_error(
    fit <- brulee_linear_reg(y ~ ., df, epochs = 2),
    regexp = NA
  )

  expect_equal(
    as.numeric(coef(fit)),
    as.numeric(coef(lm(y ~ ., df))),
    tolerance = 0.1
  )

  expect_error(
    fit <- brulee_linear_reg(y ~ ., df, epochs = 10, learn_rate = 0.1, optimizer = "SGD"),
    regexp = NA
  )

})

