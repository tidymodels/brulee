
set.seed(1)
df <- tibble::tibble(
  x1 = runif(100),
  x2 = runif(100),
  y = 3 + 2*x1 + 3*x2
)

# Log for an upcoming issue:
set.seed(1)
lantern_linear_reg(y ~ ., df, epochs = 2, verbose = TRUE)

test_that("linear regression test", {
  skip_if(!torch::torch_is_installed())
  skip_on_os("mac") # Generating slightly different results on macOS. eg.
  # - "  scaled validation loss after 1 epochs: 1.46e-12 "
  # + "  scaled validation loss after 1 epochs: 1.57e-12 "
  skip_on_os("windows") # same as above

  expect_error(
    fit <- lantern_linear_reg(y ~ ., df, epochs = 2),
    regexp = NA
  )

  # TODO
  expect_equal(
    as.numeric(unlist(coef(fit))[c(3,1,2)]),
    as.numeric(coef(lm(y ~ ., df))),
    tolerance = 1e-4
  )

  expect_error(
    fit <- lantern_linear_reg(y ~ ., df, epochs = 10, learn_rate = 0.1, optimizer = "SGD"),
    regexp = NA
  )

  # fails on windows; slightly different values from unix and macOS
  expect_snapshot({
    set.seed(1)
    fit <- lantern_linear_reg(y ~ ., df, epochs = 2, verbose = TRUE)
    fit
  })

})

