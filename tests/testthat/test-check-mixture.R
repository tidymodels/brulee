test_that("check_mixture passes through mixture = 1", {
  skip_if_not(torch::torch_is_installed())

  # All optimizers should allow mixture = 1
  expect_equal(brulee:::check_mixture(1.0, "SGD"), 1.0)
  expect_equal(brulee:::check_mixture(1.0, "LBFGS"), 1.0)
  expect_equal(brulee:::check_mixture(1.0, "RMSprop"), 1.0)
  expect_equal(brulee:::check_mixture(1.0, "Adadelta"), 1.0)
  expect_equal(brulee:::check_mixture(1.0, "Adagrad"), 1.0)

  # Even ADAMw should pass through mixture = 1 without warning
  # (the warning only applies when mixture is not exactly 0 or 1)
  expect_equal(brulee:::check_mixture(1.0, "ADAMw"), 1.0)
})

test_that("check_mixture ADAMw enforces mixture = 0", {
  skip_if_not(torch::torch_is_installed())

  # ADAMw with mixture = 0 should pass through
  expect_equal(brulee:::check_mixture(0.0, "ADAMw"), 0.0)
  expect_no_warning(brulee:::check_mixture(0.0, "ADAMw"))

  # ADAMw with 0 < mixture < 1 should warn and convert to 0
  expect_warning(
    result <- brulee:::check_mixture(0.5, "ADAMw"),
    regexp = "pure L2 penalty"
  )
  expect_equal(result, 0.0)

  expect_warning(
    result <- brulee:::check_mixture(0.1, "ADAMw"),
    regexp = "pure L2 penalty"
  )
  expect_equal(result, 0.0)

  expect_warning(
    result <- brulee:::check_mixture(0.9, "ADAMw"),
    regexp = "pure L2 penalty"
  )
  expect_equal(result, 0.0)
})

test_that("check_mixture non-ADAMw optimizers allow any mixture", {
  skip_if_not(torch::torch_is_installed())

  optimizers <- c("SGD", "LBFGS", "RMSprop", "Adadelta", "Adagrad")
  mixture_values <- c(0, 0.1, 0.5, 0.9, 1.0)

  for (opt in optimizers) {
    for (mix in mixture_values) {
      # Should pass through unchanged and not warn
      expect_equal(brulee:::check_mixture(mix, opt), mix)
      expect_no_warning(brulee:::check_mixture(mix, opt))
    }
  }
})

test_that("check_mixture handles edge cases", {
  skip_if_not(torch::torch_is_installed())

  # Exactly 0.0 should be treated as pure L2
  expect_equal(brulee:::check_mixture(0.0, "SGD"), 0.0)
  expect_equal(brulee:::check_mixture(0.0, "ADAMw"), 0.0)

  # Exactly 1.0 should be treated as pure L1
  expect_equal(brulee:::check_mixture(1.0, "SGD"), 1.0)
  expect_equal(brulee:::check_mixture(1.0, "ADAMw"), 1.0)

  # Very small non-zero values with ADAMw should warn
  expect_warning(
    result <- brulee:::check_mixture(0.001, "ADAMw"),
    regexp = "pure L2 penalty"
  )
  expect_equal(result, 0.0)

  # Values very close to 1 with ADAMw should warn
  expect_warning(
    result <- brulee:::check_mixture(0.999, "ADAMw"),
    regexp = "pure L2 penalty"
  )
  expect_equal(result, 0.0)
})
