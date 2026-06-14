test_that("penalty affects model parameters during training", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  suppressPackageStartupMessages(library(dplyr))

  data("parabolic", package = "modeldata")
  set.seed(1)
  in_train <- sample(1:nrow(parabolic), 200)
  parabolic_tr <- parabolic[in_train, ]

  # Train with no penalty
  set.seed(123)
  torch::torch_manual_seed(123)
  fit_no_penalty <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 3,
    epochs = 50L,
    learn_rate = 0.1,
    activation = "relu",
    penalty = 0,
    mixture = 0,
    batch_size = 128,
    optimizer = "SGD",
    verbose = FALSE,
    device = "cpu"
  )

  # Train with moderate penalty
  set.seed(123)
  torch::torch_manual_seed(123)
  fit_with_penalty <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 3,
    epochs = 50L,
    learn_rate = 0.1,
    activation = "relu",
    penalty = 0.5,
    mixture = 0,
    batch_size = 128,
    optimizer = "SGD",
    verbose = FALSE,
    device = "cpu"
  )

  # Get coefficients
  coef_no_penalty <- coef(fit_no_penalty)
  coef_with_penalty <- coef(fit_with_penalty)

  # Flatten all coefficient matrices into a single vector
  all_coef_no_penalty <- unlist(coef_no_penalty)
  all_coef_with_penalty <- unlist(coef_with_penalty)

  # Penalty should affect coefficient magnitudes (regularization effect)
  # They should be different, though the direction may vary by optimizer/data
  mean_abs_no_penalty <- mean(abs(all_coef_no_penalty))
  mean_abs_with_penalty <- mean(abs(all_coef_with_penalty))

  expect_false(isTRUE(all.equal(mean_abs_no_penalty, mean_abs_with_penalty)))
})

test_that("L1 penalty encourages sparsity more than L2", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  suppressPackageStartupMessages(library(dplyr))

  data("parabolic", package = "modeldata")
  set.seed(1)
  in_train <- sample(1:nrow(parabolic), 200)
  parabolic_tr <- parabolic[in_train, ]

  penalty_val <- 1

  # L2 penalty
  set.seed(123)
  torch::torch_manual_seed(123)
  fit_l2 <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 5,
    epochs = 100L,
    learn_rate = 0.1,
    activation = "relu",
    penalty = penalty_val,
    mixture = 0,
    batch_size = 128,
    optimizer = "SGD",
    verbose = FALSE,
    device = "cpu"
  )

  # L1 penalty
  set.seed(123)
  torch::torch_manual_seed(123)
  fit_l1 <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 5,
    epochs = 100L,
    learn_rate = 0.1,
    activation = "relu",
    penalty = penalty_val,
    mixture = 1,
    batch_size = 128,
    optimizer = "SGD",
    verbose = FALSE,
    device = "cpu"
  )

  # Get coefficients
  coef_l2 <- coef(fit_l2)
  coef_l1 <- coef(fit_l1)

  # Flatten all coefficient matrices into a single vector
  all_coef_l2 <- unlist(coef_l2)
  all_coef_l1 <- unlist(coef_l1)

  # L1 should produce more near-zero coefficients (sparsity)
  # Count coefficients with absolute value < 0.01
  near_zero_l2 <- sum(abs(all_coef_l2) < 0.01)
  near_zero_l1 <- sum(abs(all_coef_l1) < 0.01)

  expect_gte(near_zero_l1, near_zero_l2)
})

test_that("penalty consistency across epochs", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  data("parabolic", package = "modeldata")
  set.seed(1)
  in_train <- sample(1:nrow(parabolic), 200)
  parabolic_tr <- parabolic[in_train, ]

  set.seed(123)
  torch::torch_manual_seed(123)
  fit <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 2,
    epochs = 100L,
    learn_rate = 0.1,
    activation = "elu",
    penalty = 1,
    mixture = 0,
    batch_size = 128,
    optimizer = "SGD",
    verbose = FALSE,
    device = "cpu"
  )

  # Loss should generally decrease or stabilize (not increase erratically)
  # Check that the loss trajectory is reasonable
  expect_true(length(fit$loss) > 1)
  expect_true(all(!is.na(fit$loss)))
  expect_true(all(is.finite(fit$loss)))

  # Final loss should be less than initial loss (model is learning)
  expect_lt(tail(fit$loss, 1), fit$loss[1])
})

test_that("extreme penalty values behave reasonably", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  data("parabolic", package = "modeldata")
  set.seed(1)
  in_train <- sample(1:nrow(parabolic), 150)
  parabolic_tr <- parabolic[in_train, ]

  # Very small penalty should behave similar to no penalty
  set.seed(123)
  torch::torch_manual_seed(123)
  fit_no_penalty <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 2,
    epochs = 50L,
    learn_rate = 0.1,
    activation = "elu",
    penalty = 0,
    mixture = 0,
    batch_size = 128,
    optimizer = "SGD",
    verbose = FALSE,
    device = "cpu"
  )

  set.seed(123)
  torch::torch_manual_seed(123)
  fit_tiny_penalty <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 2,
    epochs = 50L,
    learn_rate = 0.1,
    activation = "elu",
    penalty = 1e-6,
    mixture = 0,
    batch_size = 128,
    optimizer = "SGD",
    verbose = FALSE,
    device = "cpu"
  )

  # Very small penalty should produce similar results
  expect_true(
    max(abs(fit_no_penalty$loss - fit_tiny_penalty$loss)) < 0.1
  )

  # Very large penalty should still produce valid model
  set.seed(123)
  torch::torch_manual_seed(123)
  fit_huge_penalty <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 2,
    epochs = 20L, # Fewer epochs for large penalty
    learn_rate = 0.01, # Lower learning rate for stability
    activation = "elu",
    penalty = 1000,
    mixture = 0,
    batch_size = 128,
    optimizer = "SGD",
    verbose = FALSE,
    device = "cpu"
  )

  # Should complete without error and produce finite loss
  expect_true(all(is.finite(fit_huge_penalty$loss)))
  expect_true(length(fit_huge_penalty$loss) > 0)
})

test_that("penalty works correctly with validation split", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  data("parabolic", package = "modeldata")
  set.seed(1)
  in_train <- sample(1:nrow(parabolic), 300)
  parabolic_tr <- parabolic[in_train, ]

  # With validation split
  set.seed(123)
  torch::torch_manual_seed(123)
  fit_no_penalty <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 2,
    epochs = 50L,
    learn_rate = 0.1,
    activation = "elu",
    penalty = 0,
    mixture = 0,
    validation = 0.2,
    batch_size = 128,
    optimizer = "SGD",
    verbose = FALSE,
    device = "cpu"
  )

  set.seed(123)
  torch::torch_manual_seed(123)
  fit_with_penalty <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 2,
    epochs = 50L,
    learn_rate = 0.1,
    activation = "elu",
    penalty = 5,
    mixture = 0,
    validation = 0.2,
    batch_size = 128,
    optimizer = "SGD",
    verbose = FALSE,
    device = "cpu"
  )

  # Should produce different results
  expect_false(isTRUE(all.equal(fit_no_penalty$loss, fit_with_penalty$loss)))

  # Both should have valid best_epoch
  expect_true(fit_no_penalty$best_epoch >= 0)
  expect_true(fit_with_penalty$best_epoch >= 0)
  expect_true(fit_no_penalty$best_epoch <= 50)
  expect_true(fit_with_penalty$best_epoch <= 50)
})

test_that("penalty works with different batch sizes", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  data("parabolic", package = "modeldata")
  set.seed(1)
  in_train <- sample(1:nrow(parabolic), 200)
  parabolic_tr <- parabolic[in_train, ]

  batch_sizes <- c(32, 64, 128)

  for (bs in batch_sizes) {
    set.seed(123)
    torch::torch_manual_seed(123)
    fit_no_penalty <- brulee_mlp(
      class ~ .,
      data = parabolic_tr,
      hidden_units = 2,
      epochs = 30L,
      learn_rate = 0.1,
      activation = "elu",
      penalty = 0,
      mixture = 0,
      batch_size = bs,
      optimizer = "SGD",
      verbose = FALSE,
      device = "cpu"
    )

    set.seed(123)
    torch::torch_manual_seed(123)
    fit_with_penalty <- brulee_mlp(
      class ~ .,
      data = parabolic_tr,
      hidden_units = 2,
      epochs = 30L,
      learn_rate = 0.1,
      activation = "elu",
      penalty = 1,
      mixture = 0,
      batch_size = bs,
      optimizer = "SGD",
      verbose = FALSE,
      device = "cpu"
    )

    # Penalty should make a difference regardless of batch size
    expect_false(
      isTRUE(all.equal(fit_no_penalty$loss, fit_with_penalty$loss)),
      info = paste0("batch_size = ", bs)
    )
  }
})

test_that("penalty parameter is stored correctly in model object", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  data("parabolic", package = "modeldata")
  set.seed(1)
  in_train <- sample(1:nrow(parabolic), 150)
  parabolic_tr <- parabolic[in_train, ]

  penalty_val <- 0.456
  mixture_val <- 0.3

  set.seed(123)
  torch::torch_manual_seed(123)
  fit <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 2,
    epochs = 20L,
    learn_rate = 0.1,
    activation = "elu",
    penalty = penalty_val,
    mixture = mixture_val,
    batch_size = 128,
    optimizer = "SGD",
    verbose = FALSE,
    device = "cpu"
  )

  # Check that penalty and mixture are stored in the model object
  expect_equal(fit$parameters$penalty, penalty_val)
  expect_equal(fit$parameters$mixture, mixture_val)
  expect_equal(fit$parameters$optimizer, "SGD")
})
