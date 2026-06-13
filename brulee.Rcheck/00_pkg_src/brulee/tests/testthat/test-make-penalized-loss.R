test_that("make_penalized_loss returns unmodified loss for ADAMw", {
  skip_if_not(torch::torch_is_installed())

  # Create a simple model and loss function
  model <- torch::nn_linear(5, 1)
  base_loss_fn <- function(input, target) {
    torch::nnf_mse_loss(input, target)
  }

  # For ADAMw, should return the original loss function regardless of mixture
  loss_fn_adamw <- brulee:::make_penalized_loss(
    base_loss_fn,
    model,
    penalty = 1,
    mixture = 0,
    opt = "ADAMw"
  )

  # Create test data
  input <- torch::torch_randn(10, 1)
  target <- torch::torch_randn(10, 1)

  # The penalized loss should equal the base loss (no penalty added)
  base_loss <- base_loss_fn(input, target)
  penalized_loss <- loss_fn_adamw(input, target)

  expect_equal(as.numeric(base_loss$item()), as.numeric(penalized_loss$item()))
})

test_that("make_penalized_loss returns unmodified loss for pure L2 with non-LBFGS optimizers", {
  skip_if_not(torch::torch_is_installed())

  model <- torch::nn_linear(5, 1)
  base_loss_fn <- function(input, target) {
    torch::nnf_mse_loss(input, target)
  }

  # For pure L2 (mixture = 0) with non-LBFGS optimizers, should return unmodified loss
  optimizers <- c("SGD", "RMSprop", "Adadelta", "Adagrad")

  for (opt in optimizers) {
    loss_fn <- brulee:::make_penalized_loss(
      base_loss_fn,
      model,
      penalty = 1,
      mixture = 0,
      opt = opt
    )

    input <- torch::torch_randn(10, 1)
    target <- torch::torch_randn(10, 1)

    base_loss <- base_loss_fn(input, target)
    penalized_loss <- loss_fn(input, target)

    expect_equal(
      as.numeric(base_loss$item()),
      as.numeric(penalized_loss$item()),
      info = paste0(opt, " with mixture=0 should not add penalty to loss")
    )
  }
})

test_that("make_penalized_loss adds penalty to loss for L1 (mixture = 1)", {
  skip_if_not(torch::torch_is_installed())

  model <- torch::nn_linear(5, 1)
  base_loss_fn <- function(input, target) {
    torch::nnf_mse_loss(input, target)
  }

  # For pure L1 (mixture = 1), should add penalty to loss
  loss_fn_l1 <- brulee:::make_penalized_loss(
    base_loss_fn,
    model,
    penalty = 0.1,
    mixture = 1,
    opt = "SGD"
  )

  input <- torch::torch_randn(10, 1)
  target <- torch::torch_randn(10, 1)

  base_loss <- base_loss_fn(input, target)
  penalized_loss <- loss_fn_l1(input, target)

  # Penalized loss should be greater than base loss
  expect_gt(as.numeric(penalized_loss$item()), as.numeric(base_loss$item()))
})

test_that("make_penalized_loss adds penalty to loss for elastic net", {
  skip_if_not(torch::torch_is_installed())

  model <- torch::nn_linear(5, 1)
  base_loss_fn <- function(input, target) {
    torch::nnf_mse_loss(input, target)
  }

  # For elastic net (0 < mixture < 1), should add penalty to loss
  loss_fn_en <- brulee:::make_penalized_loss(
    base_loss_fn,
    model,
    penalty = 0.1,
    mixture = 0.5,
    opt = "SGD"
  )

  input <- torch::torch_randn(10, 1)
  target <- torch::torch_randn(10, 1)

  base_loss <- base_loss_fn(input, target)
  penalized_loss <- loss_fn_en(input, target)

  # Penalized loss should be greater than base loss
  expect_gt(as.numeric(penalized_loss$item()), as.numeric(base_loss$item()))
})

test_that("make_penalized_loss LBFGS always adds penalty to loss", {
  skip_if_not(torch::torch_is_installed())

  model <- torch::nn_linear(5, 1)
  base_loss_fn <- function(input, target) {
    torch::nnf_mse_loss(input, target)
  }

  input <- torch::torch_randn(10, 1)
  target <- torch::torch_randn(10, 1)
  base_loss <- base_loss_fn(input, target)

  # LBFGS with mixture = 0 should still add penalty to loss
  loss_fn_lbfgs_l2 <- brulee:::make_penalized_loss(
    base_loss_fn,
    model,
    penalty = 0.1,
    mixture = 0,
    opt = "LBFGS"
  )
  penalized_loss_l2 <- loss_fn_lbfgs_l2(input, target)
  expect_gt(as.numeric(penalized_loss_l2$item()), as.numeric(base_loss$item()))

  # LBFGS with mixture = 1 should add penalty to loss
  loss_fn_lbfgs_l1 <- brulee:::make_penalized_loss(
    base_loss_fn,
    model,
    penalty = 0.1,
    mixture = 1,
    opt = "LBFGS"
  )
  penalized_loss_l1 <- loss_fn_lbfgs_l1(input, target)
  expect_gt(as.numeric(penalized_loss_l1$item()), as.numeric(base_loss$item()))

  # LBFGS with mixture = 0.5 should add penalty to loss
  loss_fn_lbfgs_en <- brulee:::make_penalized_loss(
    base_loss_fn,
    model,
    penalty = 0.1,
    mixture = 0.5,
    opt = "LBFGS"
  )
  penalized_loss_en <- loss_fn_lbfgs_en(input, target)
  expect_gt(as.numeric(penalized_loss_en$item()), as.numeric(base_loss$item()))
})

test_that("make_penalized_loss penalty = 0 returns unmodified loss", {
  skip_if_not(torch::torch_is_installed())

  model <- torch::nn_linear(5, 1)
  base_loss_fn <- function(input, target) {
    torch::nnf_mse_loss(input, target)
  }

  # Even with mixture = 1, if penalty = 0, should return unmodified loss
  loss_fn <- brulee:::make_penalized_loss(
    base_loss_fn,
    model,
    penalty = 0,
    mixture = 1,
    opt = "SGD"
  )

  input <- torch::torch_randn(10, 1)
  target <- torch::torch_randn(10, 1)

  base_loss <- base_loss_fn(input, target)
  penalized_loss <- loss_fn(input, target)

  expect_equal(as.numeric(base_loss$item()), as.numeric(penalized_loss$item()))
})

test_that("make_penalized_loss different mixture values produce different penalties", {
  skip_if_not(torch::torch_is_installed())

  model <- torch::nn_linear(5, 1)

  # Set specific weights to make the test deterministic
  torch::with_no_grad({
    model$weight$copy_(torch::torch_ones_like(model$weight))
    model$bias$copy_(torch::torch_ones_like(model$bias))
  })

  base_loss_fn <- function(input, target) {
    torch::nnf_mse_loss(input, target)
  }

  input <- torch::torch_randn(10, 1)
  target <- torch::torch_randn(10, 1)

  # L2 only (mixture = 0)
  loss_fn_l2 <- brulee:::make_penalized_loss(
    base_loss_fn,
    model,
    penalty = 1,
    mixture = 0,
    opt = "SGD"
  )
  loss_l2 <- loss_fn_l2(input, target)

  # L1 only (mixture = 1)
  loss_fn_l1 <- brulee:::make_penalized_loss(
    base_loss_fn,
    model,
    penalty = 1,
    mixture = 1,
    opt = "SGD"
  )
  loss_l1 <- loss_fn_l1(input, target)

  # Elastic net (mixture = 0.5)
  loss_fn_en <- brulee:::make_penalized_loss(
    base_loss_fn,
    model,
    penalty = 1,
    mixture = 0.5,
    opt = "SGD"
  )
  loss_en <- loss_fn_en(input, target)

  # All three should be different
  expect_false(isTRUE(all.equal(
    as.numeric(loss_l2$item()),
    as.numeric(loss_l1$item())
  )))
  expect_false(isTRUE(all.equal(
    as.numeric(loss_l2$item()),
    as.numeric(loss_en$item())
  )))
  expect_false(isTRUE(all.equal(
    as.numeric(loss_l1$item()),
    as.numeric(loss_en$item())
  )))
})

test_that("make_penalized_loss higher penalty produces higher loss", {
  skip_if_not(torch::torch_is_installed())

  model <- torch::nn_linear(5, 1)

  # Set specific weights to make the test deterministic
  torch::with_no_grad({
    model$weight$copy_(torch::torch_ones_like(model$weight))
    model$bias$copy_(torch::torch_ones_like(model$bias))
  })

  base_loss_fn <- function(input, target) {
    torch::nnf_mse_loss(input, target)
  }

  input <- torch::torch_randn(10, 1)
  target <- torch::torch_randn(10, 1)

  # Small penalty
  loss_fn_small <- brulee:::make_penalized_loss(
    base_loss_fn,
    model,
    penalty = 0.1,
    mixture = 1,
    opt = "SGD"
  )
  loss_small <- loss_fn_small(input, target)

  # Large penalty
  loss_fn_large <- brulee:::make_penalized_loss(
    base_loss_fn,
    model,
    penalty = 10,
    mixture = 1,
    opt = "SGD"
  )
  loss_large <- loss_fn_large(input, target)

  # Larger penalty should produce larger loss
  expect_gt(as.numeric(loss_large$item()), as.numeric(loss_small$item()))
})

test_that("make_penalized_loss works with different optimizers", {
  skip_if_not(torch::torch_is_installed())

  model <- torch::nn_linear(5, 1)
  base_loss_fn <- function(input, target) {
    torch::nnf_mse_loss(input, target)
  }

  optimizers <- c("SGD", "LBFGS", "ADAMw", "RMSprop", "Adadelta", "Adagrad")

  for (opt in optimizers) {
    # Should not error
    loss_fn <- brulee:::make_penalized_loss(
      base_loss_fn,
      model,
      penalty = 0.1,
      mixture = 0,
      opt = opt
    )

    input <- torch::torch_randn(10, 1)
    target <- torch::torch_randn(10, 1)

    loss <- loss_fn(input, target)
    expect_s3_class(loss, "torch_tensor")
  }
})
