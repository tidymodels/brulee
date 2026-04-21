test_that("set_optimizer returns correct optimizer types", {
  skip_if_not(torch::torch_is_installed())

  # Create a simple model for testing
  model <- torch::nn_linear(5, 1)

  # Test each optimizer type
  opt_sgd <- brulee:::set_optimizer("SGD", model, 0.1, 0.9, 0.01, 0)
  expect_s3_class(opt_sgd, "optim_sgd")

  opt_lbfgs <- brulee:::set_optimizer("LBFGS", model, 0.1, 0.9, 0.01, 0)
  expect_s3_class(opt_lbfgs, "optim_lbfgs")

  opt_adamw <- brulee:::set_optimizer("ADAMw", model, 0.1, 0.9, 0.01, 0)
  expect_s3_class(opt_adamw, "optim_adamw")

  opt_rmsprop <- brulee:::set_optimizer("RMSprop", model, 0.1, 0.9, 0.01, 0)
  expect_s3_class(opt_rmsprop, "optim_rmsprop")

  opt_adadelta <- brulee:::set_optimizer("Adadelta", model, 0.1, 0.9, 0.01, 0)
  expect_s3_class(opt_adadelta, "optim_adadelta")

  opt_adagrad <- brulee:::set_optimizer("Adagrad", model, 0.1, 0.9, 0.01, 0)
  expect_s3_class(opt_adagrad, "optim_adagrad")

  # Test invalid optimizer
  expect_error(
    brulee:::set_optimizer("Invalid", model, 0.1, 0.9, 0.01, 0),
    "Unsupported optimizer"
  )
})

test_that("set_optimizer sets weight_decay correctly for pure L2 (mixture = 0)", {
  skip_if_not(torch::torch_is_installed())

  model <- torch::nn_linear(5, 1)
  penalty_val <- 0.123

  # SGD with mixture = 0 should use weight_decay
  opt_sgd <- brulee:::set_optimizer(
    "SGD",
    model,
    0.1,
    0.9,
    penalty_val,
    mixture = 0
  )
  expect_equal(opt_sgd$defaults$weight_decay, penalty_val)

  # RMSprop with mixture = 0 should use weight_decay
  opt_rmsprop <- brulee:::set_optimizer(
    "RMSprop",
    model,
    0.1,
    0.9,
    penalty_val,
    mixture = 0
  )
  expect_equal(opt_rmsprop$defaults$weight_decay, penalty_val)

  # ADAMw with mixture = 0 should use weight_decay
  opt_adamw <- brulee:::set_optimizer(
    "ADAMw",
    model,
    0.1,
    0.9,
    penalty_val,
    mixture = 0
  )
  expect_equal(opt_adamw$defaults$weight_decay, penalty_val)

  # Adadelta with mixture = 0 should use weight_decay
  opt_adadelta <- brulee:::set_optimizer(
    "Adadelta",
    model,
    0.1,
    0.9,
    penalty_val,
    mixture = 0
  )
  expect_equal(opt_adadelta$defaults$weight_decay, penalty_val)

  # Adagrad with mixture = 0 should use weight_decay
  opt_adagrad <- brulee:::set_optimizer(
    "Adagrad",
    model,
    0.1,
    0.9,
    penalty_val,
    mixture = 0
  )
  expect_equal(opt_adagrad$defaults$weight_decay, penalty_val)

  # LBFGS should NOT use weight_decay (doesn't support it)
  opt_lbfgs <- brulee:::set_optimizer(
    "LBFGS",
    model,
    0.1,
    0.9,
    penalty_val,
    mixture = 0
  )
  # LBFGS doesn't have weight_decay parameter
  expect_false("weight_decay" %in% names(opt_lbfgs$defaults))
})

test_that("set_optimizer sets weight_decay to 0 for L1 penalty (mixture = 1)", {
  skip_if_not(torch::torch_is_installed())

  model <- torch::nn_linear(5, 1)
  penalty_val <- 0.456

  # SGD with mixture = 1 should NOT use weight_decay
  opt_sgd <- brulee:::set_optimizer(
    "SGD",
    model,
    0.1,
    0.9,
    penalty_val,
    mixture = 1
  )
  expect_equal(opt_sgd$defaults$weight_decay, 0)

  # RMSprop with mixture = 1 should NOT use weight_decay
  opt_rmsprop <- brulee:::set_optimizer(
    "RMSprop",
    model,
    0.1,
    0.9,
    penalty_val,
    mixture = 1
  )
  expect_equal(opt_rmsprop$defaults$weight_decay, 0)

  # Adadelta with mixture = 1 should NOT use weight_decay
  opt_adadelta <- brulee:::set_optimizer(
    "Adadelta",
    model,
    0.1,
    0.9,
    penalty_val,
    mixture = 1
  )
  expect_equal(opt_adadelta$defaults$weight_decay, 0)

  # Adagrad with mixture = 1 should NOT use weight_decay
  opt_adagrad <- brulee:::set_optimizer(
    "Adagrad",
    model,
    0.1,
    0.9,
    penalty_val,
    mixture = 1
  )
  expect_equal(opt_adagrad$defaults$weight_decay, 0)

  # LBFGS doesn't have weight_decay parameter at all
  opt_lbfgs <- brulee:::set_optimizer(
    "LBFGS",
    model,
    0.1,
    0.9,
    penalty_val,
    mixture = 1
  )
  expect_false("weight_decay" %in% names(opt_lbfgs$defaults))
})

test_that("set_optimizer sets weight_decay to 0 for elastic net (0 < mixture < 1)", {
  skip_if_not(torch::torch_is_installed())

  model <- torch::nn_linear(5, 1)
  penalty_val <- 0.789

  # SGD with mixture = 0.5 should NOT use weight_decay
  opt_sgd <- brulee:::set_optimizer(
    "SGD",
    model,
    0.1,
    0.9,
    penalty_val,
    mixture = 0.5
  )
  expect_equal(opt_sgd$defaults$weight_decay, 0)

  # RMSprop with mixture = 0.3 should NOT use weight_decay
  opt_rmsprop <- brulee:::set_optimizer(
    "RMSprop",
    model,
    0.1,
    0.9,
    penalty_val,
    mixture = 0.3
  )
  expect_equal(opt_rmsprop$defaults$weight_decay, 0)

  # Adadelta with mixture = 0.7 should NOT use weight_decay
  opt_adadelta <- brulee:::set_optimizer(
    "Adadelta",
    model,
    0.1,
    0.9,
    penalty_val,
    mixture = 0.7
  )
  expect_equal(opt_adadelta$defaults$weight_decay, 0)
})

test_that("set_optimizer ADAMw always uses weight_decay", {
  skip_if_not(torch::torch_is_installed())

  model <- torch::nn_linear(5, 1)
  penalty_val <- 0.111

  # ADAMw with mixture = 0 should use weight_decay
  opt_adamw_0 <- brulee:::set_optimizer(
    "ADAMw",
    model,
    0.1,
    0.9,
    penalty_val,
    mixture = 0
  )
  expect_equal(opt_adamw_0$defaults$weight_decay, penalty_val)

  # ADAMw with mixture = 0.5 should use weight_decay
  # (Note: check_mixture would convert this to 0 before calling set_optimizer,
  # but set_optimizer itself still uses weight_decay for ADAMw regardless)
  opt_adamw_05 <- brulee:::set_optimizer(
    "ADAMw",
    model,
    0.1,
    0.9,
    penalty_val,
    mixture = 0.5
  )
  expect_equal(opt_adamw_05$defaults$weight_decay, penalty_val)

  # ADAMw with mixture = 1 should use weight_decay
  opt_adamw_1 <- brulee:::set_optimizer(
    "ADAMw",
    model,
    0.1,
    0.9,
    penalty_val,
    mixture = 1
  )
  expect_equal(opt_adamw_1$defaults$weight_decay, penalty_val)
})

test_that("set_optimizer respects other parameters", {
  skip_if_not(torch::torch_is_installed())

  model <- torch::nn_linear(5, 1)

  # Test learning rate
  opt <- brulee:::set_optimizer("SGD", model, learn_rate = 0.123, 0.9, 0.01, 0)
  expect_equal(opt$defaults$lr, 0.123)

  # Test momentum
  opt <- brulee:::set_optimizer("SGD", model, 0.1, momentum = 0.789, 0.01, 0)
  expect_equal(opt$defaults$momentum, 0.789)

  # Test nesterov is enabled when momentum > 0
  opt <- brulee:::set_optimizer("SGD", model, 0.1, momentum = 0.5, 0.01, 0)
  expect_true(opt$defaults$nesterov)

  # Test nesterov is disabled when momentum = 0
  opt <- brulee:::set_optimizer("SGD", model, 0.1, momentum = 0, 0.01, 0)
  expect_false(opt$defaults$nesterov)

  # Test LBFGS history_size
  opt <- brulee:::set_optimizer("LBFGS", model, 0.1, 0.9, 0.01, 0)
  expect_equal(opt$defaults$history_size, 1)
})

test_that("set_optimizer weight_decay with penalty = 0", {
  skip_if_not(torch::torch_is_installed())

  model <- torch::nn_linear(5, 1)

  # All optimizers (except LBFGS) should have weight_decay = 0 when penalty = 0
  opt_sgd <- brulee:::set_optimizer(
    "SGD",
    model,
    0.1,
    0.9,
    penalty = 0,
    mixture = 0
  )
  expect_equal(opt_sgd$defaults$weight_decay, 0)

  opt_adamw <- brulee:::set_optimizer(
    "ADAMw",
    model,
    0.1,
    0.9,
    penalty = 0,
    mixture = 0
  )
  expect_equal(opt_adamw$defaults$weight_decay, 0)

  opt_rmsprop <- brulee:::set_optimizer(
    "RMSprop",
    model,
    0.1,
    0.9,
    penalty = 0,
    mixture = 0
  )
  expect_equal(opt_rmsprop$defaults$weight_decay, 0)
})

test_that("set_optimizer handles default mixture = 0", {
  skip_if_not(torch::torch_is_installed())

  model <- torch::nn_linear(5, 1)
  penalty_val <- 0.5

  # When mixture is not provided, it defaults to 0
  # So weight_decay should be used (except for LBFGS)
  opt_sgd <- brulee:::set_optimizer("SGD", model, 0.1, 0.9, penalty_val)
  expect_equal(opt_sgd$defaults$weight_decay, penalty_val)

  opt_adamw <- brulee:::set_optimizer("ADAMw", model, 0.1, 0.9, penalty_val)
  expect_equal(opt_adamw$defaults$weight_decay, penalty_val)
})
