test_that("guess_brulee_device returns cpu when no GPU available", {
  skip_if(!torch::torch_is_installed())
  skip_if(
    torch::cuda_is_available() || torch::backends_mps_is_available(),
    message = "GPU available - skipping CPU-only test"
  )

  device <- brulee:::guess_brulee_device(NULL)
  expect_equal(device, "cpu")
})

test_that("guess_brulee_device returns input when not NULL", {
  skip_if(!torch::torch_is_installed())

  expect_equal(brulee:::guess_brulee_device("cpu"), "cpu")
  expect_equal(brulee:::guess_brulee_device("cuda"), "cuda")
  expect_equal(brulee:::guess_brulee_device("mps"), "mps")
  expect_equal(brulee:::guess_brulee_device("CPU"), "cpu") # case-insensitive
})

test_that("guess_brulee_device prefers cuda on non-mac platforms", {
  skip_if(!torch::torch_is_installed())
  skip_if(
    Sys.info()[["sysname"]] == "Darwin",
    message = "macOS prefers MPS, not CUDA"
  )
  skip_if(!torch::cuda_is_available(), message = "CUDA not available")

  device <- brulee:::guess_brulee_device(NULL)
  expect_equal(device, "cuda")
})

test_that("guess_brulee_device prefers mps on macOS", {
  skip_if(!torch::torch_is_installed())
  skip_if(
    Sys.info()[["sysname"]] != "Darwin",
    message = "MPS preference is macOS-only"
  )
  skip_if(!torch::backends_mps_is_available(), message = "MPS not available")

  device <- brulee:::guess_brulee_device(NULL)
  expect_equal(device, "mps")
})

test_that("guess_brulee_device falls back to cpu on macOS without mps", {
  skip_if(!torch::torch_is_installed())
  skip_if(
    Sys.info()[["sysname"]] != "Darwin",
    message = "macOS-only test"
  )
  skip_if(torch::backends_mps_is_available(), message = "MPS available")

  device <- brulee:::guess_brulee_device(NULL)
  expect_equal(device, "cpu")
})

test_that("validate_device accepts valid devices", {
  skip_if(!torch::torch_is_installed())

  expect_equal(brulee:::validate_device("cpu"), "cpu")
  expect_equal(brulee:::validate_device("cuda"), "cuda")
  expect_equal(brulee:::validate_device("mps"), "mps")
  expect_equal(brulee:::validate_device("CPU"), "cpu") # case-insensitive
  expect_equal(brulee:::validate_device("CUDA"), "cuda")
})

test_that("validate_device rejects invalid devices", {
  skip_if(!torch::torch_is_installed())

  expect_error(
    brulee:::validate_device("gpu"),
    "device.*must be one of"
  )
  expect_error(
    brulee:::validate_device("tpu"),
    "device.*must be one of"
  )
})

test_that("get_safe_device returns cpu immediately when device is cpu", {
  skip_if(!torch::torch_is_installed())

  device <- brulee:::get_safe_device("cpu")
  expect_equal(device, "cpu")
})

test_that("get_safe_device falls back to cpu with warning when device unavailable", {
  skip_if(!torch::torch_is_installed())
  skip_if(torch::cuda_is_available(), message = "CUDA is available")

  expect_warning(
    device <- brulee:::get_safe_device("cuda"),
    "Model was trained on.*cuda.*not available"
  )
  expect_equal(device, "cpu")
})

test_that("get_safe_device returns cuda when available", {
  skip_if(!torch::torch_is_installed())
  skip_if(!torch::cuda_is_available(), message = "CUDA not available")

  device <- brulee:::get_safe_device("cuda")
  expect_equal(device, "cuda")
})

test_that("get_safe_device returns mps when available", {
  skip_if(!torch::torch_is_installed())
  skip_if(!torch::backends_mps_is_available(), message = "MPS not available")

  device <- brulee:::get_safe_device("mps")
  expect_equal(device, "mps")
})

test_that("float_32 creates tensors on cpu by default", {
  skip_if(!torch::torch_is_installed())

  tensor <- brulee:::float_32(c(1, 2, 3))
  expect_equal(tensor$device$type, "cpu")
  expect_equal(as.character(tensor$dtype), "Float")
})

test_that("float_32 creates tensors on specified device", {
  skip_if(!torch::torch_is_installed())

  tensor <- brulee:::float_32(c(1, 2, 3), device = "cpu")
  expect_equal(tensor$device$type, "cpu")
})

test_that("float_32 respects device context from with_device", {
  skip_if(!torch::torch_is_installed())

  device_type <- torch::with_device(device = "cpu", {
    tensor <- brulee:::float_32(c(1, 2, 3))
    tensor$device$type
  })
  expect_equal(device_type, "cpu")
})

# ------------------------------------------------------------------------------
# MPS device tests

test_that("linear_reg trains on MPS", {
  skip_if(!torch::torch_is_installed())
  skip_if_not_installed("modeldata")
  skip_if(
    !torch::backends_mps_is_available(),
    message = "MPS not available"
  )

  set.seed(1)
  dat <- modeldata::sim_regression(200, method = "sapp_2014_1")

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_linear_reg(
    x = dat[, -1],
    y = dat$outcome,
    epochs = 5,
    device = "mps",
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_linear_reg")
  expect_equal(fit$device, "mps")

  pred <- predict(fit, dat[1:10, -1])
  expect_equal(nrow(pred), 10)
  expect_true(".pred" %in% names(pred))
})

test_that("logistic_reg trains on MPS", {
  skip_if(!torch::torch_is_installed())
  skip_if_not_installed("modeldata")
  skip_if(
    !torch::backends_mps_is_available(),
    message = "MPS not available"
  )

  set.seed(585)
  bin_tr <- modeldata::sim_logistic(500, ~ -1 - 3 * A + 5 * B)
  bin_te <- modeldata::sim_logistic(100, ~ -1 - 3 * A + 5 * B)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_logistic_reg(
    class ~ .,
    bin_tr,
    epochs = 5,
    device = "mps"
  )

  expect_s3_class(fit, "brulee_logistic_reg")
  expect_equal(fit$device, "mps")

  pred <- predict(fit, bin_te)
  expect_equal(nrow(pred), nrow(bin_te))
})

test_that("mlp regression trains on MPS", {
  skip_if(!torch::torch_is_installed())
  skip_if_not_installed("modeldata")
  skip_if(
    !torch::backends_mps_is_available(),
    message = "MPS not available"
  )

  set.seed(1)
  dat <- modeldata::sim_regression(500)

  set.seed(1)

  torch::torch_manual_seed(1)
  fit <- brulee_mlp(
    outcome ~ .,
    dat,
    epochs = 10,
    hidden_units = 5,
    device = "mps"
  )

  expect_s3_class(fit, "brulee_mlp")
  expect_equal(fit$device, "mps")

  pred <- predict(fit, dat[1:10, -1])
  expect_equal(nrow(pred), 10)
})

test_that("mlp classification trains on MPS", {
  skip_if(!torch::torch_is_installed())
  skip_if_not_installed("modeldata")
  skip_if(
    !torch::backends_mps_is_available(),
    message = "MPS not available"
  )

  set.seed(1)
  dat <- modeldata::sim_classification(500)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_mlp(
    class ~ .,
    dat,
    epochs = 10,
    hidden_units = 5,
    device = "mps"
  )

  expect_s3_class(fit, "brulee_mlp")
  expect_equal(fit$device, "mps")

  pred <- predict(fit, dat[1:10, -1])
  expect_equal(nrow(pred), 10)
  expect_true(".pred_class" %in% names(pred))
})

test_that("multinomial_reg trains on MPS", {
  skip_if(!torch::torch_is_installed())
  skip_if_not_installed("modeldata")
  skip_if(
    !torch::backends_mps_is_available(),
    message = "MPS not available"
  )

  set.seed(585)
  mnl_tr <- modeldata::sim_multinomial(
    500,
    ~ -0.5 + 0.6 * A,
    ~ .1 * B,
    ~ -0.6 * A + 0.50 * B
  )

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_multinomial_reg(
    class ~ .,
    mnl_tr,
    epochs = 5,
    device = "mps"
  )

  expect_s3_class(fit, "brulee_multinomial_reg")
  expect_equal(fit$device, "mps")

  pred <- predict(fit, mnl_tr[1:10, -3])
  expect_equal(nrow(pred), 10)
})

test_that("resnet trains on MPS", {
  skip_if(!torch::torch_is_installed())
  skip_if_not_installed("recipes")
  skip_if(
    !torch::backends_mps_is_available(),
    message = "MPS not available"
  )

  set.seed(1)
  n <- 100
  x <- matrix(rnorm(n * 3), ncol = 3)
  colnames(x) <- c("x1", "x2", "x3")
  y <- x[, 1] + 2 * x[, 2] + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_resnet(
    x = x,
    y = y,
    hidden_units = c(5, 3),
    bottleneck_units = c(4, 4),
    epochs = 5,
    device = "mps",
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_resnet")
  expect_equal(fit$device, "mps")

  pred <- predict(fit, x[1:10, ])
  expect_equal(nrow(pred), 10)
})

test_that("saint trains on MPS", {
  skip_if(!torch::torch_is_installed())
  skip_if(
    !torch::backends_mps_is_available(),
    message = "MPS not available"
  )

  set.seed(1)
  n <- 100
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    g = factor(sample(letters[1:3], n, replace = TRUE))
  )
  df$y <- df$x1 + 2 * df$x2 + rnorm(n, sd = 0.5)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_saint(
    y ~ .,
    data = df,
    epochs = 5,
    device = "mps",
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_saint")
  expect_equal(fit$device, "mps")

  pred <- predict(fit, df)
  expect_equal(nrow(pred), n)
})

test_that("auto_int trains on MPS", {
  skip_if(!torch::torch_is_installed())
  skip_if(
    !torch::backends_mps_is_available(),
    message = "MPS not available"
  )

  set.seed(1)
  n <- 100
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    g = factor(sample(letters[1:3], n, replace = TRUE))
  )
  df$y <- df$x1 + 2 * df$x2 + rnorm(n, sd = 0.5)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 5,
    device = "mps",
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_auto_int")
  expect_equal(fit$device, "mps")

  pred <- predict(fit, df)
  expect_equal(nrow(pred), n)
})

test_that("rln trains on MPS", {
  skip_if(!torch::torch_is_installed())
  skip_if(
    !torch::backends_mps_is_available(),
    message = "MPS not available"
  )

  set.seed(1)
  n <- 100
  x <- matrix(rnorm(n * 2), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- x[, 1] + 2 * x[, 2] + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_rln(
    x = x,
    y = y,
    hidden_units = 4L,
    epochs = 5L,
    device = "mps",
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_rln")
  expect_equal(fit$device, "mps")

  pred <- predict(fit, x[1:10, ])
  expect_equal(nrow(pred), 10)
})
