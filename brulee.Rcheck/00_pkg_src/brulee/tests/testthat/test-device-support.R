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

test_that("linear_reg trains with device parameter", {
  skip_if(!torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  dat <- modeldata::sim_regression(200, method = "sapp_2014_1")

  fit <- brulee_linear_reg(
    x = dat[, -1],
    y = dat$outcome,
    epochs = 3,
    device = "cpu",
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_linear_reg")
  expect_equal(fit$device, "cpu")
})

test_that("linear_reg prediction works with device", {
  skip_if(!torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  dat <- modeldata::sim_regression(200, method = "sapp_2014_1")

  fit <- brulee_linear_reg(
    x = dat[, -1],
    y = dat$outcome,
    epochs = 3,
    device = "cpu",
    verbose = FALSE
  )

  pred <- predict(fit, dat[1:10, -1])
  expect_equal(nrow(pred), 10)
  expect_true(".pred" %in% names(pred))
})

test_that("linear_reg prediction falls back to cpu when device unavailable", {
  skip_if(!torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  dat <- modeldata::sim_regression(200, method = "sapp_2014_1")

  fit <- brulee_linear_reg(
    x = dat[, -1],
    y = dat$outcome,
    epochs = 3,
    device = "cpu",
    verbose = FALSE
  )

  # Mock as if trained on CUDA
  fit$device <- "cuda"

  expect_warning(
    pred <- predict(fit, dat[1:10, -1]),
    "Model was trained on.*cuda.*not available"
  )
  expect_equal(nrow(pred), 10)
})
