# Every model with an `epoch` argument must clamp an out-of-range value to the
# last epoch that was fit rather than erroring in `estimates[[epoch + 1]]` (#138).
#
# Two cases are pinned for each model:
#
#   * `epoch` past the end, which used to warn and then error anyway because
#     `last_epoch_note()` never returned a clamped value.
#   * `epoch` exactly equal to `length(estimates)`, which used to slip past the
#     `epoch > length(estimates)` guard with no warning at all and then error.
#     `estimates` holds epoch zero as its first element, so the largest valid
#     epoch is `length(estimates) - 1L`.
#
# `epochs = 3L` with the default `stop_iter = 5` cannot early-stop, so
# `length(estimates)` is a deterministic 4 and the snapshots are stable.

epoch_reg_data <- function(n = 100) {
  set.seed(1)
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + 2 * df$x2 + rnorm(n, sd = 0.1)
  df
}

epoch_cls_data <- function(n = 100, num_classes = 2) {
  set.seed(1)
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- factor(rep(letters[seq_len(num_classes)], length.out = n))
  df
}

# Fit an out-of-range epoch two ways and check both land on the last epoch.
expect_epoch_clamped <- function(fit, new_data) {
  last <- predict(fit, new_data, epoch = length(fit$estimates) - 1L)

  expect_snapshot(
    boundary <- predict(fit, new_data, epoch = length(fit$estimates))
  )
  expect_equal(boundary, last)

  expect_snapshot(past_end <- predict(fit, new_data, epoch = 10L))
  expect_equal(past_end, last)

  invisible(last)
}

# ------------------------------------------------------------------------------

test_that("brulee_mlp() clamps an out-of-range epoch", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  df <- epoch_reg_data()

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_mlp(
    y ~ .,
    data = df,
    epochs = 3L,
    verbose = FALSE,
    device = "cpu"
  )

  expect_epoch_clamped(fit, df)
})

test_that("brulee_linear_reg() clamps an out-of-range epoch", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  df <- epoch_reg_data()

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_linear_reg(
    y ~ .,
    data = df,
    epochs = 3L,
    verbose = FALSE,
    device = "cpu"
  )

  expect_epoch_clamped(fit, df)
})

test_that("brulee_logistic_reg() clamps an out-of-range epoch", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  df <- epoch_cls_data(num_classes = 2)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_logistic_reg(
    y ~ .,
    data = df,
    epochs = 3L,
    verbose = FALSE,
    device = "cpu"
  )

  expect_epoch_clamped(fit, df)
})

test_that("brulee_multinomial_reg() clamps an out-of-range epoch", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  df <- epoch_cls_data(num_classes = 3)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_multinomial_reg(
    y ~ .,
    data = df,
    epochs = 3L,
    verbose = FALSE,
    device = "cpu"
  )

  expect_epoch_clamped(fit, df)
})

test_that("brulee_resnet() clamps an out-of-range epoch", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  df <- epoch_reg_data()

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_resnet(
    y ~ .,
    data = df,
    hidden_units = 2L,
    num_layers = 2L,
    bottleneck_units = 5L,
    epochs = 3L,
    verbose = FALSE,
    device = "cpu"
  )

  expect_epoch_clamped(fit, df)
})

test_that("brulee_rln() clamps an out-of-range epoch", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  df <- epoch_reg_data()

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_rln(
    y ~ .,
    data = df,
    hidden_units = 4L,
    epochs = 3L,
    verbose = FALSE,
    device = "cpu"
  )

  expect_epoch_clamped(fit, df)
})

test_that("brulee_saint() clamps an out-of-range epoch", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  df <- epoch_reg_data()

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_saint(
    y ~ .,
    data = df,
    epochs = 3L,
    verbose = FALSE,
    device = "cpu"
  )

  expect_epoch_clamped(fit, df)
})

test_that("brulee_auto_int() clamps an out-of-range epoch", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  df <- epoch_reg_data()

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3L,
    verbose = FALSE,
    device = "cpu"
  )

  expect_epoch_clamped(fit, df)
})
