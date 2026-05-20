test_that("rln regression - matrix interface", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  x <- matrix(rnorm(n * 2), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- x[, 1] + 2 * x[, 2] + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_rln(
    x = x,
    y = y,
    hidden_units = 4L,
    epochs = 5L,
    batch_size = 32L,
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_rln")
  expect_true(inherits(fit$model_obj, "raw"))
  expect_true(is.list(fit$estimates))
  expect_true(is.numeric(fit$loss))
  expect_true(is.integer(fit$best_epoch))

  pred <- predict(fit, x)
  expect_s3_class(pred, "tbl_df")
  expect_equal(nrow(pred), nrow(x))
  expect_true(".pred" %in% names(pred))

  coefs <- coef(fit)
  expect_true(is.list(coefs))
})

test_that("rln regression - data.frame interface", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + 2 * df$x2 + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_rln(
    x = df[, c("x1", "x2")],
    y = df$y,
    hidden_units = 4L,
    epochs = 3L,
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_rln")

  pred <- predict(fit, df)
  expect_equal(nrow(pred), nrow(df))
})

test_that("rln regression - formula interface", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + 2 * df$x2 + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_rln(
    y ~ x1 + x2,
    data = df,
    hidden_units = 4L,
    epochs = 3L,
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_rln")

  pred <- predict(fit, df)
  expect_equal(nrow(pred), nrow(df))
})

test_that("rln regression - recipe interface", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")

  library(recipes)

  set.seed(1)
  n <- 100
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + 2 * df$x2 + rnorm(n, sd = 0.1)

  rec <- recipe(y ~ ., data = df) |>
    step_normalize(all_numeric_predictors())

  set.seed(1)
  fit <- brulee_rln(
    rec,
    data = df,
    hidden_units = 4L,
    epochs = 3L,
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_rln")

  pred <- predict(fit, df)
  expect_equal(nrow(pred), nrow(df))
})

test_that("rln regression - epoch parameter", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  x <- matrix(rnorm(n * 2), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- x[, 1] + 2 * x[, 2] + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_rln(
    x = x,
    y = y,
    hidden_units = 4L,
    epochs = 5L,
    verbose = FALSE
  )

  pred1 <- predict(fit, x, epoch = 1)
  pred2 <- predict(fit, x, epoch = 2)
  expect_false(identical(pred1, pred2))

  coef1 <- coef(fit, epoch = 1)
  coef2 <- coef(fit, epoch = 2)
  expect_false(identical(coef1, coef2))
})

test_that("rln print method works", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  x <- matrix(rnorm(n * 2), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- x[, 1] + 2 * x[, 2] + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_rln(
    x = x,
    y = y,
    hidden_units = 4L,
    epochs = 2L,
    verbose = FALSE
  )

  print_output <- capture.output(capture.output(print(fit), type = "message"))
  expect_true(any(grepl("Regularization Learning Network", print_output)))
})

test_that("rln autoplot works", {
  skip_if_not_installed("torch")
  skip_if_not_installed("ggplot2")

  set.seed(1)
  n <- 50
  x <- matrix(rnorm(n * 2), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- x[, 1] + 2 * x[, 2] + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_rln(
    x = x,
    y = y,
    hidden_units = 4L,
    epochs = 3L,
    verbose = FALSE
  )

  p <- autoplot(fit)
  expect_s3_class(p, "ggplot")
})

test_that("rln argument validation", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  x <- matrix(rnorm(n * 2), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- x[, 1] + 2 * x[, 2] + rnorm(n, sd = 0.1)

  expect_error(
    brulee_rln(x, y, penalty_type = "L3", epochs = 2L),
    "penalty_type"
  )

  expect_error(
    brulee_rln(x, y, activation = "banana", epochs = 2L),
    "activation"
  )

  expect_error(
    brulee_rln(x, y, step_rate = -1, epochs = 2L),
    "step_rate"
  )
})

test_that("rln rejects factor outcomes", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  x <- matrix(rnorm(n * 2), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- factor(sample(c("a", "b"), n, replace = TRUE))

  expect_error(
    brulee_rln(x, y, hidden_units = 4L, epochs = 2L),
    "numeric outcomes"
  )
})

test_that("predict call threading surfaces predict() not the bridge", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  x <- matrix(rnorm(n * 2), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- x[, 1] + 2 * x[, 2] + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_rln(
    x = x,
    y = y,
    hidden_units = 4L,
    epochs = 3L,
    verbose = FALSE
  )

  cnd <- rlang::catch_cnd(predict(fit, x, epoch = 9999), classes = "warning")
  expect_match(conditionMessage(cnd), "last epoch")
  expect_no_match(deparse(conditionCall(cnd)), "bridge")
})

test_that("rln stores parameters correctly", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  x <- matrix(rnorm(n * 2), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- x[, 1] + 2 * x[, 2] + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_rln(
    x = x,
    y = y,
    hidden_units = 6L,
    penalty_type = "L2",
    penalty_average = 1e-8,
    step_rate = 1e5,
    activation = "tanh",
    epochs = 3L,
    verbose = FALSE
  )

  expect_equal(fit$parameters$hidden_units, 6L)
  expect_equal(fit$parameters$penalty_type, "L2")
  expect_equal(fit$parameters$penalty_average, 1e-8)
  expect_equal(fit$parameters$step_rate, 1e5)
  expect_equal(fit$parameters$activation, "tanh")
})
