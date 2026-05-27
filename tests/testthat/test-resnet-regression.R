test_that("resnet regression - matrix interface", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")

  # Simple regression test
  set.seed(1)
  n <- 100
  x <- matrix(rnorm(n * 2), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- x[, 1] + 2 * x[, 2] + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_resnet(
    x = x,
    y = y,
    hidden_units = 2,
    num_layers = 2,
    bottleneck_units = 5,
    epochs = 5,
    batch_size = 32,
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_resnet")
  expect_true(inherits(fit$model_obj, "raw"))
  expect_true(is.list(fit$estimates))
  expect_true(is.numeric(fit$loss))
  expect_true(is.integer(fit$best_epoch))

  # Test prediction
  pred <- predict(fit, x)
  expect_s3_class(pred, "tbl_df")
  expect_equal(nrow(pred), nrow(x))
  expect_true(".pred" %in% names(pred))

  # Test coef
  coefs <- coef(fit)
  expect_true(is.list(coefs))
})

test_that("resnet regression - data.frame interface", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n)
  )
  df$y <- df$x1 + 2 * df$x2 + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_resnet(
    x = df[, c("x1", "x2")],
    y = df$y,
    hidden_units = 2,
    num_layers = 1,
    bottleneck_units = 5,
    epochs = 3,
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_resnet")

  pred <- predict(fit, df)
  expect_equal(nrow(pred), nrow(df))
})

test_that("resnet regression - formula interface", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n)
  )
  df$y <- df$x1 + 2 * df$x2 + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_resnet(
    y ~ x1 + x2,
    data = df,
    hidden_units = 2,
    num_layers = 2,
    bottleneck_units = 5,
    epochs = 3,
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_resnet")

  pred <- predict(fit, df)
  expect_equal(nrow(pred), nrow(df))
})

test_that("resnet regression - recipe interface", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")

  library(recipes)

  set.seed(1)
  n <- 100
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n)
  )
  df$y <- df$x1 + 2 * df$x2 + rnorm(n, sd = 0.1)

  rec <- recipe(y ~ ., data = df) %>%
    step_normalize(all_numeric_predictors())

  set.seed(1)
  fit <- brulee_resnet(
    rec,
    data = df,
    hidden_units = 2,
    num_layers = 2,
    bottleneck_units = 5,
    epochs = 3,
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_resnet")

  pred <- predict(fit, df)
  expect_equal(nrow(pred), nrow(df))
})

test_that("resnet regression - epoch parameter", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  x <- matrix(rnorm(n * 2), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- x[, 1] + 2 * x[, 2] + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_resnet(
    x = x,
    y = y,
    hidden_units = 2,
    num_layers = 2,
    bottleneck_units = 5,
    epochs = 5,
    verbose = FALSE
  )

  # Test prediction with different epochs
  pred1 <- predict(fit, x, epoch = 1)
  pred2 <- predict(fit, x, epoch = 2)

  expect_false(identical(pred1, pred2))

  # Test coef with different epochs
  coef1 <- coef(fit, epoch = 1)
  coef2 <- coef(fit, epoch = 2)

  expect_false(identical(coef1, coef2))
})

test_that("resnet print method works", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  x <- matrix(rnorm(n * 2), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- x[, 1] + 2 * x[, 2] + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_resnet(
    x = x,
    y = y,
    hidden_units = 2,
    num_layers = 2,
    bottleneck_units = 5,
    epochs = 2,
    verbose = FALSE
  )

  print_output <- capture.output(capture.output(print(fit), type = "message"))
  expect_true(any(grepl("Residual network", print_output)))
  expect_true(any(grepl("Bottleneck", print_output)))
})

test_that("resnet autoplot works", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")
  skip_if_not_installed("ggplot2")

  set.seed(1)
  n <- 50
  x <- matrix(rnorm(n * 2), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- x[, 1] + 2 * x[, 2] + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_resnet(
    x = x,
    y = y,
    hidden_units = 2,
    num_layers = 2,
    bottleneck_units = 5,
    epochs = 3,
    verbose = FALSE
  )

  p <- autoplot(fit)
  expect_s3_class(p, "ggplot")
})

test_that("resnet argument validation", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  x <- matrix(rnorm(n * 2), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- x[, 1] + 2 * x[, 2] + rnorm(n, sd = 0.1)

  # bottleneck_units must be >= 2
  expect_error(
    brulee_resnet(
      x,
      y,
      hidden_units = c(5, 10),
      bottleneck_units = c(1, 1),
      epochs = 2
    ),
    "bottleneck_units"
  )

  # bottleneck_units and hidden_units lengths must match
  expect_error(
    brulee_resnet(
      x,
      y,
      hidden_units = c(5, 10),
      bottleneck_units = c(3, 4, 5),
      epochs = 2
    ),
    "bottleneck_units.*hidden_units"
  )

  # residual_at values must be valid layer indices
  expect_error(
    brulee_resnet(
      x,
      y,
      hidden_units = c(5, 10),
      bottleneck_units = c(3, 4),
      residual_at = 5,
      epochs = 2
    ),
    "residual_at"
  )
})

test_that("resnet with vector hidden_units and bottleneck_units", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  x <- matrix(rnorm(n * 2), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- x[, 1] + 2 * x[, 2] + rnorm(n, sd = 0.1)

  # Test with different hidden_units and bottleneck_units per layer
  set.seed(1)
  fit <- brulee_resnet(
    x = x,
    y = y,
    hidden_units = c(8, 10, 12), # Different dimensions per layer
    bottleneck_units = c(5, 6, 7), # Different output widths
    residual_at = 3, # Single residual block
    epochs = 3,
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_resnet")
  expect_equal(fit$parameters$residual_at, 3)

  # Verify prediction works
  pred <- predict(fit, x)
  expect_equal(nrow(pred), nrow(x))

  # Test with single layer
  set.seed(1)
  fit2 <- brulee_resnet(
    x = x,
    y = y,
    hidden_units = 10,
    bottleneck_units = 5,
    epochs = 3,
    verbose = FALSE
  )

  expect_s3_class(fit2, "brulee_resnet")
  pred2 <- predict(fit2, x)
  expect_equal(nrow(pred2), nrow(x))
})
