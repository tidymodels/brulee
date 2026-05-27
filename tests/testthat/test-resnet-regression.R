test_that("resnet regression - matrix interface", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")
  skip_on_cran()

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
  skip_on_cran()

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
  skip_on_cran()

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
  skip_on_cran()

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
  skip_on_cran()

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
  skip_on_cran()

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
  skip_on_cran()

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
  skip_on_cran()

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
  skip_on_cran()

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

test_that("summary.brulee_resnet prints layers, skips, and totals", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")
  skip_on_cran()

  set.seed(1)
  n <- 100
  ames_x <- matrix(rnorm(n * 3), ncol = 3)
  colnames(ames_x) <- c("x1", "x2", "x3")
  ames_y <- ames_x[, 1] + 2 * ames_x[, 2] + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_resnet(
    x = ames_x,
    y = ames_y,
    hidden_units = c(8, 4, 6),
    bottleneck_units = c(5, 3, 4),
    residual_at = c(2, 3),
    epochs = 2,
    verbose = FALSE
  )

  out <- capture.output(result <- summary(fit))

  expect_identical(result, fit)
  expect_true(any(grepl("Residual network architecture", out)))
  expect_true(any(grepl("Residual group 1 \\(blocks 1-2", out)))
  expect_true(any(grepl("Residual group 2 \\(block 3", out)))
  expect_true(any(grepl("\\+ skip:", out)))
  expect_true(any(grepl("BatchNorm1d\\(", out)))
  expect_true(any(grepl("Linear\\(.+->.+\\)", out)))
  expect_true(any(grepl("Output head", out)))

  module <- brulee:::revive_model(fit$model_obj)
  total <- sum(vapply(module$parameters, function(p) p$numel(), integer(1)))
  expect_true(any(grepl(
    paste0("Total parameters: ", format(total, big.mark = ",")),
    out
  )))
})

test_that("summary.brulee_resnet handles no-residual and multinomial cases", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")
  skip_on_cran()

  set.seed(1)
  n <- 80
  ames_x <- matrix(rnorm(n * 3), ncol = 3)
  colnames(ames_x) <- c("x1", "x2", "x3")
  ames_y <- ames_x[, 1] + ames_x[, 2] + rnorm(n, sd = 0.1)

  set.seed(1)
  fit_no_skip <- brulee_resnet(
    x = ames_x,
    y = ames_y,
    hidden_units = c(6, 4),
    bottleneck_units = c(4, 3),
    residual_at = integer(0),
    epochs = 2,
    verbose = FALSE
  )

  out_no_skip <- capture.output(summary(fit_no_skip))
  expect_true(any(grepl("no residual connections", out_no_skip)))
  expect_false(any(grepl("\\+ skip", out_no_skip)))

  set.seed(1)
  y_cls <- factor(sample(letters[1:3], n, replace = TRUE))

  set.seed(1)
  fit_cls <- brulee_resnet(
    x = ames_x,
    y = y_cls,
    hidden_units = c(6, 4),
    bottleneck_units = c(4, 3),
    epochs = 2,
    verbose = FALSE
  )

  out_cls <- capture.output(summary(fit_cls))
  expect_true(any(grepl("Softmax", out_cls)))
  expect_true(any(grepl("output dim: 3", out_cls)))
})

test_that("resnet block structure follows Gorishniy et al. 2021", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")
  skip_on_cran()

  set.seed(1)
  n <- 50
  ames_x <- matrix(rnorm(n * 2), ncol = 2)
  colnames(ames_x) <- c("x1", "x2")
  ames_y <- ames_x[, 1] + 2 * ames_x[, 2] + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_resnet(
    x = ames_x,
    y = ames_y,
    hidden_units = 4,
    bottleneck_units = 6,
    epochs = 2,
    verbose = FALSE
  )

  module <- brulee:::revive_model(fit$model_obj)
  block <- module$layers[[1]]

  expect_equal(
    names(block$children),
    c("bn", "linear1", "act", "dropout1", "linear2", "dropout2")
  )
  expect_true(inherits(block$bn, "nn_batch_norm1d"))
  expect_true(inherits(block$linear1, "nn_linear"))
  expect_true(inherits(block$dropout1, "nn_dropout"))
  expect_true(inherits(block$linear2, "nn_linear"))
  expect_true(inherits(block$dropout2, "nn_dropout"))
})
