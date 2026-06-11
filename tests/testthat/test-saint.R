# ------------------------------------------------------------------------------
# Regression tests

test_that("saint regression - formula interface", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    g = factor(sample(letters[1:3], n, replace = TRUE))
  )
  df$y <- df$x1 + 2 * df$x2 + rnorm(n, sd = 0.5)

  set.seed(1)

  fit <- brulee_saint(y ~ ., data = df, epochs = 5, verbose = FALSE)

  expect_s3_class(fit, "brulee_saint")
  expect_true(inherits(fit$model_obj, "raw"))
  expect_true(is.list(fit$estimates))
  expect_true(is.numeric(fit$loss))
  expect_true(is.integer(fit$best_epoch))

  pred <- predict(fit, df)
  expect_s3_class(pred, "tbl_df")
  expect_equal(nrow(pred), n)
  expect_true(".pred" %in% names(pred))
})

test_that("saint regression - data.frame interface (numeric only)", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  x <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  y <- data.frame(outcome = x$x1 + rnorm(n, sd = 0.1))

  set.seed(1)
  fit <- brulee_saint(x = x, y = y, epochs = 3, verbose = FALSE)

  expect_s3_class(fit, "brulee_saint")
  expect_equal(fit$dims$p_cat, 0L)
  expect_equal(fit$dims$p_cont, 2L)

  pred <- predict(fit, x)
  expect_equal(nrow(pred), n)
})

test_that("saint regression - matrix interface", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  x <- matrix(rnorm(n * 3), ncol = 3)
  colnames(x) <- c("a", "b", "c")
  y <- x[, 1] + x[, 2] + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_saint(x = x, y = y, epochs = 3, verbose = FALSE)

  expect_s3_class(fit, "brulee_saint")
  pred <- predict(fit, x)
  expect_equal(nrow(pred), n)
})

test_that("saint regression - recipe interface", {
  skip_if_not_installed("torch")
  skip_if_not_installed("recipes")

  set.seed(1)
  n <- 80
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    g = factor(sample(c("a", "b"), n, replace = TRUE)),
    y = rnorm(n)
  )

  rec <- recipes::recipe(y ~ ., data = df)
  set.seed(1)
  fit <- brulee_saint(rec, data = df, epochs = 3, verbose = FALSE)

  expect_s3_class(fit, "brulee_saint")
  pred <- predict(fit, df)
  expect_equal(nrow(pred), n)
})

# ------------------------------------------------------------------------------
# Classification tests

test_that("saint binary classification", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 120
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    y = factor(sample(c("a", "b"), n, replace = TRUE))
  )

  set.seed(1)
  fit <- brulee_saint(y ~ ., data = df, epochs = 5, verbose = FALSE)

  expect_s3_class(fit, "brulee_saint")

  pred_class <- predict(fit, df, type = "class")
  expect_equal(nrow(pred_class), n)
  expect_true(".pred_class" %in% names(pred_class))
  expect_s3_class(pred_class$.pred_class, "factor")
  expect_equal(levels(pred_class$.pred_class), c("a", "b"))

  pred_prob <- predict(fit, df, type = "prob")
  expect_equal(nrow(pred_prob), n)
  expect_true(all(c(".pred_a", ".pred_b") %in% names(pred_prob)))
  expect_true(all(pred_prob$.pred_a >= 0 & pred_prob$.pred_a <= 1))
})

test_that("saint multiclass classification", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 150
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    y = factor(sample(c("cat", "dog", "bird"), n, replace = TRUE))
  )

  set.seed(1)
  fit <- brulee_saint(y ~ ., data = df, epochs = 5, verbose = FALSE)

  pred_class <- predict(fit, df, type = "class")
  expect_equal(nrow(pred_class), n)
  expect_equal(levels(pred_class$.pred_class), c("bird", "cat", "dog"))

  pred_prob <- predict(fit, df, type = "prob")
  expect_equal(ncol(pred_prob), 3)
  row_sums <- rowSums(as.matrix(pred_prob))
  expect_true(all(abs(row_sums - 1) < 1e-5))
})

# ------------------------------------------------------------------------------
# Attention type tests

test_that("saint with col attention type", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_saint(
    y ~ ., data = df, epochs = 3,
    attention_type = "column",
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_saint")
  expect_equal(fit$parameters$attention_type, "column")

  pred <- predict(fit, df)
  expect_equal(nrow(pred), n)
})

test_that("saint with row attention type", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_saint(
    y ~ ., data = df, epochs = 3,
    attention_type = "row",
    num_attn_blocks = 1L,
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_saint")
  expect_equal(fit$parameters$attention_type, "row")

  pred <- predict(fit, df)
  expect_equal(nrow(pred), n)
})

test_that("saint with colrow attention type (default)", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_saint(
    y ~ ., data = df, epochs = 3,
    attention_type = "both",
    num_attn_blocks = 2L,
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_saint")
  expect_equal(fit$parameters$attention_type, "both")

  pred <- predict(fit, df)
  expect_equal(nrow(pred), n)
})

# ------------------------------------------------------------------------------
# row_attention_on_predict tests

test_that("saint row_attention_on_predict=FALSE (default) gives batch-independent predictions", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(2)
  fit <- brulee_saint(
    y ~ ., data = df, epochs = 5,
    attention_type = "both", num_attn_blocks = 2L,
    verbose = FALSE
  )

  pred_full <- predict(fit, df)
  pred_single <- predict(fit, df[1, , drop = FALSE])
  expect_equal(pred_full$.pred[1], pred_single$.pred[1], tolerance = 1e-10)

  pred_subset <- predict(fit, df[c(1, 50, 70), ])
  expect_equal(pred_full$.pred[1], pred_subset$.pred[1], tolerance = 1e-10)
})

test_that("saint row_attention_on_predict=TRUE gives batch-dependent predictions", {
  skip_if_not_installed("torch")

  set.seed(1)

  n <- 80
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(2)
  fit <- brulee_saint(
    y ~ ., data = df, epochs = 5,
    attention_type = "both", num_attn_blocks = 2L,
    row_attention_on_predict = TRUE, verbose = FALSE
  )

  pred_full <- predict(fit, df)
  pred_single <- predict(fit, df[1, , drop = FALSE])
  expect_false(
    abs(pred_full$.pred[1] - pred_single$.pred[1]) < 1e-10
  )
})

# ------------------------------------------------------------------------------
# Feature handling tests

test_that("saint with only categorical predictors", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(
    g1 = factor(sample(letters[1:4], n, replace = TRUE)),
    g2 = factor(sample(c("x", "y", "z"), n, replace = TRUE)),
    y = rnorm(n)
  )

  set.seed(1)
  fit <- brulee_saint(y ~ ., data = df, epochs = 3, verbose = FALSE)

  expect_s3_class(fit, "brulee_saint")
  expect_equal(fit$dims$p_cat, 2L)
  expect_equal(fit$dims$p_cont, 0L)

  pred <- predict(fit, df)
  expect_equal(nrow(pred), n)
})

test_that("saint with only continuous predictors", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_saint(y ~ ., data = df, epochs = 3, verbose = FALSE)

  expect_equal(fit$dims$p_cat, 0L)
  expect_equal(fit$dims$p_cont, 3L)

  pred <- predict(fit, df)
  expect_equal(nrow(pred), n)
})

test_that("saint with mixed predictors", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(
    x1 = rnorm(n),
    g1 = factor(sample(letters[1:3], n, replace = TRUE)),
    x2 = rnorm(n),
    g2 = factor(sample(c("yes", "no"), n, replace = TRUE)),
    y = rnorm(n)
  )

  set.seed(1)
  fit <- brulee_saint(y ~ ., data = df, epochs = 3, verbose = FALSE)

  expect_equal(fit$dims$p_cat, 2L)
  expect_equal(fit$dims$p_cont, 2L)
  expect_equal(fit$dims$p, 4L)
})

# ------------------------------------------------------------------------------
# Hidden layer tests

test_that("saint with hidden layers", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_saint(
    y ~ ., data = df, epochs = 5,
    hidden_units = c(64L, 32L),
    hidden_activations = c("relu", "relu"),
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_saint")
  expect_equal(fit$parameters$hidden_units, c(64L, 32L))

  pred <- predict(fit, df)
  expect_equal(nrow(pred), n)
})

# ------------------------------------------------------------------------------
# Validation error tests

test_that("saint attention_type validation", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  expect_error(
    brulee_saint(y ~ ., data = df, attention_type = "invalid"),
    "attention_type"
  )
})

test_that("saint default method errors on unsupported types", {
  skip_if_not_installed("torch")

  expect_error(
    brulee_saint(list(a = 1)),
    "not defined"
  )
})

# ------------------------------------------------------------------------------
# Prediction tests

test_that("saint prediction with specific epoch", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_saint(y ~ ., data = df, epochs = 5, verbose = FALSE)

  pred_best <- predict(fit, df)
  pred_epoch1 <- predict(fit, df, epoch = 1)

  expect_equal(nrow(pred_best), n)
  expect_equal(nrow(pred_epoch1), n)
})

# ------------------------------------------------------------------------------
# Training tests

test_that("saint with validation = 0", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_saint(
    y ~ ., data = df, epochs = 3,
    validation = 0, verbose = FALSE
  )
  expect_s3_class(fit, "brulee_saint")
  expect_equal(fit$parameters$validation, 0)
})

test_that("saint with early stopping", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_saint(
    y ~ ., data = df, epochs = 100,
    stop_iter = 3, verbose = FALSE
  )
  expect_true(fit$best_epoch < 100)
})

# ------------------------------------------------------------------------------
# Print and autoplot tests

test_that("saint print method works", {
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  fit <- brulee_saint(y ~ ., data = df, epochs = 3, verbose = FALSE)

  stdout <- capture.output(
    msgs <- capture.output(print(fit), type = "message")
  )
  output <- c(stdout, msgs)
  expect_true(any(grepl("SAINT network", output)))
  expect_true(any(grepl("Attention type", output)))
})

test_that("saint autoplot works", {
  skip_if_not_installed("torch")
  skip_if_not_installed("ggplot2")

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  fit <- brulee_saint(y ~ ., data = df, epochs = 5, verbose = FALSE)

  p <- ggplot2::autoplot(fit)
  expect_s3_class(p, "ggplot")
})
