test_that("autoint regression with hidden layers - formula interface", {
  skip_if_not_installed("torch")
  skip_on_cran()

  set.seed(1)
  n <- 100
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    g = factor(sample(letters[1:3], n, replace = TRUE))
  )
  df$y <- df$x1 + 2 * df$x2 + rnorm(n, sd = 0.1)

  set.seed(1)

  fit <- brulee_auto_int(
    y ~ ., data = df,
    epochs = 5,
    hidden_units = c(32L, 16L),
    hidden_activation = "relu",
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_auto_int")
  expect_equal(fit$parameters$hidden_units, c(32L, 16L))
  expect_equal(fit$parameters$hidden_activation, c("relu", "relu"))

  pred <- predict(fit, df)
  expect_s3_class(pred, "tbl_df")
  expect_equal(nrow(pred), n)
  expect_true(".pred" %in% names(pred))
})

test_that("autoint classification with hidden layers", {
  skip_if_not_installed("torch")
  skip_on_cran()

  set.seed(1)
  n <- 100
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    y = factor(sample(c("a", "b"), n, replace = TRUE))
  )

  set.seed(1)
  fit <- brulee_auto_int(
    y ~ ., data = df,
    epochs = 5,
    hidden_units = 16L,
    hidden_activation = "relu",
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_auto_int")

  pred_class <- predict(fit, df, type = "class")
  expect_equal(nrow(pred_class), n)
  expect_true(".pred_class" %in% names(pred_class))

  pred_prob <- predict(fit, df, type = "prob")
  expect_equal(nrow(pred_prob), n)
})

test_that("autoint without hidden layers (default)", {
  skip_if_not_installed("torch")
  skip_on_cran()

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_auto_int(y ~ ., data = df, epochs = 3, verbose = FALSE)

  expect_s3_class(fit, "brulee_auto_int")
  expect_null(fit$parameters$hidden_units)
  expect_null(fit$parameters$hidden_activation)

  pred <- predict(fit, df)
  expect_equal(nrow(pred), n)
})

test_that("autoint hidden layer activation recycling", {
  skip_if_not_installed("torch")
  skip_on_cran()

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  fit <- brulee_auto_int(
    y ~ ., data = df,
    epochs = 3,
    hidden_units = c(16L, 8L),
    hidden_activation = "tanh",
    verbose = FALSE
  )

  expect_equal(fit$parameters$hidden_activation, c("tanh", "tanh"))
})

test_that("autoint hidden layer validation errors", {
  skip_if_not_installed("torch")
  skip_on_cran()

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  expect_error(
    brulee_auto_int(y ~ ., data = df, hidden_units = 16L),
    "hidden_activation"
  )

  expect_error(
    brulee_auto_int(y ~ ., data = df, hidden_activation = "relu"),
    "hidden_units"
  )

  expect_error(
    brulee_auto_int(
      y ~ ., data = df,
      hidden_units = c(16L, 8L),
      hidden_activation = c("relu", "bad_name")
    ),
    "hidden_activation"
  )

  expect_error(
    brulee_auto_int(
      y ~ ., data = df,
      hidden_units = c(16L, 8L),
      hidden_activation = c("relu", "tanh", "elu")
    ),
    "hidden_activation"
  )
})

test_that("autoint print shows hidden layer info", {
  skip_if_not_installed("torch")
  skip_on_cran()

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  fit <- brulee_auto_int(
    y ~ ., data = df,
    epochs = 3,
    hidden_units = 32L,
    hidden_activation = "relu",
    verbose = FALSE
  )

  output <- capture.output(print(fit), type = "message")
  expect_true(any(grepl("Hidden layers", output)))
})
