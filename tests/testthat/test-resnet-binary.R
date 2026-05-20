test_that("resnet binary classification - data.frame interface", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n)
  )
  df$y <- factor(ifelse(df$x1 + df$x2 > 0, "A", "B"))

  set.seed(1)
  fit <- brulee_resnet(
    x = df[, c("x1", "x2")],
    y = df$y,
    hidden_units = c(5, 3),
    batch_norm_units = c(4, 4),
    epochs = 5,
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_resnet")

  # Test class predictions
  pred_class <- predict(fit, df, type = "class")
  expect_s3_class(pred_class, "tbl_df")
  expect_equal(nrow(pred_class), nrow(df))
  expect_true(".pred_class" %in% names(pred_class))

  # Test probability predictions
  pred_prob <- predict(fit, df, type = "prob")
  expect_s3_class(pred_prob, "tbl_df")
  expect_equal(nrow(pred_prob), nrow(df))
  expect_true(all(c(".pred_A", ".pred_B") %in% names(pred_prob)))
})

test_that("resnet binary classification - formula interface", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n)
  )
  df$y <- factor(ifelse(df$x1 + df$x2 > 0, "A", "B"))

  set.seed(1)
  fit <- brulee_resnet(
    y ~ x1 + x2,
    data = df,
    hidden_units = c(5, 3),
    batch_norm_units = c(4, 4),
    epochs = 3,
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_resnet")

  pred <- predict(fit, df, type = "prob")
  expect_equal(nrow(pred), nrow(df))
})

test_that("resnet binary classification - recipe interface", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")

  library(recipes)

  set.seed(1)
  n <- 100
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n)
  )
  df$y <- factor(ifelse(df$x1 + df$x2 > 0, "A", "B"))

  rec <- recipe(y ~ ., data = df) %>%
    step_normalize(all_numeric_predictors())

  set.seed(1)
  fit <- brulee_resnet(
    rec,
    data = df,
    hidden_units = c(5, 3),
    batch_norm_units = c(4, 4),
    epochs = 3,
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_resnet")

  pred <- predict(fit, df, type = "class")
  expect_equal(nrow(pred), nrow(df))
})

test_that("resnet binary classification - class weights", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n)
  )
  df$y <- factor(ifelse(df$x1 + df$x2 > 0, "A", "B"))

  # Create imbalanced classes
  df$y[1:70] <- "A"
  df$y[71:100] <- "B"

  set.seed(1)
  fit <- brulee_resnet(
    x = df[, c("x1", "x2")],
    y = df$y,
    hidden_units = c(5, 3),
    batch_norm_units = c(4, 4),
    class_weights = c(A = 1, B = 2),
    epochs = 3,
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_resnet")
  expect_equal(fit$parameters$class_weights, c(A = 1, B = 2))
})
