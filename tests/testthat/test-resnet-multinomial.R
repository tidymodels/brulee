test_that("resnet multinomial classification - data.frame interface", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")
  skip_on_cran()

  set.seed(1)
  n <- 150
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n)
  )
  # Create 3 classes
  df$y <- factor(
    ifelse(df$x1 > 0.5, "A", ifelse(df$x2 > 0, "B", "C"))
  )

  set.seed(1)
  fit <- brulee_resnet(
    x = df[, c("x1", "x2")],
    y = df$y,
    hidden_units = 2,
    num_layers = 2,
    block_units = 5,
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
  expect_true(all(c(".pred_A", ".pred_B", ".pred_C") %in% names(pred_prob)))

  # Check probabilities sum to 1
  prob_sum <- rowSums(pred_prob[, c(".pred_A", ".pred_B", ".pred_C")])
  expect_true(all(abs(prob_sum - 1) < 1e-6))
})

test_that("resnet multinomial classification - formula interface", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")
  skip_on_cran()

  set.seed(1)
  n <- 150
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n)
  )
  df$y <- factor(
    ifelse(df$x1 > 0.5, "A", ifelse(df$x2 > 0, "B", "C"))
  )

  set.seed(1)
  fit <- brulee_resnet(
    y ~ x1 + x2,
    data = df,
    hidden_units = 2,
    num_layers = 2,
    block_units = 5,
    epochs = 3,
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_resnet")

  pred <- predict(fit, df, type = "prob")
  expect_equal(nrow(pred), nrow(df))
})

test_that("resnet multinomial classification - recipe interface", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")
  skip_on_cran()

  library(recipes)

  set.seed(1)
  n <- 150
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n)
  )
  df$y <- factor(
    ifelse(df$x1 > 0.5, "A", ifelse(df$x2 > 0, "B", "C"))
  )

  rec <- recipe(y ~ ., data = df) %>%
    step_normalize(all_numeric_predictors())

  set.seed(1)
  fit <- brulee_resnet(
    rec,
    data = df,
    hidden_units = 2,
    num_layers = 2,
    block_units = 5,
    epochs = 3,
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_resnet")

  pred <- predict(fit, df, type = "class")
  expect_equal(nrow(pred), nrow(df))
})

test_that("resnet multinomial classification - class weights", {
  skip_if_not_installed("recipes")
  skip_if_not_installed("torch")
  skip_on_cran()

  set.seed(1)
  n <- 150
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n)
  )
  df$y <- factor(
    ifelse(df$x1 > 0.5, "A", ifelse(df$x2 > 0, "B", "C"))
  )

  set.seed(1)
  fit <- brulee_resnet(
    x = df[, c("x1", "x2")],
    y = df$y,
    hidden_units = 2,
    num_layers = 1,
    block_units = 5,
    class_weights = c(A = 1, B = 2, C = 1),
    epochs = 3,
    verbose = FALSE
  )

  expect_s3_class(fit, "brulee_resnet")
  expect_equal(fit$parameters$class_weights, c(A = 1, B = 2, C = 1))
})
