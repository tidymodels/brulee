# Most tests use the real v3.5 weights and run only when they are cached
# (tab_pfn_download_weights("v3.5")).

test_that("fitting and predicting work through every interface", {
  skip_if_no_weights()
  train <- iris[-(1:5 * 10), ]
  test <- iris[1:5 * 10, ]

  set.seed(81723)
  fit_f <- brulee_tab_pfn(
    Species ~ .,
    data = train,
    num_estimators = 2,
    device = "cpu"
  )
  set.seed(81723)
  fit_xy <- brulee_tab_pfn(train[, 1:4], train$Species, num_estimators = 2)
  set.seed(81723)
  fit_m <- brulee_tab_pfn(
    as.matrix(train[, 1:4]),
    train$Species,
    num_estimators = 2
  )
  expect_s3_class(fit_f, "brulee_tab_pfn")

  pred <- predict(fit_f, test, type = "prob")
  expect_named(pred, paste0(".pred_", levels(iris$Species)))
  expect_identical(pred, predict(fit_xy, test, type = "prob"))
  expect_identical(
    pred,
    predict(fit_m, as.matrix(test[, 1:4]), type = "prob")
  )
  expect_identical(predict(fit_f, test)$.pred_class, test$Species)

  aug <- augment(fit_f, test)
  expect_named(
    aug,
    c(".pred_class", paste0(".pred_", levels(iris$Species)), names(test))
  )
  expect_identical(nrow(aug), 5L)
  expect_snapshot(print(fit_f))
})

test_that("the same seed gives the same fit", {
  skip_if_no_weights()
  set.seed(30941)
  fit_1 <- brulee_tab_pfn(Species ~ ., data = iris, num_estimators = 3)
  set.seed(30941)
  fit_2 <- brulee_tab_pfn(Species ~ ., data = iris, num_estimators = 3)
  expect_identical(fit_1$fit$members, fit_2$fit$members)
  expect_identical(fit_1$fit$seed, fit_2$fit$seed)
})

test_that("a constant outcome predicts the constant", {
  skip_if_no_weights()
  d <- data.frame(x = 1:10, y = 3)
  fit <- brulee_tab_pfn(y ~ x, data = d, device = "cpu")
  expect_identical(predict(fit, d[1:2, ])$.pred, c(3, 3))
})

test_that("bad arguments are rejected", {
  expect_snapshot(
    brulee_tab_pfn(mpg ~ ., data = mtcars, num_estimators = 0),
    error = TRUE
  )
  expect_snapshot(
    brulee_tab_pfn(mpg ~ ., data = mtcars, version = "v2"),
    error = TRUE
  )
  expect_snapshot(
    brulee_tab_pfn(mpg ~ ., data = mtcars, ignore_pretraining_limits = "yes"),
    error = TRUE
  )
  expect_snapshot(
    brulee_tab_pfn(mpg ~ ., data = mtcars, control = list()),
    error = TRUE
  )
  expect_snapshot(brulee_tab_pfn(letters), error = TRUE)
  expect_snapshot(
    brulee_tab_pfn(mpg ~ ., data = mtcars, device = "tpu"),
    error = TRUE
  )
})

test_that("data larger than the model was trained for is refused", {
  ic <- list(max_features = 3, max_samples = 50, max_classes = 2)
  x <- as.data.frame(matrix(0, 60, 4))
  y <- factor(rep(c("a", "b", "c"), 20))
  expect_snapshot(
    tabpfn_check_limits(x, y, ic, "v3.5", "cuda", FALSE, call = NULL),
    error = TRUE
  )
  expect_snapshot(
    tabpfn_check_limits(x, y, ic, "v3.5", "cuda", TRUE, call = NULL),
    error = TRUE
  )
})

test_that("the CPU row limit applies on the CPU only, with a warning", {
  ic <- list(max_features = 10, max_samples = 1e6, max_classes = 10)
  big <- data.frame(x = seq_len(5001))
  y <- seq_len(5001)
  expect_snapshot(
    tabpfn_check_limits(big, y, ic, "v3", "cpu", FALSE, call = NULL),
    error = TRUE
  )
  expect_no_error(
    tabpfn_check_limits(big, y, ic, "v3", "cuda", FALSE, call = NULL)
  )
  medium <- big[1:2000, , drop = FALSE]
  expect_snapshot(
    tabpfn_check_limits(medium, y[1:2000], ic, "v3", "cpu", FALSE, call = NULL)
  )
})
