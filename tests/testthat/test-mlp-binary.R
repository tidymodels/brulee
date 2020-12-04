context("mlp binary classification")

## -----------------------------------------------------------------------------

suppressPackageStartupMessages(library(modeldata))
suppressPackageStartupMessages(library(torch))
suppressPackageStartupMessages(library(recipes))

## -----------------------------------------------------------------------------

data("lending_club", package = "modeldata")
lending_club <- head(lending_club, 1000)

x_df <- lending_club[, c("revol_util", "open_il_24m")]
x_df_mixed <- lending_club[, c("revol_util", "open_il_24m", "emp_length")]
x_mat <- as.matrix(x_df)
y <- lending_club$Class
smol <- lending_club[, c("revol_util", "open_il_24m", "emp_length", "Class")]

rec <-
  recipe(Class ~ revol_util + open_il_24m + emp_length, data = lending_club) %>%
  step_dummy(emp_length) %>%
  step_normalize(all_predictors())

## -----------------------------------------------------------------------------

test_that('different fit interfaces', {
  skip_if(!torch::torch_is_installed())

  # matrix x
  expect_error(
    fit_mat <- lantern_mlp(x_mat, y, epochs = 10L),
    regex = NA
  )

  # data frame x (all numeric)
  expect_error(
    fit_df <- lantern_mlp(x_df, y, epochs = 10L),
    regex = NA

    # data frame x (mixed)
  )
  expect_error(
    lantern_mlp(x_df_mixed, y, epochs = 10L),
    regex = "There were some non-numeric columns in the predictors"
  )

  # formula (mixed)
  expect_error(
    fit_f <- lantern_mlp(Class ~ ., smol, epochs = 10L),
    regex = NA
  )

  # recipe (mixed)
  expect_error(
    fit_rec <- lantern_mlp(rec, lending_club, epochs = 10L),
    regex = NA
  )

})

test_that('predictions', {
  skip_if(!torch::torch_is_installed())

  set.seed(1)
  fit_df <- lantern_mlp(x_df, y, epochs = 10L)

  complete_pred <- predict(fit_df, head(x_df))
  expect_true(tibble::is_tibble(complete_pred))
  expect_true(all(names(complete_pred) == ".pred_class"))
  expect_true(nrow(complete_pred) == nrow(head(x_df)))

  has_missing <- head(x_df)
  has_missing$revol_util[1] <- NA
  incomplete_pred <- predict(fit_df, has_missing)
  expect_true(tibble::is_tibble(incomplete_pred))
  expect_true(all(names(incomplete_pred) == ".pred_class"))
  expect_true(nrow(incomplete_pred) == nrow(has_missing))
  expect_true(sum(is.na(incomplete_pred)) == 1)

  pred_prob <- predict(fit_df, head(x_df), type = "prob")
  expect_true(tibble::is_tibble(complete_pred))
  expect_true(all(names(pred_prob) == c(".pred_bad", ".pred_good")))
  expect_true(nrow(pred_prob) == nrow(head(x_df)))
  expect_equal(apply(pred_prob, 1, sum), rep(1, nrow(pred_prob)), tolerance = 1e6)

})

test_that("mlp binary learns something", {

  set.seed(1)
  x <- data.frame(x = rnorm(1000))
  y <- as.factor(x > 0)

  model <- lantern_mlp(x, y,
                     batch_size = 50,
                     epochs = 100L,
                     activation = "relu",
                     hidden_units = 5L,
                     learn_rate = 0.1,
                     dropout = 0)

  y_ <- predict(model, x)$.pred_class
  expect_true(sum(diag(table(y, y_))) > 950)

})

