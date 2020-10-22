context("mlp binary classification")

## -----------------------------------------------------------------------------

suppressPackageStartupMessages(library(modeldata))
suppressPackageStartupMessages(library(torch))
suppressPackageStartupMessages(library(recipes))

## -----------------------------------------------------------------------------

data("lending_club", package = "modeldata")

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
  fit_mat <- torch_mlp(x_mat, y, epochs = 10L),
  regex = NA
 )

 # data frame x (all numeric)
 expect_error(
  fit_df <- torch_mlp(x_df, y, epochs = 10L),
  regex = NA

 # data frame x (mixed)
 )
 expect_error(
  torch_mlp(x_df_mixed, y, epochs = 10L),
  regex = "There were some non-numeric columns in the predictors"
 )

 # formula (mixed)
 expect_error(
  fit_f <- torch_mlp(Class ~ ., smol, epochs = 10L),
  regex = NA
 )

 # recipe (mixed)
 expect_error(
  fit_rec <- torch_mlp(rec, lending_club, epochs = 10L),
  regex = NA
 )

 # Check coefs:
 for (i in 1:4) {
  expect_true(is.numeric(fit_mat$coefs[[i]]))
  expect_true(is.numeric(fit_df$coefs[[i]]))
  expect_true(is.numeric(fit_f$coefs[[i]]))
  expect_true(is.numeric(fit_rec$coefs[[i]]))
 }
})

