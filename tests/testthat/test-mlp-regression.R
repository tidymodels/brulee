context("mlp regression")

## -----------------------------------------------------------------------------

suppressPackageStartupMessages(library(modeldata))
suppressPackageStartupMessages(library(torch))
suppressPackageStartupMessages(library(recipes))

## -----------------------------------------------------------------------------

data(ames, package = "modeldata")

ames$Sale_Price <- log10(ames$Sale_Price)

ames_x_df <- ames[, c("Longitude", "Latitude")]
ames_x_df_mixed <- ames[, c("Longitude", "Latitude", "Alley")]
ames_x_mat <- as.matrix(ames_x_df)
ames_y <- ames$Sale_Price
ames_smol <- ames[, c("Longitude", "Latitude", "Alley", "Sale_Price")]

ames_rec <-
 recipe(Sale_Price ~ Longitude + Latitude + Alley, data = ames) %>%
 step_dummy(Alley) %>%
 step_normalize(all_predictors())

## -----------------------------------------------------------------------------

test_that('different fit interfaces', {
 skip_if(!torch::torch_is_installed())

 # matrix x
 expect_error(
  fit_mat <- torch_mlp(ames_x_mat, ames_y, epochs = 10L),
  regex = NA
 )

 # data frame x (all numeric)
 expect_error(
  fit_df <- torch_mlp(ames_x_df, ames_y, epochs = 10L),
  regex = NA
 )

 # data frame x (mixed)
 expect_error(
  torch_mlp(ames_x_df_mixed, ames_y, epochs = 10L),
  regex = "There were some non-numeric columns in the predictors"
 )

 # formula (mixed)
 expect_error(
  fit_f <- torch_mlp(Sale_Price ~ ., ames_smol, epochs = 10L),
  regex = NA
 )

 # recipe (mixed)
 expect_error(
  fit_rec <- torch_mlp(ames_rec, ames, epochs = 10L),
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

test_that('bad args', {
 skip_if(!torch::torch_is_installed())

 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = NA),
  "expected 'epochs' to be integer."
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 1:2),
  "expected 'epochs' to be a single integer."
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 1L),
  "expected 'epochs' to be an integer on"
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2),
  regex = NA
 )

 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, hidden_units = NA),
  "expected 'hidden_units' to be integer."
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, hidden_units = 1:2),
  "expected 'hidden_units' to be a single integer."
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, hidden_units = -1L),
  "expected 'hidden_units' to be an integer on"
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, hidden_units = 2),
  regex = NA
 )

 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, activation = NA),
  "expected 'activation' to be character"
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, activation = letters),
  "expected 'activation' to be a single character string"
 )

 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, penalty = NA),
  "expected 'penalty' to be a double"
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, penalty = runif(2)),
  "expected 'penalty' to be a single double"
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, penalty = -1.1),
  "expected 'penalty' to be a double on"
 )

 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, dropout = NA),
  "expected 'dropout' to be a double"
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, dropout = runif(2)),
  "expected 'dropout' to be a single double"
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, dropout = -1.1),
  "expected 'dropout' to be a double on"
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, dropout = 1.0),
  "expected 'dropout' to be a double on"
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, dropout = 0),
  regex = NA
 )

 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, validation = NA),
  "expected 'validation' to be a double"
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, validation = runif(2)),
  "expected 'validation' to be a single double"
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, validation = -1.1),
  "expected 'validation' to be a double on"
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, validation = 1.0),
  "expected 'validation' to be a double on"
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, validation = 0),
  regex = NA
 )


 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, learning_rate = NA),
  "expected 'learning_rate' to be a double"
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, learning_rate = runif(2)),
  "expected 'learning_rate' to be a single double"
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, learning_rate = -1.1),
  "expected 'learning_rate' to be a double on"
 )

 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, verbose = 2),
  "expected 'verbose' to be logical"
 )
 expect_error(
  torch_mlp(ames_x_mat, ames_y, epochs = 2, verbose = rep(TRUE, 10)),
  "expected 'verbose' to be a single logical"
 )

})
