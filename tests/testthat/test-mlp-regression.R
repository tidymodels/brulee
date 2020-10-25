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
})


test_that('predictions', {
  skip_if(!torch::torch_is_installed())

  set.seed(1)
  fit_df <- torch_mlp(ames_x_df, ames_y, epochs = 10L)

  complete_pred <- predict(fit_df, head(ames_x_df))
  expect_true(tibble::is_tibble(complete_pred))
  expect_true(all(names(complete_pred) == ".pred"))
  expect_true(nrow(complete_pred) == nrow(head(ames_x_df)))

  has_missing <- head(ames_x_df)
  has_missing$Longitude[1] <- NA
  incomplete_pred <- predict(fit_df, has_missing)
  expect_true(tibble::is_tibble(incomplete_pred))
  expect_true(all(names(incomplete_pred) == ".pred"))
  expect_true(nrow(incomplete_pred) == nrow(has_missing))
  expect_true(sum(is.na(incomplete_pred)) == 1)

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
  torch_mlp(ames_x_mat, ames_y, epochs = 0L),
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

 # ------------------------------------------------------------------------------

 fit_mat <- torch_mlp(ames_x_mat, ames_y, epochs = 10L)

 bad_models <- fit_mat
 bad_models$models <- "potato!"
 expect_error(
   lantern:::new_torch_mlp(
     bad_models$models,
     bad_models$models,
     bad_models$dims,
     bad_models$parameters,
     bad_models$blueprint
   ),
   "should be a list"
 )

 bad_loss <- fit_mat
 bad_loss$loss <- "potato!"
 expect_error(
   lantern:::new_torch_mlp(
     bad_loss$models,
     bad_loss$loss,
     bad_loss$dims,
     bad_loss$parameters,
     bad_loss$blueprint
   ),
   "should be a numeric vector"
 )

 bad_dims <- fit_mat
 bad_dims$dims <- "mountainous"
 expect_error(
   lantern:::new_torch_mlp(
     bad_dims$models,
     bad_dims$loss,
     bad_dims$dims,
     bad_dims$parameters,
     bad_dims$blueprint
   ),
   "should be a list"
 )


 bad_parameters <- fit_mat
 bad_parameters$dims <- "mitten"
 expect_error(
   lantern:::new_torch_mlp(
     bad_parameters$models,
     bad_parameters$loss,
     bad_parameters$dims,
     bad_parameters$parameters,
     bad_parameters$blueprint
   ),
   "should be a list"
 )


 bad_blueprint <- fit_mat
 bad_blueprint$blueprint <- "adorable"
 expect_error(
   lantern:::new_torch_mlp(
     bad_blueprint$models,
     bad_blueprint$loss,
     bad_blueprint$dims,
     bad_blueprint$parameters,
     bad_blueprint$blueprint
   ),
   "should be a hardhat blueprint"
 )

})

test_that("learning happens", {

  set.seed(1)
  x <- data.frame(x = runif(1000))
  y <- 2 * x$x

  model <- torch_mlp(x, y,
                     epochs = 100L,
                     activation = "relu",
                     hidden_units = 10L,
                     learning_rate = 0.01)
  expect_true(tail(model$loss, 1) < 0.0005)

})
