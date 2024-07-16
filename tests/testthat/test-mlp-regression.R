
test_that('basic regression mlp LBFGS', {
 skip_if(!torch::torch_is_installed())

 skip_if_not_installed("modeldata")
 skip_if_not_installed("yardstick")
 skip_if_not_installed("recipes")

 suppressPackageStartupMessages(library(dplyr))
 suppressPackageStartupMessages(library(recipes))

 # ------------------------------------------------------------------------------

 set.seed(585)
 reg_tr <- modeldata::sim_regression(5000)
 reg_te <- modeldata::sim_regression(1000)

 reg_tr_x_df <- reg_tr[, -1]
 reg_tr_x_mat <- as.matrix(reg_tr_x_df)
 reg_tr_y <- reg_tr$outcome

 reg_rec <-
  recipe(outcome ~ ., data = reg_tr) %>%
  step_normalize(all_predictors())

 # ------------------------------------------------------------------------------

 # matrix x
 expect_error({
  set.seed(1)
  mlp_reg_mat_lbfgs_fit <-
   brulee_mlp(reg_tr_x_mat, reg_tr_y, mixture = 0, learn_rate = .1)},
  regex = NA
 )

 # data frame x (all numeric)
 expect_error(
  mlp_reg_df_lbfgs_fit <- brulee_mlp(reg_tr_x_df, reg_tr_y, validation = .2),
  regex = NA
 )

 # formula (mixed)
 expect_error({
  set.seed(8373)
  mlp_reg_f_lbfgs_fit <- brulee_mlp(outcome ~ ., reg_tr)},
  regex = NA
 )

 # recipe
 expect_error({
  set.seed(8373)
  mlp_reg_rec_lbfgs_fit <- brulee_mlp(reg_rec, reg_tr)},
  regex = NA
 )

 expect_error(
  reg_pred_lbfgs <-
   predict(mlp_reg_rec_lbfgs_fit, reg_te) %>%
   bind_cols(reg_te) %>%
   select(-starts_with("predictor")),
  regex = NA)

 exp_str <-
  structure(list(.pred = numeric(0), outcome = numeric(0)),
            row.names = integer(0), class = c("tbl_df", "tbl", "data.frame"))

 expect_equal(reg_pred_lbfgs[0,], exp_str)
 expect_equal(nrow(reg_pred_lbfgs), nrow(reg_te))

 # Did it learn anything?
 reg_rmse_lbfgs <-
  reg_pred_lbfgs %>%
  yardstick::rmse(outcome, .pred)

 set.seed(382)
 shuffled <-
  reg_pred_lbfgs %>%
  mutate(outcome = sample(outcome)) %>%
  yardstick::rmse(outcome, .pred)

 expect_true(reg_rmse_lbfgs$.estimate < shuffled$.estimate )
})


test_that('bad args', {
 skip_if(!torch::torch_is_installed())

 skip_if_not_installed("recipes")

 suppressPackageStartupMessages(library(dplyr))
 suppressPackageStartupMessages(library(recipes))

 # ------------------------------------------------------------------------------

 data(ames, package = "modeldata")

 ames$Sale_Price <- log10(ames$Sale_Price)

 reg_x_df <- ames[, c("Longitude", "Latitude")]
 reg_x_df_mixed <- ames[, c("Longitude", "Latitude", "Alley")]
 reg_x_mat <- as.matrix(reg_x_df)
 reg_y <- ames$Sale_Price
 reg_smol <- ames[, c("Longitude", "Latitude", "Alley", "Sale_Price")]

 reg_rec <-
  recipe(Sale_Price ~ Longitude + Latitude + Alley, data = ames) %>%
  step_dummy(Alley) %>%
  step_normalize(all_predictors())

 # ------------------------------------------------------------------------------

 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = NA),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 1:2),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 0L),
  error = TRUE
 )
 expect_error(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2),
  regex = NA
 )

 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, hidden_units = NA),
  error = TRUE
 )

 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, hidden_units = -1L),
  error = TRUE
 )
 expect_error(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, hidden_units = 2),
  regex = NA
 )

 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, activation = NA),
  error = TRUE
 )

 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, penalty = NA),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, penalty = runif(2)),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, penalty = -1.1),
  error = TRUE
 )

 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, dropout = NA),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, dropout = runif(2)),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, dropout = -1.1),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, dropout = 1.0),
  error = TRUE
 )
 expect_error(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, dropout = 0),
  regex = NA
 )

 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, validation = NA),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, validation = runif(2)),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, validation = -1.1),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, validation = 1.0),
  error = TRUE
 )
 expect_error(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, validation = 0),
  regex = NA
 )


 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, learn_rate = NA),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, learn_rate = runif(2)),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, learn_rate = -1.1),
  error = TRUE
 )

 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, verbose = 2),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(reg_x_mat, reg_y, epochs = 2, verbose = rep(TRUE, 10)),
  error = TRUE
 )
 # ------------------------------------------------------------------------------

 fit_mat <- brulee_mlp(reg_x_mat, reg_y, epochs = 10L)

 bad_models <- fit_mat
 bad_models$model_obj <- "potato!"
 expect_snapshot(
  brulee:::new_brulee_mlp(
   model_obj = bad_models$model_obj,
   estimates = bad_models$estimates,
   best_epoch = bad_models$best_epoch,
   loss = bad_models$loss,
   dims = bad_models$dims,
   y_stats = bad_models$y_stats,
   parameters = bad_models$parameters,
   blueprint = bad_models$blueprint
  ),
  error = TRUE
 )

 bad_est <- fit_mat
 bad_est$estimates <- "potato!"
 expect_snapshot(
  brulee:::new_brulee_mlp(
   model_obj = bad_est$model_obj,
   estimates = bad_est$estimates,
   best_epoch = bad_est$best_epoch,
   loss = bad_est$loss,
   dims = bad_est$dims,
   y_stats = bad_est$y_stats,
   parameters = bad_est$parameters,
   blueprint = bad_est$blueprint
  ),
  error = TRUE
 )

 bad_loss <- fit_mat
 bad_loss$loss <- "potato!"
 expect_snapshot(
  brulee:::new_brulee_mlp(
   model_obj = bad_loss$model_obj,
   estimates = bad_loss$estimates,
   best_epoch = bad_loss$best_epoch,
   loss = bad_loss$loss,
   dims = bad_loss$dims,
   y_stats = bad_loss$y_stats,
   parameters = bad_loss$parameters,
   blueprint = bad_loss$blueprint
  ),
  error = TRUE
 )

 bad_dims <- fit_mat
 bad_dims$dims <- "mountainous"
 expect_snapshot(
  brulee:::new_brulee_mlp(
   model_obj = bad_dims$model_obj,
   estimates = bad_dims$estimates,
   best_epoch = bad_dims$best_epoch,
   loss = bad_dims$loss,
   dims = bad_dims$dims,
   y_stats = bad_dims$y_stats,
   parameters = bad_dims$parameters,
   blueprint = bad_dims$blueprint
  ),
  error = TRUE
 )


 bad_parameters <- fit_mat
 bad_parameters$dims <- "mitten"
 expect_snapshot(
  brulee:::new_brulee_mlp(
   model_obj = bad_parameters$model_obj,
   estimates = bad_parameters$estimates,
   best_epoch = bad_parameters$best_epoch,
   loss = bad_parameters$loss,
   dims = bad_parameters$dims,
   y_stats = bad_parameters$y_stats,
   parameters = bad_parameters$parameters,
   blueprint = bad_parameters$blueprint
  ),
  error = TRUE
 )


 bad_blueprint <- fit_mat
 bad_blueprint$blueprint <- "adorable"
 expect_snapshot(
  brulee:::new_brulee_mlp(
   model_obj = bad_blueprint$model_obj,
   estimates = bad_blueprint$estimates,
   best_epoch = bad_blueprint$best_epoch,
   loss = bad_blueprint$loss,
   dims = bad_blueprint$dims,
   y_stats = bad_blueprint$y_stats,
   parameters = bad_blueprint$parameters,
   blueprint = bad_blueprint$blueprint
  ),
  error = TRUE
 )
})

test_that("mlp learns something", {
 skip_if(!torch::torch_is_installed())

 # ------------------------------------------------------------------------------

 set.seed(1)
 x <- data.frame(x = rnorm(1000))
 y <- 2 * x$x

 set.seed(2)
 model <- brulee_mlp(x, y,
                     batch_size = 25,
                     epochs = 50,
                     optimizer = "SGD",
                     activation = "relu",
                     hidden_units = 5L,
                     learn_rate = 0.1,
                     dropout = 0)

 expect_true(tail(model$loss, 1) < 0.03)

})
test_that("variable hidden_units length", {
 skip_if(!torch::torch_is_installed())

 x <- data.frame(x = rnorm(1000))
 y <- 2 * x$x

 expect_error(
  model <- brulee_mlp(x, y, hidden_units = c(2, 3), epochs = 1),
  regexp = NA
 )

 expect_equal(length(unlist(coef(model))), (1*2 + 2) + (2*3 + 3) + (3*1 + 1))


 expect_snapshot(
  model <- brulee_mlp(x, y, hidden_units = c(2, 3, 4), epochs = 1,
                      activation = c("relu", "tanh")),
  error = TRUE
 )

 expect_snapshot(
  model <- brulee_mlp(x, y, hidden_units = c(1), epochs = 1,
                      activation = c("relu", "tanh")),
  error = TRUE
 )

})


test_that('two-layer networks', {
 skip_if(!torch::torch_is_installed())

 skip_if_not_installed("modeldata")
 skip_if_not_installed("yardstick")
 skip_if_not_installed("recipes")

 suppressPackageStartupMessages(library(dplyr))
 suppressPackageStartupMessages(library(recipes))

 # ------------------------------------------------------------------------------

 set.seed(585)
 reg_tr <- modeldata::sim_regression(5000)
 reg_te <- modeldata::sim_regression(1000)

 reg_tr_x_df <- reg_tr[, -1]
 reg_tr_x_mat <- as.matrix(reg_tr_x_df)
 reg_tr_y <- reg_tr$outcome

 reg_rec <-
  recipe(outcome ~ ., data = reg_tr) %>%
  step_normalize(all_predictors())

  # ------------------------------------------------------------------------------

 # matrix x
 expect_error({
  set.seed(1)
  mlp_reg_mat_two_fit <-
   brulee_mlp_two_layer(
    reg_tr_x_mat,
    reg_tr_y,
    mixture = 0,
    learn_rate = .1,
    hidden_units = 5,
    hidden_units_2 = 10,
    activation = "relu",
    activation_2 = "elu"
   )
 },
 regex = NA)

 expect_error({
  set.seed(1)
  mlp_reg_mat_two_check_fit <-
   brulee_mlp(
    reg_tr_x_mat,
    reg_tr_y,
    mixture = 0,
    learn_rate = .1,
    hidden_units = c(5, 10),
    activation = c("relu", "elu")
   )
 },
 regex = NA)

 expect_equal(mlp_reg_mat_two_fit$loss, mlp_reg_mat_two_check_fit$loss)

 # data frame x (all numeric)
 expect_error(
  mlp_reg_df_two_fit <-
   brulee_mlp_two_layer(
    reg_tr_x_df,
    reg_tr_y,
    validation = .2,
    hidden_units = 5,
    hidden_units_2 = 10,
    activation = "celu",
    activation_2 = "gelu"
   ),
  regex = NA
 )

 # formula (mixed)
 expect_error({
  set.seed(8373)
  mlp_reg_f_two_fit <- brulee_mlp_two_layer(
   outcome ~ .,
   reg_tr,
   hidden_units = 5,
   hidden_units_2 = 10,
   activation = "hardshrink",
   activation_2 = "hardsigmoid"
  )
 },
 regex = NA)

 # recipe
 expect_error({
  set.seed(8373)
  mlp_reg_rec_two_fit <- brulee_mlp_two_layer(
   reg_rec,
   reg_tr,
   hidden_units = 5,
   hidden_units_2 = 10,
   activation = "hardtanh",
   activation_2 = "sigmoid"
  )
 },
 regex = NA)

})
