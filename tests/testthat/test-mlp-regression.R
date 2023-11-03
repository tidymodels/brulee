suppressPackageStartupMessages(library(modeldata))
suppressPackageStartupMessages(library(torch))
suppressPackageStartupMessages(library(recipes))

test_that('different fit interfaces', {
 skip_if(!torch::torch_is_installed())
 skip_on_os("mac", arch = "aarch64")
 # One test here was irreducible across OSes
 skip_on_os(c("windows", "linux", "solaris"))

 # ------------------------------------------------------------------------------

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

 # ------------------------------------------------------------------------------

 # matrix x
 expect_error({
  set.seed(1)
  mlp_reg_mat_lbfgs_fit <- brulee_mlp(ames_x_mat, ames_y, epochs = 10L, mixture = 0)
 },
 regex = NA
 )

 # regression tests
 save_coef(mlp_reg_mat_lbfgs_fit)
 expect_equal(
  last_param(mlp_reg_mat_lbfgs_fit),
  load_coef(mlp_reg_mat_lbfgs_fit),
  tolerance = 0.1
 )

 # data frame x (all numeric)
 expect_error(
  mlp_reg_df_lbfgs_fit <- brulee_mlp(ames_x_df, ames_y, epochs = 10L),
  regex = NA
 )

 # data frame x (mixed)
 expect_error(
  brulee_mlp(ames_x_df_mixed, ames_y, epochs = 10L),
  regex = "There were some non-numeric columns in the predictors"
 )

 # formula (mixed)
 expect_error(
  mlp_reg_f_lbfgs_fit <- brulee_mlp(Sale_Price ~ ., ames_smol, epochs = 10L),
  regex = NA
 )

 # recipe (mixed)
 expect_error({
  set.seed(1)
  mlp_reg_rec_lbfgs_fit <- brulee_mlp(ames_rec, ames, epochs = 10L)},
  regex = NA
 )

 # NOTE this one fails across operating systems, each with different answers
 # regression tests
 save_coef(mlp_reg_rec_lbfgs_fit)
 expect_equal(
  last_param(mlp_reg_rec_lbfgs_fit),
  load_coef(mlp_reg_rec_lbfgs_fit),
  tolerance = 0.1
 )

})


test_that('predictions', {
 skip_if(!torch::torch_is_installed())
 skip_on_os("mac", arch = "aarch64")

 # ------------------------------------------------------------------------------

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

 # ------------------------------------------------------------------------------

 set.seed(1)
 fit_df <- brulee_mlp(ames_x_df, ames_y, epochs = 10L)

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
 skip_on_os("mac", arch = "aarch64")


 # ------------------------------------------------------------------------------

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

 # ------------------------------------------------------------------------------

 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = NA),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 1:2),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 0L),
  error = TRUE
 )
 expect_error(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2),
  regex = NA
 )

 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, hidden_units = NA),
  error = TRUE
 )

 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, hidden_units = -1L),
  error = TRUE
 )
 expect_error(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, hidden_units = 2),
  regex = NA
 )

 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, activation = NA),
  error = TRUE
 )

 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, penalty = NA),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, penalty = runif(2)),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, penalty = -1.1),
  error = TRUE
 )

 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, dropout = NA),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, dropout = runif(2)),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, dropout = -1.1),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, dropout = 1.0),
  error = TRUE
 )
 expect_error(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, dropout = 0),
  regex = NA
 )

 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, validation = NA),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, validation = runif(2)),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, validation = -1.1),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, validation = 1.0),
  error = TRUE
 )
 expect_error(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, validation = 0),
  regex = NA
 )


 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, learn_rate = NA),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, learn_rate = runif(2)),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, learn_rate = -1.1),
  error = TRUE
 )

 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, verbose = 2),
  error = TRUE
 )
 expect_snapshot(
  brulee_mlp(ames_x_mat, ames_y, epochs = 2, verbose = rep(TRUE, 10)),
  error = TRUE
 )
 # ------------------------------------------------------------------------------

 fit_mat <- brulee_mlp(ames_x_mat, ames_y, epochs = 10L)

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
 skip_on_os("mac", arch = "aarch64")

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
 skip_on_os("mac", arch = "aarch64")

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
