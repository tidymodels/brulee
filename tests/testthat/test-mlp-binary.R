
test_that("basic binomial mlp LBFGS", {
 skip_if_not(torch::torch_is_installed())

 skip_if_not_installed("modeldata")
 skip_if_not_installed("yardstick")

 suppressPackageStartupMessages(library(dplyr))
 suppressPackageStartupMessages(library(recipes))

 # ------------------------------------------------------------------------------

 set.seed(585)
 bin_tr <- modeldata::sim_classification(5000)
 bin_te <- modeldata::sim_classification(1000)

 rec <-
  recipe(class ~ ., data = bin_tr) %>%
  step_normalize(all_predictors())
 num_class <- length(levels(bin_tr$class))

 # ------------------------------------------------------------------------------

 expect_error({
  set.seed(392)
  bin_fit_f_lbfgs <-
   brulee_mlp(class ~ .,
              bin_tr,
              epochs = 200,
              hidden_units = 5,
              rate_schedule = "cyclic",
              learn_rate = 0.1)},
  regex = NA)


 expect_error({
  set.seed(392)
  bin_fit_lbfgs <-
   brulee_mlp(rec,
              bin_tr,
              epochs = 200,
              hidden_units = 5,
              rate_schedule = "cyclic",
              learn_rate = 0.1)},
  regex = NA)

 expect_error(
  bin_pred_lbfgs <-
   predict(bin_fit_lbfgs, bin_te) %>%
   bind_cols(predict(bin_fit_lbfgs, bin_te, type = "prob")) %>%
   bind_cols(bin_te) %>%
   select(starts_with(".pred"), class),
  regex = NA)

 fact_str <- structure(integer(0), levels = c("class_1", "class_2"), class = "factor")
 exp_str <-
  structure(
   list(.pred_class =
         fact_str,
        .pred_class_1 = numeric(0),
        .pred_class_2 = numeric(0),
        class = fact_str),
   row.names = integer(0),
   class = c("tbl_df", "tbl", "data.frame"))

 expect_equal(bin_pred_lbfgs[0,], exp_str)
 expect_equal(nrow(bin_pred_lbfgs), nrow(bin_te))

 # Did it learn anything?
 bin_brier_lbfgs <-
  bin_pred_lbfgs %>%
  yardstick::brier_class(class, .pred_class_1)

 expect_true(bin_brier_lbfgs$.estimate < (1 - 1/num_class)^2)
})


test_that("basic binomial mlp SGD", {
 skip_if_not(torch::torch_is_installed())

 skip_if_not_installed("modeldata")
 skip_if_not_installed("yardstick")

 suppressPackageStartupMessages(library(dplyr))
 suppressPackageStartupMessages(library(recipes))

 # ------------------------------------------------------------------------------

 set.seed(585)
 bin_tr <- modeldata::sim_classification(5000)
 bin_te <- modeldata::sim_classification(1000)

 rec <-
  recipe(class ~ ., data = bin_tr) %>%
  step_normalize(all_predictors())
 num_class <- length(levels(bin_tr$class))

 # ------------------------------------------------------------------------------

 expect_error({
  set.seed(392)
  bin_fit_f_sgd <-
   brulee_mlp(class ~ .,
              bin_tr,
              epochs = 200,
              penalty = 0,
              dropout = .1,
              hidden_units = 5,
              optimize = "SGD",
              batch_size = 64,
              momentum = 0.5,
              learn_rate = 0.1)},
  regex = NA)


 expect_error({
  set.seed(392)
  bin_fit_sgd <-
   brulee_mlp(rec,
              bin_tr,
              epochs = 200,
              penalty = 0,
              dropout = .1,
              hidden_units = 5,
              optimize = "SGD",
              batch_size = 64,
              momentum = 0.5,
              learn_rate = 0.1)},
  regex = NA)

 expect_error(
  bin_pred_sgd <-
   predict(bin_fit_sgd, bin_te) %>%
   bind_cols(predict(bin_fit_sgd, bin_te, type = "prob")) %>%
   bind_cols(bin_te) %>%
   select(starts_with(".pred"), class),
  regex = NA)

 fact_str <- structure(integer(0), levels = c("class_1", "class_2"), class = "factor")
 exp_str <-
  structure(
   list(.pred_class =
         fact_str,
        .pred_class_1 = numeric(0),
        .pred_class_2 = numeric(0),
        class = fact_str),
   row.names = integer(0),
   class = c("tbl_df", "tbl", "data.frame"))

 expect_equal(bin_pred_sgd[0,], exp_str)
 expect_equal(nrow(bin_pred_sgd), nrow(bin_te))

 # Did it learn anything?
 bin_brier_sgd <-
  bin_pred_sgd %>%
  yardstick::brier_class(class, .pred_class_1)

 expect_true(bin_brier_sgd$.estimate < (1 - 1/num_class)^2)
})


test_that("binomial mlp case weights", {
 skip_if_not(torch::torch_is_installed())

 skip_if_not_installed("modeldata")
 skip_if_not_installed("yardstick")

 suppressPackageStartupMessages(library(dplyr))
 suppressPackageStartupMessages(library(recipes))

 # ------------------------------------------------------------------------------

 set.seed(585)
 bin_tr <- modeldata::sim_classification(5000, intercept = 1)
 bin_te <- modeldata::sim_classification(1000, intercept = 1)

 rec <-
  recipe(class ~ ., data = bin_tr) %>%
  step_normalize(all_predictors())
 num_class <- length(levels(bin_tr$class))

 # ------------------------------------------------------------------------------

 expect_error({
  set.seed(392)
  weighted <-
   brulee_mlp(rec,
              bin_tr,
              epochs = 200,
              hidden_units = 5,
              rate_schedule = "cyclic",
              class_weights = 10,
              learn_rate = 0.1)},
  regex = NA)

 expect_error(
  weighted_pred <-
   predict(weighted, bin_te) %>%
   bind_cols(predict(weighted, bin_te, type = "prob")) %>%
   bind_cols(bin_te) %>%
   select(starts_with(".pred"), class),
  regex = NA)


 expect_error({
  set.seed(392)
  unweighted <-
   brulee_mlp(rec,
              bin_tr,
              epochs = 200,
              hidden_units = 5,
              rate_schedule = "cyclic",
              learn_rate = 0.1)},
  regex = NA)

 expect_error(
  unweighted_pred <-
   predict(unweighted, bin_te) %>%
   bind_cols(predict(unweighted, bin_te, type = "prob")) %>%
   bind_cols(bin_te) %>%
   select(starts_with(".pred"), class),
  regex = NA)

 expect_true(
  sum(weighted_pred$.pred_class == "class_2") >
   sum(unweighted_pred$.pred_class == "class_2")
 )
})

test_that('linear activations', {
 # See https://github.com/tidymodels/brulee/issues/68
 skip_if(!torch::torch_is_installed())
 skip_if_not_installed("modeldata")

 data(bivariate, package = "modeldata")
 set.seed(20)
 nn_log_biv <-
  try(
   brulee_mlp(Class ~ log(A) + log(B), data = bivariate_train,
              epochs = 150, hidden_units = 3, activation = "linear"),
   silent = TRUE)
 expect_s3_class(nn_log_biv, "brulee_mlp")

})
