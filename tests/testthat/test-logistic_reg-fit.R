
test_that("basic logistic regression LBFGS", {
 skip_if_not(torch::torch_is_installed())
 skip_if_not_installed("modeldata")
 skip_if_not_installed("yardstick")

 suppressPackageStartupMessages(library(dplyr))

 # ------------------------------------------------------------------------------

 set.seed(585)
 bin_tr <- modeldata::sim_logistic(5000, ~ -1 - 3 * A + 5 * B)
 bin_te <- modeldata::sim_logistic(1000, ~ -1 - 3 * A + 5 * B)
 num_class <- length(levels(bin_tr$class))

 # ------------------------------------------------------------------------------

 glm_fit <- glm(class ~ ., data = bin_tr, family = "binomial")

 expect_error({
  set.seed(392)
  bin_fit_lbfgs <-
   brulee_logistic_reg(class ~ ., bin_tr, penlaty = 0, epochs = 1)},
  regex = NA)

 expect_equal(
  unname(coef(glm_fit)),
  unname(coef(bin_fit_lbfgs)),
  tolerance = 1
 )

 expect_error(
  bin_pred_lbfgs <-
   predict(bin_fit_lbfgs,bin_te) %>%
   bind_cols(predict(bin_fit_lbfgs,bin_te, type = "prob")) %>%
   bind_cols(bin_te),
  regex = NA)

 fact_str <- structure(integer(0), levels = c("one", "two"), class = "factor")
 exp_str <-
  structure(
   list(.pred_class =
         fact_str,
        .pred_one = numeric(0),
        .pred_two = numeric(0),
        A = numeric(0),
        B = numeric(0),
        class = fact_str),
   row.names = integer(0),
   class = c("tbl_df", "tbl", "data.frame"))

 expect_equal(bin_pred_lbfgs[0,], exp_str)
 expect_equal(nrow(bin_pred_lbfgs), nrow(bin_te))

 # Did it learn anything?
 bin_brier_lbfgs <-
  bin_pred_lbfgs %>%
  yardstick::brier_class(class, .pred_one)

 expect_true(bin_brier_lbfgs$.estimate < (1 - 1/num_class)^2)
})

test_that("basic logistic regression SGD", {
 skip_if_not(torch::torch_is_installed())
 skip_if_not_installed("modeldata")
 skip_if_not_installed("yardstick")

 suppressPackageStartupMessages(library(dplyr))

 # ------------------------------------------------------------------------------

 set.seed(585)
 bin_tr <- modeldata::sim_logistic(5000, ~ -1 - 3 * A + 5 * B)
 bin_te <- modeldata::sim_logistic(1000, ~ -1 - 3 * A + 5 * B)
 num_class <- length(levels(bin_tr$class))

 # ------------------------------------------------------------------------------

 expect_error({
  set.seed(392)
  bin_fit_sgd <-
   brulee_logistic_reg(class ~ .,
                       bin_tr,
                       epochs = 500,
                       penalty = 0,
                       dropout = .1,
                       optimize = "SGD",
                       batch_size = 2^5,
                       learn_rate = 0.1)},
  regex = NA)

 glm_fit <- glm(class ~ ., data = bin_tr, family = "binomial")

 expect_equal(
  unname(coef(glm_fit)),
  unname(coef(bin_fit_sgd)),
  tolerance = .5
 )

 expect_error(
  bin_pred_sgd <-
   predict(bin_fit_sgd,bin_te) %>%
   bind_cols(predict(bin_fit_sgd,bin_te, type = "prob")) %>%
   bind_cols(bin_te),
  regex = NA)

 # Did it learn anything?
 bin_brier_sgd <-
  bin_pred_sgd %>%
  yardstick::brier_class(class, .pred_one)

 expect_true(bin_brier_sgd$.estimate < (1 - 1/num_class)^2)
})

test_that("coef works when recipes are used", {
 skip_if_not(torch::torch_is_installed())
 skip_if_not_installed("modeldata")
 skip_if_not_installed("recipes")
 skip_if(packageVersion("rlang") < "1.0.0")
 skip_on_os(c("windows", "linux", "solaris"))

  data("lending_club", package = "modeldata")
  lending_club <- head(lending_club, 1000)

  rec <-
   recipes::recipe(Class ~ revol_util + open_il_24m + emp_length,
                   data = lending_club) %>%
   recipes::step_dummy(emp_length, one_hot = TRUE) %>%
   recipes::step_normalize(recipes::all_predictors())

  fit_rec <- brulee_logistic_reg(rec, lending_club, epochs = 10L)

  coefs <- coef(fit_rec)
  expect_true(all(is.numeric(coefs)))
  expect_identical(
   names(coefs),
   c(
    "(Intercept)", "revol_util", "open_il_24m",
    paste0("emp_length_", levels(lending_club$emp_length))
   )
  )
})


# ------------------------------------------------------------------------------

test_that("logistic regression class weights", {
 skip_if_not(torch::torch_is_installed())
 skip_if_not_installed("modeldata")
 skip_if_not_installed("yardstick")

 suppressPackageStartupMessages(library(dplyr))

 # ------------------------------------------------------------------------------

 set.seed(585)
 bin_tr <- modeldata::sim_logistic(5000, ~ -5 - 3 * A + 5 * B)
 bin_te <- modeldata::sim_logistic(1000, ~ -5 - 3 * A + 5 * B)
 num_class <- length(levels(bin_tr$class))

 num_class <- length(levels(bin_tr$class))
 cls_xtab <- table(bin_tr$class)
 min_class <- names(sort(cls_xtab))[1]
 cls_wts <- rep(1, num_class)
 names(cls_wts) <- levels(bin_tr$class)
 cls_wts[names(cls_wts) == min_class] <- 10

 # ------------------------------------------------------------------------------

 expect_error({
  set.seed(392)
  bin_fit_lbfgs_wts <-
   brulee_logistic_reg(class ~ .,
                       bin_tr,
                       epochs = 30,
                       mixture = 0.5,
                       rate_schedule = "decay_time",
                       class_weights = cls_wts,
                       learn_rate = 0.1)},
  regex = NA)

 expect_error(
  bin_pred_lbfgs_wts <-
   predict(bin_fit_lbfgs_wts,bin_te) %>%
   bind_cols(predict(bin_fit_lbfgs_wts,bin_te, type = "prob")) %>%
   bind_cols(bin_te),
  regex = NA)

 ### matched unweighted model

 expect_error({
  set.seed(392)
  bin_fit_lbfgs_unwt <-
   brulee_logistic_reg(class ~ .,
                       bin_tr,
                       epochs = 30,
                       mixture = 0.5,
                       rate_schedule = "decay_time",
                       learn_rate = 0.1)},
  regex = NA)

 expect_error(
  bin_pred_lbfgs_unwt <-
   predict(bin_fit_lbfgs_unwt,bin_te) %>%
   bind_cols(predict(bin_fit_lbfgs_unwt,bin_te, type = "prob")) %>%
   bind_cols(bin_te),
  regex = NA)

 # did weighting predict the majority class more often?
 expect_true(
  sum(bin_pred_lbfgs_wts$.pred_class == min_class) >
   sum(bin_pred_lbfgs_unwt$.pred_class == min_class)
 )

})


