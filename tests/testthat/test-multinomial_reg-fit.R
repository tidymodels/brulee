
test_that("basic multinomial regression LBFGS", {
 skip_if_not(torch::torch_is_installed())
 skip_if_not_installed("modeldata")
 skip_if_not_installed("yardstick")

 suppressPackageStartupMessages(library(dplyr))

 # ------------------------------------------------------------------------------

 set.seed(585)
 mnl_tr <-
  modeldata::sim_multinomial(
   1000,
   ~  -0.5    +  0.6 * A,
   ~ .1 * B,
   ~ -0.6 * A + 0.50 * B)
 mnl_te <-
  modeldata::sim_multinomial(
   200,
   ~  -0.5    +  0.6 * A,
   ~ .1 * B,
   ~ -0.6 * A + 0.50 * B)
 num_class <- length(levels(mnl_tr$class))

 # ------------------------------------------------------------------------------

 expect_error({
  set.seed(392)
  mnl_fit_lbfgs <-
   brulee_multinomial_reg(class ~ .,
                          mnl_tr,
                          epochs = 200,
                          rate_schedule = "cyclic",
                          learn_rate = 0.1)},
  regex = NA)

 expect_error(
  mnl_pred_lbfgs <-
   predict(mnl_fit_lbfgs, mnl_te) |>
   bind_cols(predict(mnl_fit_lbfgs, mnl_te, type = "prob")) |>
   bind_cols(mnl_te),
  regex = NA)

 fact_str <- structure(integer(0), levels = c("one", "two", "three"), class = "factor")
 exp_str <-
  structure(
   list(.pred_class =
         fact_str,
        .pred_one = numeric(0),
        .pred_two = numeric(0),
        .pred_three = numeric(0),
        A = numeric(0),
        B = numeric(0),
        class = fact_str),
   row.names = integer(0),
   class = c("tbl_df", "tbl", "data.frame"))

 expect_equal(mnl_pred_lbfgs[0,], exp_str)
 expect_equal(nrow(mnl_pred_lbfgs), nrow(mnl_te))

 # Did it learn anything?
 mnl_brier_lbfgs <-
  mnl_pred_lbfgs |>
  yardstick::brier_class(class, .pred_one, .pred_two, .pred_three)

 expect_true(mnl_brier_lbfgs$.estimate < (1 - 1/num_class)^2)
})

test_that("basic multinomial regression SGD", {
 skip_if_not(torch::torch_is_installed())
 skip_if_not_installed("modeldata")
 skip_if_not_installed("yardstick")

 suppressPackageStartupMessages(library(dplyr))

 # ------------------------------------------------------------------------------

 set.seed(585)
 mnl_tr <-
  modeldata::sim_multinomial(
   1000,
   ~  -0.5    +  0.6 * A,
   ~ .1 * B,
   ~ -0.6 * A + 0.50 * B)
 mnl_te <-
  modeldata::sim_multinomial(
   200,
   ~  -0.5    +  0.6 * A,
   ~ .1 * B,
   ~ -0.6 * A + 0.50 * B)
 num_class <- length(levels(mnl_tr$class))

 # ------------------------------------------------------------------------------

 expect_error({
  set.seed(392)
  mnl_fit_sgd <-
   brulee_multinomial_reg(class ~ .,
                          mnl_tr,
                          epochs = 200,
                          penalty = 0,
                          dropout = .1,
                          optimize = "SGD",
                          batch_size = 64L,
                          momentum = 0.5,
                          learn_rate = 0.1)},
  regex = NA)

 expect_error(
  mnl_pred_sgd <-
   predict(mnl_fit_sgd, mnl_te) |>
   bind_cols(predict(mnl_fit_sgd, mnl_te, type = "prob")) |>
   bind_cols(mnl_te),
  regex = NA)

 # Did it learn anything?
 mnl_brier_sgd <-
  mnl_pred_sgd |>
  yardstick::brier_class(class, .pred_one, .pred_two, .pred_three)

 expect_true(mnl_brier_sgd$.estimate < (1 - 1/num_class)^2)
})


# ------------------------------------------------------------------------------

test_that("multinomial regression class weights", {
 skip_if_not(torch::torch_is_installed())
 skip_if_not_installed("modeldata")
 skip_if_not_installed("yardstick")

 suppressPackageStartupMessages(library(dplyr))

 # ------------------------------------------------------------------------------

 set.seed(585)
 mnl_tr <-
  modeldata::sim_multinomial(
   1000,
   ~  -0.5    +  0.6 * A,
   ~ .1 * B,
   ~ -0.6 * A + 0.50 * B)
 mnl_te <-
  modeldata::sim_multinomial(
   200,
   ~  -0.5    +  0.6 * A,
   ~ .1 * B,
   ~ -0.6 * A + 0.50 * B)

 num_class <- length(levels(mnl_tr$class))
 cls_xtab <- table(mnl_tr$class)
 min_class <- names(sort(cls_xtab))[1]
 cls_wts <- rep(1, num_class)
 names(cls_wts) <- levels(mnl_tr$class)
 cls_wts[names(cls_wts) == min_class] <- 10

 # ------------------------------------------------------------------------------

 expect_error({
  set.seed(392)
  mnl_fit_lbfgs_wts <-
   brulee_multinomial_reg(class ~ .,
                          mnl_tr,
                          epochs = 30,
                          mixture = 0.5,
                          rate_schedule = "decay_time",
                          class_weights = cls_wts,
                          learn_rate = 0.1)},
  regex = NA)

 expect_error(
  mnl_pred_lbfgs_wts <-
   predict(mnl_fit_lbfgs_wts, mnl_te) |>
   bind_cols(predict(mnl_fit_lbfgs_wts, mnl_te, type = "prob")) |>
   bind_cols(mnl_te),
  regex = NA)

 ### matched unweighted model

 expect_error({
  set.seed(392)
  mnl_fit_lbfgs_unwt <-
   brulee_multinomial_reg(class ~ .,
                          mnl_tr,
                          epochs = 30,
                          mixture = 0.5,
                          rate_schedule = "decay_time",
                          learn_rate = 0.1)},
  regex = NA)

 expect_error(
  mnl_pred_lbfgs_unwt <-
   predict(mnl_fit_lbfgs_unwt, mnl_te) |>
   bind_cols(predict(mnl_fit_lbfgs_unwt, mnl_te, type = "prob")) |>
   bind_cols(mnl_te),
  regex = NA)

 # did weighting predict the majority class more often?
 expect_true(
  sum(mnl_pred_lbfgs_wts$.pred_class == min_class) >
   sum(mnl_pred_lbfgs_unwt$.pred_class == min_class)
 )

})


