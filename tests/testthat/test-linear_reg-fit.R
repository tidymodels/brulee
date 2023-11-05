
test_that("basic linear regression LBFGS", {
 skip_if_not(torch::torch_is_installed())

 skip_if_not_installed("yardstick")

 suppressPackageStartupMessages(library(dplyr))

 # ------------------------------------------------------------------------------

 set.seed(1)
 lin_tr <- tibble::tibble(
  x1 = runif(1000),
  x2 = runif(1000),
  outcome = 3 + 2 * x1 + 3 * x2
 )
 lin_te <- tibble::tibble(
  x1 = runif(1000),
  x2 = runif(1000),
  outcome = 3 + 2 * x1 + 3 * x2
 )

 # ------------------------------------------------------------------------------

 lm_fit <- lm(outcome ~ ., data = lin_tr)

 expect_error({
  set.seed(392)
  lin_fit_lbfgs <-
   brulee_linear_reg(outcome ~ ., lin_tr, penlaty = 0)},
  regex = NA)

 expect_equal(
  unname(coef(lm_fit)),
  unname(coef(lin_fit_lbfgs)),
  tolerance = .1
 )

 expect_error(
  lin_pred_lbfgs <-
   predict(lin_fit_lbfgs, lin_te) %>%
   bind_cols(lin_te),
  regex = NA)

 exp_str <-
  structure(
   list(
    .pred = numeric(0),
    x1 = numeric(0),
    x2 = numeric(0),
    outcome = numeric(0)),
   row.names = integer(0),
   class = c("tbl_df", "tbl", "data.frame"))

 expect_equal(lin_pred_lbfgs[0,], exp_str)
 expect_equal(nrow(lin_pred_lbfgs), nrow(lin_te))

 # Did it learn anything?
 lin_brier_lbfgs <-
  lin_pred_lbfgs %>%
  yardstick::rmse(outcome, .pred)

 set.seed(382)
 shuffled <-
  lin_pred_lbfgs %>%
  mutate(outcome = sample(outcome)) %>%
  yardstick::rmse(outcome, .pred)

 expect_true(lin_brier_lbfgs$.estimate < shuffled$.estimate )
})

test_that("basic Linear regression sgd", {
 skip_if_not(torch::torch_is_installed())

 skip_if_not_installed("yardstick")

 suppressPackageStartupMessages(library(dplyr))

 # ------------------------------------------------------------------------------

 set.seed(1)
 lin_tr <- tibble::tibble(
  x1 = runif(1000),
  x2 = runif(1000),
  outcome = 3 + 2 * x1 + 3 * x2
 )
 lin_te <- tibble::tibble(
  x1 = runif(1000),
  x2 = runif(1000),
  outcome = 3 + 2 * x1 + 3 * x2
 )

 # ------------------------------------------------------------------------------

 lm_fit <- lm(outcome ~ ., data = lin_tr)

 expect_error({
  set.seed(392)
  lin_fit_sgd <-
   brulee_linear_reg(
    outcome ~ .,
    lin_tr,
    penlaty = 0,
    epochs = 500,
    batch_size = 2^5,
    learn_rate = 0.1,
    optimizer = "SGD",
    stop_iter = 20
   )},
  regex = NA)

 expect_equal(
  unname(coef(lm_fit)),
  unname(coef(lin_fit_sgd)),
  tolerance = .1
 )

 expect_error(
  lin_pred_sgd <-
   predict(lin_fit_sgd, lin_te) %>%
   bind_cols(lin_te),
  regex = NA)

 exp_str <-
  structure(
   list(
    .pred = numeric(0),
    x1 = numeric(0),
    x2 = numeric(0),
    outcome = numeric(0)),
   row.names = integer(0),
   class = c("tbl_df", "tbl", "data.frame"))

 expect_equal(lin_pred_sgd[0,], exp_str)
 expect_equal(nrow(lin_pred_sgd), nrow(lin_te))

 # Did it learn anything?
 lin_brier_sgd <-
  lin_pred_sgd %>%
  yardstick::rmse(outcome, .pred)

 set.seed(382)
 shuffled <-
  lin_pred_sgd %>%
  mutate(outcome = sample(outcome)) %>%
  yardstick::rmse(outcome, .pred)

 expect_true(lin_brier_sgd$.estimate < shuffled$.estimate)
})
