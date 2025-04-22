
test_that("setting class weights", {
 skip_if_not(torch::torch_is_installed())
 skip_if_not_installed("modeldata")

 suppressPackageStartupMessages(library(dplyr))

 # ------------------------------------------------------------------------------

 set.seed(585)
 mnl_tr <-
  modeldata::sim_multinomial(
   1000,
   ~  -0.5    +  0.6 * abs(A),
   ~ ifelse(A > 0 & B > 0, 1.0 + 0.2 * A / B, - 2),
   ~ -0.6 * A + 0.50 * B -  A * B)

 lvls <- levels(mnl_tr$class)
 num_class <- length(lvls)
 cls_xtab <- table(mnl_tr$class)
 min_class <- names(sort(cls_xtab))[1]

 cls_wts <- rep(1, num_class)
 names(cls_wts) <- lvls
 cls_wts[names(cls_wts) == min_class] <- 10

 bad_wts <- cls_wts
 names(bad_wts) <- letters[1:num_class]

 # ------------------------------------------------------------------------------

 expect_equal(
  brulee:::check_class_weights(1.0, lvls, cls_xtab, "fabulous") |>
   as.numeric(),
  rep(1, num_class)
 )

 expect_s3_class(
  brulee:::check_class_weights(1.0, lvls, cls_xtab, "fabulous"),
  "torch_tensor"
 )

 expect_equal(
  brulee:::check_class_weights(NULL, lvls, cls_xtab, "fabulous") |>
   as.numeric(),
  rep(1, num_class)
 )

 expect_equal(
  brulee:::check_class_weights(6.25, lvls, cls_xtab, "fabulous") |>
   as.numeric(),
  c(1, 6.25, 1)
 )

 expect_equal(
  brulee:::check_class_weights(c(1, 6.25, 1), lvls, cls_xtab, "fabulous") |>
   as.numeric(),
  c(1, 6.25, 1)
 )

 expect_null(
  brulee:::check_class_weights(1, character(0), cls_xtab, "fabulous")
 )

 expect_snapshot(
  brulee:::check_class_weights("a", lvls, cls_xtab, "fabulous"),
  error = TRUE
 )

 expect_snapshot(
  brulee:::check_class_weights(c(1, 6.25), lvls, cls_xtab, "fabulous"),
  error = TRUE
 )

 expect_snapshot(
  brulee:::check_class_weights(bad_wts, lvls, cls_xtab, "fabulous"),
  error = TRUE
 )

})
