

test_that("NaN loss due to overflow", {
 skip_if_not_installed("modeldata")
 skip_on_os(c("windows", "linux", "solaris"))
 skip_on_os("mac", arch = "x86_64")

 i <- 81872
 set.seed(i)
 data_tr <- modeldata::sim_logistic(200, ~ .1 + 2 * A - 3 * B + 1 * A *B, corr = .7)

 expect_snapshot_warning({
  set.seed(i+1)
  mlp_fit <- brulee_mlp(class ~ ., data = data_tr, hidden_units = 10,
                        stop_iter = Inf)
 })
 expect_snapshot(print(mlp_fit))
 expect_equal(length(mlp_fit$estimates), 9)

})
