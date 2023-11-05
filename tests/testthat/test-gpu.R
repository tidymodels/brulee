test_that('device type - mac ARM', {

 skip_if_not(torch::torch_is_installed())
 skip_on_os(c("windows", "linux", "solaris"))
 skip_on_os("mac", arch = "x86_64")
 expect_snapshot(guess_brulee_device())
})

test_that('device type - cpu', {

 skip_if_not(torch::torch_is_installed())
 skip_on_os("mac", arch = "aarch64")
 expect_snapshot(guess_brulee_device())
})


test_that('linear regression on gpu', {

 skip_if_not(torch::torch_is_installed())
 skip_on_os(c("windows", "linux", "solaris"))
 skip_on_os("mac", arch = "x86_64")

 skip_if_not_installed("modeldata")

 set.seed(591)
 tr <- sim_regression(1000)

 expect_error(
  fit <- brulee_linear_reg(outcome ~ ., data = tr, device = "mps"),
  regex = NA
 )
})

test_that('logistic regression on gpu', {

 skip_if_not(torch::torch_is_installed())
 skip_on_os(c("windows", "linux", "solaris"))
 skip_on_os("mac", arch = "x86_64")

 skip_if_not_installed("modeldata")

 set.seed(591)
 tr <- sim_classification(1000)

 expect_error(
  fit <- brulee_logistic_reg(class ~ ., data = tr, device = "mps"),
  regex = NA
 )
})

test_that('mlp on gpu', {

 skip_if_not(torch::torch_is_installed())
 skip_on_os(c("windows", "linux", "solaris"))
 skip_on_os("mac", arch = "x86_64")

 skip_if_not_installed("modeldata")

 set.seed(591)
 tr <- sim_regression(1000)

 expect_error(
  fit <- brulee_mlp(outcome ~ ., data = tr, device = "mps"),
  regex = NA
 )
})

