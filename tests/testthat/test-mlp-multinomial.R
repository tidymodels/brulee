
test_that("multinomial mlp", {
 skip_if_not(torch::torch_is_installed())

 # ------------------------------------------------------------------------------

 n <- 10000
 b <- cbind(c(8, -3, 5), c(-0.1, 3, 7), c(-2, -5, -5))

 set.seed(1)
 df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))

 mat <- cbind(rep(1, n), as.matrix(df))
 lps <- mat %*% b
 probs <-  binomial()$linkinv(lps)
 probs <- apply(probs, 1, function(x) exp(x)/ sum(exp(x)))
 probs <- t(probs)
 df$y <- apply(probs, 1, function(x) sample(letters[1:3], size = 1, prob = x))
 df$y <- factor(df$y)

 # ------------------------------------------------------------------------------

 expect_error({
  set.seed(1)
  mlp_mlt_mat_lbfgs_fit <- brulee_mlp(y ~ ., df, epochs = 2)
 },
 regexp = NA
 )

 # regression tests
 save_coef(mlp_mlt_mat_lbfgs_fit)
 expect_equal(
  last_param(mlp_mlt_mat_lbfgs_fit),
  load_coef(mlp_mlt_mat_lbfgs_fit),
  tolerance = 0.1
 )

})

# ------------------------------------------------------------------------------

test_that("class weights - mlp", {
 skip_if_not(torch::torch_is_installed())
 # One test here was irreducible across OSes
 skip_on_os(c("windows", "linux", "solaris"))

 # ------------------------------------------------------------------------------

 n <- 10000
 b <- cbind(c(8, -3, 5), c(-0.1, 3, 7), c(-2, -5, -5))

 set.seed(1)
 df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))

 mat <- cbind(rep(1, n), as.matrix(df))
 lps <- mat %*% b
 probs <-  binomial()$linkinv(lps)
 probs <- apply(probs, 1, function(x) exp(x)/ sum(exp(x)))
 probs <- t(probs)
 df$y <- apply(probs, 1, function(x) sample(letters[1:3], size = 1, prob = x))
 df$y <- factor(df$y)

 # ------------------------------------------------------------------------------

 expect_error({
  set.seed(1)
  mlp_bin_lbfgs_fit_20 <- brulee_mlp(y ~ ., df, class_weights = 20)
 },
 regexp = NA
 )

 # NOTE this one fails across operating systems, each with different answers
 # regression tests
 save_coef(mlp_bin_lbfgs_fit_20)
 expect_equal(
  last_param(mlp_bin_lbfgs_fit_20),
  load_coef(mlp_bin_lbfgs_fit_20),
  tolerance = 0.1
 )

 expect_error({
  set.seed(1)
  mlp_bin_lbfgs_fit_12 <- brulee_mlp(y ~ ., df, epochs = 2,
                                     class_weights = c(a = 12, b = 1, c = 1))
 },
 regexp = NA
 )

 # regression tests
 save_coef(mlp_bin_lbfgs_fit_12)
 expect_equal(
  last_param(mlp_bin_lbfgs_fit_12),
  load_coef(mlp_bin_lbfgs_fit_12),
  tolerance = 0.1
 )

 expect_error({
  set.seed(1)
  fit_bal <- brulee_mlp(y ~ ., df, learn_rate = 0.1)
 },
 regexp = NA
 )

 expect_true(
  names(sort(table(predict(fit_bal, df))))[1] == "c"
 )
 expect_true(
  names(sort(table(predict(mlp_bin_lbfgs_fit_20, df))))[3] == "c"
 )
})

