test_that("multinomial regression", {
 skip_if_not(torch::torch_is_installed())
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
  multinom_fit_lbfgs <-
   brulee_multinomial_reg(y ~ .,
                          df,
                          epochs = 2,
                          penalty = 0)},
  regexp = NA
 )

 # regression tests
 save_coef(multinom_fit_lbfgs)
 expect_equal(
  last_param(multinom_fit_lbfgs),
  load_coef(multinom_fit_lbfgs),
  tolerance = 0.1
 )

 expect_snapshot(print(multinom_fit_lbfgs))

 expect_error({
  set.seed(1)
  multinom_fit_sgd <-
   brulee_multinomial_reg(y ~ .,
                          df,
                          epochs = 10,
                          learn_rate = 0.1,
                          optimizer = "SGD")},
  regexp = NA
 )

 # regression tests
 save_coef(multinom_fit_sgd)
 expect_equal(
  last_param(multinom_fit_sgd),
  load_coef(multinom_fit_sgd),
  tolerance = 0.1
 )

})

# ------------------------------------------------------------------------------

test_that("class weights - multinomial regression", {
 skip_if_not(torch::torch_is_installed())
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
  multinom_fit_sgd_wts_20 <-
   brulee_multinomial_reg(y ~ ., df,
                          class_weights = 20,
                          optimizer = "SGD")},
  regexp = NA
 )

 # regression tests
 save_coef(multinom_fit_sgd_wts_20)
 expect_equal(
  last_param(multinom_fit_sgd_wts_20),
  load_coef(multinom_fit_sgd_wts_20),
  tolerance = 0.1
 )


 expect_error({
  set.seed(1)
  multinom_fit_lbfgs_wts_12 <-
   brulee_multinomial_reg(y ~ ., df, epochs = 2,
                          class_weights = c(a = 12, b = 1, c = 1),
                          penalty = 0)},
  regexp = NA)

 # regression tests
 save_coef(multinom_fit_lbfgs_wts_12)
 expect_equal(
  last_param(multinom_fit_lbfgs_wts_12),
  load_coef(multinom_fit_lbfgs_wts_12),
  tolerance = 0.1
 )

 expect_error({
  set.seed(1)
  fit_bal <- brulee_multinomial_reg(y ~ ., df, learn_rate = 0.1,
                                    optimizer = "SGD")
 },
 regexp = NA
 )

 expect_true(
  names(sort(table(predict(fit_bal, df))))[1] == "c"
 )
 expect_true(
  names(sort(table(predict(multinom_fit_sgd_wts_20, df))))[3] == "c"
 )
})

