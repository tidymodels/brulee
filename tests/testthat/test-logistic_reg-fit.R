test_that("logistic regression", {
 skip_if_not(torch::torch_is_installed())
 skip_if(packageVersion("rlang") < "1.0.0")
 skip_on_os(c("windows", "linux", "solaris"))

 # ------------------------------------------------------------------------------

 n <- 1000
 b <- c(-1, -3, 5)
 set.seed(1)
 df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
 lp <- b[1] + b[2] * df$x1 + b[3] * df$x2
 prob <- binomial()$linkinv(lp)
 df$y <- ifelse(prob <= runif(n), "a", "b")
 df$y <- factor(df$y)

 glm_fit <- glm(y ~ ., data = df, family = "binomial")

 # ------------------------------------------------------------------------------

 expect_snapshot({
  set.seed(1)
  logistic_reg_fit_lbfgs <- brulee_logistic_reg(y ~ ., df, epochs = 2, penalty = 0)
 })

 # regression tests
 save_coef(logistic_reg_fit_lbfgs)
 expect_equal(
  last_param(logistic_reg_fit_lbfgs),
  load_coef(logistic_reg_fit_lbfgs),
  tolerance = 0.1
 )

 expect_snapshot({
  print(logistic_reg_fit_lbfgs)
 })

 expect_error(
  logistic_reg_fit_sgd <-
   brulee_logistic_reg(y ~ ., df, epochs = 10, learn_rate = 0.1,
                       optimizer = "SGD"),
  regexp = NA
 )

 # regression tests
 save_coef(logistic_reg_fit_sgd)
 expect_equal(
  last_param(logistic_reg_fit_sgd),
  load_coef(logistic_reg_fit_sgd),
  tolerance = 0.1
 )

 expect_equal(names(coef(logistic_reg_fit_sgd)), c("(Intercept)", "x1", "x2"))
 expect_equal(sign(coef(logistic_reg_fit_sgd)), sign(coef(glm_fit)))
})

# ------------------------------------------------------------------------------

test_that("class weights - logistic regression", {
 skip_if_not(torch::torch_is_installed())
 skip_if(packageVersion("rlang") < "1.0.0")
 skip_on_os(c("windows", "linux", "solaris"))

 # ------------------------------------------------------------------------------

 n <- 1000
 b <- c(8, -3, 5)
 set.seed(1)
 df_imbal <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
 lp <- b[1] + b[2] * df_imbal$x1 + b[3] * df_imbal$x2
 prob <- binomial()$linkinv(lp)
 df_imbal$y <- ifelse(prob <= runif(n), "a", "b")
 df_imbal$y <- factor(df_imbal$y)

 # ------------------------------------------------------------------------------

 expect_error({
  set.seed(1)
  logistic_reg_fit_sgd_wts <-
   brulee_logistic_reg(
    y ~ .,
    df_imbal,
    class_weights = 20,
    optimizer = "SGD",
    penalty = 0
   )
 },
 regexp = NA
 )

 # regression tests
 save_coef(logistic_reg_fit_sgd_wts)
 expect_equal(
  last_param(logistic_reg_fit_sgd_wts),
  load_coef(logistic_reg_fit_sgd_wts),
  tolerance = 0.1
 )

 expect_error({
  set.seed(1)
  logistic_reg_fit_lbfgs_wts <-
   brulee_logistic_reg(
    y ~ .,
    df_imbal,
    epochs = 2,
    class_weights = c(a = 12, b = 1),
    penalty = 0
   )},
  regexp = NA
 )

 # regression tests
 save_coef(logistic_reg_fit_lbfgs_wts)
 expect_equal(
  last_param(logistic_reg_fit_lbfgs_wts),
  load_coef(logistic_reg_fit_lbfgs_wts),
  tolerance = 0.1
 )

 expect_error({
  set.seed(1)
  fit_bal <- brulee_logistic_reg(y ~ ., df_imbal, learn_rate = 0.1,
                                 optimizer = "SGD")},
  regexp = NA
 )

 expect_true(
  sum(predict(fit_bal, df_imbal) == "a") <
   sum(predict(logistic_reg_fit_sgd_wts, df_imbal) == "a")
 )

})

