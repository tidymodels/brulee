
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

test_that("logistic regression", {
  skip_if_not(torch::torch_is_installed())

  expect_snapshot({
    set.seed(1)
    fit <- brulee_logistic_reg(y ~ ., df, epochs = 2, verbose = TRUE)
  })

  expect_snapshot({
    fit
  })

  expect_error(
    fit <- brulee_logistic_reg(y ~ ., df, epochs = 10, learn_rate = 0.1,
                               optimizer = "SGD"),
    regexp = NA
  )

  expect_equal(names(coef(fit)), c("(Intercept)", "x1", "x2"))
  expect_equal(sign(coef(fit)), sign(coef(glm_fit)))
})

# ------------------------------------------------------------------------------

test_that("class weights - logistic regression", {
  skip_if_not(torch::torch_is_installed())

  n <- 1000
  b <- c(8, -3, 5)
  set.seed(1)
  df_imbal <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  lp <- b[1] + b[2] * df_imbal$x1 + b[3] * df_imbal$x2
  prob <- binomial()$linkinv(lp)
  df_imbal$y <- ifelse(prob <= runif(n), "a", "b")
  df_imbal$y <- factor(df_imbal$y)

  expect_snapshot({
    set.seed(1)
    fit_imbal <- brulee_logistic_reg(y ~ ., df_imbal, verbose = TRUE,
                                     class_weights = 20,
                                     optimizer = "SGD")
  })


  expect_snapshot({
    set.seed(1)
    fit <- brulee_logistic_reg(y ~ ., df_imbal, epochs = 2, verbose = TRUE,
                               class_weights = c(a = 12, b = 1))
  })

  expect_error({
    set.seed(1)
    fit_bal <- brulee_logistic_reg(y ~ ., df_imbal, learn_rate = 0.1,
                                   optimizer = "SGD")
  },
  regexp = NA
  )

  expect_true(
    sum(predict(fit_bal, df_imbal) == "a") < sum(predict(fit_imbal, df_imbal) == "a")
  )

})

