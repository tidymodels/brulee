
set.seed(1)
df <- tibble::tibble(
  x1 = rnorm(100),
  x2 = rnorm(100),
  logit = x1 + x2 + rnorm(100, sd = 0.25),
  y = as.factor(ifelse(exp(logit)/(1 + exp(logit)) > 0.6, "a", "b"))
)
df$logit <- NULL

# ------------------------------------------------------------------------------

test_that("logistic regression", {
  skip_if_not(torch::torch_is_installed())

  expect_snapshot({
    set.seed(1)
    fit <- lantern_logistic_reg(y ~ ., df, epochs = 2, verbose = TRUE)
  })

  expect_snapshot({
    fit
  })

  expect_error(
    fit <- lantern_logistic_reg(y ~ ., df, epochs = 10, learn_rate = 0.1,
                                optimizer = "SGD"),
    regexp = NA
  )
})

# ------------------------------------------------------------------------------

test_that("class weights - logistic regression", {
  skip_if_not(torch::torch_is_installed())

  set.seed(1)
  df_imbal <- tibble::tibble(
    x1 = rnorm(200),
    x2 = rnorm(200),
    logit = x1 + x2 + rnorm(200, sd = 0.25),
    y = as.factor(ifelse(exp(logit)/(1 + exp(logit)) > 0.8, "a", "b"))
  )
  df$logit <- NULL

  expect_snapshot({
    set.seed(1)
    fit_imbal <- lantern_logistic_reg(y ~ ., df_imbal, verbose = TRUE,
                                      class_weights = 20,
                                      optimizer = "SGD")
  })


  expect_snapshot({
    set.seed(1)
    fit <- lantern_logistic_reg(y ~ ., df_imbal, epochs = 2, verbose = TRUE,
                                class_weights = c(a = 12, b = 1))
  })

  expect_error({
    set.seed(1)
    fit_bal <- lantern_logistic_reg(y ~ ., df_imbal, learn_rate = 0.1,
                                    optimizer = "SGD")
  },
  regexp = NA
  )

  expect_true(
    sum(predict(fit_bal, df_imbal) == "a") < sum(predict(fit_imbal, df_imbal) == "a")
  )

})

