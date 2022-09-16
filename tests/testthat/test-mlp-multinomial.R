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

test_that("multinomial mlp", {
  skip_if_not(torch::torch_is_installed())
  skip_if(packageVersion("rlang") < "1.0.0")
  skip_on_os(c("windows", "linux", "solaris"))

  expect_snapshot({
    set.seed(1)
    fit <- brulee_mlp(y ~ ., df, epochs = 2, verbose = TRUE)
  })

  expect_snapshot({
    fit
  })

  expect_error(
    fit <- brulee_mlp(y ~ ., df, epochs = 10, learn_rate = 0.1),
    regexp = NA
  )
})

# ------------------------------------------------------------------------------

test_that("class weights - mlp", {
  skip_if_not(torch::torch_is_installed())
  skip_if(packageVersion("rlang") < "1.0.0")
  skip_on_os(c("windows", "linux", "solaris"))

  expect_snapshot({
    set.seed(1)
    fit_imbal <- brulee_mlp(y ~ ., df, verbose = TRUE,
                            class_weights = 20)
  })


  expect_snapshot({
    set.seed(1)
    fit <- brulee_mlp(y ~ ., df, epochs = 2, verbose = TRUE,
                      class_weights = c(a = 12, b = 1, c = 1))
  })

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
    names(sort(table(predict(fit_imbal, df))))[3] == "c"
  )
})

