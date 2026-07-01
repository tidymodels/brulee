test_that("basic linear regression LBFGS", {
  skip_on_cran()
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

  expect_no_error(
    {
      set.seed(392)
      torch::torch_manual_seed(392)
      lin_fit_lbfgs <-
        brulee_linear_reg(outcome ~ ., lin_tr, penlaty = 0, device = "cpu")
    }
  )

  expect_equal(
    unname(coef(lm_fit)),
    unname(coef(lin_fit_lbfgs)),
    tolerance = 0.1
  )

  lin_pred_lbfgs <- expect_no_error(
    predict(lin_fit_lbfgs, lin_te) |>
      bind_cols(lin_te)
  )

  exp_str <-
    structure(
      list(
        .pred = numeric(0),
        x1 = numeric(0),
        x2 = numeric(0),
        outcome = numeric(0)
      ),
      row.names = integer(0),
      class = c("tbl_df", "tbl", "data.frame")
    )

  expect_equal(lin_pred_lbfgs[0, ], exp_str)
  expect_equal(nrow(lin_pred_lbfgs), nrow(lin_te))

  # Did it learn anything?
  lin_brier_lbfgs <-
    lin_pred_lbfgs |>
    yardstick::rmse(outcome, .pred)

  set.seed(382)
  shuffled <-
    lin_pred_lbfgs |>
    mutate(outcome = sample(outcome)) |>
    yardstick::rmse(outcome, .pred)

  expect_true(lin_brier_lbfgs$.estimate < shuffled$.estimate)
})

test_that("basic Linear regression sgd", {
  skip_on_cran()
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

  expect_no_error(
    {
      set.seed(392)
      torch::torch_manual_seed(392)
      lin_fit_sgd <-
        brulee_linear_reg(
          outcome ~ .,
          lin_tr,
          penlaty = 0,
          epochs = 500,
          batch_size = 32L,
          learn_rate = 0.1,
          optimizer = "SGD",
          stop_iter = 20,
          device = "cpu"
        )
    }
  )

  expect_equal(
    unname(coef(lm_fit)),
    unname(coef(lin_fit_sgd)),
    tolerance = 0.1
  )

  lin_pred_sgd <- expect_no_error(
    predict(lin_fit_sgd, lin_te) |>
      bind_cols(lin_te)
  )

  exp_str <-
    structure(
      list(
        .pred = numeric(0),
        x1 = numeric(0),
        x2 = numeric(0),
        outcome = numeric(0)
      ),
      row.names = integer(0),
      class = c("tbl_df", "tbl", "data.frame")
    )

  expect_equal(lin_pred_sgd[0, ], exp_str)
  expect_equal(nrow(lin_pred_sgd), nrow(lin_te))

  # Did it learn anything?
  lin_brier_sgd <-
    lin_pred_sgd |>
    yardstick::rmse(outcome, .pred)

  set.seed(382)
  shuffled <-
    lin_pred_sgd |>
    mutate(outcome = sample(outcome)) |>
    yardstick::rmse(outcome, .pred)

  expect_true(lin_brier_sgd$.estimate < shuffled$.estimate)
})

test_that("linear_reg includes epoch zero and coef() returns the best epoch", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  set.seed(1)
  n <- 150
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 - 2 * df$x2 + rnorm(n)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_linear_reg(
    y ~ .,
    data = df,
    epochs = 10L,
    learn_rate = 0.05,
    stop_iter = 100L,
    verbose = FALSE,
    device = "cpu"
  )

  # `loss` and `estimates` include epoch zero (the initial parameters), so they
  # have one more element than the number of training epochs.
  expect_length(fit$loss, 10L + 1L)
  expect_length(fit$estimates, 10L + 1L)

  # `coef()` returns the minimum-loss epoch's parameters, matching `predict()`.
  best <- fit$estimates[[which.min(fit$loss)]]
  expect_equal(
    unname(coef(fit)),
    unname(c(best$fc1.bias, best$fc1.weight[1, ]))
  )

  # `epoch = 0` is now valid and returns the initial (pre-training) parameters.
  init <- fit$estimates[[1]]
  expect_equal(
    unname(coef(fit, epoch = 0)),
    unname(c(init$fc1.bias, init$fc1.weight[1, ]))
  )
})

test_that("print() reports the best epoch's loss (epoch-zero offset)", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  set.seed(1)
  n <- 150
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 - 2 * df$x2 + rnorm(n)

  # Strictly decreasing loss, so the best epoch's loss differs from the one
  # before it; this catches the off-by-one in the printed value.
  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_linear_reg(
    y ~ .,
    data = df,
    epochs = 8L,
    learn_rate = 0.005,
    stop_iter = 100L,
    validation = 0,
    verbose = FALSE,
    device = "cpu"
  )

  out <- capture.output(capture.output(print(fit), type = "message"))
  loss_line <- grep("loss after", out, value = TRUE)
  expect_length(loss_line, 1L)

  best_loss <- signif(fit$loss[fit$best_epoch + 1L], 3)
  expect_match(loss_line, paste0(": ", best_loss), fixed = TRUE)
})
