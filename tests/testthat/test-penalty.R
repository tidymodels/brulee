test_that("penalty works with all optimizers and mixture values", {
  skip_if_not(torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  suppressPackageStartupMessages(library(dplyr))

  # Use small dataset for speed
  data("parabolic", package = "modeldata")
  set.seed(1)
  in_train <- sample(1:nrow(parabolic), 300)
  parabolic_tr <- parabolic[in_train,]

  # Test parameters
  optimizers <- c("SGD", "LBFGS", "ADAMw", "RMSprop", "Adadelta", "Adagrad")
  mixture_values <- c(0, 0.5, 1)  # L2, elastic net, L1
  penalty_values <- c(0, 0.1, 10)

  # For each optimizer
  for (opt in optimizers) {
    # Skip long-running combinations for CRAN
    if (identical(Sys.getenv("NOT_CRAN"), "false") && opt %in% c("Adadelta", "Adagrad")) {
      next
    }

    # Test pure L2 (mixture = 0)
    set.seed(123)
    fit_no_penalty <- brulee_mlp(
      class ~ .,
      data = parabolic_tr,
      hidden_units = 2,
      epochs = 50L,
      learn_rate = if (opt == "ADAMw") 0.01 else 0.1,
      activation = "elu",
      penalty = 0,
      mixture = 0,
      batch_size = 256,
      optimizer = opt,
      verbose = FALSE
    )

    set.seed(123)
    fit_with_penalty <- brulee_mlp(
      class ~ .,
      data = parabolic_tr,
      hidden_units = 2,
      epochs = 50L,
      learn_rate = if (opt == "ADAMw") 0.01 else 0.1,
      activation = "elu",
      penalty = 10,
      mixture = 0,
      batch_size = 256,
      optimizer = opt,
      verbose = FALSE
    )

    # Penalty should make a difference
    expect_false(
      isTRUE(all.equal(fit_no_penalty$loss, fit_with_penalty$loss)),
      info = paste0(opt, " with mixture=0: penalty should affect loss")
    )

    # With penalty, loss should generally be higher or converge differently
    # (due to regularization)
    expect_true(
      !identical(tail(fit_no_penalty$loss, 1), tail(fit_with_penalty$loss, 1)),
      info = paste0(opt, " with mixture=0: final losses should differ")
    )

    # Test L1 penalty (mixture = 1) - skip for ADAMw as it requires mixture = 0
    if (opt != "ADAMw") {
      set.seed(123)
      fit_no_penalty_l1 <- brulee_mlp(
        class ~ .,
        data = parabolic_tr,
        hidden_units = 2,
        epochs = 50L,
        learn_rate = 0.1,
        activation = "elu",
        penalty = 0,
        mixture = 1,
        batch_size = 256,
        optimizer = opt,
        verbose = FALSE
      )

      set.seed(123)
      fit_with_penalty_l1 <- brulee_mlp(
        class ~ .,
        data = parabolic_tr,
        hidden_units = 2,
        epochs = 50L,
        learn_rate = 0.1,
        activation = "elu",
        penalty = 10,
        mixture = 1,
        batch_size = 256,
        optimizer = opt,
        verbose = FALSE
      )

      # Penalty should make a difference
      expect_false(
        isTRUE(all.equal(fit_no_penalty_l1$loss, fit_with_penalty_l1$loss)),
        info = paste0(opt, " with mixture=1: penalty should affect loss")
      )

      # Test elastic net (mixture = 0.5)
      set.seed(123)
      fit_no_penalty_en <- brulee_mlp(
        class ~ .,
        data = parabolic_tr,
        hidden_units = 2,
        epochs = 50L,
        learn_rate = 0.1,
        activation = "elu",
        penalty = 0,
        mixture = 0.5,
        batch_size = 256,
        optimizer = opt,
        verbose = FALSE
      )

      set.seed(123)
      fit_with_penalty_en <- brulee_mlp(
        class ~ .,
        data = parabolic_tr,
        hidden_units = 2,
        epochs = 50L,
        learn_rate = 0.1,
        activation = "elu",
        penalty = 10,
        mixture = 0.5,
        batch_size = 256,
        optimizer = opt,
        verbose = FALSE
      )

      # Penalty should make a difference
      expect_false(
        isTRUE(all.equal(fit_no_penalty_en$loss, fit_with_penalty_en$loss)),
        info = paste0(opt, " with mixture=0.5: penalty should affect loss")
      )
    }
  }
})

test_that("ADAMw enforces pure L2 penalty", {
  skip_if_not(torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  data("parabolic", package = "modeldata")
  set.seed(1)
  in_train <- sample(1:nrow(parabolic), 300)
  parabolic_tr <- parabolic[in_train,]

  # ADAMw with mixture = 0 should work fine
  expect_no_error({
    set.seed(123)
    fit_adamw_l2 <- brulee_mlp(
      class ~ .,
      data = parabolic_tr,
      hidden_units = 2,
      epochs = 50L,
      learn_rate = 0.01,
      activation = "elu",
      penalty = 0.1,
      mixture = 0,
      batch_size = 256,
      optimizer = "ADAMw",
      verbose = FALSE
    )
  })

  # ADAMw with mixture != 0 should warn and convert to 0
  expect_warning({
    set.seed(123)
    fit_adamw_mixed <- brulee_mlp(
      class ~ .,
      data = parabolic_tr,
      hidden_units = 2,
      epochs = 50L,
      learn_rate = 0.01,
      activation = "elu",
      penalty = 0.1,
      mixture = 0.5,
      batch_size = 256,
      optimizer = "ADAMw",
      verbose = FALSE
    )
  }, regexp = "pure L2 penalty")

  # The warning case should produce same result as mixture = 0
  expect_equal(fit_adamw_l2$loss, fit_adamw_mixed$loss)
})

test_that("penalty magnitude affects regularization strength", {
  skip_if_not(torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  data("parabolic", package = "modeldata")
  set.seed(1)
  in_train <- sample(1:nrow(parabolic), 300)
  parabolic_tr <- parabolic[in_train,]

  # Test with SGD - different penalty values should produce different results
  set.seed(123)
  fit_pen_0 <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 2,
    epochs = 100L,
    learn_rate = 0.1,
    activation = "elu",
    penalty = 0,
    mixture = 0,
    batch_size = 256,
    optimizer = "SGD",
    verbose = FALSE
  )

  set.seed(123)
  fit_pen_01 <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 2,
    epochs = 100L,
    learn_rate = 0.1,
    activation = "elu",
    penalty = 0.1,
    mixture = 0,
    batch_size = 256,
    optimizer = "SGD",
    verbose = FALSE
  )

  set.seed(123)
  fit_pen_10 <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 2,
    epochs = 100L,
    learn_rate = 0.1,
    activation = "elu",
    penalty = 10,
    mixture = 0,
    batch_size = 256,
    optimizer = "SGD",
    verbose = FALSE
  )

  # All three should be different
  expect_false(isTRUE(all.equal(fit_pen_0$loss, fit_pen_01$loss)))
  expect_false(isTRUE(all.equal(fit_pen_0$loss, fit_pen_10$loss)))
  expect_false(isTRUE(all.equal(fit_pen_01$loss, fit_pen_10$loss)))

  # Higher penalty should generally lead to higher final loss (more regularization)
  # or at least different convergence
  expect_true(tail(fit_pen_10$loss, 1) > tail(fit_pen_01$loss, 1))
})

test_that("L1 vs L2 vs elastic net produce different results", {
  skip_if_not(torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  data("parabolic", package = "modeldata")
  set.seed(1)
  in_train <- sample(1:nrow(parabolic), 300)
  parabolic_tr <- parabolic[in_train,]

  penalty_val <- 1

  # L2 (mixture = 0)
  set.seed(123)
  fit_l2 <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 2,
    epochs = 100L,
    learn_rate = 0.1,
    activation = "elu",
    penalty = penalty_val,
    mixture = 0,
    batch_size = 256,
    optimizer = "SGD",
    verbose = FALSE
  )

  # L1 (mixture = 1)
  set.seed(123)
  fit_l1 <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 2,
    epochs = 100L,
    learn_rate = 0.1,
    activation = "elu",
    penalty = penalty_val,
    mixture = 1,
    batch_size = 256,
    optimizer = "SGD",
    verbose = FALSE
  )

  # Elastic net (mixture = 0.5)
  set.seed(123)
  fit_en <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 2,
    epochs = 100L,
    learn_rate = 0.1,
    activation = "elu",
    penalty = penalty_val,
    mixture = 0.5,
    batch_size = 256,
    optimizer = "SGD",
    verbose = FALSE
  )

  # All three should produce different results
  expect_false(isTRUE(all.equal(fit_l2$loss, fit_l1$loss)))
  expect_false(isTRUE(all.equal(fit_l2$loss, fit_en$loss)))
  expect_false(isTRUE(all.equal(fit_l1$loss, fit_en$loss)))
})

test_that("LBFGS penalty works correctly", {
  skip_if_not(torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  data("parabolic", package = "modeldata")
  set.seed(1)
  in_train <- sample(1:nrow(parabolic), 200)
  parabolic_tr <- parabolic[in_train,]

  # LBFGS with no penalty
  set.seed(123)
  fit_no_pen <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 2,
    epochs = 50L,
    learn_rate = 0.1,
    activation = "elu",
    penalty = 0,
    mixture = 0,
    optimizer = "LBFGS",
    verbose = FALSE
  )

  # LBFGS with penalty (uses loss function, not weight_decay)
  set.seed(123)
  fit_with_pen <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 2,
    epochs = 50L,
    learn_rate = 0.1,
    activation = "elu",
    penalty = 1,
    mixture = 0,
    optimizer = "LBFGS",
    verbose = FALSE
  )

  # Should produce different results
  expect_false(isTRUE(all.equal(fit_no_pen$loss, fit_with_pen$loss)))

  # LBFGS with L1 penalty should also work
  set.seed(123)
  fit_with_l1 <- brulee_mlp(
    class ~ .,
    data = parabolic_tr,
    hidden_units = 2,
    epochs = 50L,
    learn_rate = 0.1,
    activation = "elu",
    penalty = 1,
    mixture = 1,
    optimizer = "LBFGS",
    verbose = FALSE
  )

  # Should produce different results from both no penalty and L2
  expect_false(isTRUE(all.equal(fit_no_pen$loss, fit_with_l1$loss)))
  expect_false(isTRUE(all.equal(fit_with_pen$loss, fit_with_l1$loss)))
})
