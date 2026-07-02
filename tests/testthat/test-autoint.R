# ------------------------------------------------------------------------------
# Regression tests

test_that("autoint regression - formula interface", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    g = factor(sample(letters[1:3], n, replace = TRUE))
  )
  df$y <- df$x1 + 2 * df$x2 + rnorm(n, sd = 0.5)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 5,
    verbose = FALSE,
    device = "cpu"
  )

  expect_s3_class(fit, "brulee_auto_int")
  expect_true(inherits(fit$model_obj, "raw"))
  expect_true(is.list(fit$estimates))
  expect_true(is.numeric(fit$loss))
  expect_true(is.integer(fit$best_epoch))
  expect_null(fit$parameters$hidden_units)
  expect_null(fit$parameters$hidden_activations)

  pred <- predict(fit, df)
  expect_s3_class(pred, "tbl_df")
  expect_equal(nrow(pred), n)
  expect_true(".pred" %in% names(pred))
})

test_that("autoint regression - data.frame interface (numeric only)", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  x <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  y <- data.frame(outcome = x$x1 + rnorm(n, sd = 0.1))

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    x = x,
    y = y,
    epochs = 3,
    verbose = FALSE,
    device = "cpu"
  )

  expect_s3_class(fit, "brulee_auto_int")
  expect_equal(fit$dims$p_cat, 0L)
  expect_equal(fit$dims$p_cont, 2L)

  pred <- predict(fit, x)
  expect_equal(nrow(pred), n)
})

test_that("autoint regression - matrix interface", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  x <- matrix(rnorm(n * 3), ncol = 3)
  colnames(x) <- c("a", "b", "c")
  y <- x[, 1] + x[, 2] + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    x = x,
    y = y,
    epochs = 3,
    verbose = FALSE,
    device = "cpu"
  )

  expect_s3_class(fit, "brulee_auto_int")
  pred <- predict(fit, x)
  expect_equal(nrow(pred), n)
})

test_that("autoint regression - recipe interface", {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip_if_not_installed("recipes")

  # Temp check to remind to see if issues with the gower package have been resolved
  if (packageVersion("brulee") > "1.1.0.9001") {
    cli::cli_abort("Recheck RSPM version of gower")
  }

  set.seed(1)
  n <- 80
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    g = factor(sample(c("a", "b"), n, replace = TRUE)),
    y = rnorm(n)
  )

  rec <- recipes::recipe(y ~ ., data = df)
  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    rec,
    data = df,
    epochs = 3,
    verbose = FALSE,
    device = "cpu"
  )

  expect_s3_class(fit, "brulee_auto_int")
  pred <- predict(fit, df)
  expect_equal(nrow(pred), n)
})

# ------------------------------------------------------------------------------
# Classification tests

test_that("autoint binary classification", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 120
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    y = factor(sample(c("a", "b"), n, replace = TRUE))
  )

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 5,
    verbose = FALSE,
    device = "cpu"
  )

  expect_s3_class(fit, "brulee_auto_int")

  pred_class <- predict(fit, df, type = "class")
  expect_equal(nrow(pred_class), n)
  expect_true(".pred_class" %in% names(pred_class))
  expect_s3_class(pred_class$.pred_class, "factor")
  expect_equal(levels(pred_class$.pred_class), c("a", "b"))

  pred_prob <- predict(fit, df, type = "prob")
  expect_equal(nrow(pred_prob), n)
  expect_true(all(c(".pred_a", ".pred_b") %in% names(pred_prob)))
  expect_true(all(pred_prob$.pred_a >= 0 & pred_prob$.pred_a <= 1))
})

test_that("autoint multiclass classification", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 150
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    y = factor(sample(c("cat", "dog", "bird"), n, replace = TRUE))
  )

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 5,
    verbose = FALSE,
    device = "cpu"
  )

  pred_class <- predict(fit, df, type = "class")
  expect_equal(nrow(pred_class), n)
  expect_equal(levels(pred_class$.pred_class), c("bird", "cat", "dog"))

  pred_prob <- predict(fit, df, type = "prob")
  expect_equal(ncol(pred_prob), 3)
  row_sums <- rowSums(as.matrix(pred_prob))
  expect_true(all(abs(row_sums - 1) < 1e-5))
})

test_that("autoint classification with class_weights", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    y = factor(c(rep("a", 80), rep("b", 20)))
  )

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 5,
    class_weights = c(a = 1, b = 4),
    verbose = FALSE,
    device = "cpu"
  )

  expect_s3_class(fit, "brulee_auto_int")
  pred <- predict(fit, df, type = "class")
  expect_equal(nrow(pred), n)
})

# ------------------------------------------------------------------------------
# Hidden layer tests

test_that("autoint with single hidden layer", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    g = factor(sample(letters[1:3], n, replace = TRUE))
  )
  df$y <- df$x1 + 2 * df$x2 + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 5,
    hidden_units = 32L,
    hidden_activations = "relu",
    verbose = FALSE,
    device = "cpu"
  )

  expect_s3_class(fit, "brulee_auto_int")
  expect_equal(fit$parameters$hidden_units, 32L)
  expect_equal(fit$parameters$hidden_activations, "relu")

  pred <- predict(fit, df)
  expect_equal(nrow(pred), n)
})

test_that("autoint with multiple hidden layers", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 5,
    hidden_units = c(64L, 32L),
    hidden_activations = c("relu", "tanh"),
    verbose = FALSE,
    device = "cpu"
  )

  expect_equal(fit$parameters$hidden_units, c(64L, 32L))
  expect_equal(fit$parameters$hidden_activations, c("relu", "tanh"))

  pred <- predict(fit, df)
  expect_equal(nrow(pred), n)
})

test_that("autoint hidden layer activation recycling", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    hidden_units = c(16L, 8L),
    hidden_activations = "tanh",
    verbose = FALSE,
    device = "cpu"
  )

  expect_equal(fit$parameters$hidden_activations, c("tanh", "tanh"))
})

test_that("autoint hidden layer dropout", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 5,
    hidden_units = 32L,
    hidden_activations = "relu",
    dropout = 0.5,
    verbose = FALSE,
    device = "cpu"
  )

  expect_s3_class(fit, "brulee_auto_int")
  expect_equal(fit$parameters$dropout, 0.5)

  pred <- predict(fit, df)
  expect_equal(nrow(pred), n)
})

# ------------------------------------------------------------------------------
# Validation error tests

test_that("autoint hidden layer validation errors", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  expect_snapshot(
    brulee_auto_int(y ~ ., data = df, hidden_units = 16L),
    error = TRUE
  )

  expect_snapshot(
    brulee_auto_int(y ~ ., data = df, hidden_activations = "relu"),
    error = TRUE
  )

  expect_snapshot(
    brulee_auto_int(
      y ~ .,
      data = df,
      hidden_units = c(16L, 8L),
      hidden_activations = c("relu", "bad_name")
    ),
    error = TRUE
  )

  expect_snapshot(
    brulee_auto_int(
      y ~ .,
      data = df,
      hidden_units = c(16L, 8L),
      hidden_activations = c("relu", "tanh", "elu")
    ),
    error = TRUE
  )
})

test_that("autoint attention parameter validation errors", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  expect_snapshot(
    brulee_auto_int(y ~ ., data = df, num_embedding = -1),
    error = TRUE
  )

  expect_snapshot(
    brulee_auto_int(y ~ ., data = df, num_attn_feat = 0),
    error = TRUE
  )

  expect_snapshot(
    brulee_auto_int(y ~ ., data = df, dropout_attn = 1.5),
    error = TRUE
  )

  expect_snapshot(
    brulee_auto_int(y ~ ., data = df, dropout_embedding = 1.0),
    error = TRUE
  )

  expect_snapshot(
    brulee_auto_int(y ~ ., data = df, activation = "not_real"),
    error = TRUE
  )
})

test_that("autoint gradient clipping argument validation", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(386)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  expect_snapshot(
    brulee_auto_int(y ~ ., data = df, grad_norm_clip = -1),
    error = TRUE
  )

  expect_snapshot(
    brulee_auto_int(y ~ ., data = df, grad_value_clip = -1),
    error = TRUE
  )
})

test_that("autoint gradient clipping prevents loss overflow", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(386)
  n <- 200
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
  df$class <- factor(ifelse(df$x1 - df$x2 + rnorm(n) > 0, "a", "b"))

  auto_int_args <- list(
    class ~ .,
    data = df,
    num_attn_blocks = 3L,
    num_attn_heads = 4L,
    num_embedding = 16L,
    learn_rate = 0.5,
    momentum = 0.9,
    optimizer = "SGD",
    batch_size = 16L,
    epochs = 3L,
    validation = 0,
    device = "cpu",
    verbose = FALSE
  )

  # Without clipping, the aggressive learning rate overflows the loss
  # Changed from snapshot beucase GHA and local stop at different epochs with
  # the same overflow issue
  set.seed(386)
  torch::torch_manual_seed(386)
  expect_warning(
    no_clip <- do.call(
      brulee_auto_int,
      c(auto_int_args, grad_value_clip = Inf, grad_norm_clip = Inf)
    ),
    regexp = "numerical overflow of the loss function"
  )
  expect_true(any(is.nan(no_clip$loss)))

  skip_on_os("mac")
  # With the default clipping, training completes without overflow
  set.seed(386)
  torch::torch_manual_seed(386)
  clipped <- do.call(
    brulee_auto_int,
    c(auto_int_args, grad_value_clip = 3, grad_norm_clip = 3)
  )
  expect_false(any(is.nan(clipped$loss)))
})

test_that("autoint default method errors on unsupported types", {
  skip_on_cran()
  skip_if_not_installed("torch")

  expect_snapshot(
    brulee_auto_int(list(a = 1)),
    error = TRUE
  )
})

# ------------------------------------------------------------------------------
# Feature handling tests

test_that("autoint with only categorical predictors", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(
    g1 = factor(sample(letters[1:4], n, replace = TRUE)),
    g2 = factor(sample(c("x", "y", "z"), n, replace = TRUE)),
    y = rnorm(n)
  )

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    verbose = FALSE,
    device = "cpu"
  )

  expect_s3_class(fit, "brulee_auto_int")
  expect_equal(fit$dims$p_cat, 2L)
  expect_equal(fit$dims$p_cont, 0L)

  pred <- predict(fit, df)
  expect_equal(nrow(pred), n)
})

test_that("autoint with only continuous predictors", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    verbose = FALSE,
    device = "cpu"
  )

  expect_equal(fit$dims$p_cat, 0L)
  expect_equal(fit$dims$p_cont, 3L)

  pred <- predict(fit, df)
  expect_equal(nrow(pred), n)
})

test_that("autoint with mixed predictors", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(
    x1 = rnorm(n),
    g1 = factor(sample(letters[1:3], n, replace = TRUE)),
    x2 = rnorm(n),
    g2 = factor(sample(c("yes", "no"), n, replace = TRUE)),
    y = rnorm(n)
  )

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    verbose = FALSE,
    device = "cpu"
  )

  expect_equal(fit$dims$p_cat, 2L)
  expect_equal(fit$dims$p_cont, 2L)
  expect_equal(fit$dims$p, 4L)
})

# ------------------------------------------------------------------------------
# Prediction tests

test_that("autoint prediction with specific epoch", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 5,
    verbose = FALSE,
    device = "cpu"
  )

  pred_best <- predict(fit, df)
  pred_epoch1 <- predict(fit, df, epoch = 1)

  expect_equal(nrow(pred_best), n)
  expect_equal(nrow(pred_epoch1), n)
})

test_that("autoint prediction with epoch = 0 and epoch = max", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    verbose = FALSE,
    device = "cpu"
  )

  # epoch 0 should work (initial parameters)
  pred0 <- predict(fit, df, epoch = 0)
  expect_equal(nrow(pred0), n)

  # best_epoch should work
  pred_best <- predict(fit, df, epoch = fit$best_epoch)
  expect_equal(nrow(pred_best), n)
})

# ------------------------------------------------------------------------------
# Optimizer and training tests

test_that("autoint with different optimizers", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit_sgd <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    optimizer = "SGD",
    learn_rate = 0.001,
    momentum = 0.9,
    verbose = FALSE,
    device = "cpu"
  )
  expect_s3_class(fit_sgd, "brulee_auto_int")

  set.seed(1)
  torch::torch_manual_seed(1)
  fit_lbfgs <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    optimizer = "LBFGS",
    verbose = FALSE,
    device = "cpu"
  )
  expect_s3_class(fit_lbfgs, "brulee_auto_int")
})

test_that("autoint with validation = 0", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    validation = 0,
    verbose = FALSE,
    device = "cpu"
  )
  expect_s3_class(fit, "brulee_auto_int")
  expect_equal(fit$parameters$validation, 0)
})

test_that("autoint with penalty and mixture", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    penalty = 0.1,
    verbose = FALSE,
    device = "cpu"
  )
  expect_equal(fit$parameters$penalty, 0.1)
})

test_that("autoint with batch_size", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    batch_size = 32L,
    verbose = FALSE,
    device = "cpu"
  )
  expect_equal(fit$parameters$batch_size, 32L)
})

# ------------------------------------------------------------------------------
# Attention parameter tests

test_that("autoint with various attention configurations", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    num_embedding = 8L,
    num_attn_feat = 8L,
    num_attn_heads = 4L,
    num_attn_blocks = 2L,
    activation = "tanh",
    verbose = FALSE,
    device = "cpu"
  )

  expect_equal(fit$parameters$num_embedding, 8L)
  expect_equal(fit$parameters$num_attn_feat, 8L)
  expect_equal(fit$parameters$num_attn_heads, 4L)
  expect_equal(fit$parameters$num_attn_blocks, 2L)
  expect_equal(fit$parameters$activation, "tanh")
})

test_that("autoint with dropout_attn and dropout_embedding", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    dropout_attn = 0.2,
    dropout_embedding = 0.1,
    verbose = FALSE,
    device = "cpu"
  )

  expect_equal(fit$parameters$dropout_attn, 0.2)
  expect_equal(fit$parameters$dropout_embedding, 0.1)
})

# ------------------------------------------------------------------------------
# Top interactions

test_that("autoint returns top interactions", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    x3 = rnorm(n),
    y = rnorm(n)
  )

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 5,
    verbose = FALSE,
    device = "cpu"
  )

  expect_true("top_interactions" %in% names(fit))
  expect_s3_class(fit$top_interactions, "tbl_df")
  expect_true(all(
    c("feature_1", "feature_2", "attention_weight") %in%
      names(fit$top_interactions)
  ))
  expect_true(nrow(fit$top_interactions) > 0)
})

# ------------------------------------------------------------------------------
# Print and autoplot tests

test_that("autoint print method works (no hidden layers)", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(2)
  torch::torch_manual_seed(2)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    verbose = FALSE,
    device = "cpu"
  )

  output <- capture_all_output(print(fit))
  expect_true(any(grepl("AutoInt network", output)))
  expect_true(any(grepl("Attention", output)))
  expect_false(any(grepl("Hidden layers", output)))
})

test_that("autoint print method works (with hidden layers + dropout)", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(2)
  torch::torch_manual_seed(2)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    hidden_units = 32L,
    hidden_activations = "relu",
    dropout = 0.2,
    verbose = FALSE,
    device = "cpu"
  )

  output <- capture_all_output(print(fit))
  expect_true(any(grepl("Hidden layers", output)))
  expect_true(any(grepl("dropout", output)))
})

test_that("autoint print method shows validation loss for regression", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(2)
  torch::torch_manual_seed(2)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    verbose = FALSE,
    device = "cpu"
  )
  output <- capture_all_output(print(fit))
  expect_true(any(grepl("scaled validation loss", output)))
})

test_that("autoint print method shows training loss when validation = 0", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(2)
  torch::torch_manual_seed(2)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    validation = 0,
    verbose = FALSE,
    device = "cpu"
  )
  output <- capture_all_output(print(fit))
  expect_true(any(grepl("training set loss", output)))
})

test_that("autoint print method for classification", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    y = factor(sample(c("a", "b"), n, replace = TRUE))
  )

  set.seed(2)
  torch::torch_manual_seed(2)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    verbose = FALSE,
    device = "cpu"
  )
  output <- capture_all_output(print(fit))
  expect_true(any(grepl("Classes", output)))
  expect_true(any(grepl("validation loss", output)))
})

test_that("autoint autoplot works", {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip_if_not_installed("ggplot2")

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(2)
  torch::torch_manual_seed(2)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 5,
    verbose = FALSE,
    device = "cpu"
  )

  p <- ggplot2::autoplot(fit)
  expect_s3_class(p, "ggplot")
})

# ------------------------------------------------------------------------------
# Learning rate schedule test

test_that("autoint with learning rate schedule", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 5,
    rate_schedule = "decay_time",
    verbose = FALSE,
    device = "cpu"
  )
  expect_s3_class(fit, "brulee_auto_int")
  expect_equal(fit$parameters$sched, "decay_time")
})

# ------------------------------------------------------------------------------
# Additional coverage tests

test_that("autoint print shows dropout_attn/dropout_embedding", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(2)
  torch::torch_manual_seed(2)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    dropout_attn = 0.1,
    dropout_embedding = 0.2,
    verbose = FALSE,
    device = "cpu"
  )

  output <- capture_all_output(print(fit))
  expect_true(any(grepl("Dropout.*attention", output)))
})

test_that("autoint print with LBFGS (no batch size displayed)", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(2)
  torch::torch_manual_seed(2)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    optimizer = "LBFGS",
    verbose = FALSE,
    device = "cpu"
  )

  output <- capture_all_output(print(fit))
  expect_true(any(grepl("LBFGS", output)))
  expect_false(any(grepl("Batch Size", output)))
})

test_that("autoint print with no penalty", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(2)
  torch::torch_manual_seed(2)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    penalty = 0,
    verbose = FALSE,
    device = "cpu"
  )

  output <- capture_all_output(print(fit))
  expect_false(any(grepl("Penalty", output)))
})

test_that("autoint classification with validation = 0", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    y = factor(sample(c("a", "b"), n, replace = TRUE))
  )

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    validation = 0,
    verbose = FALSE,
    device = "cpu"
  )

  output <- capture_all_output(print(fit))
  expect_true(any(grepl("training set loss", output)))

  pred <- predict(fit, df, type = "class")
  expect_equal(nrow(pred), n)
})

test_that("autoint with L1 penalty via SGD", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 80
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    optimizer = "SGD",
    learn_rate = 0.001,
    momentum = 0.9,
    penalty = 0.01,
    mixture = 1.0,
    verbose = FALSE,
    device = "cpu"
  )
  expect_s3_class(fit, "brulee_auto_int")
})

test_that("autoint with early stopping", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 100
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 100,
    stop_iter = 3,
    verbose = FALSE,
    device = "cpu"
  )
  expect_true(fit$best_epoch < 100)
})

test_that("autoint coef method returns estimates", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(2)
  torch::torch_manual_seed(2)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    verbose = FALSE,
    device = "cpu"
  )
  expect_true(is.list(fit$estimates))
  expect_true(length(fit$estimates) > 0)
})

test_that("autoint with numeric hidden_units (coerced to integer)", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    hidden_units = 16,
    hidden_activations = "relu",
    verbose = FALSE,
    device = "cpu"
  )
  expect_equal(fit$parameters$hidden_units, 16L)
})

test_that("autoint with numeric attention params (coerced to integer)", {
  skip_on_cran()
  skip_if_not_installed("torch")

  set.seed(1)
  n <- 50
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$y <- df$x1 + rnorm(n, sd = 0.1)

  set.seed(1)
  torch::torch_manual_seed(1)
  fit <- brulee_auto_int(
    y ~ .,
    data = df,
    epochs = 3,
    num_embedding = 8,
    num_attn_feat = 8,
    num_attn_heads = 2,
    num_attn_blocks = 2,
    verbose = FALSE,
    device = "cpu"
  )
  expect_equal(fit$parameters$num_embedding, 8L)
  expect_equal(fit$parameters$num_attn_feat, 8L)
})
