# Tests for the user-facing brulee_tab_icl() fit / predict, the encoders, the
# device policy, and validation. The end-to-end parity tests build a checkpoint
# directory from the engine fixtures (small random model) and check that the
# whole API path (mold -> encode -> ensemble -> spruce) reproduces the
# single-member sklearn golden, without needing the released checkpoint.

# --- device policy ------------------------------------------------------------

test_that("tabicl_resolve_device defaults to cpu and refuses mps", {
  skip_on_cran()
  skip_if_not_installed("torch")
  expect_equal(brulee:::tabicl_resolve_device("cpu"), "cpu")
  expect_equal(brulee:::tabicl_resolve_device(NULL), "cpu")
  expect_snapshot(dev <- brulee:::tabicl_resolve_device("mps"))
  expect_equal(dev, "cpu")
})

# --- encoders -----------------------------------------------------------------

test_that("tabicl encoders ordinal-encode factors and impute numerics", {
  skip_on_cran()
  df <- data.frame(
    f = factor(c("b", "a", "b", NA), levels = c("a", "b")),
    n = c(1, NA, 3, 5)
  )
  enc <- brulee:::tabicl_encode_fit(df)
  out <- brulee:::tabicl_encode_transform(enc, df)
  # Sorted categories: a -> 0, b -> 1; NA -> -1.
  expect_equal(out[, "f"], c(1, 0, 1, -1))
  # Mean impute (mean of 1, 3, 5 = 3).
  expect_equal(out[, "n"], c(1, 3, 3, 5))
})

# --- end-to-end parity (classification) ---------------------------------------

test_that("brulee_tab_icl classification reproduces the single-member golden", {
  skip_if_no_tabicl_fixtures("engine_clf")

  f <- tabicl_load_fixture("engine_clf")
  meta <- tabicl_fixture_meta("engine_clf")
  dir <- tabicl_write_model_dir(f, meta)

  x_train <- as.data.frame(as.matrix(as.array(f$X_train)))
  y_train <- factor(as.integer(as.numeric(as.array(f$y_train))))
  x_test <- as.data.frame(as.matrix(as.array(f$X_test)))
  names(x_test) <- names(x_train)

  fit <- brulee_tab_icl(
    x_train,
    y_train,
    path = dir,
    n_estimators = 1L
  )
  expect_s3_class(fit, "brulee_tab_icl")

  proba <- predict(fit, x_test, type = "prob")
  expect_equal(nrow(proba), nrow(x_test))
  expect_equal(ncol(proba), nlevels(y_train))
  expect_lt(
    max(abs(as.matrix(proba) - as.matrix(as.array(f$proba)))),
    1e-5
  )

  cls <- predict(fit, x_test, type = "class")
  expect_s3_class(cls$.pred_class, "factor")
  expect_equal(nrow(cls), nrow(x_test))
})

# --- end-to-end parity (regression) -------------------------------------------

test_that("brulee_tab_icl regression reproduces the single-member golden", {
  skip_if_no_tabicl_fixtures("engine_reg")

  f <- tabicl_load_fixture("engine_reg")
  meta <- tabicl_fixture_meta("engine_reg")
  dir <- tabicl_write_model_dir(f, meta)

  x_train <- as.data.frame(as.matrix(as.array(f$X_train)))
  y_train <- as.numeric(as.array(f$y_train))
  x_test <- as.data.frame(as.matrix(as.array(f$X_test)))
  names(x_test) <- names(x_train)

  fit <- brulee_tab_icl(x_train, y_train, path = dir, n_estimators = 1L)
  pred <- predict(fit, x_test)

  expect_equal(nrow(pred), nrow(x_test))
  expect_lt(
    max(abs(pred$.pred - as.numeric(as.array(f$mean)))),
    1e-5
  )
})

# --- validation ---------------------------------------------------------------

test_that("brulee_tab_icl errors when path is missing", {
  skip_on_cran()
  skip_if_not_installed("torch")
  if (!torch::torch_is_installed()) {
    skip("libtorch not installed")
  }
  x_train <- data.frame(a = rnorm(10), b = rnorm(10))
  y_train <- factor(sample(c("a", "b"), 10, replace = TRUE))
  expect_snapshot(error = TRUE, brulee_tab_icl(x_train, y_train))
})

test_that("brulee_tab_icl errors on task / checkpoint mismatch", {
  skip_if_no_tabicl_fixtures("engine_clf")

  f <- tabicl_load_fixture("engine_clf")
  meta <- tabicl_fixture_meta("engine_clf")
  dir <- tabicl_write_model_dir(f, meta)

  x_train <- as.data.frame(as.matrix(as.array(f$X_train)))
  y_num <- as.numeric(as.array(f$y_train))

  # Numeric outcome looks for the regression checkpoint, which is absent in a
  # classification-only directory. The error names the (temporary) path, so match
  # on the stable part of the message.
  expect_error(
    brulee_tab_icl(x_train, y_num, path = dir),
    "No regression checkpoint found"
  )
})
