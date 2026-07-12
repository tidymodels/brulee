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
  tabicl_local_cache(f, meta)

  x_train <- as.data.frame(as.matrix(as.array(f$X_train)))
  y_train <- factor(as.integer(as.numeric(as.array(f$y_train))))
  x_test <- as.data.frame(as.matrix(as.array(f$X_test)))
  names(x_test) <- names(x_train)

  fit <- brulee_tab_icl(x_train, y_train, num_estimators = 1L)
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
  tabicl_local_cache(f, meta)

  x_train <- as.data.frame(as.matrix(as.array(f$X_train)))
  y_train <- as.numeric(as.array(f$y_train))
  x_test <- as.data.frame(as.matrix(as.array(f$X_test)))
  names(x_test) <- names(x_train)

  fit <- brulee_tab_icl(x_train, y_train, num_estimators = 1L)
  pred <- predict(fit, x_test)

  expect_equal(nrow(pred), nrow(x_test))
  expect_lt(
    max(abs(pred$.pred - as.numeric(as.array(f$mean)))),
    1e-5
  )
})

# --- subsample helper ---------------------------------------------------------

test_that("tabicl_subsample_indices returns NULL when limit is Inf or n <= limit", {
  set.seed(20260618)
  outcome <- factor(sample(c("a", "b"), 100, replace = TRUE))
  expect_null(brulee:::tabicl_subsample_indices(outcome, Inf))
  expect_null(brulee:::tabicl_subsample_indices(outcome, 100))
  expect_null(brulee:::tabicl_subsample_indices(outcome, 200))
})

test_that("tabicl_subsample_indices keeps exactly `limit` rows", {
  set.seed(20260618)
  outcome_factor <- factor(sample(c("a", "b", "c"), 500, replace = TRUE))
  outcome_num <- rnorm(500)

  idx_factor <- brulee:::tabicl_subsample_indices(outcome_factor, 120)
  expect_length(idx_factor, 120L)
  expect_true(all(idx_factor %in% seq_along(outcome_factor)))

  idx_num <- brulee:::tabicl_subsample_indices(outcome_num, 120)
  expect_length(idx_num, 120L)
  expect_true(all(idx_num %in% seq_along(outcome_num)))
})

test_that("tabicl_subsample_indices stratifies classification samples", {
  set.seed(20260618)
  outcome <- factor(c(
    rep("rare", 5),
    rep("common", 95),
    rep("medium", 50)
  ))
  idx <- brulee:::tabicl_subsample_indices(outcome, 30)
  expect_length(idx, 30L)
  kept <- outcome[idx]
  expect_named(table(kept), levels(outcome), ignore.order = TRUE)
  expect_all_true(as.integer(table(kept)) >= 1L)
})

test_that("tabicl_subsample_indices errors when limit < number of classes", {
  outcome <- factor(c("a", "b", "c", "d", "a", "b"))
  expect_snapshot(
    error = TRUE,
    brulee:::tabicl_subsample_indices(outcome, 3)
  )
})

# --- validation ---------------------------------------------------------------

test_that("brulee_tab_icl errors when no checkpoint is cached", {
  skip_on_cran()
  skip_if_not_installed("torch")
  if (!torch::torch_is_installed()) {
    skip("libtorch not installed")
  }
  # Point the cache at an empty directory so nothing is found. The error names
  # the (temporary) cache path, so match the stable part.
  withr::local_options(brulee.tabicl_cache_dir = withr::local_tempdir())
  x_train <- data.frame(a = rnorm(10), b = rnorm(10))
  y_train <- factor(sample(c("a", "b"), 10, replace = TRUE))
  expect_error(
    brulee_tab_icl(x_train, y_train),
    "No cached Classification"
  )
})

test_that("brulee_tab_icl errors when the task's checkpoint is not cached", {
  skip_if_no_tabicl_fixtures("engine_clf")

  f <- tabicl_load_fixture("engine_clf")
  meta <- tabicl_fixture_meta("engine_clf")
  tabicl_local_cache(f, meta) # caches the classification checkpoint only

  x_train <- as.data.frame(as.matrix(as.array(f$X_train)))
  y_num <- as.numeric(as.array(f$y_train))

  # A numeric outcome needs the regression checkpoint, which is not cached. The
  # error names the (temporary) cache path, so match the stable part.
  expect_error(
    brulee_tab_icl(x_train, y_num),
    "No cached Regression"
  )
})
