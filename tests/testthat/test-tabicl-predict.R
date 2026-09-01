# Tests for the user-facing predict() types of brulee_tab_icl() regression fits
# (R/tabicl-predict.R).
#
# There is no Python golden for ensemble-level quantiles or variance -- the dump
# script lives in the uncommitted dev/ tree -- so these are compositional: the
# per-member readouts are parity-tested in test-tabicl-quantile.R, the engine's
# use of them in test-tabicl-ensemble.R, and what is checked here is the shape,
# the pooling invariants, and the affine equivariance of the returned columns.
# A future golden fixture would need `quantiles` at known alphas and `variance`
# from TabICLRegressor.predict(output_type = ...) at n_estimators = 1.
#
# `tabicl_local_cache()` scopes the fake weight cache to its caller's frame, so
# it has to be called inside each test rather than from a shared setup helper.

test_that("type = 'quantile' returns a quantile_pred column", {
  skip_if_no_tabicl_fixtures("engine_reg")
  f <- tabicl_load_fixture("engine_reg")
  tabicl_local_cache(f, tabicl_fixture_meta("engine_reg"))
  d <- tabicl_reg_data(f)

  fit <- brulee_tab_icl(d$x_train, d$y_train, num_estimators = 1L)
  out <- predict(fit, d$x_test, type = "quantile")

  expect_named(out, ".pred_quantile")
  expect_equal(nrow(out), nrow(d$x_test))
  expect_s3_class(out$.pred_quantile, "quantile_pred")
  expect_equal(attr(out$.pred_quantile, "quantile_levels"), (1:9) / 10)
  expect_equal(dim(as.matrix(out$.pred_quantile)), c(nrow(d$x_test), 9L))
})

test_that("quantile predictions are monotone across levels", {
  skip_if_no_tabicl_fixtures("engine_reg")
  f <- tabicl_load_fixture("engine_reg")
  tabicl_local_cache(f, tabicl_fixture_meta("engine_reg"))
  d <- tabicl_reg_data(f)

  fit <- brulee_tab_icl(d$x_train, d$y_train, num_estimators = 1L)

  # Averaging monotone quantile curves preserves monotonicity exactly (floating
  # point addition and division by a positive constant are monotone), so this
  # needs no tolerance.
  for (lvls in list((1:9) / 10, c(0.01, 0.5, 0.99), c(0.2, 0.4, 0.6, 0.8))) {
    out <- predict(fit, d$x_test, type = "quantile", quantile_levels = lvls)
    qm <- as.matrix(out$.pred_quantile)
    expect_equal(ncol(qm), length(lvls))
    expect_true(all(apply(qm, 1, function(x) all(diff(x) >= 0))))
  }
})

test_that("a single quantile level agrees with the ported median", {
  skip_if_no_tabicl_fixtures("engine_reg")
  f <- tabicl_load_fixture("engine_reg")
  tabicl_local_cache(f, tabicl_fixture_meta("engine_reg"))
  d <- tabicl_reg_data(f)

  fit <- brulee_tab_icl(d$x_train, d$y_train, num_estimators = 1L)
  out <- predict(fit, d$x_test, type = "quantile", quantile_levels = 0.5)
  qm <- as.matrix(out$.pred_quantile)
  expect_equal(dim(qm), c(nrow(d$x_test), 1L))

  # Not a tautology: tabicl_qdist_median() passes a scalar tensor and takes the
  # squeeze branch of the icdf, while a length-1 vector of levels takes the
  # other one. This checks the two entry points agree.
  loaded <- brulee:::tabicl_load_model(
    fit$path,
    task = "regression",
    device = "cpu"
  )
  ref <- tabicl_reference_dist(
    loaded,
    as.matrix(d$x_train),
    d$y_train,
    as.matrix(d$x_test)
  )
  expect_equal(
    as.numeric(qm),
    as.numeric(brulee:::tabicl_qdist_median(ref$dist)$squeeze(1)) *
      ref$y_scale +
      ref$y_mean,
    tolerance = 1e-6
  )
})

test_that("type = 'variance' returns non-negative variances", {
  skip_if_no_tabicl_fixtures("engine_reg")
  f <- tabicl_load_fixture("engine_reg")
  tabicl_local_cache(f, tabicl_fixture_meta("engine_reg"))
  d <- tabicl_reg_data(f)

  fit <- brulee_tab_icl(d$x_train, d$y_train, num_estimators = 1L)
  out <- predict(fit, d$x_test, type = "variance")

  expect_named(out, ".pred_variance")
  expect_equal(nrow(out), nrow(d$x_test))
  expect_type(out$.pred_variance, "double")
  expect_true(all(out$.pred_variance >= 0))
  expect_true(all(is.finite(out$.pred_variance)))
})

test_that("type = 'numeric' is unchanged and ignores quantile_levels", {
  skip_if_no_tabicl_fixtures("engine_reg")
  f <- tabicl_load_fixture("engine_reg")
  tabicl_local_cache(f, tabicl_fixture_meta("engine_reg"))
  d <- tabicl_reg_data(f)

  fit <- brulee_tab_icl(d$x_train, d$y_train, num_estimators = 1L)
  base <- predict(fit, d$x_test)
  expect_identical(base, predict(fit, d$x_test, type = "numeric"))
  expect_identical(
    base,
    predict(fit, d$x_test, quantile_levels = c(0.25, 0.75))
  )
})

test_that("predictions are equivariant under an affine rescaling of y", {
  skip_if_no_tabicl_fixtures("engine_reg")
  f <- tabicl_load_fixture("engine_reg")
  tabicl_local_cache(f, tabicl_fixture_meta("engine_reg"))
  d <- tabicl_reg_data(f)

  a <- 100
  b <- 3
  fit_1 <- brulee_tab_icl(d$x_train, d$y_train, num_estimators = 1L)
  fit_2 <- brulee_tab_icl(d$x_train, a + b * d$y_train, num_estimators = 1L)

  # The internal target scaler maps both outcomes to the same standardized
  # values, so the forward passes agree and every returned statistic must
  # transform the way its units say it should.
  expect_equal(
    predict(fit_2, d$x_test)$.pred,
    a + b * predict(fit_1, d$x_test)$.pred,
    tolerance = 1e-5
  )
  expect_equal(
    as.matrix(predict(fit_2, d$x_test, type = "quantile")$.pred_quantile),
    a +
      b * as.matrix(predict(fit_1, d$x_test, type = "quantile")$.pred_quantile),
    tolerance = 1e-5
  )
  # The direct guard on the deliberate deviation from the reference, which
  # would scale the variance by b (and shift it) rather than by b^2.
  expect_equal(
    predict(fit_2, d$x_test, type = "variance")$.pred_variance,
    b^2 * predict(fit_1, d$x_test, type = "variance")$.pred_variance,
    tolerance = 1e-5
  )
})

test_that("predict() rejects bad types and quantile levels", {
  skip_if_no_tabicl_fixtures("engine_reg")
  f <- tabicl_load_fixture("engine_reg")
  tabicl_local_cache(f, tabicl_fixture_meta("engine_reg"))
  d <- tabicl_reg_data(f)

  fit <- brulee_tab_icl(d$x_train, d$y_train, num_estimators = 1L)
  x_test <- d$x_test

  expect_snapshot(error = TRUE, predict(fit, x_test, type = "prob"))
  expect_snapshot(error = TRUE, predict(fit, x_test, type = "bogus"))
  expect_snapshot(
    error = TRUE,
    predict(fit, x_test, type = "quantile", quantile_levels = c(0.5, 0.1))
  )
  expect_snapshot(
    error = TRUE,
    predict(fit, x_test, type = "quantile", quantile_levels = c(0, 0.5))
  )
  expect_snapshot(
    error = TRUE,
    predict(fit, x_test, type = "quantile", quantile_levels = c(0.5, 1))
  )
  expect_snapshot(
    error = TRUE,
    predict(fit, x_test, type = "quantile", quantile_levels = c(0.1, 0.1))
  )
  expect_snapshot(
    error = TRUE,
    predict(fit, x_test, type = "quantile", quantile_levels = numeric(0))
  )
  expect_snapshot(
    error = TRUE,
    predict(fit, x_test, type = "quantile", quantile_levels = "a")
  )
  expect_snapshot(
    error = TRUE,
    predict(fit, x_test, type = "quantile", quantile_levels = c(0.1, NA))
  )
})

test_that("a classification fit rejects the regression-only types", {
  skip_if_no_tabicl_fixtures("engine_clf")
  f <- tabicl_load_fixture("engine_clf")
  tabicl_local_cache(f, tabicl_fixture_meta("engine_clf"))

  x_train <- as.data.frame(as.matrix(as.array(f$X_train)))
  y_train <- factor(as.integer(as.array(f$y_train)))
  x_test <- as.data.frame(as.matrix(as.array(f$X_test)))
  names(x_test) <- names(x_train)

  fit <- brulee_tab_icl(x_train, y_train, num_estimators = 1L)
  expect_snapshot(error = TRUE, predict(fit, x_test, type = "quantile"))
  expect_snapshot(error = TRUE, predict(fit, x_test, type = "variance"))
})
