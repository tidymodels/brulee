# Tests for the TabICL prediction engine (R/tabicl-ensemble.R).
#
# The single-member, identity-shuffle, "none"-normalization configuration is the
# deterministic one the sklearn wrappers use at num_estimators = 1. The committed
# fixtures (engine_clf / engine_reg) run that exact pipeline through the real
# preprocessing + a small random model, so the end-to-end engine (preprocess ->
# model -> softmax / quantile-mean) is validated without the released checkpoint.

tabicl_engine_weight_keys <- function(f) {
  grep(
    "^(col_embedder|row_interactor|icl_predictor)\\.",
    names(f),
    value = TRUE
  )
}

tabicl_load_engine_model <- function(f, meta) {
  model <- brulee:::tabicl_model(meta$config)
  brulee:::load_tabicl_weights(model, f[tabicl_engine_weight_keys(f)])
  model$eval()
  list(model = model, config = meta$config)
}

test_that("tabicl_softmax_rows applies a temperature softmax per row", {
  skip_on_cran()
  logits <- matrix(c(1, 2, 3, 0, 0, 0), nrow = 2, byrow = TRUE)
  p <- brulee:::tabicl_softmax_rows(logits, temperature = 0.9)
  expect_equal(rowSums(p), c(1, 1), tolerance = 1e-6)
  expect_equal(p[2, ], rep(1 / 3, 3), tolerance = 1e-6) # equal logits -> uniform
  # Reference for row 1.
  ref <- exp(c(1, 2, 3) / 0.9)
  expect_equal(p[1, ], ref / sum(ref), tolerance = 1e-6)
})

test_that("tabicl_unique_filter drops constant features", {
  skip_on_cran()
  x <- cbind(c(1, 2, 3), c(5, 5, 5), c(0, 1, 0))
  expect_equal(brulee:::tabicl_unique_filter(x), c(TRUE, FALSE, TRUE))
})

test_that("tabicl_classifier_proba matches the single-member sklearn pipeline", {
  skip_if_no_tabicl_fixtures("engine_clf")

  f <- tabicl_load_fixture("engine_clf")
  meta <- tabicl_fixture_meta("engine_clf")
  loaded <- tabicl_load_engine_model(f, meta)

  x_train <- as.matrix(as.array(f$X_train))
  x_test <- as.matrix(as.array(f$X_test))
  y_train <- as.integer(as.numeric(as.array(f$y_train)))

  members <- list(brulee:::tabicl_single_member(
    ncol(x_train),
    n_classes = length(unique(y_train))
  ))
  proba <- brulee:::tabicl_classifier_proba(
    loaded,
    x_train,
    y_train,
    x_test,
    members
  )

  expect_equal(dim(proba), dim(as.matrix(as.array(f$proba))))
  expect_lt(max(abs(proba - as.matrix(as.array(f$proba)))), 1e-5)
})

test_that("tabicl_regressor_mean matches the single-member sklearn pipeline", {
  skip_if_no_tabicl_fixtures("engine_reg")

  f <- tabicl_load_fixture("engine_reg")
  meta <- tabicl_fixture_meta("engine_reg")
  loaded <- tabicl_load_engine_model(f, meta)

  x_train <- as.matrix(as.array(f$X_train))
  x_test <- as.matrix(as.array(f$X_test))
  y_train <- as.numeric(as.array(f$y_train))

  members <- list(brulee:::tabicl_single_member(ncol(x_train)))
  out <- brulee:::tabicl_regressor_mean(
    loaded,
    x_train,
    y_train,
    x_test,
    members
  )

  expect_equal(length(out), length(as.numeric(as.array(f$mean))))
  expect_lt(max(abs(out - as.numeric(as.array(f$mean)))), 1e-5)
})

# --- tabicl_regressor_stats ---------------------------------------------------
#
# There is no Python golden for ensemble-level quantiles or variance (the dump
# script lives in the uncommitted dev/ tree), so these validate compositionally:
# the per-member readouts (`tabicl_qdist_*`) are parity-tested against the
# reference in test-tabicl-quantile.R, and what is checked here is that the
# engine calls them correctly and pools members the way it claims to.

test_that("tabicl_regressor_stats reproduces the per-member readouts", {
  skip_if_no_tabicl_fixtures("engine_reg")

  f <- tabicl_load_fixture("engine_reg")
  meta <- tabicl_fixture_meta("engine_reg")
  loaded <- tabicl_load_engine_model(f, meta)

  x_train <- as.matrix(as.array(f$X_train))
  x_test <- as.matrix(as.array(f$X_test))
  y_train <- as.numeric(as.array(f$y_train))
  members <- list(brulee:::tabicl_single_member(ncol(x_train)))
  alphas <- c(0.05, 0.25, 0.5, 0.75, 0.95)

  out <- brulee:::tabicl_regressor_stats(
    loaded,
    x_train,
    y_train,
    x_test,
    members,
    output_type = c("mean", "variance", "quantiles"),
    alphas = alphas
  )

  ref <- tabicl_reference_dist(loaded, x_train, y_train, x_test)

  expect_equal(
    out$quantiles,
    as.matrix(brulee:::tabicl_qdist_quantiles(ref$dist, alphas)$squeeze(1)) *
      ref$y_scale +
      ref$y_mean,
    tolerance = 1e-6
  )
  # A variance is in squared outcome units, so only y_scale^2 applies. The
  # reference implementation instead adds the location shift; brulee does not.
  expect_equal(
    out$variance,
    as.numeric(brulee:::tabicl_qdist_variance(ref$dist)$squeeze(1)) *
      ref$y_scale^2,
    tolerance = 1e-6
  )
})

test_that("tabicl_regressor_stats leaves the mean path byte-identical", {
  skip_if_no_tabicl_fixtures("engine_reg")

  f <- tabicl_load_fixture("engine_reg")
  meta <- tabicl_fixture_meta("engine_reg")
  loaded <- tabicl_load_engine_model(f, meta)

  x_train <- as.matrix(as.array(f$X_train))
  x_test <- as.matrix(as.array(f$X_test))
  y_train <- as.numeric(as.array(f$y_train))
  members <- list(brulee:::tabicl_single_member(ncol(x_train)))

  # Asking for more statistics must not perturb the mean.
  solo <- brulee:::tabicl_regressor_mean(
    loaded,
    x_train,
    y_train,
    x_test,
    members
  )
  combined <- brulee:::tabicl_regressor_stats(
    loaded,
    x_train,
    y_train,
    x_test,
    members,
    output_type = c("mean", "variance", "quantiles"),
    alphas = c(0.1, 0.9)
  )
  expect_identical(combined$mean, solo)
})

test_that("tabicl_regressor_stats pools members per statistic", {
  skip_if_no_tabicl_fixtures("engine_reg")

  f <- tabicl_load_fixture("engine_reg")
  meta <- tabicl_fixture_meta("engine_reg")
  loaded <- tabicl_load_engine_model(f, meta)

  x_train <- as.matrix(as.array(f$X_train))
  x_test <- as.matrix(as.array(f$X_test))
  y_train <- as.numeric(as.array(f$y_train))
  alphas <- c(0.1, 0.5, 0.9)

  # Two deterministic, genuinely different members (no RNG involved).
  n_feat <- sum(brulee:::tabicl_unique_filter(x_train))
  m1 <- brulee:::tabicl_member("none", seq_len(n_feat), NULL)
  m2 <- brulee:::tabicl_member("YeoJohnson", rev(seq_len(n_feat)), NULL)

  run <- function(members) {
    brulee:::tabicl_regressor_stats(
      loaded,
      x_train,
      y_train,
      x_test,
      members,
      output_type = c("mean", "variance", "quantiles"),
      alphas = alphas
    )
  }
  one <- run(list(m1))
  two <- run(list(m2))
  both <- run(list(m1, m2))

  # Locations pool arithmetically (Vincentization for the quantile curves).
  expect_equal(both$mean, (one$mean + two$mean) / 2, tolerance = 1e-8)
  expect_equal(
    both$quantiles,
    (one$quantiles + two$quantiles) / 2,
    tolerance = 1e-8
  )

  # The variance pools geometrically, which is strictly smaller than the
  # arithmetic mean whenever the members disagree.
  expect_equal(
    both$variance,
    exp((log(one$variance) + log(two$variance)) / 2),
    tolerance = 1e-8
  )
  expect_true(all(both$variance <= (one$variance + two$variance) / 2))
  expect_false(isTRUE(all.equal(
    both$variance,
    (one$variance + two$variance) / 2
  )))
})

test_that("geometric pooling sends a degenerate zero-variance member to zero", {
  # Documents the intended semantics of the log-space accumulation: no warning,
  # no NaN, and a single zero member collapses the pooled value.
  expect_warning(res <- exp(mean(log(pmax(c(0, 1, 4), 0)))), NA)
  expect_identical(res, 0)
})

test_that("tabicl_regressor_stats validates its arguments", {
  expect_snapshot(
    error = TRUE,
    brulee:::tabicl_regressor_stats(
      loaded = NULL,
      x_train = NULL,
      y_train = NULL,
      x_test = NULL,
      members = NULL,
      output_type = "quantiles"
    )
  )
  expect_snapshot(
    error = TRUE,
    brulee:::tabicl_regressor_stats(
      loaded = NULL,
      x_train = NULL,
      y_train = NULL,
      x_test = NULL,
      members = NULL,
      output_type = "bogus"
    )
  )
})
