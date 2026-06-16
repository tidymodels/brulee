# Tests for the TabICL prediction engine (R/tabicl-ensemble.R).
#
# The single-member, identity-shuffle, "none"-normalization configuration is the
# deterministic one the sklearn wrappers use at n_estimators = 1. The committed
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
