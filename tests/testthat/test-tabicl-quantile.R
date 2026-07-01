# Parity tests for the regression head (R/tabicl-quantile.R) against the Python
# reference (QuantileToDistribution with exp tails and sort crossing). The
# fixture (dev/tabicl/dump_primitives.py) holds noisy predicted quantiles plus
# the reference mean / variance / median / quantiles / monotonized quantiles.
# The requested alphas include points in both tails and the spline interior.

test_that("tabicl_quantile_dist matches the Python reference", {
  skip_if_no_tabicl_fixtures("quantile_dist")

  f <- tabicl_load_fixture("quantile_dist")
  meta <- tabicl_fixture_meta("quantile_dist")

  dist <- brulee:::tabicl_quantile_dist(f$quantiles)

  expect_lt(tabicl_max_abs_diff(dist$quantiles, f$raw_quantiles), 1e-5)
  expect_lt(tabicl_max_abs_diff(brulee:::tabicl_qdist_mean(dist), f$mean), 1e-5)
  expect_lt(
    tabicl_max_abs_diff(brulee:::tabicl_qdist_variance(dist), f$variance),
    1e-5
  )
  expect_lt(
    tabicl_max_abs_diff(brulee:::tabicl_qdist_median(dist), f$median),
    1e-5
  )

  qs <- brulee:::tabicl_qdist_quantiles(dist, meta$alphas)
  expect_equal(dim(qs), dim(f$quantiles_at_alphas))
  expect_lt(tabicl_max_abs_diff(qs, f$quantiles_at_alphas), 1e-5)
})

test_that("tabicl_quantile_dist monotonizes crossing quantiles", {
  skip_on_cran()
  skip_if_not_installed("torch")
  if (!torch::torch_is_installed()) {
    skip("libtorch not installed")
  }

  # Deliberately decreasing quantiles must come back sorted.
  q <- torch::torch_tensor(matrix(c(5, 1, 3, 2, 4), nrow = 1))
  dist <- brulee:::tabicl_quantile_dist(q)
  vals <- as.numeric(dist$quantiles)
  expect_equal(vals, sort(vals))
})
