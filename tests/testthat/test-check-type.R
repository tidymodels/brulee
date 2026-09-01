# Tests for the shared prediction-type helper (R/mlp-predict.R). It is used by
# every predict() method in the package, so the default behaviour is pinned here
# as a guard: the `numeric_types` argument that `brulee_tab_icl()` uses to allow
# "quantile" / "variance" must not leak into the set the other models accept.

stub_model <- function(outcome) {
  list(blueprint = list(ptypes = list(outcomes = tibble::tibble(y = outcome))))
}

num_model <- stub_model(numeric())
fct_model <- stub_model(factor(levels = c("a", "b")))

test_that("check_type picks the natural type for the outcome", {
  expect_identical(brulee:::check_type(num_model, NULL), "numeric")
  expect_identical(brulee:::check_type(fct_model, NULL), "class")
  expect_identical(brulee:::check_type(num_model, "numeric"), "numeric")
  expect_identical(brulee:::check_type(fct_model, "prob"), "prob")
})

test_that("check_type defaults still reject distributional types", {
  # The guard: models that did not opt in must not gain the TabICL types.
  expect_snapshot(error = TRUE, brulee:::check_type(num_model, "quantile"))
  expect_snapshot(error = TRUE, brulee:::check_type(num_model, "variance"))
  expect_snapshot(error = TRUE, brulee:::check_type(num_model, "prob"))
  expect_snapshot(error = TRUE, brulee:::check_type(fct_model, "numeric"))
})

test_that("check_type honours a widened numeric_types", {
  wide <- c("numeric", "quantile", "variance")
  expect_identical(
    brulee:::check_type(num_model, "quantile", numeric_types = wide),
    "quantile"
  )
  expect_identical(
    brulee:::check_type(num_model, "variance", numeric_types = wide),
    "variance"
  )
  expect_identical(
    brulee:::check_type(num_model, NULL, numeric_types = wide),
    "numeric"
  )
})

test_that("a widened numeric_types still rejects bad input", {
  wide <- c("numeric", "quantile", "variance")
  # A factor outcome must not reach the regression-only types.
  expect_snapshot(
    error = TRUE,
    brulee:::check_type(fct_model, "quantile", numeric_types = wide)
  )
  expect_snapshot(
    error = TRUE,
    brulee:::check_type(num_model, "bogus", numeric_types = wide)
  )
})
