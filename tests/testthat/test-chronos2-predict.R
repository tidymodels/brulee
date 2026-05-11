# predict.brulee_chronos() tests beyond the basic shape checks in
# test-chronos2-fit.R. Most of these depend on the same stubbing helpers
# already defined in test-chronos2-fit.R; testthat sources every file
# alphabetically into the same env, so `stub_chronos_loaders()` and
# `chicago_subset()` are available here.

skip_if_not_installed("hardhat")
skip_if_not_installed("recipes")
skip_if_not_installed("modeldata")

# ------------------------------------------------------------------------------
# prediction_length / quantile_levels overrides

test_that("predict falls back to the object's prediction_length / quantile_levels", {
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = chi,
    id_column = "series_id",
    timestamp_column = "date",
    prediction_length = 7L,
    quantile_levels = c(0.2, 0.5, 0.8)
  )

  out <- predict(mod)
  expect_equal(nrow(out), 7L)
  expect_equal(ncol(as.matrix(out$.pred_quantile)), 3L)
  expect_equal(
    attr(out$.pred_quantile, "quantile_levels"),
    c(0.2, 0.5, 0.8)
  )
})

# ------------------------------------------------------------------------------
# new_data round-trips

test_that("predict with new_data forges covariates from a formula model", {
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    ridership ~ Clark_Lake + Austin,
    data = Chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  out <- predict(mod, new_data = Chi, prediction_length = 3L)
  expect_s3_class(out, "tbl_df")
  expect_named(out, c(".pred", ".pred_quantile"))
  expect_equal(nrow(out), 3L)
})

test_that("predict with new_data forges through a recipe blueprint", {
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()

  rec <- recipes::recipe(ridership ~ ., data = Chi) |>
    recipes::update_role(series_id, new_role = "id") |>
    recipes::update_role(date, new_role = "time")
  mod <- brulee_chronos(rec, data = Chi)

  out <- predict(mod, new_data = Chi, prediction_length = 4L)
  expect_s3_class(out, "tbl_df")
  expect_equal(nrow(out), 4L)
})

test_that("predict with new_data on a multi-series model returns id column", {
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()
  multi_series <- rbind(
    transform(Chi[, c("series_id", "date", "ridership")], series_id = "A"),
    transform(Chi[, c("series_id", "date", "ridership")], series_id = "B")
  )

  mod <- brulee_chronos(
    ridership ~ .,
    data = multi_series,
    id_column = "series_id",
    timestamp_column = "date"
  )

  out <- predict(mod, new_data = multi_series, prediction_length = 3L)
  expect_named(out, c("series_id", ".pred", ".pred_quantile"))
  expect_setequal(out$series_id, c("A", "B"))
})

test_that("predict with new_data missing the id column errors", {
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    ridership ~ .,
    data = Chi[, c("series_id", "date", "ridership")],
    id_column = "series_id",
    timestamp_column = "date"
  )

  bad <- Chi[, c("date", "ridership")]
  expect_snapshot(error = TRUE, {
    predict(mod, new_data = bad, prediction_length = 3L)
  })
})

test_that("predict with new_data containing a non-numeric target errors", {
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  bad <- chi
  bad$ridership <- as.character(bad$ridership)
  expect_snapshot(error = TRUE, {
    predict(mod, new_data = bad, prediction_length = 3L)
  })
})

# ------------------------------------------------------------------------------
# future_df validation

test_that("predict with valid future_df runs", {
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    ridership ~ Clark_Lake + Austin,
    data = Chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  future_df <- data.frame(
    series_id = rep("L", 5),
    date = seq(max(Chi$date) + 1, by = "day", length.out = 5),
    Clark_Lake = rnorm(5),
    Austin = rnorm(5)
  )

  out <- predict(mod, future_df = future_df, prediction_length = 5L)
  expect_s3_class(out, "tbl_df")
  expect_equal(nrow(out), 5L)
})

test_that("future_df missing the id column errors", {
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    ridership ~ Clark_Lake,
    data = Chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  future_df <- data.frame(
    date = seq(max(Chi$date) + 1, by = "day", length.out = 3),
    Clark_Lake = rnorm(3)
  )

  expect_snapshot(error = TRUE, {
    predict(mod, future_df = future_df, prediction_length = 3L)
  })
})

test_that("future_df missing the timestamp column errors", {
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    ridership ~ Clark_Lake,
    data = Chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  future_df <- data.frame(
    series_id = rep("L", 3),
    Clark_Lake = rnorm(3)
  )

  expect_snapshot(error = TRUE, {
    predict(mod, future_df = future_df, prediction_length = 3L)
  })
})

test_that("future_df with an unknown covariate column errors", {
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    ridership ~ Clark_Lake,
    data = Chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  future_df <- data.frame(
    series_id = rep("L", 3),
    date = seq(max(Chi$date) + 1, by = "day", length.out = 3),
    something_else = rnorm(3)
  )

  expect_snapshot(error = TRUE, {
    predict(mod, future_df = future_df, prediction_length = 3L)
  })
})

# ------------------------------------------------------------------------------
# chronos2_pull_column helper

test_that("chronos2_pull_column errors when the column is missing", {
  expect_snapshot(error = TRUE, {
    brulee:::chronos2_pull_column(
      data.frame(a = 1:3),
      "missing_col",
      "id_column"
    )
  })
})

test_that("chronos2_pull_column returns the column value when present", {
  res <- brulee:::chronos2_pull_column(
    data.frame(a = 1:3, b = letters[1:3]),
    "b",
    "x"
  )
  expect_equal(res, letters[1:3])
})
