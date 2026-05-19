# brulee_chronos() dispatch + validation tests.
#
# These tests cover the S3 dispatch, hardhat blueprint, validation, and
# context-splitting logic. Torch model construction and weight loading
# are mocked so the tests don't depend on the pretrained model files
# (and don't trigger the segfault in the safetensors loader).

# `stub_chronos_loaders()` and `chicago_subset()` live in
# tests/testthat/helper-chronos2.R so they're shared with other chronos
# test files.

# ------------------------------------------------------------------------------

test_that("default method errors for unsupported types", {
  expect_snapshot(error = TRUE, brulee_chronos(42L))
  expect_snapshot(error = TRUE, brulee_chronos("oops"))
})

test_that("formula method builds context with no covariates", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("recipes")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  expect_s3_class(mod, "brulee_chronos")
  expect_equal(mod$context$id_column, "series_id")
  expect_equal(mod$context$timestamp_column, "date")
  expect_equal(mod$context$target_column, "ridership")
  expect_equal(length(mod$context$covariate_cols), 0L)
  expect_equal(length(mod$context$item_ids), 1L)

  expect_equal(length(mod$context$series_target[[1]]), nrow(chi))
})

test_that("model_id and revision are recorded on the object", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  expect_equal(mod$model_id, "amazon/chronos-2")
  expect_match(mod$revision, "^[0-9a-f]{40}$")
  expect_equal(mod$revision, brulee:::chronos2_default_revision())
})

test_that("an explicit SHA is passed through verbatim (no API call)", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]
  some_sha <- "1234567890abcdef1234567890abcdef12345678"

  mod <- brulee_chronos(
    ridership ~ .,
    data = chi,
    id_column = "series_id",
    timestamp_column = "date",
    revision = some_sha
  )
  expect_equal(mod$revision, some_sha)
})

test_that("chronos2_resolve_revision returns SHAs unchanged offline", {
  sha <- "1234567890abcdef1234567890abcdef12345678"
  expect_equal(
    brulee:::chronos2_resolve_revision("amazon/chronos-2", sha),
    sha
  )
})

test_that("formula method builds context with covariates", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    ridership ~ Clark_Lake + Austin,
    data = Chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  expect_equal(mod$context$covariate_cols, c("Clark_Lake", "Austin"))
  expect_equal(mod$context$target_column, "ridership")
})

test_that("recipe method detects id/time roles via hardhat extras", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("recipes")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  rec <- recipes::recipe(ridership ~ ., data = Chi) |>
    recipes::update_role(series_id, new_role = "id") |>
    recipes::update_role(date, new_role = "time")

  mod <- brulee_chronos(rec, data = Chi)

  expect_equal(mod$context$id_column, "series_id")
  expect_equal(mod$context$timestamp_column, "date")
  expect_equal(mod$context$target_column, "ridership")
  expect_setequal(mod$context$covariate_cols, c("Clark_Lake", "Austin"))
})

test_that("recipe method synthesizes id when the id role is missing", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("recipes")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi_no_id <- Chi[, c("date", "ridership", "Clark_Lake", "Austin")]

  rec_no_id <- recipes::recipe(ridership ~ ., data = chi_no_id) |>
    recipes::update_role(date, new_role = "time")
  mod <- brulee_chronos(rec_no_id, data = chi_no_id)

  expect_equal(mod$context$id_column, ".id_column")
  expect_true(isTRUE(mod$context$id_synthetic))
  expect_equal(length(mod$context$item_ids), 1L)
  expect_equal(mod$context$item_ids, "default")
})

test_that("recipe method synthesizes timestamp when the time role is missing", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("recipes")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi_no_date <- Chi[, c("series_id", "ridership", "Clark_Lake", "Austin")]

  rec_no_time <- recipes::recipe(ridership ~ ., data = chi_no_date) |>
    recipes::update_role(series_id, new_role = "id")
  mod <- brulee_chronos(rec_no_time, data = chi_no_date)

  expect_equal(mod$context$timestamp_column, ".timestamp_column")
  expect_true(isTRUE(mod$context$timestamp_synthetic))
  expect_equal(mod$context$series_target[[1]], chi_no_date$ridership)
})

test_that("recipe method synthesizes both when neither role is set", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("recipes")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi_numeric <- Chi[, c("ridership", "Clark_Lake", "Austin")]

  rec <- recipes::recipe(ridership ~ ., data = chi_numeric)
  mod <- brulee_chronos(rec, data = chi_numeric)

  expect_true(isTRUE(mod$context$id_synthetic))
  expect_true(isTRUE(mod$context$timestamp_synthetic))
  expect_equal(mod$context$item_ids, "default")
})

test_that("data.frame method builds context with item_id/timestamp vectors", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  mod_df <- brulee_chronos(
    x = Chi[, c("Clark_Lake", "Austin")],
    y = Chi$ridership,
    item_id = Chi$series_id,
    timestamp = Chi$date,
    id_column = "series_id",
    timestamp_column = "date"
  )
  expect_equal(mod_df$context$covariate_cols, c("Clark_Lake", "Austin"))
  expect_false(isTRUE(mod_df$context$id_synthetic))
  expect_false(isTRUE(mod_df$context$timestamp_synthetic))
})

test_that("matrix input dispatches to the default method and errors", {
  skip_if_not_installed("hardhat")
  stub_chronos_loaders()
  expect_snapshot(error = TRUE, {
    brulee_chronos(
      matrix(rnorm(20), nrow = 10, ncol = 2),
      y = rnorm(10)
    )
  })
})

test_that("non-numeric predictors are rejected when mold doesn't encode them", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("recipes")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  Chi$cat <- factor(rep(c("a", "b"), length.out = nrow(Chi)))

  rec <- recipes::recipe(ridership ~ ., data = Chi) |>
    recipes::update_role(series_id, new_role = "id") |>
    recipes::update_role(date, new_role = "time")

  expect_snapshot(error = TRUE, brulee_chronos(rec, data = Chi))
})

test_that("formula factor predictors are encoded by hardhat and pass the check", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  Chi$cat <- factor(rep(c("a", "b"), length.out = nrow(Chi)))

  mod <- brulee_chronos(
    ridership ~ Clark_Lake + cat,
    data = Chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  expect_true("Clark_Lake" %in% mod$context$covariate_cols)
  expect_true(any(grepl("^cat", mod$context$covariate_cols)))
})

test_that("length mismatch between y and item_id errors", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      x = Chi[, "Clark_Lake", drop = FALSE],
      y = Chi$ridership,
      item_id = Chi$series_id[-1],
      timestamp = Chi$date,
      id_column = "series_id",
      timestamp_column = "date"
    )
  })
})

test_that("length mismatch between y and timestamp errors", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      x = Chi[, "Clark_Lake", drop = FALSE],
      y = Chi$ridership,
      item_id = Chi$series_id,
      timestamp = Chi$date[-1],
      id_column = "series_id",
      timestamp_column = "date"
    )
  })
})

test_that("quantile_levels outside (0,1) errors", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      ridership ~ .,
      data = chi,
      id_column = "series_id",
      timestamp_column = "date",
      quantile_levels = c(0, 0.5)
    )
  })
})

test_that("negative prediction_length errors", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      ridership ~ .,
      data = chi,
      id_column = "series_id",
      timestamp_column = "date",
      prediction_length = -1
    )
  })
})

test_that("quantile_levels not available in model errors", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      ridership ~ .,
      data = chi,
      id_column = "series_id",
      timestamp_column = "date",
      quantile_levels = 0.123
    )
  })
})

# ------------------------------------------------------------------------------
# Forge round-trips on the stored blueprint

test_that("formula blueprint forges new_data back to predictors + outcome", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    ridership ~ Clark_Lake + Austin,
    data = Chi,
    id_column = "series_id",
    timestamp_column = "date"
  )
  forged <- hardhat::forge(Chi, mod$blueprint, outcomes = TRUE)

  expect_setequal(names(forged$predictors), c("Clark_Lake", "Austin"))
  expect_equal(names(forged$outcomes), "ridership")
})

test_that("recipe blueprint forges id/time through extras roles", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("recipes")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  rec <- recipes::recipe(ridership ~ ., data = Chi) |>
    recipes::update_role(series_id, new_role = "id") |>
    recipes::update_role(date, new_role = "time")
  mod <- brulee_chronos(rec, data = Chi)

  forged <- hardhat::forge(Chi, mod$blueprint, outcomes = TRUE)

  expect_setequal(names(forged$extras$roles), c("id", "time"))
  expect_equal(forged$extras$roles$id[[1]], Chi$series_id)
  expect_equal(forged$extras$roles$time[[1]], Chi$date)
})

# ------------------------------------------------------------------------------
# predict.brulee_chronos output shape / types

test_that("predict() returns a tibble with .pred + .pred_quantile (single series)", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = chi,
    id_column = "series_id",
    timestamp_column = "date"
  )
  out <- predict(mod, prediction_length = 5L)

  expect_s3_class(out, "tbl_df")
  expect_named(out, c(".pred", ".pred_quantile"))
  expect_equal(nrow(out), 5L)
  expect_type(out$.pred, "double")
  expect_s3_class(out$.pred_quantile, "quantile_pred")
  expect_equal(ncol(as.matrix(out$.pred_quantile)), 9L)
  expect_equal(out$.pred, as.numeric(stats::median(out$.pred_quantile)))
})

test_that("predict() keeps the id column when there are multiple series", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
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
  out <- predict(mod, prediction_length = 4L)

  expect_s3_class(out, "tbl_df")
  expect_named(out, c("series_id", ".pred", ".pred_quantile"))
  expect_setequal(out$series_id, c("A", "B"))
  expect_equal(nrow(out), 2L * 4L)
  expect_s3_class(out$.pred_quantile, "quantile_pred")
})

test_that("predict() respects a custom quantile_levels override", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = chi,
    id_column = "series_id",
    timestamp_column = "date"
  )
  out <- predict(
    mod,
    prediction_length = 3L,
    quantile_levels = c(0.1, 0.5, 0.9)
  )

  expect_equal(nrow(out), 3L)
  expect_s3_class(out$.pred_quantile, "quantile_pred")
  expect_equal(ncol(as.matrix(out$.pred_quantile)), 3L)
  expect_equal(
    attr(out$.pred_quantile, "quantile_levels"),
    c(0.1, 0.5, 0.9)
  )
})

test_that("predict() does not return any timestamp column", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = chi,
    id_column = "series_id",
    timestamp_column = "date"
  )
  out <- predict(mod, prediction_length = 2L)

  expect_false("date" %in% names(out))
  expect_false("timestamp" %in% names(out))
  is_time_col <- vapply(
    out,
    function(col) inherits(col, c("Date", "POSIXct", "POSIXlt")),
    logical(1)
  )
  expect_false(any(is_time_col))
})

# ------------------------------------------------------------------------------
# Optional id_column / timestamp_column on the formula method

test_that("formula synthesizes id and timestamp when both are omitted", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()
  target_only <- Chi[, "ridership", drop = FALSE]

  mod <- brulee_chronos(ridership ~ ., data = target_only)

  expect_equal(mod$context$id_column, ".id_column")
  expect_equal(mod$context$timestamp_column, ".timestamp_column")
  expect_true(isTRUE(mod$context$id_synthetic))
  expect_true(isTRUE(mod$context$timestamp_synthetic))
  expect_equal(mod$context$item_ids, "default")

  out <- predict(mod, prediction_length = 3L)
  expect_named(out, c(".pred", ".pred_quantile"))
})

test_that("formula tidyselect with c() resolves id and timestamp columns", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    ridership ~ Clark_Lake,
    data = Chi,
    id_column = c(series_id),
    timestamp_column = c(date)
  )

  expect_equal(mod$context$id_column, "series_id")
  expect_equal(mod$context$timestamp_column, "date")
  expect_false(isTRUE(mod$context$id_synthetic))
  expect_false(isTRUE(mod$context$timestamp_synthetic))
})

test_that("formula tidyselect with bare names resolves id and timestamp", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    ridership ~ Clark_Lake,
    data = Chi,
    id_column = series_id,
    timestamp_column = date
  )

  expect_equal(mod$context$id_column, "series_id")
  expect_equal(mod$context$timestamp_column, "date")
})

test_that("formula accepts character strings for id and timestamp (back-compat)", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    ridership ~ Clark_Lake,
    data = Chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  expect_equal(mod$context$id_column, "series_id")
  expect_equal(mod$context$timestamp_column, "date")
})

test_that("formula errors when id_column tidyselect picks more than one column", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      ridership ~ Clark_Lake,
      data = Chi,
      id_column = c(series_id, date)
    )
  })
})

test_that("formula errors when id_column tidyselect picks no column", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      ridership ~ Clark_Lake,
      data = Chi,
      id_column = tidyselect::starts_with("nope_")
    )
  })
})

test_that("formula errors when string id_column is not a column in data", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      ridership ~ Clark_Lake,
      data = Chi,
      id_column = "nope"
    )
  })
})

# ------------------------------------------------------------------------------
# Optional item_id / timestamp on the data.frame method

test_that("data.frame method synthesizes id and timestamp when omitted", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    x = Chi[, "Clark_Lake", drop = FALSE],
    y = Chi$ridership
  )

  expect_equal(mod$context$id_column, ".id_column")
  expect_equal(mod$context$timestamp_column, ".timestamp_column")
  expect_true(isTRUE(mod$context$id_synthetic))
  expect_true(isTRUE(mod$context$timestamp_synthetic))
  expect_equal(mod$context$item_ids, "default")

  out <- predict(mod, prediction_length = 3L)
  expect_named(out, c(".pred", ".pred_quantile"))
})

# ------------------------------------------------------------------------------
# predict round-trip with new_data on a model that has no id / timestamp

test_that("predict accepts new_data without id/timestamp when model was synthesized", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()
  chi_simple <- Chi[, c("ridership", "Clark_Lake")]

  mod <- brulee_chronos(ridership ~ Clark_Lake, data = chi_simple)

  out <- predict(mod, new_data = chi_simple, prediction_length = 4L)
  expect_s3_class(out, "tbl_df")
  expect_named(out, c(".pred", ".pred_quantile"))
  expect_equal(nrow(out), 4L)
})

# ------------------------------------------------------------------------------
# Print method

test_that("print.brulee_chronos lists model + context details", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  out <- capture.output(print(mod), type = "output")
  msg <- capture.output(print(mod), type = "message")
  out_str <- paste(c(out, msg), collapse = "\n")

  expect_match(out_str, "Chronos-2 Pretrained Forecasting Model")
  expect_match(out_str, "amazon/chronos-2")
  expect_match(out_str, "Prediction length:")
  expect_match(out_str, "Quantiles:")
  expect_match(out_str, "Device:")
  expect_match(out_str, "Context: 1 series")
})

test_that("print uses 'unknown' when revision is missing", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = chi,
    id_column = "series_id",
    timestamp_column = "date"
  )
  mod$revision <- NULL

  out <- capture.output(print(mod), type = "output")
  msg <- capture.output(print(mod), type = "message")
  out_str <- paste(c(out, msg), collapse = "\n")
  expect_match(out_str, "unknown")
})

test_that("print handles invalid external pointer gracefully", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  mod$device <- unserialize(serialize(torch::torch_device("cpu"), NULL))

  out <- capture.output(print(mod), type = "output")
  msg <- capture.output(print(mod), type = "message")
  out_str <- paste(c(out, msg), collapse = "\n")

  expect_match(out_str, "Chronos-2 Pretrained Forecasting Model")
  expect_match(out_str, "not available")
  expect_match(out_str, "invalid external pointer")
})

# ------------------------------------------------------------------------------
# Bridge validation extras

test_that("prediction_length above the model maximum errors", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      ridership ~ .,
      data = chi,
      id_column = "series_id",
      timestamp_column = "date",
      prediction_length = 2000L
    )
  })
})

test_that("non-numeric model_id errors", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      ridership ~ .,
      data = chi,
      id_column = "series_id",
      timestamp_column = "date",
      model_id = 42
    )
  })
})

test_that("NA values in item_id error", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  na_id <- Chi$series_id
  na_id[1] <- NA
  expect_snapshot(error = TRUE, {
    brulee_chronos(
      x = Chi[, "Clark_Lake", drop = FALSE],
      y = Chi$ridership,
      item_id = na_id,
      timestamp = Chi$date
    )
  })
})

test_that("NA values in timestamp error", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  na_ts <- Chi$date
  na_ts[1] <- NA
  expect_snapshot(error = TRUE, {
    brulee_chronos(
      x = Chi[, "Clark_Lake", drop = FALSE],
      y = Chi$ridership,
      item_id = Chi$series_id,
      timestamp = na_ts
    )
  })
})

test_that("non-numeric target errors", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      x = Chi[, "Clark_Lake", drop = FALSE],
      y = as.character(Chi$ridership),
      item_id = Chi$series_id,
      timestamp = Chi$date
    )
  })
})

test_that("quantile_levels of length zero errors", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      ridership ~ .,
      data = chi,
      id_column = "series_id",
      timestamp_column = "date",
      quantile_levels = numeric(0)
    )
  })
})

# ------------------------------------------------------------------------------
# Recipe role validation extras

test_that("recipe with more than one id role errors", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("recipes")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  Chi$series_id2 <- Chi$series_id

  rec <- recipes::recipe(ridership ~ ., data = Chi) |>
    recipes::update_role(series_id, new_role = "id") |>
    recipes::update_role(series_id2, new_role = "id") |>
    recipes::update_role(date, new_role = "time")

  expect_snapshot(error = TRUE, brulee_chronos(rec, data = Chi))
})

test_that("recipe with more than one time role errors", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("recipes")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  Chi$date2 <- Chi$date

  rec <- recipes::recipe(ridership ~ ., data = Chi) |>
    recipes::update_role(series_id, new_role = "id") |>
    recipes::update_role(date, new_role = "time") |>
    recipes::update_role(date2, new_role = "time")

  expect_snapshot(error = TRUE, brulee_chronos(rec, data = Chi))
})

# ------------------------------------------------------------------------------
# chronos2_split_by_series

test_that("chronos2_split_by_series sorts each series by timestamp", {
  unsorted_target <- c(3, 1, 2, 30, 10, 20)
  unsorted_ts <- c(3, 1, 2, 3, 1, 2)
  item_id <- c("A", "A", "A", "B", "B", "B")
  covariates <- data.frame(cov = unsorted_target * 10)

  res <- brulee:::chronos2_split_by_series(
    target = unsorted_target,
    covariates = covariates,
    item_id = item_id,
    timestamp = unsorted_ts,
    id_column = "iid",
    timestamp_column = "ts",
    target_column = "y"
  )

  expect_equal(res$item_ids, c("A", "B"))
  expect_equal(res$series_target[[1]], c(1, 2, 3))
  expect_equal(res$series_target[[2]], c(10, 20, 30))
  expect_equal(res$series_timestamp[[1]], c(1, 2, 3))
  expect_equal(res$series_covars[[1]]$cov, c(10, 20, 30))
  expect_equal(res$covariate_cols, "cov")
  expect_false(isTRUE(res$id_synthetic))
  expect_false(isTRUE(res$timestamp_synthetic))
})

test_that("formula method drops covariates correctly when only timestamp is named", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("date", "ridership", "Clark_Lake")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = chi,
    timestamp_column = date
  )

  expect_true(isTRUE(mod$context$id_synthetic))
  expect_false(isTRUE(mod$context$timestamp_synthetic))
  expect_setequal(mod$context$covariate_cols, "Clark_Lake")
})

# ------------------------------------------------------------------------------
# torch not installed guard

test_that("bridge errors when torch is not installed", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  testthat::local_mocked_bindings(
    torch_is_installed = function() FALSE,
    .package = "torch"
  )

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      ridership ~ .,
      data = chi,
      id_column = "series_id",
      timestamp_column = "date"
    )
  })
})

# ------------------------------------------------------------------------------
# Device validation

test_that("non-character device argument errors", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      ridership ~ .,
      data = chi,
      id_column = "series_id",
      timestamp_column = "date",
      device = 123
    )
  })
})

# ------------------------------------------------------------------------------
# numeric prediction_length coercion

test_that("numeric prediction_length is coerced to integer without error", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = chi,
    id_column = "series_id",
    timestamp_column = "date",
    prediction_length = 10.0
  )

  expect_identical(mod$prediction_length, 10L)
})

# ------------------------------------------------------------------------------
# quantile_levels boundary checks

test_that("quantile_levels at 1.0 errors", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      ridership ~ .,
      data = chi,
      id_column = "series_id",
      timestamp_column = "date",
      quantile_levels = c(0.5, 1.0)
    )
  })
})

test_that("non-numeric quantile_levels errors", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      ridership ~ .,
      data = chi,
      id_column = "series_id",
      timestamp_column = "date",
      quantile_levels = "half"
    )
  })
})

# ------------------------------------------------------------------------------
# Formula LHS validation

test_that("formula with multiple LHS variables errors", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      cbind(ridership, Clark_Lake) ~ Austin,
      data = Chi,
      id_column = "series_id",
      timestamp_column = "date"
    )
  })
})

test_that("formula with target not in data errors (no-covariate path)", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      not_a_column ~ 1,
      data = Chi,
      id_column = "series_id",
      timestamp_column = "date"
    )
  })
})

# ------------------------------------------------------------------------------
# Multiple series via formula

test_that("formula method handles multiple series correctly", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  multi <- rbind(
    transform(Chi, series_id = "A"),
    transform(Chi, series_id = "B")
  )

  mod <- brulee_chronos(
    ridership ~ Clark_Lake,
    data = multi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  expect_equal(length(mod$context$item_ids), 2L)
  expect_setequal(mod$context$item_ids, c("A", "B"))
  expect_equal(length(mod$context$series_target), 2L)
  expect_equal(length(mod$context$series_target[[1]]), nrow(Chi))
})

# ------------------------------------------------------------------------------
# data.frame method with zero-column covariates

test_that("data.frame method with empty covariate frame works", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    x = Chi[, character(0), drop = FALSE],
    y = Chi$ridership,
    item_id = Chi$series_id,
    timestamp = Chi$date,
    id_column = "series_id",
    timestamp_column = "date"
  )

  expect_s3_class(mod, "brulee_chronos")
  expect_equal(length(mod$context$covariate_cols), 0L)
})

# ------------------------------------------------------------------------------
# formula with `~ .` drops id and timestamp from predictors

test_that("formula with `~ .` excludes id and timestamp from covariates", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    ridership ~ .,
    data = Chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  expect_false("series_id" %in% mod$context$covariate_cols)
  expect_false("date" %in% mod$context$covariate_cols)
  expect_setequal(mod$context$covariate_cols, c("Clark_Lake", "Austin"))
})

# ------------------------------------------------------------------------------
# chronos2_split_by_series with a single series (synthetic flags)

test_that("chronos2_split_by_series works for single series with synthetic flags", {
  target <- c(10, 20, 30)
  ts <- c(3, 1, 2)
  item_id <- rep("X", 3)
  covariates <- data.frame(a = c(100, 200, 300))

  res <- brulee:::chronos2_split_by_series(
    target = target,
    covariates = covariates,
    item_id = item_id,
    timestamp = ts,
    id_column = "id",
    timestamp_column = "ts",
    target_column = "y",
    id_synthetic = TRUE,
    timestamp_synthetic = FALSE
  )

  expect_equal(res$item_ids, "X")
  expect_equal(res$series_target[[1]], c(20, 30, 10))
  expect_equal(res$series_covars[[1]]$a, c(200, 300, 100))
  expect_true(isTRUE(res$id_synthetic))
  expect_false(isTRUE(res$timestamp_synthetic))
})

# ------------------------------------------------------------------------------
# chronos2_resolve_column edge cases

test_that("chronos2_resolve_column errors for string not in data", {
  df <- data.frame(a = 1:3, b = 4:6)
  quo <- rlang::quo("missing_col")

  expect_snapshot(error = TRUE, {
    brulee:::chronos2_resolve_column(quo, df, "test_arg")
  })
})

test_that("chronos2_resolve_column errors on invalid tidyselect expression", {
  skip_if_not_installed("tidyselect")
  df <- data.frame(a = 1:3, b = 4:6)
  quo <- rlang::quo(c(not_a_col))

  expect_snapshot(error = TRUE, {
    brulee:::chronos2_resolve_column(quo, df, "test_arg")
  })
})

# ------------------------------------------------------------------------------
# print with multiple series and covariates

test_that("print works with multiple series and covariates", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  multi <- rbind(
    transform(Chi, series_id = "A"),
    transform(Chi, series_id = "B")
  )

  mod <- brulee_chronos(
    ridership ~ Clark_Lake + Austin,
    data = multi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  out <- capture.output(print(mod), type = "output")
  msg <- capture.output(print(mod), type = "message")
  out_str <- paste(c(out, msg), collapse = "\n")

  expect_match(out_str, "2 series")
  expect_match(out_str, "2 covariate")
})

# ------------------------------------------------------------------------------
# Additional bridge validation

test_that("non-character revision argument errors", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      ridership ~ .,
      data = chi,
      id_column = "series_id",
      timestamp_column = "date",
      revision = 123
    )
  })
})

test_that("non-character cache_dir argument errors", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  expect_snapshot(error = TRUE, {
    brulee_chronos(
      ridership ~ .,
      data = chi,
      id_column = "series_id",
      timestamp_column = "date",
      cache_dir = 999
    )
  })
})

test_that("prediction_length defaults to model max when NULL", {
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  expect_identical(mod$prediction_length, 1024L)
})
