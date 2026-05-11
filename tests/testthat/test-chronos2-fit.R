# brulee_chronos() dispatch + validation tests.
#
# These tests cover the S3 dispatch, hardhat blueprint, validation, and
# context-splitting logic. Torch model construction and weight loading
# are mocked so the tests don't depend on the pretrained model files
# (and don't trigger the segfault in the safetensors loader).

skip_if_not_installed("hardhat")
skip_if_not_installed("recipes")
skip_if_not_installed("modeldata")

# `stub_chronos_loaders()` and `chicago_subset()` live in
# tests/testthat/helper-chronos2.R so they're shared with other chronos
# test files.

# ------------------------------------------------------------------------------

test_that("default method errors for unsupported types", {
  expect_error(brulee_chronos(42L), "is not defined for a 'integer'")
  expect_error(brulee_chronos("oops"), "is not defined for a 'character'")
})

test_that("formula method builds context with no covariates", {
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
  # Default revision is the pinned 40-char SHA
  expect_match(mod$revision, "^[0-9a-f]{40}$")
  expect_equal(mod$revision, brulee:::chronos2_default_revision())
})

test_that("an explicit SHA is passed through verbatim (no API call)", {
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
  stub_chronos_loaders()
  Chi <- chicago_subset()
  # Drop the series_id column so it isn't treated as a (non-numeric) predictor.
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
  stub_chronos_loaders()
  Chi <- chicago_subset()
  # Drop the date column so it isn't treated as a (non-numeric) predictor.
  chi_no_date <- Chi[, c("series_id", "ridership", "Clark_Lake", "Austin")]

  rec_no_time <- recipes::recipe(ridership ~ ., data = chi_no_date) |>
    recipes::update_role(series_id, new_role = "id")
  mod <- brulee_chronos(rec_no_time, data = chi_no_date)

  expect_equal(mod$context$timestamp_column, ".timestamp_column")
  expect_true(isTRUE(mod$context$timestamp_synthetic))
  # Row order is preserved within each series (synthesized timestamps are seq_len).
  expect_equal(mod$context$series_target[[1]], chi_no_date$ridership)
})

test_that("recipe method synthesizes both when neither role is set", {
  stub_chronos_loaders()
  Chi <- chicago_subset()
  # All-numeric data so that nothing un-roled trips the predictor check.
  chi_numeric <- Chi[, c("ridership", "Clark_Lake", "Austin")]

  rec <- recipes::recipe(ridership ~ ., data = chi_numeric)
  mod <- brulee_chronos(rec, data = chi_numeric)

  expect_true(isTRUE(mod$context$id_synthetic))
  expect_true(isTRUE(mod$context$timestamp_synthetic))
  expect_equal(mod$context$item_ids, "default")
})

test_that("data.frame method builds context with item_id/timestamp vectors", {
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
  stub_chronos_loaders()
  expect_snapshot(error = TRUE, {
    brulee_chronos(
      matrix(rnorm(20), nrow = 10, ncol = 2),
      y = rnorm(10)
    )
  })
})

test_that("non-numeric predictors are rejected when mold doesn't encode them", {
  stub_chronos_loaders()
  Chi <- chicago_subset()
  Chi$cat <- factor(rep(c("a", "b"), length.out = nrow(Chi)))

  rec <- recipes::recipe(ridership ~ ., data = Chi) |>
    recipes::update_role(series_id, new_role = "id") |>
    recipes::update_role(date, new_role = "time")

  expect_error(
    brulee_chronos(rec, data = Chi),
    "must be numeric"
  )
})

test_that("formula factor predictors are encoded by hardhat and pass the check", {
  stub_chronos_loaders()
  Chi <- chicago_subset()
  Chi$cat <- factor(rep(c("a", "b"), length.out = nrow(Chi)))

  mod <- brulee_chronos(
    ridership ~ Clark_Lake + cat,
    data = Chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  # mold's formula blueprint dummies the factor; the surviving predictors
  # should all be numeric (Clark_Lake plus a `catb` dummy).
  expect_true("Clark_Lake" %in% mod$context$covariate_cols)
  expect_true(any(grepl("^cat", mod$context$covariate_cols)))
})

test_that("length mismatches between y and item_id/timestamp error", {
  stub_chronos_loaders()
  Chi <- chicago_subset()

  expect_error(
    brulee_chronos(
      x = Chi[, "Clark_Lake", drop = FALSE],
      y = Chi$ridership,
      item_id = Chi$series_id[-1],
      timestamp = Chi$date,
      id_column = "series_id",
      timestamp_column = "date"
    ),
    "item_id"
  )
  expect_error(
    brulee_chronos(
      x = Chi[, "Clark_Lake", drop = FALSE],
      y = Chi$ridership,
      item_id = Chi$series_id,
      timestamp = Chi$date[-1],
      id_column = "series_id",
      timestamp_column = "date"
    ),
    "timestamp"
  )
})

test_that("invalid quantile_levels and prediction_length error", {
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  expect_error(
    brulee_chronos(
      ridership ~ .,
      data = chi,
      id_column = "series_id",
      timestamp_column = "date",
      quantile_levels = c(0, 0.5)
    ),
    "open interval"
  )
  expect_error(
    brulee_chronos(
      ridership ~ .,
      data = chi,
      id_column = "series_id",
      timestamp_column = "date",
      prediction_length = -1
    ),
    "prediction_length"
  )
  # Quantile not present in the model
  expect_error(
    brulee_chronos(
      ridership ~ .,
      data = chi,
      id_column = "series_id",
      timestamp_column = "date",
      quantile_levels = 0.123
    ),
    "not available in model"
  )
})

# ------------------------------------------------------------------------------
# Forge round-trips on the stored blueprint

test_that("formula blueprint forges new_data back to predictors + outcome", {
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
  # single series -> id column omitted; timestamp never present
  expect_named(out, c(".pred", ".pred_quantile"))
  expect_equal(nrow(out), 5L)
  expect_type(out$.pred, "double")
  expect_s3_class(out$.pred_quantile, "quantile_pred")
  # default quantile_levels has 9 levels
  expect_equal(ncol(as.matrix(out$.pred_quantile)), 9L)
  # `.pred` should equal median(.pred_quantile) elementwise
  expect_equal(out$.pred, as.numeric(stats::median(out$.pred_quantile)))
})

test_that("predict() keeps the id column when there are multiple series", {
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()
  # Two synthetic series, same shape
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
  # quantile labels stored on the quantile_pred match the request
  expect_equal(
    attr(out$.pred_quantile, "quantile_levels"),
    c(0.1, 0.5, 0.9)
  )
})

test_that("predict() does not return any timestamp column", {
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
  # also no Date / POSIXct columns sneaking through under another name
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
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()
  chi_simple <- Chi[, c("ridership", "Clark_Lake")]

  mod <- brulee_chronos(ridership ~ Clark_Lake, data = chi_simple)

  # new_data has only the target + covariate; no id, no timestamp
  out <- predict(mod, new_data = chi_simple, prediction_length = 4L)
  expect_s3_class(out, "tbl_df")
  expect_named(out, c(".pred", ".pred_quantile"))
  expect_equal(nrow(out), 4L)
})

# ------------------------------------------------------------------------------
# Print method

test_that("print.brulee_chronos lists model + context details", {
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  # cli_bullets writes via cli's own sink; capture both stdout and stderr.
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

# ------------------------------------------------------------------------------
# Bridge validation extras

test_that("prediction_length above the model maximum errors", {
  stub_chronos_loaders()
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  # Model maximum is config$max_output_patches * config$output_patch_size
  # = 64 * 16 = 1024 in stub_chronos_loaders.
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

test_that("non-numeric model_id / cache_dir error", {
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

test_that("NA values in item_id or timestamp error", {
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
  stub_chronos_loaders()
  Chi <- chicago_subset()
  Chi$series_id2 <- Chi$series_id

  rec <- recipes::recipe(ridership ~ ., data = Chi) |>
    recipes::update_role(series_id, new_role = "id") |>
    recipes::update_role(series_id2, new_role = "id") |>
    recipes::update_role(date, new_role = "time")

  expect_snapshot(error = TRUE, {
    brulee_chronos(rec, data = Chi)
  })
})

test_that("recipe with more than one time role errors", {
  stub_chronos_loaders()
  Chi <- chicago_subset()
  Chi$date2 <- Chi$date

  rec <- recipes::recipe(ridership ~ ., data = Chi) |>
    recipes::update_role(series_id, new_role = "id") |>
    recipes::update_role(date, new_role = "time") |>
    recipes::update_role(date2, new_role = "time")

  expect_snapshot(error = TRUE, {
    brulee_chronos(rec, data = Chi)
  })
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
  # The timestamp column should be dropped from the predictors.
  expect_setequal(mod$context$covariate_cols, "Clark_Lake")
})
