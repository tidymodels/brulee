# brulee_chronos() dispatch + validation tests.
#
# These tests cover the S3 dispatch, hardhat blueprint, validation, and
# context-splitting logic. Torch model construction and weight loading
# are mocked so the tests don't depend on the pretrained model files
# (and don't trigger the segfault in the safetensors loader).

skip_if_not_installed("hardhat")
skip_if_not_installed("recipes")
skip_if_not_installed("modeldata")

stub_chronos_loaders <- function(also_mock_predict_core = FALSE) {
  fake_dir <- file.path(
    tempdir(check = TRUE),
    paste0("chronos-stub-", as.integer(Sys.time()))
  )
  dir.create(fake_dir, recursive = TRUE, showWarnings = FALSE)

  bindings <- list(
    chronos2_download = function(model_id, revision, cache_dir) {
      list(
        model_dir = fake_dir,
        sha = if (grepl("^[0-9a-f]{40}$", revision)) {
          revision
        } else {
          "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        }
      )
    },
    chronos2_parse_config = function(path) {
      list(
        d_model = 384L,
        num_layers = 12L,
        num_heads = 12L,
        output_patch_size = 16L,
        max_output_patches = 64L,
        quantiles = (1:9) / 10
      )
    },
    chronos2_model = function(config) {
      structure(list(config = config), class = "fake_chronos_module")
    },
    load_chronos2_weights = function(model, path) invisible(NULL)
  )

  if (also_mock_predict_core) {
    # Return a deterministic [n_series, n_model_quantiles, prediction_length]
    # array. Quantiles are fixed at config$quantiles = (1:9)/10 (see above).
    bindings$chronos2_predict_core <- function(
      object,
      context,
      prediction_length = NULL,
      past_covariates = NULL,
      future_covariates = NULL
    ) {
      n_series <- if (is.list(context)) length(context) else 1L
      n_q <- length(object$config$quantiles)
      preds <- array(0, dim = c(n_series, n_q, prediction_length))
      for (s in seq_len(n_series)) {
        for (q in seq_len(n_q)) {
          for (t in seq_len(prediction_length)) {
            preds[s, q, t] <- s * 100 + object$config$quantiles[q] * 10 + t
          }
        }
      }
      list(
        predictions = preds,
        quantiles = object$config$quantiles,
        prediction_length = prediction_length
      )
    }
  }

  do.call(
    testthat::local_mocked_bindings,
    c(bindings, list(.package = "brulee", .env = parent.frame()))
  )
  # `model$to(...)` and `model$eval()` need to no-op
  assign(
    "$.fake_chronos_module",
    function(x, name) {
      if (name %in% c("to", "eval")) {
        return(function(...) invisible(NULL))
      }
      unclass(x)[[name]]
    },
    envir = globalenv()
  )
  withr::defer(rm("$.fake_chronos_module", envir = globalenv()), parent.frame())
}

chicago_subset <- function(n = 200) {
  data(Chicago, package = "modeldata", envir = environment())
  Chi <- Chicago[seq_len(n), c("date", "ridership", "Clark_Lake", "Austin")]
  Chi$series_id <- "L"
  Chi[, c("series_id", "date", "ridership", "Clark_Lake", "Austin")]
}

# ------------------------------------------------------------------------------

test_that("default method errors for unsupported types", {
  expect_error(brulee_chronos(42L), "is not defined for a 'integer'")
  expect_error(brulee_chronos("oops"), "is not defined for a 'character'")
})

test_that("formula method builds context with no covariates", {
  stub_chronos_loaders()
  Chi <- chicago_subset()
  d <- Chi[, c("series_id", "date", "ridership")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = d,
    id_column = "series_id",
    timestamp_column = "date"
  )

  expect_s3_class(mod, "brulee_chronos")
  expect_equal(mod$context$id_column, "series_id")
  expect_equal(mod$context$timestamp_column, "date")
  expect_equal(mod$context$target_column, "ridership")
  expect_equal(length(mod$context$covariate_cols), 0L)
  expect_equal(length(mod$context$item_ids), 1L)
  expect_equal(length(mod$context$series_target[[1]]), nrow(d))
})

test_that("model_id and revision are recorded on the object", {
  stub_chronos_loaders()
  Chi <- chicago_subset()
  d <- Chi[, c("series_id", "date", "ridership")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = d,
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
  d <- Chi[, c("series_id", "date", "ridership")]
  some_sha <- "1234567890abcdef1234567890abcdef12345678"

  mod <- brulee_chronos(
    ridership ~ .,
    data = d,
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

test_that("recipe method errors when id or time role is missing", {
  stub_chronos_loaders()
  Chi <- chicago_subset()

  rec_no_id <- recipes::recipe(ridership ~ ., data = Chi) |>
    recipes::update_role(date, new_role = "time")
  expect_error(
    brulee_chronos(rec_no_id, data = Chi),
    "exactly one variable with role"
  )

  rec_no_time <- recipes::recipe(ridership ~ ., data = Chi) |>
    recipes::update_role(series_id, new_role = "id")
  expect_error(
    brulee_chronos(rec_no_time, data = Chi),
    "exactly one variable with role"
  )
})

test_that("data.frame and matrix methods build matching context", {
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

  mod_mat <- brulee_chronos(
    x = as.matrix(Chi[, c("Clark_Lake", "Austin")]),
    y = Chi$ridership,
    item_id = Chi$series_id,
    timestamp = Chi$date,
    id_column = "series_id",
    timestamp_column = "date"
  )
  expect_equal(mod_mat$context$covariate_cols, c("Clark_Lake", "Austin"))
  expect_equal(
    mod_df$context$series_target[[1]],
    mod_mat$context$series_target[[1]]
  )
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
  d <- Chi[, c("series_id", "date", "ridership")]

  expect_error(
    brulee_chronos(
      ridership ~ .,
      data = d,
      id_column = "series_id",
      timestamp_column = "date",
      quantile_levels = c(0, 0.5)
    ),
    "open interval"
  )
  expect_error(
    brulee_chronos(
      ridership ~ .,
      data = d,
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
      data = d,
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
  d <- Chi[, c("series_id", "date", "ridership")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = d,
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
  d <- rbind(
    transform(Chi[, c("series_id", "date", "ridership")], series_id = "A"),
    transform(Chi[, c("series_id", "date", "ridership")], series_id = "B")
  )

  mod <- brulee_chronos(
    ridership ~ .,
    data = d,
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
  d <- Chi[, c("series_id", "date", "ridership")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = d,
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
  d <- Chi[, c("series_id", "date", "ridership")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = d,
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
