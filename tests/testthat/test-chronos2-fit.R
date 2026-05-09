# brulee_chronos() dispatch + validation tests.
#
# These tests cover the S3 dispatch, hardhat blueprint, validation, and
# context-splitting logic. Torch model construction and weight loading
# are mocked so the tests don't depend on the pretrained model files
# (and don't trigger the segfault in the safetensors loader).

skip_if_not_installed("hardhat")
skip_if_not_installed("recipes")
skip_if_not_installed("modeldata")

stub_chronos_loaders <- function() {
  testthat::local_mocked_bindings(
    chronos2_model = function(config) {
      structure(list(config = config), class = "fake_chronos_module")
    },
    load_chronos2_weights = function(model, path) invisible(NULL),
    .package = "brulee",
    .env = parent.frame()
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
