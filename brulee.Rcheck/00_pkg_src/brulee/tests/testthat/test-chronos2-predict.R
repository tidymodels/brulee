# predict.brulee_chronos() tests.
#
# These cover: prediction_length / quantile_levels overrides, new_data
# round-trips (formula, recipe, x_y), future_df validation, extra columns,
# and the low-level prediction engine (chronos2_predict_core,
# chronos2_build_inputs, chronos2_run_with_covariates) with real torch ops.

# ------------------------------------------------------------------------------
# prediction_length / quantile_levels overrides

test_that("predict falls back to the object's prediction_length / quantile_levels", {
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
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

test_that("predict with prediction_length override uses the new value", {
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()
  chi <- Chi[, c("series_id", "date", "ridership")]

  mod <- brulee_chronos(
    ridership ~ .,
    data = chi,
    id_column = "series_id",
    timestamp_column = "date",
    prediction_length = 10L
  )

  out <- predict(mod, prediction_length = 3L)
  expect_equal(nrow(out), 3L)
})

test_that("predict uses stored context when new_data is NULL", {
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("recipes")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    ridership ~ Clark_Lake,
    data = Chi,
    id_column = "series_id",
    timestamp_column = "date",
    prediction_length = 6L
  )

  out <- predict(mod)
  expect_s3_class(out, "tbl_df")
  expect_equal(nrow(out), 6L)
})

# ------------------------------------------------------------------------------
# new_data round-trips

test_that("predict with new_data forges covariates from a formula model", {
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("recipes")
  skip_if_not_installed("modeldata")
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
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("recipes")
  skip_if_not_installed("modeldata")
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
  skip_on_cran()
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

  out <- predict(mod, new_data = multi_series, prediction_length = 3L)
  expect_named(out, c("series_id", ".pred", ".pred_quantile"))
  expect_setequal(out$series_id, c("A", "B"))
})

test_that("predict with new_data missing the id column errors", {
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
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
  skip_on_cran()
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

  bad <- chi
  bad$ridership <- as.character(bad$ridership)
  expect_snapshot(error = TRUE, {
    predict(mod, new_data = bad, prediction_length = 3L)
  })
})

test_that("predict with new_data on no-covariate model pulls target by name", {
  skip_on_cran()
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

  out <- predict(mod, new_data = chi, prediction_length = 5L)
  expect_s3_class(out, "tbl_df")
  expect_equal(nrow(out), 5L)
  expect_named(out, c(".pred", ".pred_quantile"))
})

test_that("predict with new_data missing the timestamp column errors", {
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("recipes")
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

  bad <- chi[, c("series_id", "ridership")]
  expect_snapshot(error = TRUE, {
    predict(mod, new_data = bad, prediction_length = 3L)
  })
})

test_that("predict with new_data on no-covariate x_y model works", {
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("recipes")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    x = Chi[, character(0), drop = FALSE],
    y = Chi$ridership
  )

  new_df <- data.frame(.outcome = Chi$ridership[1:50])
  out <- predict(mod, new_data = new_df, prediction_length = 3L)
  expect_s3_class(out, "tbl_df")
  expect_equal(nrow(out), 3L)
  expect_named(out, c(".pred", ".pred_quantile"))
})

test_that("predict with new_data on multi-series recipe model has id column", {
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("recipes")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()
  multi <- rbind(
    transform(Chi, series_id = "A"),
    transform(Chi, series_id = "B")
  )

  rec <- recipes::recipe(ridership ~ ., data = multi) |>
    recipes::update_role(series_id, new_role = "id") |>
    recipes::update_role(date, new_role = "time")
  mod <- brulee_chronos(rec, data = multi)

  out <- predict(mod, new_data = multi, prediction_length = 4L)
  expect_named(out, c("series_id", ".pred", ".pred_quantile"))
  expect_setequal(out$series_id, c("A", "B"))
  expect_equal(nrow(out), 8L)
})

test_that("predict errors when new_data has non-numeric covariates (no-forge path)", {
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    ridership ~ Clark_Lake + Austin,
    data = Chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  bad <- Chi
  bad$Clark_Lake <- as.character(bad$Clark_Lake)

  expect_error(
    predict(mod, new_data = bad, prediction_length = 3L),
    "numeric|convert"
  )
})

# ------------------------------------------------------------------------------
# future_df validation

test_that("predict with valid future_df runs", {
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
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
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
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
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
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

test_that("future_df with an unknown covariate column silently ignores it", {
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
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

  out <- predict(mod, future_df = future_df, prediction_length = 3L)
  expect_s3_class(out, "tbl_df")
  expect_equal(nrow(out), 3L)
})

test_that("future_df works when model has synthesized id and timestamp", {
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    ridership ~ Clark_Lake + Austin,
    data = Chi[, c("ridership", "Clark_Lake", "Austin")]
  )

  future_df <- data.frame(
    Clark_Lake = rnorm(5),
    Austin = rnorm(5)
  )

  out <- predict(mod, future_df = future_df, prediction_length = 5L)
  expect_s3_class(out, "tbl_df")
  expect_equal(nrow(out), 5L)
})

test_that("future_df with subset of covariates provides only those to model", {
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    ridership ~ Clark_Lake + Austin,
    data = Chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  future_df <- data.frame(
    series_id = rep("L", 4),
    date = seq(max(Chi$date) + 1, by = "day", length.out = 4),
    Clark_Lake = rnorm(4)
  )

  out <- predict(mod, future_df = future_df, prediction_length = 4L)
  expect_s3_class(out, "tbl_df")
  expect_equal(nrow(out), 4L)
})

test_that("future_df is split by series and sorted by timestamp", {
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders(also_mock_predict_core = TRUE)
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

  future_dates <- seq(max(Chi$date) + 1, by = "day", length.out = 3)
  future_df <- data.frame(
    series_id = rep(c("B", "A"), each = 3),
    date = rep(future_dates, 2),
    Clark_Lake = rnorm(6)
  )

  out <- predict(mod, future_df = future_df, prediction_length = 3L)
  expect_named(out, c("series_id", ".pred", ".pred_quantile"))
  expect_setequal(out$series_id, c("A", "B"))
  expect_equal(nrow(out), 6L)
})

test_that("future_df with wrong row count per series errors", {
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
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
    Clark_Lake = rnorm(3)
  )

  expect_snapshot(error = TRUE, {
    predict(mod, future_df = future_df, prediction_length = 5L)
  })
})

# ------------------------------------------------------------------------------
# extra columns in new_data and future_df are silently ignored

test_that("new_data with extra columns works (formula model with covariates)", {
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    ridership ~ Clark_Lake + Austin,
    data = Chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  extra <- Chi
  extra$extra_col <- rnorm(nrow(extra))
  extra$another_one <- "hello"

  out <- predict(mod, new_data = extra, prediction_length = 3L)
  expect_s3_class(out, "tbl_df")
  expect_equal(nrow(out), 3L)
})

test_that("new_data with extra columns works (no-covariate model)", {
  skip_on_cran()
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

  extra <- chi
  extra$extra_col <- rnorm(nrow(extra))
  extra$another_one <- "hello"

  out <- predict(mod, new_data = extra, prediction_length = 3L)
  expect_s3_class(out, "tbl_df")
  expect_equal(nrow(out), 3L)
})

test_that("future_df with extra columns silently ignores them", {
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
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
    Austin = rnorm(5),
    extra_col = rnorm(5),
    unrelated = letters[1:5]
  )

  out <- predict(mod, future_df = future_df, prediction_length = 5L)
  expect_s3_class(out, "tbl_df")
  expect_equal(nrow(out), 5L)
})

test_that("new_data still errors when a required column is missing", {
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
  stub_chronos_loaders(also_mock_predict_core = TRUE)
  Chi <- chicago_subset()

  mod <- brulee_chronos(
    ridership ~ Clark_Lake + Austin,
    data = Chi,
    id_column = "series_id",
    timestamp_column = "date"
  )

  bad <- Chi[, c("series_id", "date", "ridership", "Clark_Lake")]
  expect_snapshot(
    error = TRUE,
    predict(mod, new_data = bad, prediction_length = 3L)
  )
})

test_that("future_df still errors when the id column is missing", {
  skip_on_cran()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("modeldata")
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
    Clark_Lake = rnorm(3),
    extra_col = rnorm(3)
  )

  expect_snapshot(
    error = TRUE,
    predict(mod, future_df = future_df, prediction_length = 3L)
  )
})

# ------------------------------------------------------------------------------
# chronos2_pull_column helper

test_that("chronos2_pull_column errors when the column is missing", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  expect_snapshot(error = TRUE, {
    brulee:::chronos2_pull_column(
      data.frame(a = 1:3),
      "missing_col",
      "id_column"
    )
  })
})

test_that("chronos2_pull_column returns the column value when present", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  res <- brulee:::chronos2_pull_column(
    data.frame(a = 1:3, b = letters[1:3]),
    "b",
    "x"
  )
  expect_equal(res, letters[1:3])
})

# ------------------------------------------------------------------------------
# chronos2_build_inputs (torch-level unit tests)

test_that("chronos2_build_inputs handles a single numeric context vector", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  result <- brulee:::chronos2_build_inputs(
    context = c(1, 2, 3, 4, 5),
    past_covariates = list(data.frame(cov1 = c(10, 20, 30, 40, 50))),
    future_covariates = list(data.frame(cov1 = c(60, 70, 80))),
    prediction_length = 3L
  )

  expect_length(result, 1L)
  expect_equal(result[[1]]$target, c(1, 2, 3, 4, 5))
  expect_equal(result[[1]]$past_covariates$cov1, c(10, 20, 30, 40, 50))
  expect_equal(result[[1]]$future_covariates$cov1, c(60, 70, 80))
})

test_that("chronos2_build_inputs handles list of contexts (multiple series)", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  result <- brulee:::chronos2_build_inputs(
    context = list(c(1, 2, 3), c(4, 5, 6, 7)),
    past_covariates = list(
      data.frame(x = c(10, 20, 30)),
      data.frame(x = c(40, 50, 60, 70))
    ),
    future_covariates = NULL,
    prediction_length = 2L
  )

  expect_length(result, 2L)
  expect_equal(result[[1]]$target, c(1, 2, 3))
  expect_equal(result[[2]]$target, c(4, 5, 6, 7))
  expect_equal(result[[1]]$past_covariates$x, c(10, 20, 30))
  expect_equal(result[[2]]$future_covariates, list())
})

test_that("chronos2_build_inputs converts torch tensor context", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  ctx_tensor <- torch::torch_tensor(c(1, 2, 3), dtype = torch::torch_float32())

  result <- brulee:::chronos2_build_inputs(
    context = ctx_tensor,
    past_covariates = list(data.frame(a = c(10, 20, 30))),
    future_covariates = NULL,
    prediction_length = 2L
  )

  expect_length(result, 1L)
  expect_equal(result[[1]]$target, c(1, 2, 3))
})

test_that("chronos2_build_inputs converts 2D torch tensor context", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  ctx_tensor <- torch::torch_tensor(
    matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, byrow = TRUE),
    dtype = torch::torch_float32()
  )

  result <- brulee:::chronos2_build_inputs(
    context = ctx_tensor,
    past_covariates = list(
      data.frame(a = c(10, 20, 30)),
      data.frame(a = c(40, 50, 60))
    ),
    future_covariates = NULL,
    prediction_length = 2L
  )

  expect_length(result, 2L)
  expect_equal(result[[1]]$target, c(1, 2, 3))
  expect_equal(result[[2]]$target, c(4, 5, 6))
})

test_that("chronos2_build_inputs handles data.frame past_covariates (single task)", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  result <- brulee:::chronos2_build_inputs(
    context = list(c(1, 2, 3)),
    past_covariates = data.frame(x = c(10, 20, 30)),
    future_covariates = data.frame(x = c(40, 50)),
    prediction_length = 2L
  )

  expect_length(result, 1L)
  expect_equal(result[[1]]$past_covariates$x, c(10, 20, 30))
  expect_equal(result[[1]]$future_covariates$x, c(40, 50))
})

# ------------------------------------------------------------------------------
# chronos2_predict_core and chronos2_run_with_covariates (torch-level)

test_that("chronos2_predict_core runs simple path (no covariates, list context)", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  tiny_config <- list(
    d_model = 32L,
    d_ff = 64L,
    d_kv = 16L,
    num_heads = 2L,
    num_layers = 1L,
    dropout_rate = 0.0,
    layer_norm_epsilon = 1e-6,
    rope_theta = 10000,
    vocab_size = 2L,
    pad_token_id = 0L,
    reg_token_id = 1L,
    context_length = 64L,
    input_patch_size = 4L,
    input_patch_stride = 4L,
    output_patch_size = 4L,
    max_output_patches = 4L,
    quantiles = c(0.1, 0.5, 0.9),
    use_arcsinh = FALSE,
    use_reg_token = TRUE,
    time_encoding_scale = 1.0
  )
  tiny_model <- brulee:::chronos2_model(tiny_config)
  tiny_model$eval()

  obj <- list(
    model = tiny_model,
    config = tiny_config,
    device = torch::torch_device("cpu"),
    prediction_length = 4L
  )

  result <- brulee:::chronos2_predict_core(
    obj,
    context = list(rnorm(20)),
    prediction_length = 4L
  )

  expect_true(inherits(result$predictions, "torch_tensor"))
  expect_equal(result$prediction_length, 4L)
  # Shape: [1, n_quantiles, prediction_length]
  expect_equal(result$predictions$size(1), 1L)
  expect_equal(result$predictions$size(2), 3L)
  expect_equal(result$predictions$size(3), 4L)
})

test_that("chronos2_predict_core runs with covariates", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  tiny_config <- list(
    d_model = 32L,
    d_ff = 64L,
    d_kv = 16L,
    num_heads = 2L,
    num_layers = 1L,
    dropout_rate = 0.0,
    layer_norm_epsilon = 1e-6,
    rope_theta = 10000,
    vocab_size = 2L,
    pad_token_id = 0L,
    reg_token_id = 1L,
    context_length = 64L,
    input_patch_size = 4L,
    input_patch_stride = 4L,
    output_patch_size = 4L,
    max_output_patches = 4L,
    quantiles = c(0.1, 0.5, 0.9),
    use_arcsinh = FALSE,
    use_reg_token = TRUE,
    time_encoding_scale = 1.0
  )
  tiny_model <- brulee:::chronos2_model(tiny_config)
  tiny_model$eval()

  obj <- list(
    model = tiny_model,
    config = tiny_config,
    device = torch::torch_device("cpu"),
    prediction_length = 4L
  )

  result <- brulee:::chronos2_predict_core(
    obj,
    context = list(rnorm(20)),
    prediction_length = 4L,
    past_covariates = list(data.frame(x = rnorm(20))),
    future_covariates = list(data.frame(x = rnorm(4)))
  )

  expect_true(inherits(result$predictions, "torch_tensor"))
  expect_equal(result$prediction_length, 4L)
})

test_that("chronos2_predict_core handles multiple series with covariates", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  tiny_config <- list(
    d_model = 32L,
    d_ff = 64L,
    d_kv = 16L,
    num_heads = 2L,
    num_layers = 1L,
    dropout_rate = 0.0,
    layer_norm_epsilon = 1e-6,
    rope_theta = 10000,
    vocab_size = 2L,
    pad_token_id = 0L,
    reg_token_id = 1L,
    context_length = 64L,
    input_patch_size = 4L,
    input_patch_stride = 4L,
    output_patch_size = 4L,
    max_output_patches = 4L,
    quantiles = c(0.1, 0.5, 0.9),
    use_arcsinh = FALSE,
    use_reg_token = TRUE,
    time_encoding_scale = 1.0
  )
  tiny_model <- brulee:::chronos2_model(tiny_config)
  tiny_model$eval()

  obj <- list(
    model = tiny_model,
    config = tiny_config,
    device = torch::torch_device("cpu"),
    prediction_length = 4L
  )

  result <- brulee:::chronos2_predict_core(
    obj,
    context = list(rnorm(16), rnorm(20)),
    prediction_length = 4L,
    past_covariates = list(
      data.frame(x = rnorm(16)),
      data.frame(x = rnorm(20))
    ),
    future_covariates = list(
      data.frame(x = rnorm(4)),
      data.frame(x = rnorm(4))
    )
  )

  expect_true(inherits(result$predictions, "torch_tensor"))
  expect_equal(result$prediction_length, 4L)
})

test_that("chronos2_predict_core simple path with numeric vector", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  tiny_config <- list(
    d_model = 32L,
    d_ff = 64L,
    d_kv = 16L,
    num_heads = 2L,
    num_layers = 1L,
    dropout_rate = 0.0,
    layer_norm_epsilon = 1e-6,
    rope_theta = 10000,
    vocab_size = 2L,
    pad_token_id = 0L,
    reg_token_id = 1L,
    context_length = 64L,
    input_patch_size = 4L,
    input_patch_stride = 4L,
    output_patch_size = 4L,
    max_output_patches = 4L,
    quantiles = c(0.1, 0.5, 0.9),
    use_arcsinh = FALSE,
    use_reg_token = TRUE,
    time_encoding_scale = 1.0
  )
  tiny_model <- brulee:::chronos2_model(tiny_config)
  tiny_model$eval()

  obj <- list(
    model = tiny_model,
    config = tiny_config,
    device = torch::torch_device("cpu"),
    prediction_length = 4L
  )

  # Passing a raw numeric vector (not a list)
  result <- brulee:::chronos2_predict_core(
    obj,
    context = rnorm(20),
    prediction_length = 4L
  )

  expect_true(inherits(result$predictions, "torch_tensor"))
  expect_equal(result$predictions$size(3), 4L)
})

test_that("chronos2_predict_core simple path with torch tensor", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  tiny_config <- list(
    d_model = 32L,
    d_ff = 64L,
    d_kv = 16L,
    num_heads = 2L,
    num_layers = 1L,
    dropout_rate = 0.0,
    layer_norm_epsilon = 1e-6,
    rope_theta = 10000,
    vocab_size = 2L,
    pad_token_id = 0L,
    reg_token_id = 1L,
    context_length = 64L,
    input_patch_size = 4L,
    input_patch_stride = 4L,
    output_patch_size = 4L,
    max_output_patches = 4L,
    quantiles = c(0.1, 0.5, 0.9),
    use_arcsinh = FALSE,
    use_reg_token = TRUE,
    time_encoding_scale = 1.0
  )
  tiny_model <- brulee:::chronos2_model(tiny_config)
  tiny_model$eval()

  obj <- list(
    model = tiny_model,
    config = tiny_config,
    device = torch::torch_device("cpu"),
    prediction_length = 4L
  )

  # Passing a pre-built 2D tensor
  ctx <- torch::torch_randn(1, 20)
  result <- brulee:::chronos2_predict_core(
    obj,
    context = ctx,
    prediction_length = 4L
  )

  expect_true(inherits(result$predictions, "torch_tensor"))
  expect_equal(result$predictions$size(1), 1L)
})

test_that("chronos2_predict_core uses object$prediction_length when NULL", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  tiny_config <- list(
    d_model = 32L,
    d_ff = 64L,
    d_kv = 16L,
    num_heads = 2L,
    num_layers = 1L,
    dropout_rate = 0.0,
    layer_norm_epsilon = 1e-6,
    rope_theta = 10000,
    vocab_size = 2L,
    pad_token_id = 0L,
    reg_token_id = 1L,
    context_length = 64L,
    input_patch_size = 4L,
    input_patch_stride = 4L,
    output_patch_size = 4L,
    max_output_patches = 4L,
    quantiles = c(0.1, 0.5, 0.9),
    use_arcsinh = FALSE,
    use_reg_token = TRUE,
    time_encoding_scale = 1.0
  )
  tiny_model <- brulee:::chronos2_model(tiny_config)
  tiny_model$eval()

  obj <- list(
    model = tiny_model,
    config = tiny_config,
    device = torch::torch_device("cpu"),
    prediction_length = 8L
  )

  result <- brulee:::chronos2_predict_core(
    obj,
    context = list(rnorm(20)),
    prediction_length = NULL
  )

  expect_equal(result$prediction_length, 8L)
  expect_equal(result$predictions$size(3), 8L)
})

test_that("chronos2_predict_core simple path with list containing torch tensor", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  tiny_config <- list(
    d_model = 32L,
    d_ff = 64L,
    d_kv = 16L,
    num_heads = 2L,
    num_layers = 1L,
    dropout_rate = 0.0,
    layer_norm_epsilon = 1e-6,
    rope_theta = 10000,
    vocab_size = 2L,
    pad_token_id = 0L,
    reg_token_id = 1L,
    context_length = 64L,
    input_patch_size = 4L,
    input_patch_stride = 4L,
    output_patch_size = 4L,
    max_output_patches = 4L,
    quantiles = c(0.1, 0.5, 0.9),
    use_arcsinh = FALSE,
    use_reg_token = TRUE,
    time_encoding_scale = 1.0
  )
  tiny_model <- brulee:::chronos2_model(tiny_config)
  tiny_model$eval()

  obj <- list(
    model = tiny_model,
    config = tiny_config,
    device = torch::torch_device("cpu"),
    prediction_length = 4L
  )

  # List containing a pre-built tensor (hits the "else x" branch)
  ctx_list <- list(torch::torch_tensor(
    rnorm(16),
    dtype = torch::torch_float32()
  ))
  result <- brulee:::chronos2_predict_core(
    obj,
    context = ctx_list,
    prediction_length = 4L
  )

  expect_true(inherits(result$predictions, "torch_tensor"))
})

test_that("chronos2_predict_core simple path with 1D torch tensor", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  tiny_config <- list(
    d_model = 32L,
    d_ff = 64L,
    d_kv = 16L,
    num_heads = 2L,
    num_layers = 1L,
    dropout_rate = 0.0,
    layer_norm_epsilon = 1e-6,
    rope_theta = 10000,
    vocab_size = 2L,
    pad_token_id = 0L,
    reg_token_id = 1L,
    context_length = 64L,
    input_patch_size = 4L,
    input_patch_stride = 4L,
    output_patch_size = 4L,
    max_output_patches = 4L,
    quantiles = c(0.1, 0.5, 0.9),
    use_arcsinh = FALSE,
    use_reg_token = TRUE,
    time_encoding_scale = 1.0
  )
  tiny_model <- brulee:::chronos2_model(tiny_config)
  tiny_model$eval()

  obj <- list(
    model = tiny_model,
    config = tiny_config,
    device = torch::torch_device("cpu"),
    prediction_length = 4L
  )

  # 1D tensor triggers the unsqueeze(1) path
  ctx <- torch::torch_tensor(rnorm(20), dtype = torch::torch_float32())
  result <- brulee:::chronos2_predict_core(
    obj,
    context = ctx,
    prediction_length = 4L
  )

  expect_true(inherits(result$predictions, "torch_tensor"))
  expect_equal(result$predictions$size(1), 1L)
})

test_that("chronos2_run_with_covariates handles no-covariate tasks (empty keys)", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  tiny_config <- list(
    d_model = 32L,
    d_ff = 64L,
    d_kv = 16L,
    num_heads = 2L,
    num_layers = 1L,
    dropout_rate = 0.0,
    layer_norm_epsilon = 1e-6,
    rope_theta = 10000,
    vocab_size = 2L,
    pad_token_id = 0L,
    reg_token_id = 1L,
    context_length = 64L,
    input_patch_size = 4L,
    input_patch_stride = 4L,
    output_patch_size = 4L,
    max_output_patches = 4L,
    quantiles = c(0.1, 0.5, 0.9),
    use_arcsinh = FALSE,
    use_reg_token = TRUE,
    time_encoding_scale = 1.0
  )
  tiny_model <- brulee:::chronos2_model(tiny_config)
  tiny_model$eval()

  # Build inputs with empty covariate lists (hits else branches at lines 536, 563)
  inputs <- list(
    list(
      target = rnorm(16),
      past_covariates = list(),
      future_covariates = list()
    )
  )

  result <- brulee:::chronos2_run_with_covariates(
    tiny_model,
    tiny_config,
    torch::torch_device("cpu"),
    inputs,
    prediction_length = 4L,
    num_output_patches = 1L
  )

  expect_true(inherits(result$predictions, "torch_tensor"))
})

test_that("chronos2_run_with_covariates handles past-only covariates (no future)", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  skip_if_not(torch::torch_is_installed())

  tiny_config <- list(
    d_model = 32L,
    d_ff = 64L,
    d_kv = 16L,
    num_heads = 2L,
    num_layers = 1L,
    dropout_rate = 0.0,
    layer_norm_epsilon = 1e-6,
    rope_theta = 10000,
    vocab_size = 2L,
    pad_token_id = 0L,
    reg_token_id = 1L,
    context_length = 64L,
    input_patch_size = 4L,
    input_patch_stride = 4L,
    output_patch_size = 4L,
    max_output_patches = 4L,
    quantiles = c(0.1, 0.5, 0.9),
    use_arcsinh = FALSE,
    use_reg_token = TRUE,
    time_encoding_scale = 1.0
  )
  tiny_model <- brulee:::chronos2_model(tiny_config)
  tiny_model$eval()

  obj <- list(
    model = tiny_model,
    config = tiny_config,
    device = torch::torch_device("cpu"),
    prediction_length = 4L
  )

  result <- brulee:::chronos2_predict_core(
    obj,
    context = list(rnorm(16)),
    prediction_length = 4L,
    past_covariates = list(data.frame(cov1 = rnorm(16), cov2 = rnorm(16))),
    future_covariates = NULL
  )

  expect_true(inherits(result$predictions, "torch_tensor"))
})
