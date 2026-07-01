#' Chronos-2 pretrained forecasting model
#'
#' `brulee_chronos()` loads a pretrained Chronos-2 time series forecasting
#' quantile regression model from HuggingFace and ingests historical
#' ("context") data so that the returned object is ready to forecast. Unlike
#' other brulee models, no training is performed; the network has fixed
#' pretrained weights.
#'
#' @details
#'
#' ## Computing Requirements
#'
#' This model can be used with or without a graphics processing unit (GPU).
#' However, it may be computationally slow when used with a CPU (and no GPU).
#'
#' ## Model Weight File Download
#'
#' Keep in mind that, on the first usage of the fitting function, the package
#' will attempt to download the model weights file. This file can require about
#' 500MB and is locally cached.
#'
#' ## Interface Overview
#'
#' Every Chronos-2 forecast needs at most four pieces of information about
#' the historical (context) data:
#'
#'   * A __target__ column with the values to forecast (always required),
#'   * An optional __id__ column that distinguishes one time series from
#'     another (e.g. a city, store, or sensor); when omitted, all rows are
#'     treated as a single series,
#'   * An optional __timestamp__ column with the time index of each
#'     observation; when omitted, rows are read in their existing order,
#'   * Any number of __past covariates__, additional numeric columns
#'     measured alongside the target.
#'
#' `brulee_chronos()` is a generic with three interfaces for supplying that
#' information; this intended to add flexibility in how you declare the model as
#' well as what data are given as inputs. All three produce an object that
#' behaves the same way at predict time.
#'
#' To contrast these approaches, consider the `Chicago` data contained in the
#' \pkg{modeldata} package. The goal is to predict daily train `ridership`.
#' There is a `date` column, as well as a set of 14-day lagged ridership data
#' from our station of interest and from others in the Chicago system.
#'
#' You could use Chronos in the simplest way by just passing in the column
#' containing past ridership values. It assumes that there are no gaps in the
#' data and that the data are arranged/sorted in the proper order (past to
#' present). The simplest interfaces to use in this case are the formula and
#' matrix ones.
#'
#' We could add the `date` column, but this is primarily used to label the data.
#' Here, we would want the formula or recipe interface.
#'
#' In these data, only one station's ridership is modeled. Suppose we did this
#' for all stations. In that case, we would _stack_ the ridership data and use
#' the `id` argument to specify which station corresponds to each row. In this
#' implementation, that is equivalent to running the function separately for each
#' station; it is just a simpler interface with some small computational gains.
#'
#' If we wanted to use covariates in our model, such as lagged ridership data,
#' we can do so with the formula or recipe interfaces (see below).
#'
#' ## Formula interface
#'
#' Use a formula when your data is a single tidy data frame and you want to
#' name the covariates inline. The `id_column` and `timestamp_column`
#' arguments use tidyselect, so bare column names, `c()` selections, and
#' character strings all work:
#'
#' ```r
#' brulee_chronos(target ~ cov1 + cov2, data = df,
#'                id_column = c(series_id), timestamp_column = c(date))
#'
#' # bare names also work
#' brulee_chronos(target ~ cov1 + cov2, data = df,
#'                id_column = series_id, timestamp_column = date)
#'
#' # character strings still work for back compatibility
#' brulee_chronos(target ~ cov1 + cov2, data = df,
#'                id_column = "series_id", timestamp_column = "date")
#' ```
#'
#' If you have no covariates, use `target ~ .`. The id and timestamp
#' columns are excluded automatically. Categorical covariates on the
#' right hand side are converted to numeric dummy variables (just like
#' `lm()`).
#'
#' If you have a single series and no useful timestamp, you can omit both
#' columns entirely:
#'
#' ```r
#' brulee_chronos(target ~ ., data = df_single_series)
#' ```
#'
#' ## Recipe interface
#'
#' Use a [recipes::recipe()] when you want to apply preprocessing steps
#' (e.g. normalizing or encoding columns) before the data reaches the
#' model. With the recipe interface, the id and timestamp columns are
#' identified by their __role__, not by name:
#'
#' ```r
#' rec <- recipe(target ~ ., data = df) |>
#'   update_role(item_id,   new_role = "id") |>
#'   update_role(timestamp, new_role = "time") |>
#'   step_normalize(all_numeric_predictors())
#'
#' brulee_chronos(rec, data = df)
#' ```
#'
#' Both the `id` and `time` roles are optional. If neither role is set,
#' `brulee_chronos()` treats the recipe data as a single series in row
#' order. All non numeric covariates must be encoded numerically by the
#' recipe (e.g. with [recipes::step_dummy()]).
#'
#' ## Data-frame (`x` and `y`) interface
#'
#' Use the `x_y` interface when you already have your covariates and target
#' separated. `x` is a data frame of past covariates (zero columns is
#' allowed when there are no covariates), `y` is the numeric target vector,
#' and `item_id` / `timestamp` are optional vectors of length `nrow(x)`:
#'
#' ```r
#' brulee_chronos(x = df[, c("cov1", "cov2")], y = df$target,
#'                item_id = df$item_id, timestamp = df$timestamp)
#'
#' # single series, no timestamp:
#' brulee_chronos(x = df[, c("cov1", "cov2")], y = df$target)
#' ```
#'
#' ## Multiple time series
#'
#' All three interfaces support multiple series in one call. Stack the
#' series end to end in a single long format data frame and let the id
#' column distinguish them. `brulee_chronos()` sorts each series by
#' timestamp before forecasting. When you omit the id column, every row
#' is treated as part of one series called `"default"`.
#'
#' ## Pre-sorted input
#'
#' When you omit the timestamp, `brulee_chronos()` uses each series' row
#' order as its time order. Pre-sort each series before calling
#' `brulee_chronos()` if you take this shortcut.
#'
#' ## What happens at `predict()` time
#'
#' The model is pretrained and performs no training, so the historical context
#' is always the data supplied at construction; [predict.brulee_chronos()]
#' forecasts forward from that context. By default it returns the full
#' `prediction_length` horizon. To forecast a specific future window---and to
#' supply known future values of any covariate (e.g., holiday flags, planned
#' promotions)---pass that window as `new_data`. Its per-series row count sets
#' how many future steps are returned (at most `prediction_length`). When the
#' model has no covariates, `new_data` only needs the id and timestamp columns
#' that describe the future steps.
#'
#' @param x Depending on the context:
#'
#'   * A __data frame__ of past covariates.
#'   * A __recipe__ specifying preprocessing and roles for `target`,
#'     `id`, and `time` columns.
#'
#'  Pass an empty data frame when there are no covariates.
#'
#' @param y A numeric vector of target values, of length `nrow(x)`.
#' @param item_id Optional vector of time series identifiers, of length
#'   `nrow(x)`. Default: `NULL`, which treats all rows as a single series.
#' @param timestamp Optional vector of timestamps (Date, POSIXct, or
#'   numeric), of length `nrow(x)`. Default: `NULL`, which uses row order
#'   within each series.
#' @param data When a __recipe__ or __formula__ is used, `data` is the
#'   training set with columns for the id, timestamp, target, and any
#'   covariates.
#' @param formula A formula of the form `target ~ cov1 + cov2`. Use
#'   `target ~ .` when there are no covariates. The id and timestamp
#'   columns (if named) are dropped before the formula is evaluated.
#' @param id_column For the formula method, a tidyselect expression
#'   selecting the id column in `data` (e.g. `c(series_id)`, `series_id`,
#'   or `"series_id"`). For the data frame `x_y` method, a character
#'   string is used as the output label only (the actual id values come from
#'   `item_id`). Default: `NULL` for the formula method and `".id_column"`
#'   for the `x_y` method. When omitted, all rows are treated as one
#'   series. For the recipe method, identify the id column with
#'   `recipes::update_role(..., new_role = "id")`.
#' @param timestamp_column For the formula method, a tidyselect expression
#'   selecting the timestamp column in `data`. For the data frame `x_y`
#'   method, a character string is used as the output label only. Default:
#'   `NULL` for the formula method and `".timestamp_column"` for the `x_y`
#'   method. When omitted, row order is used as the time order. For the
#'   recipe method, identify the timestamp column with
#'   `recipes::update_role(..., new_role = "time")`.
#' @param model_id A character string identifying the HuggingFace model
#'   repository to download. Default: `"amazon/chronos-2"` (120M parameters).
#' @param revision A character string identifying which version of the
#'   weights to load. May be a 40-character commit SHA, a tag, or a branch
#'   name on the HuggingFace repo (e.g. `"main"`). Default: a commit SHA
#'   pinned by brulee so the weights cannot change without you opting in.
#'   The resolved SHA is recorded on the returned object as
#'   `object$revision` and printed by `print()`.
#' @param prediction_length An integer for the number of future time steps to
#'   forecast. Default: `NULL` (uses the model maximum). Must not exceed the
#'   model maximum. Can be overridden at `predict()` time.
#' @param quantile_levels A numeric vector of quantile levels to produce in
#'   predictions. Must be a subset of the model's trained quantiles. Default:
#'   `(1:9) / 10`. Can be overridden at `predict()` time.
#' @param device A character string for the computation device: `"cpu"`,
#'   `"cuda"`, or `"mps"`. Default: `NULL` (auto-detects best available).
#' @param cache_dir Path to a directory for caching downloaded model files.
#'   Default: `"~/.cache/chronos-r"`.
#' @param ... Currently unused.
#'
#' @references
#' Ansari, A. F., Shchur, O., Küken, J., Auer, A., Han, B., Mercado, P., ... &
#'   Bohlke-Schneider, M. (2025). "Chronos-2: From univariate to universal
#'   forecasting." _arXiv preprint arXiv:2510.15821_.
#'
#' Ansari, A. F., Shchur, O., Küken, J., Auer, A., Han, B., Mercado, P., ... &
#'   Bohlke-Schneider, M. (2026). "A foundation model for multivariate time
#'   series forecasting.", https://doi.org/10.21203/rs.3.rs-9096522/v1
#'
#' @returns A `brulee_chronos` object with elements:
#'
#'   * `model`: The torch `nn_module` (in eval mode, on the specified device).
#'   * `config`: Parsed model configuration list.
#'   * `device`: The torch device in use.
#'   * `prediction_length`: Validated prediction length.
#'   * `quantile_levels`: Validated quantile levels.
#'   * `model_id`: The HuggingFace repository the weights came from.
#'   * `revision`: The 40-character commit SHA of the weights actually loaded.
#'   * `blueprint`: The hardhat blueprint for processing new data.
#'   * `context`: A list with the per-series target, covariates, timestamps,
#'     and column-name metadata that `predict()` uses by default.
#'
#' @examplesIf !brulee:::is_cran_check()
#' pkgs <- c("recipes", "lubridate", "modeldata", "ggplot2")
#'
#' \dontrun{
#' if (torch::torch_is_installed() && rlang::is_installed(pkgs)) {
#'  library(dplyr)
#'  library(ggplot2)
#'
#'  n <- nrow(modeldata::Chicago)
#'
#'  prior_data <- modeldata::Chicago[-((n-13):n),]
#'  test_data <-
#'   modeldata::Chicago[(n-13):n,] |>
#'   mutate(day = lubridate::wday(date, label = TRUE))
#'
#'  # ------------------------------------------------------------------------------
#'  # Simple, no covariate model
#'
#'  mod_1 <-
#'   brulee_chronos(
#'    ridership ~ 1,
#'    data = prior_data,
#'    # Removing `timestamp_column` does not affect the fit
#'    timestamp_column = c(date),
#'    prediction_length = 14)
#'
#'  # ------------------------------------------------------------------------------
#'  # Some covariates via the formula method
#'
#' mod_2 <-
#'   brulee_chronos(
#'    ridership ~ Clark_Lake + Belmont + Harlem + Monroe,
#'    data = prior_data,
#'    timestamp_column = c(date),
#'    prediction_length = 14)
#'
#'  # ------------------------------------------------------------------------------
#'  # Covariates using recipes
#'
#'  rec <-
#'   recipe(ridership ~ ., data = prior_data) |>
#'   update_role(date, new_role = "time")
#'
#'  mod_3 <- brulee_chronos(rec, data = prior_data, prediction_length = 14)
#' }
#' }
#' @export
brulee_chronos <- function(x, ...) {
  UseMethod("brulee_chronos")
}

#' @export
#' @rdname brulee_chronos
brulee_chronos.default <- function(x, ...) {
  stop(
    "`brulee_chronos()` is not defined for a '",
    class(x)[1],
    "'.",
    call. = FALSE
  )
}

# XY method - data frame

#' @export
#' @rdname brulee_chronos
brulee_chronos.data.frame <- function(
  x,
  y,
  item_id = NULL,
  timestamp = NULL,
  id_column = ".id_column",
  timestamp_column = ".timestamp_column",
  model_id = "amazon/chronos-2",
  revision = chronos2_default_revision(),
  prediction_length = NULL,
  quantile_levels = (1:9) / 10,
  device = NULL,
  cache_dir = file.path(Sys.getenv("HOME"), ".cache", "chronos-r"),
  ...
) {
  processed <- hardhat::mold(x, y)

  id_synthetic <- is.null(item_id)
  timestamp_synthetic <- is.null(timestamp)
  if (id_synthetic) {
    item_id <- rep("default", length(y))
    id_column <- ".id_column"
  }
  if (timestamp_synthetic) {
    timestamp <- seq_along(y)
    timestamp_column <- ".timestamp_column"
  }

  brulee_chronos_bridge(
    processed,
    item_id = item_id,
    timestamp = timestamp,
    id_column = id_column,
    timestamp_column = timestamp_column,
    target_column = ".outcome",
    id_synthetic = id_synthetic,
    timestamp_synthetic = timestamp_synthetic,
    model_id = model_id,
    revision = revision,
    prediction_length = prediction_length,
    quantile_levels = quantile_levels,
    device = device,
    cache_dir = cache_dir,
    ...
  )
}

# Formula method

#' @export
#' @rdname brulee_chronos
brulee_chronos.formula <- function(
  formula,
  data,
  id_column = NULL,
  timestamp_column = NULL,
  model_id = "amazon/chronos-2",
  revision = chronos2_default_revision(),
  prediction_length = NULL,
  quantile_levels = (1:9) / 10,
  device = NULL,
  cache_dir = file.path(Sys.getenv("HOME"), ".cache", "chronos-r"),
  ...
) {
  id_name <- chronos2_resolve_column(
    rlang::enquo(id_column),
    data,
    "id_column"
  )
  ts_name <- chronos2_resolve_column(
    rlang::enquo(timestamp_column),
    data,
    "timestamp_column"
  )

  id_synthetic <- is.null(id_name)
  timestamp_synthetic <- is.null(ts_name)

  if (id_synthetic) {
    item_id <- rep("default", nrow(data))
    id_column <- ".id_column"
  } else {
    item_id <- data[[id_name]]
    id_column <- id_name
  }
  if (timestamp_synthetic) {
    timestamp <- seq_len(nrow(data))
    timestamp_column <- ".timestamp_column"
  } else {
    timestamp <- data[[ts_name]]
    timestamp_column <- ts_name
  }

  # Hide id/timestamp from the formula-mold pass so `target ~ .` doesn't pick
  # them up as predictors.
  drop_cols <- intersect(c(id_name, ts_name), names(data))
  data_for_mold <- data[,
    setdiff(names(data), drop_cols),
    drop = FALSE
  ]

  target_var <- all.vars(formula[[2]])
  if (length(target_var) != 1L) {
    cli::cli_abort(
      "{.arg formula} must have exactly one variable on the left-hand side."
    )
  }
  rhs_terms <- attr(stats::terms(formula, data = data_for_mold), "term.labels")
  has_covariates <- length(rhs_terms) > 0L

  if (has_covariates) {
    processed <- hardhat::mold(formula, data_for_mold)
  } else {
    # No covariates: bypass the formula blueprint (which rejects `~ 1`) and
    # mold via the x_y interface with a zero-column predictor frame.
    if (!target_var %in% names(data_for_mold)) {
      cli::cli_abort(
        "Target column {.val {target_var}} not found in {.arg data}."
      )
    }
    y <- data_for_mold[[target_var]]
    empty_x <- data_for_mold[, character(0), drop = FALSE]
    processed <- hardhat::mold(empty_x, y)
  }

  brulee_chronos_bridge(
    processed,
    item_id = item_id,
    timestamp = timestamp,
    id_column = id_column,
    timestamp_column = timestamp_column,
    target_column = target_var,
    id_synthetic = id_synthetic,
    timestamp_synthetic = timestamp_synthetic,
    model_id = model_id,
    revision = revision,
    prediction_length = prediction_length,
    quantile_levels = quantile_levels,
    device = device,
    cache_dir = cache_dir,
    ...
  )
}

# Recipe method

#' @export
#' @rdname brulee_chronos
brulee_chronos.recipe <- function(
  x,
  data,
  model_id = "amazon/chronos-2",
  revision = chronos2_default_revision(),
  prediction_length = NULL,
  quantile_levels = (1:9) / 10,
  device = NULL,
  cache_dir = file.path(Sys.getenv("HOME"), ".cache", "chronos-r"),
  ...
) {
  processed <- hardhat::mold(x, data)

  # hardhat passes columns with non-standard recipe roles through
  # `processed$extras$roles` (one tibble per role). The id and time roles
  # are both optional; when absent we fall back to a single synthesized
  # series and row order, respectively.
  roles <- processed$extras$roles
  id_role <- roles$id
  time_role <- roles$time

  if (is.null(id_role)) {
    id_synthetic <- TRUE
    item_id <- rep("default", nrow(data))
    id_column <- ".id_column"
  } else if (ncol(id_role) != 1L) {
    cli::cli_abort(c(
      "The recipe must have at most one variable with role {.val id}."
    ))
  } else {
    id_synthetic <- FALSE
    id_column <- names(id_role)
    item_id <- id_role[[1L]]
  }

  if (is.null(time_role)) {
    timestamp_synthetic <- TRUE
    timestamp <- seq_len(nrow(data))
    timestamp_column <- ".timestamp_column"
  } else if (ncol(time_role) != 1L) {
    cli::cli_abort(c(
      "The recipe must have at most one variable with role {.val time}."
    ))
  } else {
    timestamp_synthetic <- FALSE
    timestamp_column <- names(time_role)
    timestamp <- time_role[[1L]]
  }

  outcome_names <- names(processed$blueprint$ptypes$outcomes)
  if (length(outcome_names) != 1L) {
    cli::cli_abort(
      "The recipe must have exactly one variable with role {.val outcome}."
    )
  }
  target_column <- outcome_names

  brulee_chronos_bridge(
    processed,
    item_id = item_id,
    timestamp = timestamp,
    id_column = id_column,
    timestamp_column = timestamp_column,
    target_column = target_column,
    id_synthetic = id_synthetic,
    timestamp_synthetic = timestamp_synthetic,
    model_id = model_id,
    revision = revision,
    prediction_length = prediction_length,
    quantile_levels = quantile_levels,
    device = device,
    cache_dir = cache_dir,
    ...
  )
}

# ------------------------------------------------------------------------------
# Bridge

brulee_chronos_bridge <- function(
  processed,
  item_id,
  timestamp,
  id_column,
  timestamp_column,
  target_column,
  id_synthetic = FALSE,
  timestamp_synthetic = FALSE,
  model_id,
  revision,
  prediction_length,
  quantile_levels,
  device,
  cache_dir,
  ...
) {
  if (!torch::torch_is_installed()) {
    cli::cli_abort(
      "The torch backend has not been installed; use {.run torch::install_torch()}."
    )
  }

  # Pretrained-model arg validation
  check_string(model_id)
  check_string(revision)
  check_string(cache_dir)
  if (!is.null(device)) {
    check_string(device)
  }
  if (!is.null(prediction_length)) {
    if (is.numeric(prediction_length) && !is.integer(prediction_length)) {
      prediction_length <- as.integer(prediction_length)
    }
    check_integer(prediction_length, single = TRUE, x_min = 1L)
  }
  if (!is.numeric(quantile_levels) || length(quantile_levels) < 1L) {
    cli::cli_abort(
      "{.arg quantile_levels} must be a non-empty numeric vector."
    )
  }
  if (any(quantile_levels <= 0 | quantile_levels >= 1)) {
    cli::cli_abort(
      "{.arg quantile_levels} must be in the open interval (0, 1)."
    )
  }

  # Predictor / outcome validation
  predictors <- processed$predictors
  outcome <- processed$outcomes[[1]]

  if (!is.numeric(outcome)) {
    cli::cli_abort(
      "The target ({.arg y} / left-hand side of the formula) must be numeric."
    )
  }

  non_numeric <- vapply(predictors, function(col) !is.numeric(col), logical(1))
  if (any(non_numeric)) {
    bad <- names(predictors)[non_numeric]
    cli::cli_abort(c(
      "All past covariates must be numeric. Non-numeric column{?s}: {.val {bad}}.",
      "i" = "Use a recipe (e.g. {.fn recipes::step_dummy}) to encode them as numeric."
    ))
  }

  # Length checks for id/timestamp
  n <- length(outcome)
  if (length(item_id) != n) {
    cli::cli_abort(
      "{.arg item_id} has length {length(item_id)} but {.arg y} has length {n}."
    )
  }
  if (length(timestamp) != n) {
    cli::cli_abort(
      "{.arg timestamp} has length {length(timestamp)} but {.arg y} has length {n}."
    )
  }
  if (anyNA(item_id)) {
    cli::cli_abort("{.arg item_id} must not contain {.code NA}.")
  }
  if (anyNA(timestamp)) {
    cli::cli_abort("{.arg timestamp} must not contain {.code NA}.")
  }

  # Resolve device
  torch_device <- guess_brulee_device(device)

  # Download, parse, build, load. Returns the resolved commit SHA so we
  # can record exactly which version of the weights ended up in the object.
  download_info <- chronos2_download(
    model_id,
    revision = revision,
    cache_dir = cache_dir
  )
  resolved_sha <- download_info$sha
  config <- chronos2_parse_config(
    file.path(download_info$model_dir, "config.json")
  )

  max_prediction_length <- config$max_output_patches * config$output_patch_size
  if (is.null(prediction_length)) {
    prediction_length <- as.integer(max_prediction_length)
  }
  if (prediction_length > max_prediction_length) {
    cli::cli_abort(
      "{.arg prediction_length} ({prediction_length}) exceeds model maximum ({max_prediction_length})."
    )
  }

  unavailable <- setdiff(quantile_levels, config$quantiles)
  if (length(unavailable) > 0) {
    cli::cli_abort(c(
      "Requested quantile levels not available in model: {.val {unavailable}}.",
      "i" = "Available: {.val {config$quantiles}}"
    ))
  }

  model <- chronos2_model(config)
  load_chronos2_weights(
    model,
    file.path(download_info$model_dir, "model.safetensors")
  )
  model$to(device = torch_device)
  model$eval()

  # Per-series context split
  context <- chronos2_split_by_series(
    target = as.numeric(outcome),
    covariates = as.data.frame(predictors),
    item_id = item_id,
    timestamp = timestamp,
    id_column = id_column,
    timestamp_column = timestamp_column,
    target_column = target_column,
    id_synthetic = id_synthetic,
    timestamp_synthetic = timestamp_synthetic
  )

  structure(
    list(
      model = model,
      config = config,
      device = torch_device,
      prediction_length = as.integer(prediction_length),
      quantile_levels = quantile_levels,
      model_id = model_id,
      revision = resolved_sha,
      blueprint = processed$blueprint,
      context = context
    ),
    class = "brulee_chronos"
  )
}

#' @export
print.brulee_chronos <- function(x, ...) {
  cat(
    cli::style_bold("Chronos-2 Pretrained Forecasting Model"),
    "\n\n",
    sep = ""
  )

  n_series <- length(x$context$item_ids)
  history_lengths <- lengths(x$context$series_target)
  n_covars <- length(x$context$covariate_cols)

  device_label <- tryCatch(
    as.character(x$device),
    error = function(e) "<not available; model has not been reloaded>"
  )

  if (is.null(x$revision)) {
    short_sha <- "unknown"
  } else {
    short_sha <- substr(x$revision, 1, 8)
  }
  mod_lst <- c(
    " " = "Source: {x$model_id} @ {short_sha}",
    " " = "Model dim: {x$config$d_model}",
    " " = "Layers: {x$config$num_layers}",
    " " = "Attention heads: {x$config$num_heads}",
    " " = "Prediction length: {x$prediction_length}",
    " " = "Quantiles: {x$quantile_levels}",
    " " = "Device: {device_label}",
    " " = "Context: {n_series} series, max history {max(history_lengths)}, {n_covars} covariate{?s}"
  )

  cli::cli_bullets(mod_lst)

  if (identical(device_label, "<not available; model has not been reloaded>")) {
    cli::cli_alert_warning(
      "Object contains an invalid external pointer (e.g. loaded from an RDS without serialization). Re-fit or reload the model to restore full functionality."
    )
  }

  invisible(x)
}

# ------------------------------------------------------------------------------
# Internal helpers

# Download / revision-resolution helpers live in chronos2-misc.R alongside
# the safetensors loader and config parser.

# Split a long-format context (target vector + covariate frame + per-row id and
# timestamp vectors) into per-series structures keyed by unique item_id and
# sorted by timestamp.
chronos2_split_by_series <- function(
  target,
  covariates,
  item_id,
  timestamp,
  id_column,
  timestamp_column,
  target_column,
  id_synthetic = FALSE,
  timestamp_synthetic = FALSE
) {
  covariates <- as.data.frame(covariates)
  item_ids <- unique(item_id)

  series_target <- vector("list", length(item_ids))
  series_covars <- vector("list", length(item_ids))
  series_timestamp <- vector("list", length(item_ids))

  for (i in seq_along(item_ids)) {
    mask <- item_id == item_ids[i]
    sub_ts <- timestamp[mask]
    ord <- order(sub_ts)
    series_target[[i]] <- as.numeric(target[mask][ord])
    series_covars[[i]] <- covariates[mask, , drop = FALSE][ord, , drop = FALSE]
    series_timestamp[[i]] <- sub_ts[ord]
  }

  list(
    item_ids = item_ids,
    series_target = series_target,
    series_covars = series_covars,
    series_timestamp = series_timestamp,
    covariate_cols = colnames(covariates),
    id_column = id_column,
    timestamp_column = timestamp_column,
    target_column = target_column,
    id_synthetic = id_synthetic,
    timestamp_synthetic = timestamp_synthetic
  )
}

# Resolve a tidyselect-style id/timestamp column argument against a data
# frame. Accepts NULL (returns NULL), bare names / `c(name)` / tidyselect
# expressions, and legacy single-string column names. Errors when the
# expression does not select exactly one column.
chronos2_resolve_column <- function(quo, data, arg_name) {
  if (rlang::quo_is_null(quo)) {
    return(NULL)
  }

  expr <- rlang::quo_get_expr(quo)
  if (is.character(expr) && length(expr) == 1L) {
    if (!expr %in% names(data)) {
      cli::cli_abort(
        "Column {.val {expr}} (from {.arg {arg_name}}) not found in {.arg data}."
      )
    }
    return(expr)
  }

  pos <- tryCatch(
    tidyselect::eval_select(quo, data),
    error = function(e) {
      cli::cli_abort(
        "Couldn't resolve {.arg {arg_name}}: {conditionMessage(e)}",
        call = NULL
      )
    }
  )
  if (length(pos) != 1L) {
    cli::cli_abort(
      "{.arg {arg_name}} must select exactly one column, got {length(pos)}."
    )
  }
  names(pos)
}
