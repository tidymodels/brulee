#' Predict from a `brulee_chronos` model
#'
#' @param object A `brulee_chronos` object returned by [brulee_chronos()].
#' @param new_data A data frame in long format with columns for item
#'   identifiers, timestamps, and target values. Additional numeric columns are
#'   treated as past covariates.
#' @param future_df Optional data frame with future covariate values. Must have
#'   the same `id_column` and `timestamp_column` as `new_data`, plus covariate
#'   columns (a subset of extra columns in `new_data`). Each series must have
#'   exactly `prediction_length` rows.
#' @param prediction_length Number of future time steps to forecast. Defaults
#'   to the value set in [brulee_chronos()].
#' @param id_column Name of the column containing time series identifiers.
#'   Default: `"item_id"`.
#' @param timestamp_column Name of the column containing timestamps.
#'   Default: `"timestamp"`.
#' @param target Name of the column containing target values to forecast.
#'   Default: `"target"`.
#' @param quantile_levels Numeric vector of quantile levels. Defaults to the
#'   value set in [brulee_chronos()].
#' @param ... Not used.
#'
#' @returns A data frame with columns:
#'   \describe{
#'     \item{`<id_column>`}{The time series identifier.}
#'     \item{`<timestamp_column>`}{Future timestamps.}
#'     \item{mean}{Point forecast (the 0.5 quantile).}
#'     \item{`0.1`, `0.2`, ...}{One column per requested quantile level.}
#'   }
#'
#' @examples
#' \dontrun{
#' mod <- brulee_chronos()
#'
#' df <- data.frame(
#'   item_id = rep("air_passengers", 144),
#'   timestamp = seq(as.Date("1949-01-01"), by = "month", length.out = 144),
#'   target = as.numeric(AirPassengers)
#' )
#'
#' predict(mod, df, prediction_length = 24)
#'
#' # With covariates
#' df$temperature <- rnorm(144)
#' future_df <- data.frame(
#'   item_id = rep("air_passengers", 24),
#'   timestamp = seq(as.Date("1961-01-01"), by = "month", length.out = 24),
#'   temperature = rnorm(24)
#' )
#' predict(mod, df, future_df = future_df, prediction_length = 24)
#' }
#' @export
predict.brulee_chronos <- function(
  object,
  new_data,
  future_df = NULL,
  prediction_length = NULL,
  id_column = "item_id",
  timestamp_column = "timestamp",
  target = "target",
  quantile_levels = NULL,
  ...
) {
  if (is.null(prediction_length)) {
    prediction_length <- object$prediction_length
  }
  if (is.null(quantile_levels)) {
    quantile_levels <- object$quantile_levels
  }

  # Validate columns
  if (!id_column %in% names(new_data)) {
    cli::cli_abort("Column {.val {id_column}} not found in {.arg new_data}.")
  }
  if (!timestamp_column %in% names(new_data)) {
    cli::cli_abort(
      "Column {.val {timestamp_column}} not found in {.arg new_data}."
    )
  }
  if (!target %in% names(new_data)) {
    cli::cli_abort("Column {.val {target}} not found in {.arg new_data}.")
  }

  # Identify covariates
  reserved_cols <- c(id_column, timestamp_column, target)
  covariate_cols <- setdiff(names(new_data), reserved_cols)
  covariate_cols <- covariate_cols[vapply(
    new_data[covariate_cols],
    is.numeric,
    logical(1)
  )]

  # Identify known-future vs past-only covariates
  if (!is.null(future_df)) {
    if (!id_column %in% names(future_df)) {
      cli::cli_abort("Column {.val {id_column}} not found in {.arg future_df}.")
    }
    if (!timestamp_column %in% names(future_df)) {
      cli::cli_abort(
        "Column {.val {timestamp_column}} not found in {.arg future_df}."
      )
    }
    future_cov_cols <- setdiff(names(future_df), c(id_column, timestamp_column))
    bad_cols <- setdiff(future_cov_cols, covariate_cols)
    if (length(bad_cols) > 0) {
      cli::cli_abort(c(
        "Columns in {.arg future_df} not found as covariates in {.arg new_data}: {.val {bad_cols}}.",
        "i" = "Available covariate columns: {.val {covariate_cols}}"
      ))
    }
  } else {
    future_cov_cols <- character(0)
  }
  has_covariates <- length(covariate_cols) > 0

  # Split by item_id
  item_ids <- unique(new_data[[id_column]])
  series_list <- lapply(item_ids, function(id) {
    sub <- new_data[new_data[[id_column]] == id, ]
    sub[order(sub[[timestamp_column]]), ]
  })

  if (!is.null(future_df)) {
    future_list <- lapply(item_ids, function(id) {
      sub <- future_df[future_df[[id_column]] == id, ]
      sub[order(sub[[timestamp_column]]), ]
    })
  }

  # Infer future timestamps
  future_timestamps <- lapply(series_list, function(sub) {
    infer_future_timestamps(sub[[timestamp_column]], prediction_length)
  })

  # Build prediction inputs
  contexts <- lapply(series_list, function(sub) as.numeric(sub[[target]]))

  if (has_covariates) {
    past_cov_list <- lapply(series_list, function(sub) {
      sub[, covariate_cols, drop = FALSE]
    })

    if (length(future_cov_cols) > 0 && !is.null(future_df)) {
      future_cov_list <- lapply(seq_along(item_ids), function(i) {
        future_sub <- future_list[[i]]
        if (nrow(future_sub) != prediction_length) {
          cli::cli_abort(
            "Series {.val {item_ids[i]}}: {.arg future_df} has {nrow(future_sub)} rows, expected {prediction_length}."
          )
        }
        future_sub[, future_cov_cols, drop = FALSE]
      })
    } else {
      future_cov_list <- NULL
    }

    result <- chronos2_predict_core(
      object,
      contexts,
      prediction_length = prediction_length,
      past_covariates = past_cov_list,
      future_covariates = future_cov_list
    )
  } else {
    result <- chronos2_predict_core(
      object,
      contexts,
      prediction_length = prediction_length
    )
  }

  # Validate quantile levels against model

  model_quantiles <- object$config$quantiles
  quantile_indices <- match(quantile_levels, model_quantiles)

  median_idx <- match(0.5, model_quantiles)
  if (is.na(median_idx)) {
    median_idx <- which.min(abs(model_quantiles - 0.5))
  }

  # Build output data frame
  out_rows <- vector("list", length(item_ids))
  for (i in seq_along(item_ids)) {
    preds_i <- as.matrix(result$predictions[i, , ])

    row_df <- data.frame(
      id = rep(item_ids[i], prediction_length),
      timestamp = future_timestamps[[i]],
      mean = as.numeric(preds_i[median_idx, ])
    )
    names(row_df)[1] <- id_column
    names(row_df)[2] <- timestamp_column

    for (q_idx in seq_along(quantile_levels)) {
      row_df[[as.character(quantile_levels[q_idx])]] <- as.numeric(preds_i[
        quantile_indices[q_idx],
      ])
    }

    out_rows[[i]] <- row_df
  }

  do.call(rbind, out_rows)
}

# ─── Internal prediction engine ──────────────────────────────────────────────

chronos2_predict_core <- function(
  object,
  context,
  prediction_length = NULL,
  past_covariates = NULL,
  future_covariates = NULL
) {
  model <- object$model
  config <- object$config
  device <- object$device

  if (is.null(prediction_length)) {
    prediction_length <- object$prediction_length
  }

  num_output_patches <- ceiling(prediction_length / config$output_patch_size)
  num_output_patches <- min(num_output_patches, config$max_output_patches)

  if (!is.null(past_covariates)) {
    inputs <- chronos2_build_inputs(
      context,
      past_covariates,
      future_covariates,
      prediction_length
    )
    return(chronos2_run_with_covariates(
      model,
      config,
      device,
      inputs,
      prediction_length,
      num_output_patches
    ))
  }

  # Simple path (no covariates)
  if (is.numeric(context) && !inherits(context, "torch_tensor")) {
    context_tensor <- torch::torch_tensor(
      context,
      dtype = torch::torch_float32()
    )$unsqueeze(1)
  } else if (is.list(context)) {
    tensors <- lapply(context, function(x) {
      if (is.numeric(x)) {
        torch::torch_tensor(x, dtype = torch::torch_float32())
      } else {
        x
      }
    })
    context_tensor <- left_pad_and_stack(tensors)
  } else {
    context_tensor <- context
    if (context_tensor$dim() == 1) {
      context_tensor <- context_tensor$unsqueeze(1)
    }
  }

  context_tensor <- context_tensor$to(device = device)

  torch::with_no_grad({
    quantile_preds <- model(
      context_tensor,
      num_output_patches = num_output_patches
    )
  })

  quantile_preds <- quantile_preds[,, 1:prediction_length]$detach()$cpu()

  list(
    predictions = quantile_preds,
    quantiles = config$quantiles,
    prediction_length = prediction_length
  )
}

chronos2_build_inputs <- function(
  context,
  past_covariates,
  future_covariates,
  prediction_length
) {
  if (is.numeric(context) && !inherits(context, "torch_tensor")) {
    context <- list(context)
  } else if (inherits(context, "torch_tensor")) {
    if (context$dim() == 1) {
      context <- list(as.numeric(context))
    } else {
      context <- lapply(seq_len(context$size(1)), function(i) {
        as.numeric(context[i, ])
      })
    }
  }

  n_tasks <- length(context)

  if (is.data.frame(past_covariates) || is.matrix(past_covariates)) {
    past_covariates <- list(as.data.frame(past_covariates))
  }

  if (is.null(future_covariates)) {
    future_covariates <- vector("list", n_tasks)
  } else if (is.data.frame(future_covariates) || is.matrix(future_covariates)) {
    future_covariates <- list(as.data.frame(future_covariates))
  }

  lapply(seq_len(n_tasks), function(i) {
    past_df <- as.data.frame(past_covariates[[i]])
    past_covs <- as.list(past_df)

    future_covs <- list()
    if (!is.null(future_covariates[[i]])) {
      future_df <- as.data.frame(future_covariates[[i]])
      future_covs <- as.list(future_df)
    }

    list(
      target = context[[i]],
      past_covariates = past_covs,
      future_covariates = future_covs
    )
  })
}

chronos2_run_with_covariates <- function(
  model,
  config,
  device,
  inputs,
  prediction_length,
  num_output_patches
) {
  context_list <- list()
  future_covariates_list <- list()
  group_ids_list <- list()
  target_idx_ranges <- list()
  row_offset <- 0L

  for (i in seq_along(inputs)) {
    task <- inputs[[i]]
    target_vec <- task$target
    if (is.numeric(target_vec)) {
      target_vec <- torch::torch_tensor(
        target_vec,
        dtype = torch::torch_float32()
      )
    }
    if (target_vec$dim() == 1) {
      target_vec <- target_vec$unsqueeze(1)
    }
    n_targets <- target_vec$size(1)
    history_length <- target_vec$size(2)

    if (!is.null(task$past_covariates)) {
      past_covs <- task$past_covariates
    } else {
      past_covs <- list()
    }
    if (!is.null(task$future_covariates)) {
      future_covs <- task$future_covariates
    } else {
      future_covs <- list()
    }

    future_cov_keys <- sort(names(future_covs))
    past_only_keys <- sort(setdiff(names(past_covs), future_cov_keys))
    ordered_keys <- c(past_only_keys, future_cov_keys)

    if (length(ordered_keys) > 0) {
      past_cov_tensors <- lapply(ordered_keys, function(key) {
        v <- past_covs[[key]]
        if (is.numeric(v)) {
          v <- torch::torch_tensor(v, dtype = torch::torch_float32())
        }
        v$unsqueeze(1)
      })
      past_cov_tensor <- torch::torch_cat(past_cov_tensors, dim = 1)
    } else {
      past_cov_tensor <- torch::torch_zeros(0, history_length)
    }

    context_tensor <- torch::torch_cat(
      list(target_vec, past_cov_tensor),
      dim = 1
    )
    group_size <- context_tensor$size(1)

    future_target_pad <- torch::torch_full(
      c(n_targets, prediction_length),
      fill_value = NaN
    )
    if (length(ordered_keys) > 0) {
      future_cov_rows <- lapply(ordered_keys, function(key) {
        if (key %in% future_cov_keys) {
          v <- future_covs[[key]]
          if (is.numeric(v)) {
            v <- torch::torch_tensor(v, dtype = torch::torch_float32())
          }
          v$unsqueeze(1)
        } else {
          torch::torch_full(c(1, prediction_length), fill_value = NaN)
        }
      })
      future_cov_tensor <- torch::torch_cat(future_cov_rows, dim = 1)
    } else {
      future_cov_tensor <- torch::torch_zeros(0, prediction_length)
    }
    future_covariates_tensor <- torch::torch_cat(
      list(future_target_pad, future_cov_tensor),
      dim = 1
    )

    group_ids <- torch::torch_full(
      group_size,
      fill_value = as.integer(i - 1L),
      dtype = torch::torch_long()
    )

    context_list[[i]] <- context_tensor
    future_covariates_list[[i]] <- future_covariates_tensor
    group_ids_list[[i]] <- group_ids
    target_idx_ranges[[i]] <- c(row_offset + 1L, row_offset + n_targets)
    row_offset <- row_offset + group_size
  }

  batch_context <- left_pad_and_cat_2D(context_list)$to(device = device)
  batch_future_covariates <- torch::torch_cat(
    future_covariates_list,
    dim = 1
  )$to(device = device)
  batch_group_ids <- torch::torch_cat(group_ids_list)$to(device = device)

  torch::with_no_grad({
    quantile_preds <- model(
      batch_context,
      num_output_patches = num_output_patches,
      group_ids = batch_group_ids,
      future_covariates = batch_future_covariates
    )
  })

  quantile_preds <- quantile_preds[,, 1:prediction_length]$detach()$cpu()

  predictions <- lapply(target_idx_ranges, function(range) {
    quantile_preds[range[1]:range[2], , ]
  })

  if (length(predictions) == 1 && predictions[[1]]$size(1) == 1) {
    predictions <- predictions[[1]]
  } else if (all(vapply(predictions, function(p) p$size(1), integer(1)) == 1)) {
    predictions <- torch::torch_cat(predictions, dim = 1)
  }

  list(
    predictions = predictions,
    quantiles = config$quantiles,
    prediction_length = prediction_length
  )
}

# ─── Timestamp inference ─────────────────────────────────────────────────────

infer_future_timestamps <- function(ts_col, prediction_length) {
  n <- length(ts_col)
  if (n < 2) {
    cli::cli_abort(
      "Each time series must have at least 2 observations to infer frequency."
    )
  }
  last_ts <- ts_col[n]

  if (inherits(ts_col, "Date")) {
    diffs <- as.numeric(diff(tail(ts_col, min(n, 12))), units = "days")
    if (all(diffs >= 28 & diffs <= 31)) {
      return(seq(last_ts, by = "month", length.out = prediction_length + 1)[-1])
    }
    freq_days <- median(diffs)
    return(
      last_ts +
        as.difftime(freq_days, units = "days") * seq_len(prediction_length)
    )
  } else if (inherits(ts_col, "POSIXct") || inherits(ts_col, "POSIXlt")) {
    diffs <- as.numeric(diff(tail(ts_col, min(n, 12))), units = "secs")
    median_secs <- median(diffs)
    if (abs(median_secs - 3600) < 60) {
      return(seq(last_ts, by = "hour", length.out = prediction_length + 1)[-1])
    } else if (abs(median_secs - 86400) < 60) {
      return(seq(last_ts, by = "day", length.out = prediction_length + 1)[-1])
    } else if (all(diffs >= 28 * 86400 & diffs <= 31 * 86400)) {
      return(seq(last_ts, by = "month", length.out = prediction_length + 1)[-1])
    }
    return(last_ts + median_secs * seq_len(prediction_length))
  } else {
    freq <- ts_col[n] - ts_col[n - 1]
    return(last_ts + freq * seq_len(prediction_length))
  }
}
