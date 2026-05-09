#' Predict from a `brulee_chronos` model
#'
#' @param object A `brulee_chronos` object returned by [brulee_chronos()].
#' @param new_data Optional data frame in the same long format as the data
#'   used to build `object` --- it must contain the id, timestamp, target,
#'   and covariate columns named in `object`. If `NULL` (the default), the
#'   context stored in `object` is used.
#' @param future_df Optional data frame with future covariate values. Must
#'   contain the id and timestamp columns plus any covariate columns to
#'   provide for the future window (a subset of the past covariates). Each
#'   series must have exactly `prediction_length` rows.
#' @param prediction_length Number of future time steps to forecast. Defaults
#'   to the value stored in `object`.
#' @param quantile_levels Numeric vector of quantile levels. Defaults to the
#'   value stored in `object`.
#' @param ... Not used.
#'
#' @returns A data frame with columns:
#'   \describe{
#'     \item{`<id_column>`}{The time series identifier.}
#'     \item{`<timestamp_column>`}{Future timestamps.}
#'     \item{mean}{Point forecast (the 0.5 quantile).}
#'     \item{`0.1`, `0.2`, ...}{One column per requested quantile level.}
#'   }
#' @examplesIf !brulee:::is_cran_check()
#' \dontrun{
#' data(Chicago, package = "modeldata")
#' chi <- Chicago[, c("date", "ridership")]
#' chi$series_id <- "L"
#'
#' mod <- brulee_chronos(
#'   ridership ~ .,
#'   data = chi[, c("series_id", "date", "ridership")],
#'   id_column = "series_id",
#'   timestamp_column = "date"
#' )
#' predict(mod, prediction_length = 14)
#' }
#' @export
predict.brulee_chronos <- function(
  object,
  new_data = NULL,
  future_df = NULL,
  prediction_length = NULL,
  quantile_levels = NULL,
  ...
) {
  if (is.null(prediction_length)) {
    prediction_length <- object$prediction_length
  }
  if (is.null(quantile_levels)) {
    quantile_levels <- object$quantile_levels
  }

  id_column <- object$context$id_column
  timestamp_column <- object$context$timestamp_column
  target_column <- object$context$target_column
  covariate_cols <- object$context$covariate_cols
  has_stored_covariates <- length(covariate_cols) > 0L

  # Resolve context: stored or forged from new_data
  if (is.null(new_data)) {
    ctx <- object$context
  } else {
    if (has_stored_covariates) {
      forged <- hardhat::forge(
        new_data,
        object$blueprint,
        outcomes = TRUE
      )

      # Recipe-built models pass id/time through `forged$extras$roles`.
      # Formula / x_y blueprints don't have that mechanism, so fall back to
      # pulling those columns by name from `new_data`.
      roles <- forged$extras$roles
      if (!is.null(roles) && !is.null(roles$id) && !is.null(roles$time)) {
        item_id <- roles$id[[1L]]
        timestamp <- roles$time[[1L]]
      } else {
        item_id <- chronos2_pull_column(new_data, id_column, "id_column")
        timestamp <- chronos2_pull_column(
          new_data,
          timestamp_column,
          "timestamp_column"
        )
      }

      target <- forged$outcomes[[1]]
      covariates <- as.data.frame(forged$predictors)
    } else {
      # No covariates: skip forge entirely. Pull target / id / timestamp by
      # stored names. (For x_y-built models, target_column == ".outcome", so
      # the user must include a column with that name.)
      target <- chronos2_pull_column(new_data, target_column, "target")
      item_id <- chronos2_pull_column(new_data, id_column, "id_column")
      timestamp <- chronos2_pull_column(
        new_data,
        timestamp_column,
        "timestamp_column"
      )
      covariates <- as.data.frame(matrix(
        NA_real_,
        nrow = length(target),
        ncol = 0
      ))
    }

    if (!is.numeric(target)) {
      cli::cli_abort("The target column must be numeric.")
    }
    non_numeric <- vapply(
      covariates,
      function(col) !is.numeric(col),
      logical(1)
    )
    if (any(non_numeric)) {
      bad <- names(covariates)[non_numeric]
      cli::cli_abort(c(
        "All past covariates must be numeric. Non-numeric column{?s}: {.val {bad}}."
      ))
    }

    ctx <- chronos2_split_by_series(
      target = as.numeric(target),
      covariates = covariates,
      item_id = item_id,
      timestamp = timestamp,
      id_column = id_column,
      timestamp_column = timestamp_column,
      target_column = target_column
    )
  }

  has_covariates <- length(ctx$covariate_cols) > 0

  # Future-covariate handling
  future_cov_cols <- character(0)
  future_list <- NULL
  if (!is.null(future_df)) {
    if (!id_column %in% names(future_df)) {
      cli::cli_abort(
        "Column {.val {id_column}} not found in {.arg future_df}."
      )
    }
    if (!timestamp_column %in% names(future_df)) {
      cli::cli_abort(
        "Column {.val {timestamp_column}} not found in {.arg future_df}."
      )
    }
    future_cov_cols <- setdiff(
      names(future_df),
      c(id_column, timestamp_column)
    )
    bad_cols <- setdiff(future_cov_cols, ctx$covariate_cols)
    if (length(bad_cols) > 0) {
      cli::cli_abort(c(
        "Columns in {.arg future_df} not found as covariates: {.val {bad_cols}}.",
        "i" = "Available covariate columns: {.val {ctx$covariate_cols}}"
      ))
    }

    future_list <- lapply(ctx$item_ids, function(id) {
      sub <- future_df[future_df[[id_column]] == id, , drop = FALSE]
      sub[order(sub[[timestamp_column]]), , drop = FALSE]
    })
  }

  # Per-series target / covariate / timestamp lists from context
  contexts <- ctx$series_target
  past_cov_list <- ctx$series_covars
  future_timestamps <- lapply(ctx$series_timestamp, function(ts_col) {
    infer_future_timestamps(ts_col, prediction_length)
  })

  if (has_covariates) {
    if (length(future_cov_cols) > 0 && !is.null(future_list)) {
      future_cov_list <- lapply(seq_along(ctx$item_ids), function(i) {
        future_sub <- future_list[[i]]
        if (nrow(future_sub) != prediction_length) {
          cli::cli_abort(
            "Series {.val {ctx$item_ids[i]}}: {.arg future_df} has {nrow(future_sub)} rows, expected {prediction_length}."
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

  # Map requested quantile levels to model quantile indices
  model_quantiles <- object$config$quantiles
  quantile_indices <- match(quantile_levels, model_quantiles)

  median_idx <- match(0.5, model_quantiles)
  if (is.na(median_idx)) {
    median_idx <- which.min(abs(model_quantiles - 0.5))
  }

  # Build output data frame
  out_rows <- vector("list", length(ctx$item_ids))
  for (i in seq_along(ctx$item_ids)) {
    preds_i <- as.matrix(result$predictions[i, , ])

    row_df <- data.frame(
      id = rep(ctx$item_ids[i], prediction_length),
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

chronos2_pull_column <- function(data, column, arg_label) {
  if (!column %in% names(data)) {
    cli::cli_abort(
      "Column {.val {column}} (from {.arg {arg_label}}) not found in {.arg new_data}."
    )
  }
  data[[column]]
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
