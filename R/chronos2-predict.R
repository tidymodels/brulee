#' Predict from a `brulee_chronos` model
#'
#' @param object A `brulee_chronos` object returned by [brulee_chronos()].
#' @param new_data Optional data frame describing the future window to
#'   forecast for. It should contain the id and timestamp columns (when those
#'   were supplied at construction) plus any known future covariate values (a
#'   subset of the past covariates). The number of rows per series is the
#'   number of future time steps to return and may be at most
#'   `prediction_length`; supplying more is an error. When a series has fewer
#'   rows than `prediction_length`, the missing future covariates are treated
#'   as unknown and the forecast is truncated to the rows provided. If `NULL`
#'   (the default), the full `prediction_length` horizon is forecast from the
#'   context stored in `object`. The model is pretrained, so the historical
#'   context is always the data passed to [brulee_chronos()] and is never
#'   overridden here.
#' @param type A single string for the type of prediction to return. The
#'   default `"all"` returns both the point forecast (`.pred`) and the quantile
#'   forecast (`.pred_quantile`). Use `"numeric"` for only `.pred` or
#'   `"quantile"` for only `.pred_quantile`.
#' @param prediction_length Number of future time steps to forecast. Defaults
#'   to the value stored in `object`.
#' @param quantile_levels Numeric vector of quantile levels. Defaults to the
#'   value stored in `object`.
#' @param ... Not used.
#'
#' @returns A [tibble][tibble::tibble] with one row per forecast time step
#'   per series (up to `nrow(new_data)` rows per series, or
#'   `prediction_length` rows when `new_data` is `NULL`). Columns depend on
#'   `type`:
#'   \describe{
#'     \item{`<id_column>`}{The time series identifier. Omitted when the
#'       context contains a single series.}
#'     \item{`.pred`}{Point forecast, i.e. the median of `.pred_quantile`.
#'       Returned when `type` is `"all"` or `"numeric"`.}
#'     \item{`.pred_quantile`}{A [hardhat::quantile_pred()] vector packing
#'       all requested quantile levels into a single column. Returned when
#'       `type` is `"all"` or `"quantile"`.}
#'   }
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
#'  pred_1 <- predict(mod_1, new_data = test_data)
#'  pred_1
#'
#'  pred_1 |>
#'   bind_cols(test_data) |>
#'   ggplot(aes(date)) +
#'   geom_point(aes(y = ridership, col = day)) +
#'   geom_line(aes(y = .pred)) +
#'   labs(title = "No covariates: Meh") +
#'   theme_bw()
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
#'  pred_2 <- predict(mod_2, new_data = test_data)
#'
#'  pred_2 |>
#'   bind_cols(test_data) |>
#'   ggplot(aes(date)) +
#'   geom_point(aes(y = ridership, col = day)) +
#'   geom_line(aes(y = .pred)) +
#'   labs(title = "Four covariates: Pretty good") +
#'   theme_bw()
#'
#'  # ------------------------------------------------------------------------------
#'  # Covariates using recipes
#'
#'  rec <-
#'   recipe(ridership ~ ., data = prior_data) |>
#'   update_role(date, new_role = "time")
#'
#'  mod_3 <- brulee_chronos(rec, data = prior_data, prediction_length = 14)
#'
#'  pred_3 <- predict(mod_3, new_data = test_data)
#'
#'  pred_3 |>
#'   bind_cols(test_data) |>
#'   ggplot(aes(date)) +
#'   geom_point(aes(y = ridership, col = day)) +
#'   geom_line(aes(y = .pred)) +
#'   labs(title = "All covariates: Better Saturdays") +
#'   theme_bw()
#' }
#' }
#' @export
predict.brulee_chronos <- function(
  object,
  new_data = NULL,
  type = "all",
  prediction_length = NULL,
  quantile_levels = NULL,
  ...
) {
  type <- rlang::arg_match(type, c("all", "numeric", "quantile"))
  if (is.null(prediction_length)) {
    prediction_length <- object$prediction_length
  }
  if (is.null(quantile_levels)) {
    quantile_levels <- object$quantile_levels
  }
  if (is.data.frame(new_data) && length(new_data) == 0L) {
    new_data <- NULL
  }

  id_column <- object$context$id_column
  timestamp_column <- object$context$timestamp_column
  id_synthetic <- isTRUE(object$context$id_synthetic)
  timestamp_synthetic <- isTRUE(object$context$timestamp_synthetic)

  # The context is always the data supplied to `brulee_chronos()`: the model
  # is pretrained and performs no training, so there is nothing to re-fit at
  # predict time.
  ctx <- object$context
  has_covariates <- length(ctx$covariate_cols) > 0

  # `new_data` is the future window: the timestamps (and any known future
  # covariate values) to forecast for. Its per-series row count is the number
  # of forecast steps requested, which may be at most `prediction_length`.
  # When `new_data` is NULL we forecast the full `prediction_length`.
  requested_lengths <- rep(prediction_length, length(ctx$item_ids))
  future_cov_cols <- character(0)
  future_list <- NULL
  if (!is.null(new_data)) {
    if (!id_synthetic && !id_column %in% names(new_data)) {
      cli::cli_abort(
        "Column {.val {id_column}} not found in {.arg new_data}."
      )
    }
    if (!timestamp_synthetic && !timestamp_column %in% names(new_data)) {
      cli::cli_abort(
        "Column {.val {timestamp_column}} not found in {.arg new_data}."
      )
    }
    keep_cols <- c(
      if (!id_synthetic) id_column,
      if (!timestamp_synthetic) timestamp_column,
      ctx$covariate_cols
    )
    new_data <- new_data[,
      intersect(names(new_data), keep_cols),
      drop = FALSE
    ]
    drop_in_future <- c(
      if (!id_synthetic) id_column,
      if (!timestamp_synthetic) timestamp_column
    )
    future_cov_cols <- setdiff(names(new_data), drop_in_future)

    sub_data <- function(id) {
      if (id_synthetic) {
        sub <- new_data
      } else {
        sub <- new_data[new_data[[id_column]] == id, , drop = FALSE]
      }
      if (timestamp_synthetic) {
        sub
      } else {
        sub[order(sub[[timestamp_column]]), , drop = FALSE]
      }
    }

    future_list <- purrr::map(ctx$item_ids, sub_data)

    # The future window sets the per-series forecast horizon. More rows than
    # `prediction_length` has no meaning; fewer is padded internally and
    # truncated from the output below.
    requested_lengths <- purrr::map_int(future_list, nrow)
    too_long <- which(requested_lengths > prediction_length)
    if (length(too_long) > 0L) {
      i <- too_long[1L]
      cli::cli_abort(
        "Series {.val {ctx$item_ids[i]}}: {.arg new_data} has {requested_lengths[i]} rows, more than the prediction length ({prediction_length})."
      )
    }
  }

  # Per-series target / covariate lists from context. Future timestamps are
  # not needed downstream. Prediction order is determined by horizon
  # alone and the timestamp column is intentionally not returned.
  contexts <- ctx$series_target
  past_cov_list <- ctx$series_covars

  if (has_covariates) {
    if (length(future_cov_cols) > 0 && !is.null(future_list)) {
      future_cov_list <- lapply(seq_along(ctx$item_ids), function(i) {
        cov <- future_list[[i]][, future_cov_cols, drop = FALSE]
        n <- nrow(cov)
        # Pad short future windows up to `prediction_length` with NA. The model
        # reads NaN future covariates as "unknown" (mask 0), so the padded tail
        # does not inform the forecast; it is truncated from the output below.
        if (n < prediction_length) {
          cov <- rbind(
            cov,
            cov[rep(NA_integer_, prediction_length - n), , drop = FALSE]
          )
        }
        cov
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

  # Build output tibble. Quantile predictions are packed into a single
  # `.pred_quantile` column via hardhat::quantile_pred(); `.pred` is the
  # median pulled out of that quantile_pred. `type` selects which of these
  # columns to return. The id column is omitted when the context contains a
  # single series; the timestamp column is never returned. Each series is
  # truncated to the number of forecast steps requested via `new_data` (the
  # full `prediction_length` when `new_data` is NULL).
  multiple_series <- length(ctx$item_ids) > 1L

  out_rows <- vector("list", length(ctx$item_ids))
  for (i in seq_along(ctx$item_ids)) {
    preds_i <- as.matrix(result$predictions[i, , ])
    n_i <- requested_lengths[i]

    # rows = future timesteps, cols = requested quantile levels
    q_mat <- t(preds_i[quantile_indices, , drop = FALSE])
    q_mat <- q_mat[seq_len(n_i), , drop = FALSE]
    qp <- hardhat::quantile_pred(q_mat, quantile_levels)

    row_tbl <- switch(
      type,
      all = tibble::tibble(
        .pred = as.numeric(stats::median(qp)),
        .pred_quantile = qp
      ),
      numeric = tibble::tibble(.pred = as.numeric(stats::median(qp))),
      quantile = tibble::tibble(.pred_quantile = qp)
    )
    if (multiple_series) {
      row_tbl <- tibble::add_column(
        row_tbl,
        !!id_column := rep(ctx$item_ids[i], n_i),
        .before = 1L
      )
    }

    out_rows[[i]] <- row_tbl
  }

  dplyr::bind_rows(out_rows)
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
      future_cov_df <- as.data.frame(future_covariates[[i]])
      future_covs <- as.list(future_cov_df)
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
