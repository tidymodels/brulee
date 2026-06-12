#' Predict from a `brulee_chronos` model
#'
#' @param object A `brulee_chronos` object returned by [brulee_chronos()].
#' @param new_data Optional data frame in the same long format as the data
#'   used to build `object`. It should contain the target and covariate
#'   columns named in `object`, plus the id and timestamp columns when
#'   those were supplied at construction. (If the model was built without
#'   an id column, every row of `new_data` is treated as part of the same
#'   single series; similarly, if the model was built without a timestamp
#'   column, row order is used as the time order.) If `NULL` (the
#'   default), the context stored in `object` is used.
#' @param future_df Optional data frame with future covariate values. Must
#'   contain the id and timestamp columns (when present in the original
#'   model) plus any covariate columns to provide for the future window (a
#'   subset of the past covariates). Each series must have exactly
#'   `prediction_length` rows.
#' @param prediction_length Number of future time steps to forecast. Defaults
#'   to the value stored in `object`.
#' @param quantile_levels Numeric vector of quantile levels. Defaults to the
#'   value stored in `object`.
#' @param ... Not used.
#'
#' @returns A [tibble][tibble::tibble] with one row per future time step
#'   per series, in the same order as the rows of `new_data` (or the stored
#'   context). Columns:
#'   \describe{
#'     \item{`<id_column>`}{The time series identifier. Omitted when the
#'       context contains a single series.}
#'     \item{`.pred`}{Point forecast, i.e. the median of `.pred_quantile`.}
#'     \item{`.pred_quantile`}{A [hardhat::quantile_pred()] vector packing
#'       all requested quantile levels into a single column.}
#'   }
#' @examplesIf !brulee:::is_cran_check()
#' pkgs <- c("recipes", "lubridate", "modeldata", "ggplot2")
#'
#' \dontrun{
#' if (torch::torch_is_installed() & rlang::is_installed(pkgs)) {
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
#'  pred_1 <- predict(mod_1, test_data)
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
#'  pred_2 <- predict(mod_2, future_df = test_data)
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
#'  pred_3 <- predict(mod_3, future_df = test_data)
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
  if (is.data.frame(new_data) & length(new_data) == 0L) {
    new_data <- NULL
  }

  id_column <- object$context$id_column
  timestamp_column <- object$context$timestamp_column
  target_column <- object$context$target_column
  covariate_cols <- object$context$covariate_cols
  id_synthetic <- isTRUE(object$context$id_synthetic)
  timestamp_synthetic <- isTRUE(object$context$timestamp_synthetic)
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
      # pulling those columns by name from `new_data` (or synthesizing,
      # when the model was built without an id / timestamp column).
      roles <- forged$extras$roles
      n_rows <- nrow(new_data)

      if (id_synthetic) {
        item_id <- rep("default", n_rows)
      } else if (!is.null(roles) && !is.null(roles$id)) {
        item_id <- roles$id[[1L]]
      } else {
        item_id <- chronos2_pull_column(new_data, id_column, "id_column")
      }

      if (timestamp_synthetic) {
        timestamp <- seq_len(n_rows)
      } else if (!is.null(roles) && !is.null(roles$time)) {
        timestamp <- roles$time[[1L]]
      } else {
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
      n_rows <- length(target)

      if (id_synthetic) {
        item_id <- rep("default", n_rows)
      } else {
        item_id <- chronos2_pull_column(new_data, id_column, "id_column")
      }
      if (timestamp_synthetic) {
        timestamp <- seq_len(n_rows)
      } else {
        timestamp <- chronos2_pull_column(
          new_data,
          timestamp_column,
          "timestamp_column"
        )
      }
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
      target_column = target_column,
      id_synthetic = id_synthetic,
      timestamp_synthetic = timestamp_synthetic
    )
  }

  has_covariates <- length(ctx$covariate_cols) > 0

  # Future-covariate handling
  future_cov_cols <- character(0)
  future_list <- NULL
  if (!is.null(future_df)) {
    if (!id_synthetic && !id_column %in% names(future_df)) {
      cli::cli_abort(
        "Column {.val {id_column}} not found in {.arg future_df}."
      )
    }
    if (!timestamp_synthetic && !timestamp_column %in% names(future_df)) {
      cli::cli_abort(
        "Column {.val {timestamp_column}} not found in {.arg future_df}."
      )
    }
    keep_cols <- c(
      if (!id_synthetic) id_column,
      if (!timestamp_synthetic) timestamp_column,
      ctx$covariate_cols
    )
    future_df <- future_df[,
      intersect(names(future_df), keep_cols),
      drop = FALSE
    ]
    drop_in_future <- c(
      if (!id_synthetic) id_column,
      if (!timestamp_synthetic) timestamp_column
    )
    future_cov_cols <- setdiff(names(future_df), drop_in_future)

    future_list <- lapply(ctx$item_ids, function(id) {
      sub <- if (id_synthetic) {
        future_df
      } else {
        future_df[future_df[[id_column]] == id, , drop = FALSE]
      }
      if (timestamp_synthetic) {
        sub
      } else {
        sub[order(sub[[timestamp_column]]), , drop = FALSE]
      }
    })
  }

  # Per-series target / covariate lists from context. Future timestamps are
  # not needed downstream. Prediction order is determined by horizon
  # alone and the timestamp column is intentionally not returned.
  contexts <- ctx$series_target
  past_cov_list <- ctx$series_covars

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

  # Build output tibble. Quantile predictions are packed into a single
  # `.pred_quantile` column via hardhat::quantile_pred(); `.pred` is the
  # median pulled out of that quantile_pred. The id column is omitted when
  # the context contains a single series; the timestamp column is never
  # returned.
  multiple_series <- length(ctx$item_ids) > 1L

  out_rows <- vector("list", length(ctx$item_ids))
  for (i in seq_along(ctx$item_ids)) {
    preds_i <- as.matrix(result$predictions[i, , ])

    # rows = future timesteps, cols = requested quantile levels
    q_mat <- t(preds_i[quantile_indices, , drop = FALSE])
    qp <- hardhat::quantile_pred(q_mat, quantile_levels)

    row_tbl <- tibble::tibble(
      .pred = as.numeric(stats::median(qp)),
      .pred_quantile = qp
    )
    if (multiple_series) {
      row_tbl <- tibble::add_column(
        row_tbl,
        !!id_column := rep(ctx$item_ids[i], prediction_length),
        .before = 1L
      )
    }

    out_rows[[i]] <- row_tbl
  }

  dplyr::bind_rows(out_rows)
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
