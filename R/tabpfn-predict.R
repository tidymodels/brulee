#' Predict from a `brulee_tab_pfn`
#'
#' @param object A `brulee_tab_pfn` object.
#' @param new_data A data frame or matrix of new predictors.
#' @param type A single character. The type of predictions to generate.
#' Valid options are:
#'
#' - `"numeric"` for numeric predictions (the mean of the predictive
#'   distribution).
#' - `"quantile"` for quantiles of the predictive distribution.
#' - `"class"` for hard class predictions.
#' - `"prob"` for soft class predictions (i.e., class probabilities).
#'
#' The default (`NULL`) is `"class"` for classification and `"numeric"` for
#' regression.
#'
#' @param quantile_levels A numeric vector of probabilities in (0, 1) for
#' `type = "quantile"`.
#' @param ... Not used, but required for extensibility.
#' @return A tibble of predictions with one row per row of `new_data`:
#' `.pred_class`, `.pred_{level}` columns, `.pred`, or `.pred_quantile` (a
#' [hardhat::quantile_pred()] column), depending on `type`.
#' @details
#' TabPFN runs the network when predicting: each call feeds the stored
#' training rows and `new_data` through the model.
#'
#' R torch currently lacks a memory-efficient attention kernel for Apple GPUs
#' that PyTorch (>= 2.13) has, so with `device = "mps"` attention is computed
#' in chunks of rows to keep it within the GPU's memory.
#'
#' @examplesIf FALSE
#' fit <- brulee_tab_pfn(mpg ~ ., data = mtcars)
#' predict(fit, mtcars[1:3, ])
#' predict(fit, mtcars[1:3, ], type = "quantile", quantile_levels = c(0.1, 0.9))
#' @export
predict.brulee_tab_pfn <- function(
  object,
  new_data,
  type = NULL,
  quantile_levels = (1:9) / 10,
  ...
) {
  call <- rlang::current_env()
  check_dots_empty(call = call)
  type <- check_type(
    object,
    type,
    numeric_types = c("numeric", "quantile"),
    call = call
  )
  if (identical(type, "quantile")) {
    check_quantile_levels(quantile_levels, call = call)
  } else {
    quantile_levels <- NULL
  }
  res <- tabpfn_predict(object, new_data, quantile_levels)
  predictions <- switch(
    type,
    class = res[".pred_class"],
    prob = res[paste0(".pred_", object$levels)],
    numeric = res[".pred"],
    quantile = res[".pred_quantile"]
  )
  hardhat::validate_prediction_size(predictions, new_data)
  predictions
}

# All the predictions of a type from one pass of the model: class and
# probabilities for classification; the mean and, when `quantile_levels` is
# given, the quantiles for regression.
tabpfn_predict <- function(object, new_data, quantile_levels = NULL) {
  forged <- hardhat::forge(new_data, object$blueprint)$predictors
  fit <- object$fit
  x_new <- tabpfn_encode(fit$encoder, as.data.frame(forged))
  if (!is.null(object$levels)) {
    probs <- tabpfn_predict_classification(fit, x_new)
    return(tabpfn_classification_tibble(probs, object$levels, fit$classes))
  }
  res <- tabpfn_predict_regression(fit, x_new, quantile_levels)
  out <- tibble::tibble(.pred = res$mean)
  if (!is.null(quantile_levels)) {
    out$.pred_quantile <- hardhat::quantile_pred(
      res$quantiles,
      quantile_levels
    )
  }
  out
}

#' @rdname brulee-augment
#' @export
augment.brulee_tab_pfn <- function(x, new_data, quantile_levels = NULL, ...) {
  call <- rlang::current_env()
  check_dots_empty(call = call)
  classification <- brulee_mode(x, call = call) == "classification"
  if (!is.null(quantile_levels)) {
    if (classification) {
      cli::cli_abort(
        "{.arg quantile_levels} is only used for regression fits.",
        call = call
      )
    }
    check_quantile_levels(quantile_levels, call = call)
  }
  res <- tabpfn_predict(x, new_data, quantile_levels)
  if (classification) {
    res <- res[c(".pred_class", paste0(".pred_", x$levels))]
  } else {
    res <- brulee_add_resid(res, x, new_data, call = call)
    if (".resid" %in% names(res)) {
      res <- dplyr::relocate(res, ".resid", .after = ".pred")
    }
  }
  dplyr::bind_cols(res, new_data)
}

tabpfn_model_inputs <- function(fit, x_new, y_members) {
  x_all <- rbind(fit$x_train, x_new)
  tabpfn_member_inputs(
    fit$members,
    fit$preprocessors,
    x_all,
    nrow(fit$x_train),
    y_members,
    fit$categorical,
    fit$outlier_std
  )
}

tabpfn_predict_classification <- function(fit, x_new) {
  y_members <- lapply(fit$members, function(m) {
    m$class_permutation[fit$y_train + 1L]
  })
  inputs <- tabpfn_model_inputs(fit, x_new, y_members)
  model <- tabpfn_fit_model(fit)
  out <- tabpfn_forward(model, inputs$x, inputs$y, fit$task_type, fit$device)
  class_counts <- NULL
  if (fit$balance_probabilities) {
    class_counts <- fit$class_counts
  }
  tabpfn_postprocess_classification(
    out,
    fit$members,
    length(fit$classes),
    temperature = fit$softmax_temperature,
    average_before_softmax = fit$average_before_softmax,
    class_counts = class_counts
  )
}

tabpfn_predict_regression <- function(fit, x_new, quantile_levels = NULL) {
  n_new <- nrow(x_new)
  if (!is.null(fit$constant)) {
    return(list(
      mean = rep(fit$constant, n_new),
      quantiles = matrix(fit$constant, n_new, length(quantile_levels))
    ))
  }
  y_members <- lapply(fit$members, function(m) {
    tabpfn_apply_target_transform(fit$y_train, m$target_transform)
  })
  inputs <- tabpfn_model_inputs(fit, x_new, y_members)
  model <- tabpfn_fit_model(fit)
  out <- tabpfn_forward(model, inputs$x, inputs$y, fit$task_type, fit$device)
  borders <- model$borders()$to(device = "cpu")
  log_probs <- tabpfn_postprocess_regression(
    out,
    fit$members,
    borders,
    temperature = fit$softmax_temperature,
    average_before_softmax = fit$average_before_softmax
  )
  tabpfn_decode_regression(log_probs, borders, fit$y_scale, quantile_levels)
}

tabpfn_fit_model <- function(fit) {
  info <- tabpfn_version_info(fit$version, fit$task, fit$file)
  tabpfn_load(info, fit$device)$model
}

tabpfn_classification_tibble <- function(probs, levels, classes) {
  full <- matrix(0, nrow(probs), length(levels))
  full[, match(classes, levels)] <- probs
  colnames(full) <- paste0(".pred_", levels)
  out <- tibble::as_tibble(as.data.frame(full))
  out$.pred_class <- factor(
    levels[max.col(full, ties.method = "first")],
    levels = levels
  )
  out
}

# ------------------------------------------------------------------------------

# Loaded models are kept for the session, keyed by checkpoint path and
# device, so repeated fits and predictions don't reload the weights.

tabpfn_load <- function(info, device, call = caller_env()) {
  path <- tabpfn_checkpoint_path(info, call = call)
  key <- paste(path, device, sep = "|")
  entry <- tabpfn_env$models[[key]]
  if (is.null(entry)) {
    checkpoint <- tabpfn_read_checkpoint(path, call = call)
    model <- tabpfn_build_model(checkpoint, info$architecture, call = call)
    model$to(device = device)
    entry <- list(
      model = model,
      inference_config = checkpoint$inference_config,
      path = path
    )
    tabpfn_env$models[[key]] <- entry
  }
  entry
}

# Unlike `guess_brulee_device()`, the default doesn't pick MPS: on Apple GPUs
# TabPFN was not faster than on the CPU in our tests.
tabpfn_resolve_device <- function(device = NULL, call = caller_env()) {
  if (is.null(device)) {
    if (torch::cuda_is_available()) {
      return("cuda")
    }
    return("cpu")
  }
  device <- arg_match(device, c("cpu", "cuda", "mps"), error_call = call)
  if (device == "cuda" && !torch::cuda_is_available()) {
    cli::cli_abort("CUDA is not available.", call = call)
  }
  if (device == "mps" && !torch::backends_mps_is_available()) {
    cli::cli_abort("MPS is not available.", call = call)
  }
  device
}

# Run the model over the ensemble members. `xs` holds one (rows, C_i) input
# per member; members of equal width are batched along dim 2, at most
# `brulee.tabpfn_rows_per_pass` rows (summed over members) and 768 million cells
# per forward pass (Python `EstimatorBatchBudget` uses 32768 rows). Returns
# (test rows, members, outputs) on the CPU, in member order.
tabpfn_forward <- function(model, xs, y, task_type, device) {
  widths <- vapply(xs, function(x) x$size(2), numeric(1))
  rows <- xs[[1]]$size(1)
  budget <- getOption("brulee.tabpfn_rows_per_pass", 32768)
  outs <- vector("list", length(xs))
  for (w in unique(widths)) {
    members <- which(widths == w)
    per_pass <- max(1, min(budget %/% rows, floor(768e6 / (rows * w))))
    for (s in seq(1, length(members), by = per_pass)) {
      idx <- members[s:min(s + per_pass - 1, length(members))]
      x <- torch::torch_stack(xs[idx], dim = 2)
      y_idx <- y$index_select(
        2,
        torch::torch_tensor(idx, dtype = torch::torch_long())
      )
      out <- torch::with_no_grad(
        model(x$to(device = device), y_idx$to(device = device), task_type)
      )$to(device = "cpu")
      for (j in seq_along(idx)) {
        outs[[idx[j]]] <- out$select(2, j)
      }
    }
  }
  torch::torch_stack(outs, dim = 2)
}

# ------------------------------------------------------------------------------

# From model outputs to predictions, ported from Python
# `TabPFNClassifier.forward` / `logits_to_probabilities` and
# `TabPFNRegressor._iter_forward_executor` / `_logits_to_output`.

# Class probabilities from the model's logits (M, B, max_classes): each
# member's logits are put back in the original class order, then softmax and
# averaging across members (in that order unless `average_before_softmax`),
# then optional balancing by the training class frequencies.
tabpfn_postprocess_classification <- function(
  out,
  members,
  n_classes,
  temperature = 1,
  average_before_softmax = FALSE,
  class_counts = NULL
) {
  long <- torch::torch_long()
  logits <- torch::torch_stack(
    lapply(seq_along(members), function(b) {
      perm <- members[[b]]$class_permutation %||% (seq_len(n_classes) - 1L)
      out[, b, ]$index_select(2, torch::torch_tensor(perm + 1L, dtype = long))
    })
  )$to(dtype = torch::torch_float32())
  if (temperature != 1) {
    logits <- logits / temperature
  }
  if (average_before_softmax) {
    probs <- torch::nnf_softmax(logits$mean(dim = 1), dim = -1)
  } else {
    probs <- torch::nnf_softmax(logits, dim = -1)$mean(dim = 1)
  }
  if (!is.null(class_counts)) {
    prior <- torch::torch_tensor(
      class_counts / sum(class_counts),
      dtype = probs$dtype
    )
    probs <- probs / prior
    probs <- probs / probs$sum(dim = -1, keepdim = TRUE)
  }
  probs <- as.matrix(torch::as_array(probs$to(dtype = torch::torch_float64())))
  probs / rowSums(probs)
}

# Averaged log-probabilities (M, buckets) on the model's z-space grid from the
# model's bar logits (M, B, buckets). Members with a target transform have
# their distribution moved from the transformed grid onto the common one.
tabpfn_postprocess_regression <- function(
  out,
  members,
  borders,
  temperature = 1,
  average_before_softmax = FALSE
) {
  std <- as.numeric(torch::as_array(borders))
  acc <- NULL
  for (b in seq_along(members)) {
    logits <- out[, b, ]$to(dtype = torch::torch_float32())
    if (temperature != 1) {
      logits <- logits / temperature
    }
    transform <- members[[b]]$target_transform
    if (is.null(transform)) {
      frm <- std
    } else {
      fixed <- tabpfn_inverse_target_borders(borders, transform)
      frm <- fixed$borders
      if (!is.null(fixed$cancel)) {
        logits[, which(fixed$cancel)] <- -Inf
      }
    }
    probs <- tabpfn_translate_probs(logits, frm, std)
    if (average_before_softmax) {
      probs <- probs$log()
    }
    if (is.null(acc)) {
      acc <- probs
    } else {
      acc <- acc + probs
    }
  }
  n <- length(members)
  if (average_before_softmax) {
    torch::nnf_softmax(acc / n, dim = -1)$log()
  } else {
    (acc / n)$log()
  }
}

# Mean, median, and quantiles on the original target scale.
tabpfn_decode_regression <- function(
  log_probs,
  borders,
  y_scale,
  quantiles = NULL
) {
  to_raw <- function(z) {
    as.numeric(torch::as_array(z$to(dtype = torch::torch_float64()))) *
      y_scale$std +
      y_scale$mean
  }
  out <- list(
    mean = to_raw(tabpfn_bar_mean(log_probs, borders)),
    median = to_raw(tabpfn_bar_icdf(log_probs, borders, 0.5))
  )
  if (length(quantiles) > 0) {
    out$quantiles <- vapply(
      quantiles,
      function(q) to_raw(tabpfn_bar_icdf(log_probs, borders, q)),
      numeric(log_probs$size(1))
    )
    if (!is.matrix(out$quantiles)) {
      out$quantiles <- matrix(out$quantiles, ncol = length(quantiles))
    }
  }
  out
}

# ------------------------------------------------------------------------------

# The regression output: a "bar distribution" over buckets with half-normal
# tails in the two outer buckets (Python `FullSupportBarDistribution`), and
# the translation of a member's distribution from its own bucket grid to the
# common one (Python `translate_probs_across_borders`).
#
# The arithmetic follows the Python code's dtypes and operation order (float32
# decoding, float64 remap weights) so that results agree to float32
# precision.

# Scale of a half-normal with half of its mass below `width`, computed as
# torch's `HalfNormal(1).icdf(0.5)` does in the dtype of `width`.
tabpfn_halfnormal_scale <- function(width) {
  half <- torch::torch_tensor(0.5, dtype = width$dtype)
  icdf <- torch::torch_erfinv(2 * ((half + 1) / 2) - 1) * sqrt(2)
  width / icdf
}

tabpfn_halfnormal_icdf <- function(scale, p) {
  scale * torch::torch_erfinv(2 * ((p + 1) / 2) - 1) * sqrt(2)
}

# ---- Translation between grids ----------------------------------------------

# Weights that move bucket masses from grid `frm` to grid `to` (numeric
# vectors): interior source buckets are uniform bars, the outer ones
# half-normal tails. Computed in double precision, as Python
# `_remap_weights` does in float64. Indices are 1-based.
tabpfn_remap_weights <- function(frm, to) {
  n_frm <- length(frm) - 1
  n_to <- length(to) - 1
  edges <- sort(unique(c(frm[2:n_frm], to)))
  ne <- length(edges)
  mids <- (edges[-ne] + edges[-1]) / 2
  # Number of borders strictly below each midpoint = torch.searchsorted().
  source <- findInterval(mids, frm, left.open = TRUE) # 1-based bucket
  destination <- pmin(pmax(findInterval(mids, to, left.open = TRUE), 1), n_to)
  interior <- source >= 2 & source <= n_frm - 1
  piece <- (edges[-1] - edges[-ne])[interior]
  source <- source[interior]
  destination <- destination[interior]
  weight <- piece / (frm[source + 1] - frm[source])

  # Zero-width interior buckets are point masses, assigned to the
  # destination bucket containing their point.
  point <- which(frm[2:(n_frm - 1)] == frm[3:n_frm]) + 1
  if (length(point) > 0) {
    pd <- pmin(pmax(findInterval(frm[point], to), 1), n_to)
    source <- c(source, point)
    destination <- c(destination, pd)
    weight <- c(weight, rep(1, length(point)))
  }
  ord <- order(destination)
  source <- source[ord]
  destination <- destination[ord]
  weight <- weight[ord]

  lower <- to[-(n_to + 1)]
  upper <- to[-1]
  lower[1] <- -Inf
  upper[n_to] <- Inf
  icdf_half <- tabpfn_halfnormal_icdf_half()
  survival <- function(d, width) {
    sigma <- max(width, .Machine$double.xmin) / icdf_half
    2 * stats::pnorm(-(d / (sigma * sqrt(2))) * sqrt(2))
  }
  lower_survival <- function(y) {
    survival(frm[2] - pmin(y, frm[2]), frm[2] - frm[1])
  }
  upper_survival <- function(y) {
    survival(pmax(y, frm[n_frm]) - frm[n_frm], frm[n_frm + 1] - frm[n_frm])
  }
  f32 <- torch::torch_float32()
  list(
    source = source,
    destination = destination,
    weight = torch::torch_tensor(weight, dtype = f32),
    lower_tail = torch::torch_tensor(
      pmax(lower_survival(upper) - lower_survival(lower), 0),
      dtype = f32
    ),
    upper_tail = torch::torch_tensor(
      pmax(upper_survival(lower) - upper_survival(upper), 0),
      dtype = f32
    )
  )
}

# `HalfNormal(1).icdf(0.5)` in float64, computed as torch does.
tabpfn_halfnormal_icdf_half <- function() {
  half <- torch::torch_tensor(0.5, dtype = torch::torch_float64())
  (torch::torch_erfinv(2 * ((half + 1) / 2) - 1) * sqrt(2))$item()
}

# Probabilities on grid `to` from logits (rows, buckets) on grid `frm`.
# `frm` and `to` are numeric vectors of float32 values.
tabpfn_translate_probs <- function(logits, frm, to) {
  probs <- torch::nnf_softmax(logits, dim = -1)
  if (identical(frm, to)) {
    return(probs)
  }
  w <- tabpfn_remap_weights(frm, to)
  nb <- probs$size(2)
  long <- torch::torch_long()
  out <- probs[, 1:1] * w$lower_tail + probs[, nb:nb] * w$upper_tail
  contrib <- probs$index_select(
    2,
    torch::torch_tensor(w$source, dtype = long)
  )$mul(
    w$weight
  )
  out$index_add_(2, torch::torch_tensor(w$destination, dtype = long), contrib)
  out
}

# ---- Decoding ----------------------------------------------------------------

tabpfn_bar_mean <- function(logits, borders) {
  nb <- borders$size(1) - 1
  widths <- borders[2:(nb + 1)] - borders[1:nb]
  means <- borders[1:nb] + widths / 2
  s0 <- tabpfn_halfnormal_scale(widths[1])
  s1 <- tabpfn_halfnormal_scale(widths[nb])
  means[1] <- -(s0 * sqrt(2 / pi)) + borders[2]
  means[nb] <- s1 * sqrt(2 / pi) + borders[nb]
  torch::torch_matmul(torch::nnf_softmax(logits, dim = -1), means)
}

tabpfn_bar_icdf <- function(logits, borders, q) {
  nb <- borders$size(1) - 1
  rows <- logits$size(1)
  if (q == 0) {
    return(torch::torch_full(rows, -Inf))
  }
  if (q == 1) {
    return(torch::torch_full(rows, Inf))
  }
  widths <- borders[2:(nb + 1)] - borders[1:nb]
  probs <- torch::nnf_softmax(logits, dim = -1)
  cumprobs <- torch::torch_cumsum(probs, dim = -1)
  target <- torch::torch_full(c(rows, 1), q, dtype = logits$dtype)
  idx <- torch::torch_searchsorted(cumprobs, target)$squeeze(-1)$clamp(
    0L,
    as.integer(nb - 1)
  )$to(dtype = torch::torch_long())
  before <- torch::torch_cat(
    list(torch::torch_zeros_like(cumprobs[, 1:1]), cumprobs[, 1:(nb - 1)]),
    dim = -1
  )
  gi <- (idx + 1L)$unsqueeze(-1)$to(dtype = torch::torch_long())
  selected <- probs$gather(-1, gi)$squeeze(-1)
  cond <- ((q - before$gather(-1, gi)$squeeze(-1)) / selected)$clamp(0, 1)
  values <- borders[idx + 1L] + widths[idx + 1L] * cond
  left <- borders[2] -
    tabpfn_halfnormal_icdf(tabpfn_halfnormal_scale(widths[1]), 1 - cond)
  right <- borders[nb] +
    tabpfn_halfnormal_icdf(tabpfn_halfnormal_scale(widths[nb]), cond)
  values <- torch::torch_where(idx == 0L, left, values)
  torch::torch_where(idx == nb - 1L, right, values)
}
