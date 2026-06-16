# Feature preprocessing for TabICL, ported for fidelity from
# `src/tabicl/_sklearn/preprocessing.py`. Operates on numeric matrices (the
# sklearn pipeline works on numpy arrays), before the data becomes a torch
# tensor at the model input.
#
# Pipeline (per ensemble member): standard scale + clip to [-100, 100] ->
# optional normalization -> two-stage z-score outlier clipping. The default
# normalization methods are "none" and "power" (Yeo-Johnson); "quantile",
# "quantile_rtdl", and "robust" are not yet ported (they are not used by the
# default ensemble) and error if requested.
#
# Parameters are learned on the training rows and applied to all rows.

# Population standard deviation per column (numpy std default, ddof = 0).
tabicl_col_pop_sd <- function(x) {
  m <- colMeans(x, na.rm = TRUE)
  sqrt(colMeans(sweep(x, 2, m)^2, na.rm = TRUE))
}

# Sample standard deviation per column (ddof = 1), NA-aware (numpy nanstd ddof=1).
tabicl_col_sample_sd <- function(x) {
  apply(x, 2, stats::sd, na.rm = TRUE)
}

# ------------------------------------------------------------------------------
# Standard scaler with clipping (CustomStandardScaler)

tabicl_standard_scaler_fit <- function(x, epsilon = 1e-6) {
  list(
    mean = colMeans(x),
    scale = tabicl_col_pop_sd(x) + epsilon
  )
}

tabicl_standard_scaler_transform <- function(
  params,
  x,
  clip_min = -100,
  clip_max = 100
) {
  scaled <- sweep(sweep(x, 2, params$mean), 2, params$scale, "/")
  pmin(pmax(scaled, clip_min), clip_max)
}

# ------------------------------------------------------------------------------
# Two-stage z-score outlier clipping (OutlierRemover)

tabicl_outlier_remover_fit <- function(x, threshold = 4.0) {
  n <- nrow(x)
  means <- colMeans(x, na.rm = TRUE)
  stds <- if (n > 1) tabicl_col_sample_sd(x) else tabicl_col_pop_sd(x)
  stds <- pmax(stds, 1e-6)

  lower <- means - threshold * stds
  upper <- means + threshold * stds

  # Mask values outside the initial bounds, then recompute statistics.
  x_clean <- x
  outlier <- sweep(x, 2, lower, "<") | sweep(x, 2, upper, ">")
  x_clean[outlier] <- NA

  means <- colMeans(x_clean, na.rm = TRUE)
  stds <- if (n > 1) {
    tabicl_col_sample_sd(x_clean)
  } else {
    tabicl_col_pop_sd(x_clean)
  }
  stds <- pmax(stds, 1e-6)

  list(
    lower = means - threshold * stds,
    upper = means + threshold * stds
  )
}

tabicl_outlier_remover_transform <- function(params, x) {
  # Soft, log-based clipping toward the learned bounds.
  log_abs <- log1p(abs(x))
  x <- pmax(sweep(-log_abs, 2, params$lower, "+"), x)
  pmin(sweep(log_abs, 2, params$upper, "+"), x)
}

# ------------------------------------------------------------------------------
# Yeo-Johnson power transform (PowerTransformer, standardize = TRUE)

# Elementwise Yeo-Johnson transform of a numeric vector with a scalar lambda.
tabicl_yeo_johnson <- function(x, lambda, eps = 1e-8) {
  out <- numeric(length(x))
  pos <- x >= 0
  neg <- !pos
  if (abs(lambda) < eps) {
    out[pos] <- log1p(x[pos])
  } else {
    out[pos] <- ((x[pos] + 1)^lambda - 1) / lambda
  }
  if (abs(lambda - 2) < eps) {
    out[neg] <- -log1p(-x[neg])
  } else {
    out[neg] <- -((-x[neg] + 1)^(2 - lambda) - 1) / (2 - lambda)
  }
  out
}

# Optimal lambda per column via Brent minimization of the negative
# log-likelihood (matching sklearn's `_yeo_johnson_optimize`).
tabicl_yeo_johnson_optimize <- function(x) {
  neg_log_likelihood <- function(lambda) {
    xt <- tabicl_yeo_johnson(x, lambda)
    n <- length(x)
    variance <- mean((xt - mean(xt))^2)
    loglike <- -n /
      2 *
      log(variance) +
      (lambda - 1) * sum(sign(x) * log1p(abs(x)))
    -loglike
  }
  stats::optimize(neg_log_likelihood, interval = c(-5, 5), tol = 1e-8)$minimum
}

tabicl_power_transformer_fit <- function(x) {
  lambdas <- apply(x, 2, tabicl_yeo_johnson_optimize)
  transformed <- vapply(
    seq_len(ncol(x)),
    function(j) tabicl_yeo_johnson(x[, j], lambdas[j]),
    numeric(nrow(x))
  )
  list(
    lambdas = lambdas,
    mean = colMeans(transformed),
    scale = tabicl_col_pop_sd(transformed)
  )
}

tabicl_power_transformer_transform <- function(params, x) {
  transformed <- vapply(
    seq_len(ncol(x)),
    function(j) tabicl_yeo_johnson(x[, j], params$lambdas[j]),
    numeric(nrow(x))
  )
  sweep(sweep(transformed, 2, params$mean), 2, params$scale, "/")
}

# ------------------------------------------------------------------------------
# Full pipeline (PreprocessingPipeline)

tabicl_preprocess_fit <- function(
  x,
  normalization_method = "power",
  outlier_threshold = 4.0
) {
  scaler <- tabicl_standard_scaler_fit(x)
  x_scaled <- tabicl_standard_scaler_transform(scaler, x)

  normalizer <- NULL
  if (!identical(normalization_method, "none")) {
    if (identical(normalization_method, "power")) {
      normalizer <- tabicl_power_transformer_fit(x_scaled)
    } else {
      cli::cli_abort(
        "Normalization method {.val {normalization_method}} is not yet implemented in the brulee port."
      )
    }
    x_norm <- tabicl_power_transformer_transform(normalizer, x_scaled)
  } else {
    x_norm <- x_scaled
  }

  remover <- tabicl_outlier_remover_fit(x_norm, threshold = outlier_threshold)

  structure(
    list(
      normalization_method = normalization_method,
      scaler = scaler,
      normalizer = normalizer,
      remover = remover
    ),
    class = "tabicl_preprocessor"
  )
}

tabicl_preprocess_transform <- function(pp, x) {
  x <- tabicl_standard_scaler_transform(pp$scaler, x)
  if (!is.null(pp$normalizer)) {
    x <- tabicl_power_transformer_transform(pp$normalizer, x)
  }
  tabicl_outlier_remover_transform(pp$remover, x)
}
