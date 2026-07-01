# Regression head: turn the model's predicted quantiles into a distribution and
# read off statistics. Ported from the INFERENCE path of
# `src/tabicl/_model/quantile_dist.py` (the training-only CRPS loss, log_prob,
# pdf, and sampling are not needed for prediction and are omitted).
#
# The released regressor uses tail_type = "exp" and crossing_method = "sort"
# (the QuantileToDistribution defaults), so that is what is implemented:
# monotonize the quantiles by sorting, fit exponential tails by log-space
# regression on the extreme quantiles, and interpolate with a piecewise-linear
# spline between knots. icdf extrapolates with the exp tails outside the knot
# range. The "mean"/"variance" outputs are simple reductions of the monotonized
# quantiles (matching `predict_stats`).

tabicl_qd_config <- list(
  TOL = 1e-6,
  MIN_BETA = 0.01,
  MAX_BETA = 100.0,
  TAIL_QUANTILES_FOR_ESTIMATION = 20L
)

# Monotonize quantiles to remove crossing. Only "sort" (the released default) is
# implemented; "cummax" is trivial and "isotonic" (PAVA, via stats::isoreg) can
# be added if a checkpoint ever needs it.
tabicl_enforce_monotonicity <- function(quantiles, method = "sort") {
  if (method == "sort") {
    torch_sort(quantiles, dim = -1)[[1]]
  } else if (method == "cummax") {
    torch_cummax(quantiles, dim = -1)[[1]]
  } else {
    cli::cli_abort(
      "Crossing method {.val {method}} is not implemented in the brulee port."
    )
  }
}

# Estimate exponential-tail scale parameters beta_l / beta_r by regressing the
# extreme quantiles on log(alpha) (left) and log(1 - alpha) (right). Mirrors
# `estimate_exp_tail_params`.
tabicl_estimate_exp_tail_params <- function(
  quantiles,
  alpha_levels,
  num_tail = 20L
) {
  tol <- tabicl_qd_config$TOL
  n <- quantiles$size(-1)
  k <- min(num_tail, n %/% 4L)

  alpha_left <- alpha_levels[1:k]
  q_left <- quantiles$narrow(dim = -1, start = 1, length = k)
  ln_alpha_left <- torch_log(alpha_left$clamp(min = tol))
  ln_alpha_centered <- ln_alpha_left - ln_alpha_left$mean()
  q_left_centered <- q_left - q_left$mean(dim = -1, keepdim = TRUE)
  cov_left <- (q_left_centered * ln_alpha_centered)$mean(dim = -1)
  var_ln_alpha_left <- (ln_alpha_centered^2)$mean()
  beta_l <- cov_left / var_ln_alpha_left$clamp(min = tol)
  beta_l <- torch_clamp(
    beta_l$abs(),
    min = tabicl_qd_config$MIN_BETA,
    max = tabicl_qd_config$MAX_BETA
  )

  alpha_right <- alpha_levels[(n - k + 1):n]
  q_right <- quantiles$narrow(dim = -1, start = n - k + 1, length = k)
  ln_one_minus_alpha <- torch_log((1 - alpha_right)$clamp(min = tol))
  ln_1ma_centered <- ln_one_minus_alpha - ln_one_minus_alpha$mean()
  q_right_centered <- q_right - q_right$mean(dim = -1, keepdim = TRUE)
  cov_right <- (q_right_centered * ln_1ma_centered)$mean(dim = -1)
  var_ln_1ma <- (ln_1ma_centered^2)$mean()
  beta_r <- -cov_right / var_ln_1ma$clamp(min = tol)
  beta_r <- torch_clamp(
    beta_r$abs(),
    min = tabicl_qd_config$MIN_BETA,
    max = tabicl_qd_config$MAX_BETA
  )

  list(beta_l = beta_l, beta_r = beta_r)
}

# Build a quantile-distribution object (a plain list of precomputed tensors)
# from predicted quantiles. `alpha_levels` defaults to the interior of an
# evenly spaced grid, matching QuantileToDistribution.
tabicl_quantile_dist <- function(
  quantiles,
  alpha_levels = NULL,
  tail_type = "exp",
  fix_crossing = TRUE,
  crossing_method = "sort"
) {
  if (!identical(tail_type, "exp")) {
    cli::cli_abort(
      "Only exp tails are implemented in the brulee port (got {.val {tail_type}})."
    )
  }
  tol <- tabicl_qd_config$TOL
  num_quantiles <- quantiles$size(-1)

  if (is.null(alpha_levels)) {
    alpha_levels <- torch_linspace(
      0,
      1,
      steps = num_quantiles + 2L,
      dtype = quantiles$dtype,
      device = quantiles$device
    )[2:(num_quantiles + 1L)]
  }

  if (fix_crossing) {
    quantiles <- tabicl_enforce_monotonicity(
      quantiles,
      method = crossing_method
    )
  }

  # Spline knots / segments.
  q_lo <- quantiles$narrow(dim = -1, start = 1, length = num_quantiles - 1L)
  q_hi <- quantiles$narrow(dim = -1, start = 2, length = num_quantiles - 1L)
  alpha_lo_1d <- alpha_levels[1:(num_quantiles - 1L)]
  alpha_hi_1d <- alpha_levels[2:num_quantiles]
  # Batch-expanded segment boundaries (..., num_segments), matching q_lo/q_hi,
  # so they can be gathered with the per-element segment index.
  alpha_lo <- alpha_lo_1d$expand_as(q_lo)
  alpha_hi <- alpha_hi_1d$expand_as(q_hi)

  q_l <- quantiles$select(dim = -1, index = 1)
  q_r <- quantiles$select(dim = -1, index = num_quantiles)
  alpha_l <- as.numeric(alpha_levels[1]$item())
  alpha_r <- as.numeric(alpha_levels[num_quantiles]$item())

  tails <- tabicl_estimate_exp_tail_params(
    quantiles,
    alpha_levels,
    num_tail = tabicl_qd_config$TAIL_QUANTILES_FOR_ESTIMATION
  )

  alpha_l_safe <- max(alpha_l, tol)
  tail_a_l <- tails$beta_l
  tail_b_l <- q_l - tail_a_l * log(alpha_l_safe)

  alpha_r_safe <- min(alpha_r, 1 - tol)
  tail_a_r <- -tails$beta_r
  tail_b_r <- q_r - tail_a_r * log(1 - alpha_r_safe)

  structure(
    list(
      quantiles = quantiles,
      alpha_levels = alpha_levels,
      num_quantiles = num_quantiles,
      num_segments = num_quantiles - 1L,
      q_lo = q_lo,
      q_hi = q_hi,
      alpha_lo = alpha_lo,
      alpha_hi = alpha_hi,
      alpha_lo_1d = alpha_lo_1d,
      q_l = q_l,
      q_r = q_r,
      alpha_l = alpha_l,
      alpha_r = alpha_r,
      tail_a_l = tail_a_l,
      tail_b_l = tail_b_l,
      tail_a_r = tail_a_r,
      tail_b_r = tail_b_r
    ),
    class = "tabicl_quantile_dist"
  )
}

# Expand a per-batch parameter (..., ) to broadcast against alpha (..., n).
tabicl_qd_expand <- function(param, alpha) {
  n_expand <- length(dim(alpha)) - length(dim(param))
  for (i in seq_len(n_expand)) {
    param <- param$unsqueeze(-1)
  }
  param$expand_as(alpha)
}

# Quantile function Q(alpha) = F^{-1}(alpha): exp tails outside [alpha_l,
# alpha_r], piecewise-linear spline inside. Mirrors `icdf`.
tabicl_qdist_icdf <- function(dist, alpha) {
  tol <- tabicl_qd_config$TOL

  squeeze_output <- FALSE
  if (length(dim(alpha)) == 0L) {
    alpha <- alpha$unsqueeze(1)
    squeeze_output <- TRUE
  }

  batch_shape <- utils::head(dim(dist$quantiles), -1)
  if (length(dim(alpha)) == 1L) {
    alpha_expanded <- alpha
    for (i in seq_along(batch_shape)) {
      alpha_expanded <- alpha_expanded$unsqueeze(1)
    }
    alpha_expanded <- alpha_expanded$expand(c(batch_shape, dim(alpha)))
  } else {
    alpha_expanded <- alpha
  }

  # Left exp tail.
  a_l <- tabicl_qd_expand(dist$tail_a_l, alpha_expanded)
  b_l <- tabicl_qd_expand(dist$tail_b_l, alpha_expanded)
  q_left <- a_l * torch_log(alpha_expanded$clamp(min = tol)) + b_l

  # Right exp tail.
  a_r <- tabicl_qd_expand(dist$tail_a_r, alpha_expanded)
  b_r <- tabicl_qd_expand(dist$tail_b_r, alpha_expanded)
  q_right <- a_r * torch_log((1 - alpha_expanded)$clamp(min = tol)) + b_r

  # Spline interior. searchsorted returns the same 0-based count as Python;
  # gather is 1-based, so the R segment index is clamp(searchsorted, 1, S).
  seg_idx <- torch_searchsorted(
    dist$alpha_lo_1d$contiguous(),
    alpha_expanded$contiguous(),
    right = TRUE
  )
  seg_idx <- seg_idx$clamp(min = 1, max = dist$num_segments)$to(
    dtype = torch_long()
  )

  q_lo_g <- dist$q_lo$gather(dim = -1, index = seg_idx)
  q_hi_g <- dist$q_hi$gather(dim = -1, index = seg_idx)
  alpha_lo_g <- dist$alpha_lo$gather(dim = -1, index = seg_idx)
  alpha_hi_g <- dist$alpha_hi$gather(dim = -1, index = seg_idx)

  t <- (alpha_expanded - alpha_lo_g) /
    (alpha_hi_g - alpha_lo_g)$clamp(min = tol)
  q_spline <- q_lo_g + t$clamp(0, 1) * (q_hi_g - q_lo_g)
  q_r_exp <- tabicl_qd_expand(dist$q_r, alpha_expanded)
  q_spline <- torch_where(alpha_expanded >= dist$alpha_r, q_r_exp, q_spline)

  # Region selection.
  result <- torch_where(
    alpha_expanded < dist$alpha_l,
    q_left,
    torch_where(alpha_expanded > dist$alpha_r, q_right, q_spline)
  )

  if (squeeze_output) {
    result <- result$squeeze(-1)
  }
  result
}

tabicl_qdist_mean <- function(dist) {
  dist$quantiles$mean(dim = -1)
}

tabicl_qdist_variance <- function(dist) {
  torch_var(dist$quantiles, dim = -1)
}

tabicl_qdist_median <- function(dist) {
  tabicl_qdist_icdf(
    dist,
    torch_scalar_tensor(
      0.5,
      dtype = dist$quantiles$dtype,
      device = dist$quantiles$device
    )
  )
}

tabicl_qdist_quantiles <- function(dist, alphas) {
  tabicl_qdist_icdf(
    dist,
    torch_tensor(
      alphas,
      dtype = dist$quantiles$dtype,
      device = dist$quantiles$device
    )
  )
}
