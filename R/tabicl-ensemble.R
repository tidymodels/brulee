# Ensemble prediction engine for TabICL. Reproduces the sklearn wrappers'
# predict path: per ensemble member, preprocess (fit on train, transform
# train+test), optionally permute features and shuffle class labels, run the
# in-context forward pass, then combine across members.
#
# For classification, member logits are averaged (after undoing the class
# shuffle) and a temperature softmax is applied. For regression, the model's
# quantiles become a per-member mean which is inverse-scaled and averaged.
#
# NOTE on reproducibility: the sklearn `EnsembleGenerator` chooses member
# configurations (feature permutations, member ordering) with Python's
# `random.Random`, whose stream cannot be reproduced in R. The single-member
# configuration (`num_estimators = 1`: one "none" member, identity shuffles) is
# fully deterministic and matches Python exactly; multi-member ensembles here
# are a faithful reimplementation of the same algorithm, not bit-identical.

# Features with more than one unique value in the training set (UniqueFeatureFilter).
tabicl_unique_filter <- function(x_train, threshold = 1L) {
  if (nrow(x_train) <= threshold) {
    return(rep(TRUE, ncol(x_train)))
  }
  apply(x_train, 2, function(col) length(unique(col)) > threshold)
}

# Row-wise temperature softmax of a logit matrix (n, n_classes).
tabicl_softmax_rows <- function(logits, temperature = 0.9) {
  z <- logits / temperature
  z <- z - apply(z, 1, max) # column-major recycling subtracts the row max
  e <- exp(z)
  e / rowSums(e)
}

# An ensemble member: normalization method, 1-based feature permutation over the
# kept features, and a 0-based class-label pattern (identity for regression).
tabicl_member <- function(norm = "none", feat = NULL, class_shuffle = NULL) {
  list(norm = norm, feat = feat, class_shuffle = class_shuffle)
}

# Build the model input tensor for one member: preprocess (cached per norm
# method), concatenate train + test rows, permute feature columns.
tabicl_member_input <- function(member, pp_cache, xtr, xte, device) {
  nm <- member$norm
  if (is.null(pp_cache$store[[nm]])) {
    pp <- tabicl_preprocess_fit(xtr, normalization_method = nm)
    pp_cache$store[[nm]] <- list(
      tr = tabicl_preprocess_transform(pp, xtr),
      te = tabicl_preprocess_transform(pp, xte)
    )
  }
  cached <- pp_cache$store[[nm]]
  x_cat <- rbind(cached$tr, cached$te)[, member$feat, drop = FALSE]
  torch::torch_tensor(x_cat, dtype = torch::torch_float())$unsqueeze(1)$to(
    device = device
  )
}

# Class probabilities for test rows, averaging member logits. `y_train` are
# integer labels in 0..(C-1) (label-encoded). Returns an (n_test, C) matrix.
tabicl_classifier_proba <- function(
  loaded,
  x_train,
  y_train,
  x_test,
  members,
  temperature = 0.9,
  device = "cpu"
) {
  model <- loaded$model
  n_classes <- length(unique(y_train))
  keep <- tabicl_unique_filter(x_train)
  xtr <- x_train[, keep, drop = FALSE]
  xte <- x_test[, keep, drop = FALSE]
  n_train <- nrow(xtr)

  pp_cache <- new.env()
  pp_cache$store <- list()

  logit_sum <- matrix(0, nrow = nrow(xte), ncol = n_classes)
  for (member in members) {
    x_t <- tabicl_member_input(member, pp_cache, xtr, xte, device)
    shuffled <- member$class_shuffle[y_train + 1L] # apply class pattern
    y_t <- torch::torch_tensor(
      matrix(shuffled, nrow = 1),
      dtype = torch::torch_float()
    )$to(device = device)

    out <- torch::with_no_grad(
      model(x_t, y_t, return_logits = TRUE, softmax_temperature = temperature)
    )
    logits <- as.matrix(out$squeeze(1)$cpu())
    # Undo the class shuffle: column j corresponds to original class j.
    logit_sum <- logit_sum + logits[, member$class_shuffle + 1L, drop = FALSE]
  }

  avg <- logit_sum / length(members)
  proba <- tabicl_softmax_rows(avg, temperature)
  proba / rowSums(proba)
}

# Mean regression prediction for test rows, averaging inverse-scaled per-member
# means. `y_train` are the raw numeric targets.
tabicl_regressor_mean <- function(
  loaded,
  x_train,
  y_train,
  x_test,
  members,
  device = "cpu"
) {
  model <- loaded$model

  # Target StandardScaler (ddof = 0), as in the sklearn regressor.
  y_mean <- mean(y_train)
  y_scale <- sqrt(mean((y_train - y_mean)^2))
  if (y_scale == 0) {
    y_scale <- 1
  }
  y_scaled <- (y_train - y_mean) / y_scale

  keep <- tabicl_unique_filter(x_train)
  xtr <- x_train[, keep, drop = FALSE]
  xte <- x_test[, keep, drop = FALSE]

  pp_cache <- new.env()
  pp_cache$store <- list()

  mean_sum <- numeric(nrow(xte))
  for (member in members) {
    x_t <- tabicl_member_input(member, pp_cache, xtr, xte, device)
    y_t <- torch::torch_tensor(
      matrix(y_scaled, nrow = 1),
      dtype = torch::torch_float()
    )$to(device = device)

    quantiles <- torch::with_no_grad(model(x_t, y_t))
    dist <- tabicl_quantile_dist(quantiles)
    member_mean <- as.numeric(tabicl_qdist_mean(dist)$squeeze(1)$cpu())
    mean_sum <- mean_sum + (member_mean * y_scale + y_mean) # inverse-scale
  }

  mean_sum / length(members)
}

# Deterministic single "none" member with identity feature / class shuffles --
# the configuration that matches the sklearn wrappers at num_estimators = 1.
tabicl_single_member <- function(n_features, n_classes = NULL) {
  tabicl_member(
    norm = "none",
    feat = seq_len(n_features),
    class_shuffle = if (is.null(n_classes)) NULL else seq_len(n_classes) - 1L
  )
}
