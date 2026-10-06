# Fit-time encoding of the predictors into a numeric matrix, as Python
# `tabpfn` does before building the ensemble (`clean.py`,
# `modality_detection.py`):
#
# * factor, character, and logical columns are categorical;
# * a numeric column is constant when it has at most one distinct value
#   (NA counts as a value), and categorical when there are more than
#   `min_samples` rows (100 for the v3.5 checkpoints) and fewer than
#   `min_unique` distinct values (4);
# * categorical values are coded 0..k-1 in sorted order of the observed
#   values (C-locale order for strings, as Python sorts by code point);
#   missing values are NaN and, at predict time, unseen values are -1;
# * constant columns keep their value; at predict time they are overwritten
#   with the value from the first training row.

tabpfn_fit_encoder <- function(
  x,
  min_samples = 100L,
  min_unique = 4L,
  call = caller_env()
) {
  n <- nrow(x)
  cols <- lapply(names(x), function(nm) {
    v <- x[[nm]]
    if (is.factor(v) || is.character(v) || is.logical(v)) {
      levels <- sort(unique(as.character(v[!is.na(v)])), method = "radix")
      return(list(type = "categorical", levels = levels))
    }
    if (!is.numeric(v)) {
      cli::cli_abort(
        "Column {.field {nm}} has an unsupported type ({.cls {class(v)}}).",
        call = call
      )
    }
    n_unique <- length(unique(v))
    if (n_unique <= 1) {
      return(list(type = "constant", value = as.double(v[1])))
    }
    if (n > min_samples && n_unique < min_unique) {
      levels <- sort(unique(v[!is.na(v)]))
      return(list(type = "categorical", levels = levels))
    }
    list(type = "numeric")
  })
  names(cols) <- names(x)
  structure(list(columns = cols), class = "tabpfn_encoder")
}

# Encode a data frame with a fitted encoder: an n x p double matrix with NaN
# for missing values.
tabpfn_encode <- function(encoder, x, call = caller_env()) {
  cols <- encoder$columns
  out <- matrix(NaN, nrow = nrow(x), ncol = length(cols))
  for (j in seq_along(cols)) {
    info <- cols[[j]]
    v <- x[[names(cols)[j]]]
    out[, j] <- switch(
      info$type,
      categorical = {
        if (is.numeric(info$levels)) {
          key <- v
        } else {
          key <- as.character(v)
        }
        code <- match(key, info$levels) - 1
        code[is.na(code) & !is.na(v)] <- -1
        code[is.na(v)] <- NaN
        code
      },
      constant = rep(info$value, nrow(x)),
      numeric = {
        v <- as.double(v)
        if (any(is.infinite(v))) {
          cli::cli_abort(
            "Column {.field {names(cols)[j]}} has infinite values.",
            call = call
          )
        }
        v[is.na(v)] <- NaN
        v
      }
    )
  }
  colnames(out) <- names(cols)
  out
}

tabpfn_categorical_columns <- function(encoder) {
  unname(vapply(
    encoder$columns,
    function(x) x$type == "categorical",
    logical(1)
  ))
}

# ------------------------------------------------------------------------------

# Ensemble members. Each member sees the data through its own random choices,
# kept as plain data so that tests can inject the choices Python made:
#
#   preprocessor       index into the checkpoint's preprocessing configs
#   features           columns of the encoded matrix the member uses (1-based)
#   class_permutation  0-based permutation of the class labels, or NULL
#   category_maps      per encoded categorical column, a 0-based permutation
#                      of its codes (one extra slot when the column has
#                      missing training values, as in Python)
#   shuffle            0-based column permutation, appended columns included
#   target_transform   NULL or a fitted "safepower" transform (regression)
#
# The deterministic parts (which columns are dropped as constant, the column
# layout, which categoricals are encoded) come from `tabpfn_member_layout()`.
#
# The random choices follow the structure of Python's ensemble generation
# (`preprocessing/ensemble.py`) but use R's RNG, so they differ from
# Python's draws.

tabpfn_make_members <- function(
  x_train,
  categorical,
  n_estimators,
  task_type,
  preprocessors,
  n_classes = NULL,
  y_train = NULL,
  subsampling = "balanced"
) {
  p <- ncol(x_train)
  n_train <- nrow(x_train)
  plan <- tabpfn_member_plan(n_estimators, length(preprocessors), task_type)
  sizes <- vapply(
    plan$preprocessor,
    function(i) min(p, preprocessors[[i]]$max_features),
    numeric(1)
  )
  subsets <- tabpfn_feature_subsets(p, sizes, subsampling)
  if (task_type == "multiclass") {
    class_perms <- tabpfn_class_permutations(n_estimators, n_classes)
  } else {
    class_perms <- NULL
  }
  lapply(seq_len(n_estimators), function(i) {
    pre <- preprocessors[[plan$preprocessor[i]]]
    features <- subsets[[i]]
    layout <- tabpfn_member_layout(
      x_train[, features, drop = FALSE],
      categorical[features],
      pre
    )
    width <- length(layout$columns) +
      tabpfn_svd_width(pre, n_train, layout) +
      1L
    member <- list(
      preprocessor = plan$preprocessor[i],
      features = features,
      class_permutation = class_perms[[i]],
      category_maps = lapply(layout$category_sizes, function(k) {
        sample.int(k) - 1L
      }),
      shuffle = sample.int(width) - 1L,
      target_transform = NULL
    )
    if (identical(plan$target[i], "safepower")) {
      member$target_transform <- tabpfn_fit_safepower(y_train)
    }
    member
  })
}

# Which preprocessing config (and, for regression, target transform) each
# member uses: every combination for an equal number of consecutive members,
# the remainder taken from the front (Python `_balance`).
tabpfn_member_plan <- function(n_estimators, n_preprocessors, task_type) {
  if (task_type == "regression") {
    combos <- expand.grid(
      target = c("none", "safepower"),
      preprocessor = seq_len(n_preprocessors),
      stringsAsFactors = FALSE
    )
  } else {
    combos <- data.frame(
      preprocessor = seq_len(n_preprocessors),
      target = "none"
    )
  }
  k <- nrow(combos)
  idx <- c(
    rep(seq_len(k), each = n_estimators %/% k),
    seq_len(n_estimators %% k)
  )
  combos[idx, c("preprocessor", "target")]
}

# The member's column layout, from its training rows (Python's CPU steps
# `RemoveConstantFeaturesStep`, `ReshapeFeatureDistributionsStep`, and
# `EncodeCategoricalFeaturesStep`):
#
#   columns         member columns (into its feature subset) in output order
#   categorical     whether each output column is categorical for the model
#   gpu             the torch transform of each output column: "none",
#                   "quantile", or "squash"
#   n_encoded       the first n_encoded columns are categorical codes to
#                   permute with `category_maps`
#   category_sizes  their numbers of code slots
tabpfn_member_layout <- function(x_train, categorical, pre) {
  n_train <- nrow(x_train)
  keep <- vapply(
    seq_len(ncol(x_train)),
    function(j) {
      v <- x_train[, j]
      same <- v[1] == v
      same[is.na(same)] <- FALSE
      !all(is.nan(v)) && mean(same) < 1
    },
    logical(1)
  )
  if (!any(keep)) {
    cli::cli_abort("All predictors are constant in the training data.")
  }
  kept <- which(keep)
  cats <- kept[categorical[kept]]
  nums <- kept[!categorical[kept]]

  # Reshape: the GPU transform applies to the numeric columns, or to all
  # columns for the "numeric" categorical encoding.
  apply_to_cat <- identical(pre$categorical, "numeric")
  if (apply_to_cat) {
    transformed <- c(cats, nums)
  } else {
    transformed <- nums
  }
  if (identical(pre$append, "auto")) {
    append <- length(kept) < 500 && length(kept) <= pre$max_features / 2
  } else {
    append <- isTRUE(pre$append)
  }
  if (append) {
    columns <- c(kept, transformed)
    is_cat <- c(categorical[kept], rep(FALSE, length(transformed)))
    gpu <- c(rep("none", length(kept)), rep(pre$gpu, length(transformed)))
  } else if (apply_to_cat) {
    columns <- c(cats, nums)
    is_cat <- rep(FALSE, length(columns))
    gpu <- rep(pre$gpu, length(columns))
  } else {
    columns <- c(cats, nums)
    is_cat <- c(rep(TRUE, length(cats)), rep(FALSE, length(nums)))
    gpu <- c(rep("none", length(cats)), rep(pre$gpu, length(nums)))
  }

  # Encode: the encoded categoricals move to the front. With the "very
  # common" encoding, only categoricals whose every value (missing counts as
  # one) has at least 10 training rows, and that have fewer than n / 10
  # values, are encoded; the others become numeric.
  encoded <- integer()
  if (identical(pre$categorical, "ordinal_shuffled")) {
    encoded <- which(is_cat)
  } else if (
    identical(pre$categorical, "ordinal_very_common_categories_shuffled")
  ) {
    encoded <- which(is_cat)[vapply(
      which(is_cat),
      function(j) {
        counts <- table(x_train[, columns[j]], useNA = "ifany")
        min(counts) >= 10 && length(counts) < n_train %/% 10
      },
      logical(1)
    )]
    is_cat[setdiff(which(is_cat), encoded)] <- FALSE
  }
  ord <- c(encoded, setdiff(seq_along(columns), encoded))
  sizes <- vapply(
    columns[encoded],
    function(j) {
      v <- x_train[, j]
      length(unique(v[!is.nan(v)])) + any(is.nan(v))
    },
    integer(1)
  )
  list(
    columns = columns[ord],
    categorical = is_cat[ord],
    gpu = gpu[ord],
    n_encoded = length(encoded),
    category_sizes = unname(sizes)
  )
}

# Number of SVD columns the member appends (Python `get_svd_n_components`
# for "svd_quarter_components").
tabpfn_svd_width <- function(pre, n_train, layout) {
  n_features <- length(layout$columns)
  if (!isTRUE(pre$svd) || n_features < 2) {
    return(0L)
  }
  k <- max(1, min(n_train %/% 10 + 1, n_features %/% 4))
  as.integer(min(k, n_train, n_features))
}

# Feature subsets of the given sizes. "balanced": members draw from a pool
# shared across members, so that together they cover the columns evenly;
# "random": each member draws independently. A member whose size covers all
# columns uses all of them.
tabpfn_feature_subsets <- function(p, sizes, method = "balanced") {
  if (all(sizes >= p)) {
    return(rep(list(seq_len(p)), length(sizes)))
  }
  if (method == "random") {
    return(lapply(sizes, function(size) {
      if (size >= p) {
        seq_len(p)
      } else {
        sort(sample.int(p)[seq_len(size)])
      }
    }))
  }
  shuffled <- sample.int(p)
  pool <- integer()
  lapply(sizes, function(budget) {
    if (budget >= p) {
      return(seq_len(p))
    }
    taken <- integer()
    while (length(taken) < budget) {
      if (length(pool) == 0) {
        pool <<- setdiff(seq_len(p), taken)
        pool <<- pool[sample.int(length(pool))]
      }
      fresh <- setdiff(pool, taken)
      n_take <- min(budget - length(taken), length(fresh))
      taken <- c(taken, fresh[seq_len(n_take)])
      pool <<- setdiff(pool, fresh[seq_len(n_take)])
      if (length(fresh) == 0) {
        pool <<- integer()
      }
    }
    sort(shuffled[taken])
  })
}

# Class permutations: random permutations, deduplicated and sorted, each used
# for an equal number of consecutive members, with the remainder drawn at
# random (Python's "shuffle" class shift).
tabpfn_class_permutations <- function(n_estimators, n_classes) {
  if (n_classes == 1) {
    draws <- matrix(0L, nrow = 3 * n_estimators)
  } else {
    draws <- t(replicate(3 * n_estimators, sample.int(n_classes) - 1L))
  }
  uniq <- unique(draws)
  uniq <- uniq[do.call(order, as.data.frame(uniq)), , drop = FALSE]
  reps <- n_estimators %/% nrow(uniq)
  idx <- rep(seq_len(nrow(uniq)), each = reps)
  rest <- n_estimators %% nrow(uniq)
  if (rest > 0) {
    idx <- c(idx, sample.int(nrow(uniq), rest, replace = TRUE))
  }
  lapply(idx, function(i) uniq[i, ])
}

# Run `code` with R's RNG seeded by `seed`, restoring the caller's RNG state.
tabpfn_with_seed <- function(seed, code) {
  # `set.seed(NULL)` re-seeds at random, which would silently make fits
  # irreproducible.
  stopifnot(is.numeric(seed), length(seed) == 1, !is.na(seed))
  if (exists(".Random.seed", envir = globalenv())) {
    old <- get(".Random.seed", envir = globalenv())
  } else {
    old <- NULL
  }
  on.exit({
    if (is.null(old)) {
      rm(".Random.seed", envir = globalenv())
    } else {
      assign(".Random.seed", old, envir = globalenv())
    }
  })
  set.seed(seed)
  code
}

# ------------------------------------------------------------------------------

# Per-member preprocessing, ported from Python's CPU pipeline
# (`preprocessing/pipeline_factory.py`) and its torch steps
# (`preprocessing/torch/`), with GPU preprocessing on as in the v3 and v3.5
# checkpoints:
#
#   CPU, float64:  member columns -> drop constant columns -> reshape
#                  layout -> permute the encoded categorical codes (missing
#                  and unseen values become NaN); see `tabpfn_member_layout()`
#   torch, float32: quantile or squashing transform of the marked columns
#                  -> append SVD features (when configured) -> append the
#                  fingerprint column -> shuffle columns -> soft-clip the
#                  numeric columns, with all statistics from the train rows

# The member's CPU steps on the encoded matrix (train rows, then test rows).
tabpfn_member_cpu <- function(member, x, layout) {
  x <- x[, member$features, drop = FALSE][, layout$columns, drop = FALSE]
  for (j in seq_len(layout$n_encoded)) {
    v <- x[, j]
    ok <- !is.nan(v) & v >= 0
    v[ok] <- member$category_maps[[j]][v[ok] + 1]
    v[!ok] <- NaN
    x[, j] <- v
  }
  x
}

# The member's torch steps: (rows, C) float32 -> (rows, C') float32.
tabpfn_member_torch <- function(x, num_train, layout, pre, shuffle, n_sigma) {
  is_cat <- layout$categorical
  long <- torch::torch_long()
  apply_cols <- function(x, cols, fn) {
    if (length(cols) == 0) {
      return(x)
    }
    idx <- torch::torch_tensor(cols, dtype = long)
    x$index_copy(2, idx, fn(x$index_select(2, idx)))
  }
  x <- apply_cols(x, which(layout$gpu == "quantile"), function(v) {
    tabpfn_quantile_transform(v, num_train, pre$quantile)
  })
  x <- apply_cols(x, which(layout$gpu == "squash"), function(v) {
    tabpfn_squash(v, num_train, pre$squash_max)
  })
  k <- tabpfn_svd_width(pre, num_train, layout)
  if (k > 0) {
    x <- torch::torch_cat(
      list(x, tabpfn_svd_features(x, num_train, k)),
      dim = 2
    )
    is_cat <- c(is_cat, rep(FALSE, k))
  }
  fingerprint <- tabpfn_fingerprint(x, num_train)
  x <- torch::torch_cat(list(x, fingerprint$unsqueeze(2)), dim = 2)
  is_cat <- c(is_cat, FALSE)
  perm <- shuffle + 1L
  x <- x$index_select(2, torch::torch_tensor(perm, dtype = long))
  is_cat <- is_cat[perm]
  if (!is.null(n_sigma)) {
    x <- apply_cols(x, which(!is_cat), function(v) {
      tabpfn_soft_clip(v, num_train, n_sigma)
    })
  }
  x
}

# Model inputs: one (rows, C_i) float32 tensor per member (members can differ
# in width) and the train targets y (N, B) float32.
tabpfn_member_inputs <- function(
  members,
  preprocessors,
  x_all,
  num_train,
  y_members,
  categorical,
  n_sigma
) {
  xs <- lapply(members, function(m) {
    pre <- preprocessors[[m$preprocessor]]
    x_train <- x_all[seq_len(num_train), m$features, drop = FALSE]
    layout <- tabpfn_member_layout(x_train, categorical[m$features], pre)
    x <- tabpfn_member_cpu(m, x_all, layout)
    x <- torch::torch_tensor(x, dtype = torch::torch_float32())
    tabpfn_member_torch(x, num_train, layout, pre, m$shuffle, n_sigma)
  })
  list(
    x = xs,
    y = torch::torch_tensor(
      do.call(cbind, y_members),
      dtype = torch::torch_float32()
    )
  )
}

# ---- Quantile transform --------------------------------------------------------

# Map each column to [0, 1] through quantiles of its train rows (Python
# `TorchQuantileTransformer`), averaging the forward and backward
# interpolation as sklearn does for tied quantiles. "extrapolate" maps values
# beyond the train range linearly into [-1, 0] and [1, 2].
tabpfn_quantile_transform <- function(x, num_train, preset) {
  f32 <- torch::torch_float32()
  train <- x$narrow(1, 1, num_train)
  if (num_train <= 1) {
    refs <- torch::torch_tensor(c(0, 1), dtype = f32)
    q <- torch::torch_stack(list(train[1], train[1]))
  } else {
    if (preset == "coarse") {
      n_q <- max(num_train %/% 10, 2)
    } else {
      n_q <- max(num_train %/% 5, 2)
    }
    n_q <- min(max(1, min(n_q, num_train, 20000)), num_train)
    refs <- torch::torch_linspace(0, 1, n_q, dtype = f32)
    q <- tabpfn_nanquantile(train, refs)
    q <- torch::torch_cummax(q, dim = 1)[[1]]
  }
  x_t <- x$t()$contiguous()
  q_t <- q$t()$contiguous()
  refs_t <- refs$unsqueeze(1)$expand_as(q_t)
  forward <- tabpfn_interp(x_t, q_t, refs_t)
  backward <- tabpfn_interp(
    -x_t,
    (-q_t)$flip(2)$contiguous(),
    (-refs_t)$flip(2)$contiguous()
  )
  res <- (0.5 * (forward - backward))$t()
  if (preset == "extrapolate") {
    x_min <- q[1]
    x_max <- q[q$size(1)]
    range <- x_max - x_min
    constant <- range == 0
    norm <- (x - x_min) /
      torch::torch_where(constant, torch::torch_ones_like(range), range)
    below <- (x < x_min) & !constant
    above <- (x > x_max) & !constant
    res <- torch::torch_where(below, norm$clamp(-1, 0), res)
    res <- torch::torch_where(above, norm$clamp(1, 2), res)
  } else {
    res <- res$clamp(0, 1)
  }
  torch::torch_where(torch::torch_isnan(x), x, res)
}

# Column quantiles ignoring NaN, linear interpolation: (rows, C) ->
# (length(q), C). As in torch 2.14's `nanquantile`, the ranks q * (n - 1) are
# computed in double precision and the interpolation in the data's dtype;
# R torch's bundled libtorch computes the ranks differently, which changes
# the last bit of some quantiles.
tabpfn_nanquantile <- function(x, q) {
  long <- torch::torch_long()
  sorted <- x$sort(dim = 1)[[1]]
  n_valid <- (!torch::torch_isnan(x))$sum(dim = 1, keepdim = TRUE)
  ranks <- q$to(dtype = torch::torch_float64())$unsqueeze(2) *
    (n_valid - 1L)$clamp(min = 0)$to(dtype = torch::torch_float64())
  below <- ranks$floor()
  weight <- (ranks - below)$to(dtype = x$dtype)
  v_below <- sorted$gather(1, below$to(dtype = long) + 1L)
  v_above <- sorted$gather(1, ranks$ceil()$to(dtype = long) + 1L)
  torch::torch_lerp(v_below, v_above, weight)
}

# Batched `numpy.interp` over rows: x (F, n), breakpoints xp and values fp
# (F, q).
tabpfn_interp <- function(x, xp, fp) {
  n_q <- xp$size(2)
  long <- torch::torch_long()
  constant <- xp[, 1] == xp[, n_q]
  mid <- 0.5 * (fp[, 1] + fp[, n_q])
  idx <- torch::torch_searchsorted(xp, x)$clamp(1L, as.integer(n_q - 1))$to(
    dtype = long
  )
  x_low <- xp$gather(2, idx)
  x_high <- xp$gather(2, idx + 1L)
  f_low <- fp$gather(2, idx)
  f_high <- fp$gather(2, idx + 1L)
  dx <- x_high - x_low
  dx <- torch::torch_where(dx == 0, torch::torch_ones_like(dx), dx)
  res <- f_low + (f_high - f_low) / dx * (x - x_low)
  res <- torch::torch_where(x <= xp$narrow(2, 1, 1), fp$narrow(2, 1, 1), res)
  res <- torch::torch_where(
    x >= xp$narrow(2, n_q, 1),
    fp$narrow(2, n_q, 1),
    res
  )
  torch::torch_where(constant$unsqueeze(2), mid$unsqueeze(2), res)
}

# ---- Squashing scaler ----------------------------------------------------------

# Robust scaling by the median and interquartile range (or the range when the
# quartiles coincide), then the soft squash z / sqrt(1 + (z / b)^2) into
# (-b, b) (Python `TorchSquashingScaler`).
tabpfn_squash <- function(x, num_train, b) {
  train <- x$narrow(1, 1, num_train)
  if (num_train <= 1) {
    center <- torch::torch_zeros(x$size(2))
    scale <- torch::torch_ones(x$size(2))
    zero <- torch::torch_ones(x$size(2), dtype = torch::torch_bool())
  } else {
    masked <- torch::torch_where(torch::torch_isinf(train), NaN, train)
    is_nan <- torch::torch_isnan(masked)
    col_min <- torch::torch_where(is_nan, Inf, masked)$amin(dim = 1)
    col_max <- torch::torch_where(is_nan, -Inf, masked)$amax(dim = 1)
    all_nan <- is_nan$all(dim = 1)
    col_min <- torch::torch_where(all_nan, NaN, col_min)
    col_max <- torch::torch_where(all_nan, NaN, col_max)
    qs <- torch::torch_tensor(c(0.25, 0.5, 0.75), dtype = x$dtype)
    q <- tabpfn_nanquantile(masked, qs)
    zero <- col_max == col_min
    minmax <- (q[1] == q[3]) & !zero
    robust <- !(zero | minmax)
    tiny <- torch::torch_finfo(x$dtype)$tiny
    center <- torch::torch_zeros_like(col_min)
    scale <- torch::torch_ones_like(col_min)
    center <- torch::torch_where(robust, q[2], center)
    scale <- torch::torch_where(robust, q[3] - q[1], scale)
    center <- torch::torch_where(minmax, q[2], center)
    scale <- torch::torch_where(minmax, (col_max - col_min + tiny) / 2, scale)
  }
  out <- torch::torch_where(torch::torch_isinf(x), NaN, x)
  out <- (out - center) / scale
  finite_zero <- (!torch::torch_isnan(x)) & zero
  out <- out$masked_fill(finite_zero, 0)
  out <- out / ((out / b)$pow(2) + 1)$sqrt()
  out <- out$masked_fill(torch::torch_isposinf(x), b)
  out$masked_fill(torch::torch_isneginf(x), -b)
}

# ---- SVD features --------------------------------------------------------------

# k SVD features of all columns (Python `TorchAddSVDFeaturesStep`): columns
# are scaled by their train standard deviation (no centering; missing values
# take the train mean), and the scaled rows are projected onto the top k
# right singular vectors of the scaled train rows. Python uses a randomized
# SVD when n * F > 1e6; this always uses the exact SVD.
tabpfn_svd_features <- function(x, num_train, k) {
  train <- x$narrow(1, 1, num_train)
  masked <- torch::torch_where(torch::torch_isinf(train), NaN, train)
  is_nan <- torch::torch_isnan(masked)
  zeros <- torch::torch_zeros_like(masked)
  n_valid <- (!is_nan)$to(dtype = x$dtype)$sum(dim = 1)
  mean <- torch::torch_where(is_nan, zeros, masked)$sum(dim = 1) /
    n_valid$clamp(min = 1)
  sq <- torch::torch_where(is_nan, zeros, (masked - mean)$pow(2))$sum(dim = 1)
  std <- torch::torch_sqrt(sq / max(num_train, 1))
  std <- torch::torch_where(
    std == 0 | torch::torch_isnan(std),
    torch::torch_ones_like(std),
    std
  )
  if (num_train == 1) {
    std <- torch::torch_ones_like(std)
  }
  eps <- torch::torch_finfo(x$dtype)$eps
  scaled <- torch::torch_where(torch::torch_isinf(x), NaN, x)
  scaled <- torch::torch_where(
    torch::torch_isnan(scaled),
    mean$expand_as(scaled),
    scaled
  )
  scaled <- (scaled / (std + eps))$clip(-100, 100)
  scaled <- torch::torch_where(
    torch::torch_isfinite(scaled),
    scaled,
    torch::torch_zeros_like(scaled)
  )

  svd <- torch::linalg_svd(
    scaled$narrow(1, 1, num_train)$cpu(),
    full_matrices = FALSE
  )
  vh <- svd[[3]]$narrow(1, 1, k)
  # Make the largest-magnitude entry of each component positive (leftmost on
  # ties), as Python `_svd_flip_stable`.
  vh_val <- as.matrix(torch::as_array(vh))
  signs <- vapply(
    seq_len(k),
    function(i) {
      s <- sign(vh_val[i, which.max(abs(vh_val[i, ]))])
      if (s == 0) {
        1
      } else {
        s
      }
    },
    numeric(1)
  )
  vh <- (vh * torch::torch_tensor(signs, dtype = vh$dtype)$unsqueeze(2))$to(
    device = x$device
  )
  torch::torch_matmul(scaled, vh$t())
}

# ---- Fingerprint ---------------------------------------------------------------

# A per-row hash feature in [0, 1] that lets the model tell identical rows
# apart (Python `AddFingerprintFeaturesStep`, torch path): SHA-256 of the
# row's float32 values rounded to 12 decimals, salted with the number of
# training cells. Duplicate training rows get successive salts; test rows
# always get their first hash.
tabpfn_fingerprint <- function(x, num_train) {
  n_rows <- x$size(1)
  salt <- num_train * x$size(2)
  rounded <- as.matrix(torch::as_array((x * 1e12)$round() / 1e12))
  raw_x <- as.matrix(torch::as_array(x))
  out <- numeric(n_rows)
  seen <- new.env(hash = TRUE, parent = emptyenv())
  counter <- new.env(hash = TRUE, parent = emptyenv())
  for (i in seq_len(n_rows)) {
    row <- writeBin(rounded[i, ], raw(), size = 4, endian = "little")
    h_base <- tabpfn_row_hash(row, salt)
    if (i > num_train) {
      out[i] <- h_base
      next
    }
    key <- sprintf("%.17g", h_base)
    offset <- counter[[key]] %||% 0
    if (offset == 0) {
      h <- h_base
    } else {
      h <- tabpfn_row_hash(row, salt + offset)
    }
    all_nan <- all(is.nan(raw_x[i, ]))
    retries <- 0
    while (!is.null(seen[[sprintf("%.17g", h)]]) && !all_nan) {
      offset <- offset + 1
      retries <- retries + 1
      if (retries > 100) {
        cli::cli_abort("Could not resolve a fingerprint hash collision.")
      }
      h <- tabpfn_row_hash(row, salt + offset)
    }
    out[i] <- h
    seen[[sprintf("%.17g", h)]] <- TRUE
    counter[[key]] <- offset + 1
  }
  torch::torch_tensor(out, dtype = torch::torch_float32())
}

# The last 8 bytes of SHA-256(row || salt as uint64 little-endian), read as
# an unsigned integer, divided by 2^64 - 1.
tabpfn_row_hash <- function(row, salt) {
  salt_bytes <- as.raw((salt %/% 256^(0:7)) %% 256)
  hex <- cli::hash_raw_sha256(c(row, salt_bytes))
  hi <- strtoi(substr(hex, 49, 56), 16L)
  lo <- strtoi(substr(hex, 57, 64), 16L)
  # strtoi() returns NA above 2^31 - 1.
  if (is.na(hi)) {
    hi <- as.numeric(paste0("0x", substr(hex, 49, 56)))
  }
  if (is.na(lo)) {
    lo <- as.numeric(paste0("0x", substr(hex, 57, 64)))
  }
  (hi * 2^32 + lo) / 18446744073709551615
}

# ---- Outlier clipping ----------------------------------------------------------

# Soft clipping at n_sigma standard deviations (Python `TorchSoftClipOutliers`):
# bounds are fitted on the training rows in two passes, the second ignoring
# the first pass's outliers; values beyond a bound are pulled in
# logarithmically.
tabpfn_soft_clip <- function(x, num_train, n_sigma) {
  if (num_train <= 1) {
    return(x)
  }
  train <- x$narrow(1, 1, num_train)
  mean <- tabpfn_nanmean(train)
  std <- tabpfn_nanstd(train)
  outlier <- (train > mean + std * n_sigma) | (train < mean - std * n_sigma)
  train <- torch::torch_where(
    outlier,
    torch::torch_full_like(train, NaN),
    train
  )
  mean <- tabpfn_nanmean(train)
  std <- tabpfn_nanstd(train)
  lower <- mean - std * n_sigma
  upper <- mean + std * n_sigma
  clamped <- torch::torch_maximum(
    -torch::torch_log(1 + torch::torch_abs(x)) + lower,
    x
  )
  torch::torch_minimum(
    torch::torch_log(1 + torch::torch_abs(clamped)) + upper,
    clamped
  )
}

# ------------------------------------------------------------------------------

# Regression targets. The model sees z-scored targets; half of the ensemble
# members additionally apply the "safepower" transform: a Yeo-Johnson power
# transform with a maximum-likelihood lambda, then standardization (Python
# `SafePowerTransformer` followed by sklearn's `StandardScaler`).

tabpfn_znorm_fit <- function(y) {
  mean <- mean(y)
  std <- sqrt(mean((y - mean)^2)) + 1e-20
  list(mean = mean, std = std)
}

tabpfn_yeo_johnson <- function(x, lambda) {
  out <- numeric(length(x))
  pos <- x >= 0
  eps <- .Machine$double.eps
  if (abs(lambda) < eps) {
    out[pos] <- log1p(x[pos])
  } else {
    out[pos] <- expm1(lambda * log1p(x[pos])) / lambda
  }
  if (abs(lambda - 2) > eps) {
    out[!pos] <- -expm1((2 - lambda) * log1p(-x[!pos])) / (2 - lambda)
  } else {
    out[!pos] <- -log1p(-x[!pos])
  }
  out
}

# Maximum-likelihood Yeo-Johnson lambda, with the search bounds of Python's
# `_yeojohnson_normmax` that keep the transform from overflowing.
tabpfn_yeo_johnson_lambda <- function(x) {
  if (all(x == 0)) {
    return(1)
  }
  n <- length(x)
  log_sum <- sum(sign(x) * log1p(abs(x)))
  neg_llf <- function(lambda) {
    trans <- tabpfn_yeo_johnson(x, lambda)
    v <- mean((trans - mean(trans))^2)
    if (!is.finite(v) || v < .Machine$double.xmin) {
      return(Inf)
    }
    -(-n / 2 * log(v) + (lambda - 1) * log_sum)
  }
  log1p_max_x <- log1p(20 * max(abs(x)))
  log_eps <- log(.Machine$double.eps)
  lb <- ((log(.Machine$double.xmin) - log_eps) / 2) / log1p_max_x
  ub <- ((log(.Machine$double.xmax) + log_eps) / 2) / log1p_max_x
  if (all(x < 0)) {
    bounds <- c(2 - ub, 2 - lb)
  } else if (any(x < 0)) {
    bounds <- c(max(2 - ub, lb), min(2 - lb, ub))
  } else {
    bounds <- c(lb, ub)
  }
  stats::optimize(neg_llf, bounds, tol = 1.48e-8)$minimum
}

tabpfn_fit_safepower <- function(y) {
  lambda <- tabpfn_yeo_johnson_lambda(y)
  trans <- tabpfn_yeo_johnson(y, lambda)
  mean <- mean(trans)
  scale <- sqrt(mean((trans - mean)^2))
  if (scale < 10 * .Machine$double.eps) {
    scale <- 1
  }
  list(name = "safepower", lambda = lambda, mean = mean, scale = scale)
}

tabpfn_apply_target_transform <- function(y, transform) {
  if (is.null(transform)) {
    return(y)
  }
  (tabpfn_yeo_johnson(y, transform$lambda) - transform$mean) / transform$scale
}

# Map the model's z-space bucket borders back through a member's target
# transform, so that its bar distribution can be put on the common grid
# (Python `transform_borders_one`). `borders` is a float32 tensor; the
# arithmetic follows sklearn's: the scaler inverse runs in float32 with
# float32 constants, and the power inverse runs in float64 (lambda is a
# float64 numpy scalar) before the result is stored as float32.
tabpfn_inverse_target_borders <- function(borders, transform) {
  b <- as.numeric(torch::as_array(borders))
  # Products and sums of two float32 values are exact in float64, so
  # rounding them to float32 gives the float32 result.
  x <- tabpfn_f32(
    tabpfn_f32(b * tabpfn_f32(transform$scale)) + tabpfn_f32(transform$mean)
  )

  lambda <- transform$lambda
  log_max32 <- tabpfn_f32(log(3.4028234663852886e38))
  max_arg <- log_max32 - 4 * 2^(floor(log2(log_max32)) - 23)
  clip <- function(v) pmin(pmax(v, -max_arg), max_arg)
  out <- numeric(length(x))
  pos <- x >= 0
  eps <- .Machine$double.eps
  if (abs(lambda) < eps) {
    out[pos] <- expm1(clip(x[pos]))
  } else {
    out[pos] <- expm1(clip(log1p(x[pos] * lambda) / lambda))
  }
  if (abs(lambda - 2) > eps) {
    out[!pos] <- -expm1(clip(log1p(-(2 - lambda) * x[!pos]) / (2 - lambda)))
  } else {
    out[!pos] <- -expm1(clip(-x[!pos]))
  }
  # Round to float32, as numpy stores the result in a float32 array.
  out <- tabpfn_f32(out)
  tabpfn_fix_borders(out)
}

# Borders that the inverse transform sent to non-finite or extreme values
# are pinned to the nearest good border, and the logits of the buckets they
# bound are cancelled; then degenerate outer buckets are widened (Python
# `_cancel_nan_borders` and `_repair_borders`). Values are float32.
tabpfn_fix_borders <- function(b) {
  n <- length(b)
  broken <- !is.finite(b) | b > 1e3 | b < -1e3
  cancel <- NULL
  if (any(broken)) {
    right_edges <- which(broken[-n] & !broken[-1])
    left_edges <- which(!broken[-n] & broken[-1])
    if (length(right_edges)) {
      first_good <- right_edges[1] + 1
      b[seq_len(first_good - 1)] <- b[first_good]
      b[1] <- tabpfn_f32(b[2] - 1)
    }
    if (length(left_edges)) {
      last_good <- left_edges[1]
      b[(last_good + 1):n] <- b[last_good]
      b[n] <- tabpfn_f32(b[n - 1] + 1)
    }
    cancel <- broken[-1] | broken[-n]
  }
  if (is.nan(b[n])) {
    b[is.nan(b)] <- max(b[!is.nan(b)])
    b[n] <- tabpfn_f32(b[n] + abs(b[n]))
  }
  if (b[n] - b[n - 1] < 1e-6) {
    b[n] <- tabpfn_f32(b[n] + abs(tabpfn_f32(b[n] * 0.1)))
  }
  if (b[1] == b[2]) {
    b[1] <- tabpfn_f32(b[1] - abs(tabpfn_f32(b[1] * 0.1)))
  }
  list(borders = b, cancel = cancel)
}

tabpfn_f32 <- function(x) {
  as.numeric(torch::as_array(torch::torch_tensor(
    x,
    dtype = torch::torch_float32()
  )))
}
