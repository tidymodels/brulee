# User-facing fit for the TabICL tabular foundation model.
#
# TabICL is an in-context learner: there is no training. `brulee_tab_icl()`
# validates and stores the (encoded) training data; the model runs at predict
# time, conditioning on the stored training rows.
#
# Until weight hosting is finalized, `path` must point at a converted checkpoint
# directory (config.json + model.safetensors); see dev/tabicl/convert_ckpt.py.

# ------------------------------------------------------------------------------
# Device policy

# Resolve the compute device for TabICL. Unlike `guess_brulee_device()` (which
# prefers MPS on Apple Silicon), this defaults to CPU and refuses MPS: the
# bundled libtorch MPS backend crashes the regressor column embedder with a hard
# Metal assertion. CUDA is used when available.
tabicl_resolve_device <- function(device = NULL, call = rlang::caller_env()) {
  if (is.null(device)) {
    device <- "cpu"
  }
  device <- rlang::arg_match0(
    device,
    c("cpu", "cuda", "mps"),
    arg_nm = "device"
  )

  if (device == "cuda" && !torch::cuda_is_available()) {
    cli::cli_warn("CUDA is not available; using {.val cpu}.", call = call)
    device <- "cpu"
  }
  if (device == "mps") {
    cli::cli_warn(
      c(
        "The MPS backend is not supported for TabICL; using {.val cpu}.",
        "i" = "The bundled libtorch MPS kernels crash on parts of the model."
      ),
      call = call
    )
    device <- "cpu"
  }
  device
}

# ------------------------------------------------------------------------------
# Numerical encoding of predictors (TransformToNumerical)

# Learn per-column encoders: ordinal (sorted categories, unknown/missing -> -1)
# for factor/character columns, mean imputation for numeric columns.
tabicl_encode_fit <- function(predictors) {
  encoders <- lapply(names(predictors), function(nm) {
    col <- predictors[[nm]]
    if (is.factor(col) || is.character(col)) {
      list(type = "ordinal", levels = sort(unique(as.character(col))))
    } else {
      list(type = "numeric", mean = mean(as.numeric(col), na.rm = TRUE))
    }
  })
  names(encoders) <- names(predictors)
  encoders
}

# Apply learned encoders, returning a numeric matrix.
tabicl_encode_transform <- function(encoders, predictors) {
  cols <- lapply(names(encoders), function(nm) {
    enc <- encoders[[nm]]
    col <- predictors[[nm]]
    if (enc$type == "ordinal") {
      idx <- match(as.character(col), enc$levels) - 1L
      idx[is.na(idx)] <- -1L
      as.numeric(idx)
    } else {
      v <- as.numeric(col)
      v[is.na(v)] <- enc$mean
      v
    }
  })
  out <- matrix(
    unlist(cols, use.names = FALSE),
    ncol = length(cols)
  )
  colnames(out) <- names(encoders)
  out
}

# ------------------------------------------------------------------------------
# Training-set subsampling

# Pick row indices for a (possibly stratified) subsample of size `limit`. For
# a factor outcome we draw proportionally from each class so no class is
# dropped from the in-context training set. Returns NULL when no subsampling
# is needed (limit is Inf or n <= limit).
tabicl_subsample_indices <- function(
  outcome,
  limit,
  call = rlang::caller_env()
) {
  n <- length(outcome)
  if (!is.finite(limit) || n <= limit) {
    return(NULL)
  }
  limit <- as.integer(limit)

  if (is.factor(outcome)) {
    lvls <- levels(outcome)
    n_classes <- length(lvls)
    if (limit < n_classes) {
      cli::cli_abort(
        "{.arg training_set_limit} ({limit}) is smaller than the number of
         outcome classes ({n_classes}); cannot keep at least one row per class.",
        call = call
      )
    }
    counts <- tabulate(as.integer(outcome), nbins = n_classes)
    alloc <- pmax(1L, as.integer(round(limit * counts / sum(counts))))
    drift <- limit - sum(alloc)
    while (drift != 0L) {
      if (drift > 0L) {
        cand <- which(alloc < counts)
        if (length(cand) == 0L) {
          break
        }
        j <- cand[which.max(counts[cand])]
        alloc[j] <- alloc[j] + 1L
        drift <- drift - 1L
      } else {
        cand <- which(alloc > 1L)
        j <- cand[which.max(counts[cand])]
        alloc[j] <- alloc[j] - 1L
        drift <- drift + 1L
      }
    }
    idx <- unlist(lapply(seq_len(n_classes), function(j) {
      pool <- which(as.integer(outcome) == j)
      sample(pool, size = min(alloc[j], length(pool)))
    }))
  } else {
    idx <- sample.int(n, size = limit)
  }
  idx
}

# ------------------------------------------------------------------------------
# Ensemble member configuration

# Build the ensemble member configs. n_estimators = 1 yields the single
# deterministic "none" member with identity shuffles (matching the reference
# exactly). For more members, feature permutations are drawn with R's RNG (a
# faithful reimplementation of the reference's diversity, not bit-identical),
# and class labels use deterministic circular shifts.
tabicl_make_members <- function(
  n_estimators,
  n_features,
  n_classes,
  normalization,
  classification
) {
  feat_id <- seq_len(n_features)
  if (classification) {
    class_id <- seq_len(n_classes) - 1L
  } else {
    class_id <- NULL
  }

  if (n_estimators == 1L) {
    return(list(tabicl_member("none", feat_id, class_id)))
  }

  norms <- rep(normalization, length.out = n_estimators)
  members <- vector("list", n_estimators)
  for (k in seq_len(n_estimators)) {
    if (k == 1L) {
      feat <- feat_id
    } else {
      feat <- sample(feat_id)
    }
    cls <- class_id
    if (classification && k > 1L) {
      shift <- (k - 1L) %% n_classes
      cls <- ((class_id + shift) %% n_classes)
    }
    members[[k]] <- tabicl_member(norms[k], feat, cls)
  }
  members
}

# ------------------------------------------------------------------------------
# Fit methods

#' Fit a TabICL tabular foundation model
#'
#' `brulee_tab_icl()` prepares the pre-trained TabICL (Tabular In-Context
#' Learning) foundation model from Qu _et al_ (2025) for prediction. TabICL is a
#' transformer that makes predictions on tabular data by _in-context learning_:
#' it is not trained on your data. Instead, the released pre-trained weights are
#' loaded and the model conditions on your training rows at prediction time,
#' much like a few-shot language model conditions on its prompt. Both
#' classification and regression are supported.
#'
#' @inheritParams brulee_mlp
#' @param n_estimators An integer for the number of ensemble members (default
#'   `8`). Each member preprocesses, permutes features, and (for classification)
#'   shuffles class labels differently; their predictions are averaged. Use `1`
#'   for a single, fully deterministic member.
#' @param normalization A character vector of per-member normalization methods.
#'   Currently `"none"` (standardize only) and `"YeoJohnson"` (Yeo-Johnson
#'   power transform on top of standardization) are supported.
#' @param softmax_temperature A number for the temperature applied to the
#'   classification softmax. Only used for classification.
#' @param training_set_limit A single number giving the maximum number of
#'   training rows kept as in-context examples. When the training data has
#'   more rows than this, a subsample of exactly `training_set_limit` rows
#'   is drawn (stratified by the outcome for classification, simple random
#'   for regression). The default is `Inf`, which keeps every row. Useful
#'   for capping memory and prediction time on large training sets, since
#'   the entire (kept) training set is stored on the fitted object and
#'   re-sent through the network on every call to `predict()`.
#' @param device A character string for the compute device: `"cpu"` (the
#'   default) or `"cuda"`. See the **Device support** section.
#' @param ... Not currently used.
#'
#' @details
#'
#' ## In-context learning
#'
#' Unlike the other \pkg{brulee} models, `brulee_tab_icl()` does not train any
#' parameters. The pre-trained network is fixed; "fitting" simply validates and
#' stores the (encoded) training predictors and outcomes together with a
#' reference to the checkpoint. At [predict()] time, the model is given the
#' training rows as labelled context alongside the new rows and produces
#' predictions in a single forward pass. Because the training data are stored on
#' the fitted object, larger training sets make the object larger and prediction
#' slower.
#'
#' ## Architecture
#'
#' TabICL processes a table through three transformer stages:
#'
#' 1. **Column embedding**: a per-column set transformer turns each cell into a
#'    distribution-aware embedding, optionally informed by the target.
#' 2. **Row interaction**: a transformer with rotary position encoding mixes the
#'    feature embeddings within each row and aggregates them with learnable CLS
#'    tokens.
#' 3. **In-context learning**: a dataset-level transformer lets the test rows
#'    attend to the labelled training rows to produce class logits
#'    (classification) or quantiles (regression).
#'
#' ## Preprocessing
#'
#' TabICL applies its own preprocessing to mirror the reference implementation,
#' so most data shaping that other tabular models require is unnecessary (and
#' in some cases counter-productive). The pipeline runs in two stages.
#'
#' **Stage 1: numeric encoding (always, at fit time).**
#'
#' Each predictor column is converted to a numeric value:
#'
#' - Factor and character columns are **ordinal-encoded**: the unique values
#'   seen during fitting are sorted lexicographically and mapped to 0-based
#'   integers. _Do not pre-encode factors as indicator (dummy) variables._
#'   TabICL is a per-column tokenized transformer; one ordinal column gives
#'   the model one token per row, while a wide one-hot expansion bloats the
#'   sequence length, blows up the row-interaction stage, and degrades
#'   prediction quality.
#' - Numeric columns are taken as-is.
#'
#' The training predictors are stored on the fitted object in this encoded
#' form so that they can serve as context at [predict()] time.
#'
#' **Stage 2: per-member normalization (at predict time).**
#'
#' For each ensemble member, the encoded predictors pass through a small
#' pipeline before being handed to the network:
#'
#' 1. **Standardization** — center by column mean and divide by the
#'    population standard deviation (with a small epsilon and a soft clip to
#'    \eqn{\pm 100}). This always runs.
#' 2. **Optional Yeo-Johnson** — when the member's `normalization` slot is
#'    `"YeoJohnson"`, a per-column Yeo-Johnson power transform is inserted
#'    between standardization and outlier clipping. The Yeo-Johnson
#'    \eqn{\lambda} for each column is fit on the standardized training data
#'    via maximum likelihood, then the transformed values are re-standardized
#'    so the downstream stages see the same mean/scale as the `"none"` path.
#'    The transform is helpful when individual columns are heavily skewed or
#'    heavy-tailed. The `normalization` argument is a vector because the default
#'    ensemble intentionally mixes `"none"` and `"YeoJohnson"` across members to
#'    boost predictive diversity.
#' 3. **Outlier clipping** — a two-stage z-score clipper trims extreme
#'    values. This always runs.
#'
#' All parameters in stage 2 (means, standard deviations, Yeo-Johnson
#' lambdas, clip bounds) are estimated on the training rows alone and then
#' applied to both training and new rows.
#'
#' For regression, the outcome is standardized internally and predictions
#' are returned on the original scale. For classification, the outcome is
#' label-encoded.
#'
#' ## Missing Values
#'
#' Missing values do not need to be imputed by the user.
#'
#' - **Numeric columns**: at fit time the column mean (ignoring `NA`) is
#'   learned and reused to fill any `NA` in both the training context and
#'   the prediction rows.
#' - **Factor and character columns**: missing values, as well as any
#'   _new_ factor levels seen at prediction time that were not present
#'   during fitting, are mapped to the sentinel code `-1` and treated as a
#'   distinct "unknown" category by the model.
#'
#' Pre-imputation by the user is still allowed and is sometimes desirable
#' (for example, when a domain-appropriate imputation outperforms a column
#' mean), but it is not required for the model to run.
#'
#' ## Ensembling
#'
#' With `n_estimators > 1`, several views of the data are created by permuting
#' features and (for classification) shuffling class labels, each optionally with
#' a different normalization. Class logits are averaged across members and
#' converted to probabilities with a temperature softmax; regression means are
#' averaged. `n_estimators = 1` uses a single deterministic member (no shuffles,
#' `"none"` normalization). Note that with more than one member the feature
#' permutations are drawn with R's random number generator, so results are a
#' faithful reproduction of the reference ensemble but not bit-for-bit identical
#' to it; set the seed for reproducibility across runs.
#'
#' ## Device support
#'
#' Computation runs on CPU by default and on CUDA when `device = "cuda"` and a
#' GPU is available. The Apple `"mps"` backend is **not** supported: the bundled
#' libtorch MPS kernels crash on parts of the model, so requesting `"mps"` issues
#' a warning and falls back to CPU.
#'
#' ## Pre-trained weights
#'
#' The estimated parameters from the pre-trained Python model are used. On
#' first use, the values are downloaded and cached locally and are more than
#' 200MB.
#'
#' @references
#'
#' Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. (2025). TabICL: A
#' Tabular Foundation Model for In-Context Learning on Large Data. arXiv preprint
#' arXiv:2502.05564.
#'
#' @seealso [predict.brulee_tab_icl()]
#'
#' @return
#'
#' A `brulee_tab_icl` object with elements:
#'  * `path`: the cached checkpoint directory the weights are loaded from.
#'  * `config`: the parsed model configuration.
#'  * `task`: `"classification"` or `"regression"`.
#'  * `levels`: the outcome factor levels (classification only).
#'  * `encoders`: the fitted per-column predictor encoders.
#'  * `train_x`, `train_y`: the encoded training context.
#'  * `n_estimators`, `normalization`, `softmax_temperature`: ensemble settings.
#'  * `device`: the resolved compute device.
#'  * `blueprint`: the `hardhat` blueprint.
#'
#' @examples
#' \dontrun{
#' # Requires converted TabICL weights cached under ~/.cache/TabICL/ (see the
#' # "Pre-trained weights" section and dev/tabicl/).
#'
#' if (torch::torch_is_installed() && rlang::is_installed("modeldata")) {
#'   data(penguins, package = "modeldata")
#'   penguins <- na.omit(penguins)
#'
#'   in_train <- sample(seq_len(nrow(penguins)), 250)
#'   tr <- penguins[in_train, ]
#'   te <- penguins[-in_train, ]
#'
#'   # Classification (uses the cached classification checkpoint)
#'   cls_fit <- brulee_tab_icl(species ~ ., data = tr)
#'   predict(cls_fit, te)
#'   predict(cls_fit, te, type = "prob")
#'
#'   # Regression (uses the cached regression checkpoint)
#'   reg_fit <- brulee_tab_icl(body_mass_g ~ ., data = tr)
#'   predict(reg_fit, te)
#' }
#' }
#' @rdname brulee_tab_icl
#' @export
brulee_tab_icl <- function(x, ...) {
  UseMethod("brulee_tab_icl")
}

#' @export
#' @rdname brulee_tab_icl
brulee_tab_icl.default <- function(x, ...) {
  cli::cli_abort(
    "{.fn brulee_tab_icl} is not defined for a {.cls {class(x)[1]}}."
  )
}

#' @export
#' @rdname brulee_tab_icl
brulee_tab_icl.data.frame <- function(
  x,
  y,
  n_estimators = 8L,
  normalization = c("none", "YeoJohnson"),
  softmax_temperature = 0.9,
  training_set_limit = Inf,
  device = NULL,
  ...
) {
  processed <- hardhat::mold(x, y)
  tabicl_bridge(
    processed,
    n_estimators,
    normalization,
    softmax_temperature,
    training_set_limit,
    device
  )
}

#' @export
#' @rdname brulee_tab_icl
brulee_tab_icl.matrix <- function(
  x,
  y,
  n_estimators = 8L,
  normalization = c("none", "YeoJohnson"),
  softmax_temperature = 0.9,
  training_set_limit = Inf,
  device = NULL,
  ...
) {
  processed <- hardhat::mold(x, y)
  tabicl_bridge(
    processed,
    n_estimators,
    normalization,
    softmax_temperature,
    training_set_limit,
    device
  )
}

#' @export
#' @rdname brulee_tab_icl
brulee_tab_icl.formula <- function(
  formula,
  data,
  n_estimators = 8L,
  normalization = c("none", "YeoJohnson"),
  softmax_temperature = 0.9,
  training_set_limit = Inf,
  device = NULL,
  ...
) {
  processed <- hardhat::mold(
    formula,
    data,
    blueprint = hardhat::default_formula_blueprint(indicators = "none")
  )
  tabicl_bridge(
    processed,
    n_estimators,
    normalization,
    softmax_temperature,
    training_set_limit,
    device
  )
}

#' @export
#' @rdname brulee_tab_icl
brulee_tab_icl.recipe <- function(
  x,
  data,
  n_estimators = 8L,
  normalization = c("none", "YeoJohnson"),
  softmax_temperature = 0.9,
  training_set_limit = Inf,
  device = NULL,
  ...
) {
  processed <- hardhat::mold(x, data)
  tabicl_bridge(
    processed,
    n_estimators,
    normalization,
    softmax_temperature,
    training_set_limit,
    device
  )
}

# ------------------------------------------------------------------------------
# Bridge

tabicl_bridge <- function(
  processed,
  n_estimators,
  normalization,
  softmax_temperature,
  training_set_limit,
  device,
  call = rlang::caller_env()
) {
  if (!torch::torch_is_installed()) {
    cli::cli_abort(
      "The torch backend has not been installed; use {.run torch::install_torch()}.",
      call = call
    )
  }

  normalization <- rlang::arg_match(
    normalization,
    c("none", "YeoJohnson"),
    multiple = TRUE,
    call = call
  )
  if (
    !is.numeric(training_set_limit) ||
      length(training_set_limit) != 1L ||
      is.na(training_set_limit) ||
      training_set_limit < 1
  ) {
    cli::cli_abort(
      "{.arg training_set_limit} must be a single number >= 1 (or {.code Inf}).",
      call = call
    )
  }
  device <- tabicl_resolve_device(device, call = call)

  outcome <- validate_mlp_outcome(processed$outcomes[[1]], call = call)
  classification <- is.factor(outcome)

  sub_idx <- tabicl_subsample_indices(outcome, training_set_limit, call = call)
  if (!is.null(sub_idx)) {
    processed$predictors <- processed$predictors[sub_idx, , drop = FALSE]
    outcome <- outcome[sub_idx]
  }

  # Locate the cached checkpoint for the task (errors if none is cached).
  if (classification) {
    task <- "classification"
  } else {
    task <- "regression"
  }
  files <- tabicl_checkpoint_files(task)
  path <- tabicl_cache_lookup(task, call = call)
  config <- tabicl_parse_config(file.path(path, files$config))
  # Guard against a mislabeled file (e.g. a classification config renamed to the
  # regression filename).
  if (classification && config$max_classes <= 0) {
    cli::cli_abort(
      "{.path {files$config}} in {.path {path}} is not a classification checkpoint.",
      call = call
    )
  }
  if (!classification && config$max_classes > 0) {
    cli::cli_abort(
      "{.path {files$config}} in {.path {path}} is not a regression checkpoint.",
      call = call
    )
  }

  encoders <- tabicl_encode_fit(processed$predictors)
  train_x <- tabicl_encode_transform(encoders, processed$predictors)

  if (classification) {
    levels <- levels(outcome)
    if (length(levels) > config$max_classes) {
      cli::cli_abort(
        c(
          "Outcome has {length(levels)} classes, exceeding the model's
           {config$max_classes}.",
          "i" = "Many-class (hierarchical) classification is not yet implemented."
        ),
        call = call
      )
    }
    train_y <- as.integer(outcome) - 1L # 0-based label encoding
  } else {
    levels <- NULL
    train_y <- as.numeric(outcome)
  }

  new_brulee_tab_icl(
    path = path,
    config = config,
    task = task,
    levels = levels,
    encoders = encoders,
    train_x = train_x,
    train_y = train_y,
    n_estimators = as.integer(n_estimators),
    normalization = normalization,
    softmax_temperature = softmax_temperature,
    device = device,
    blueprint = processed$blueprint
  )
}

# ------------------------------------------------------------------------------
# Constructor

new_brulee_tab_icl <- function(
  path,
  config,
  task,
  levels,
  encoders,
  train_x,
  train_y,
  n_estimators,
  normalization,
  softmax_temperature,
  device,
  blueprint
) {
  hardhat::new_model(
    path = path,
    config = config,
    task = task,
    levels = levels,
    encoders = encoders,
    train_x = train_x,
    train_y = train_y,
    n_estimators = n_estimators,
    normalization = normalization,
    softmax_temperature = softmax_temperature,
    device = device,
    blueprint = blueprint,
    class = "brulee_tab_icl"
  )
}

#' @export
print.brulee_tab_icl <- function(x, ...) {
  cli::cli_text("TabICL {x$task} model")
  cli::cli_text(
    "{nrow(x$train_x)} training rows, {ncol(x$train_x)} predictors,
     {x$n_estimators} ensemble member{?s}"
  )
  if (x$task == "classification") {
    cli::cli_text("{length(x$levels)} class{?es}: {.val {x$levels}}")
  }
  cli::cli_text("device: {.val {x$device}}")
  invisible(x)
}
