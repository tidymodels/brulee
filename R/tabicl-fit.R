# User-facing fit for the TabICL tabular foundation model.
#
# TabICL is an in-context learner: there is no training. `brulee_tab_icl()`
# validates and stores the (encoded) training data and a reference to the
# pretrained checkpoint; the model runs at predict time, conditioning on the
# stored training rows. Classification uses the classifier checkpoint, regression
# the regressor checkpoint (selected by the outcome type).
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
  norm_methods,
  classification
) {
  feat_id <- seq_len(n_features)
  class_id <- if (classification) seq_len(n_classes) - 1L else NULL

  if (n_estimators == 1L) {
    return(list(tabicl_member("none", feat_id, class_id)))
  }

  norms <- rep(norm_methods, length.out = n_estimators)
  members <- vector("list", n_estimators)
  for (k in seq_len(n_estimators)) {
    feat <- if (k == 1L) feat_id else sample(feat_id)
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
#' `brulee_tab_icl()` prepares the pretrained TabICL (Tabular In-Context
#' Learning) foundation model from Qu _et al_ (2025) for prediction. TabICL is a
#' transformer that makes predictions on tabular data by _in-context learning_:
#' it is not trained on your data. Instead, the released pretrained weights are
#' loaded and the model conditions on your training rows at prediction time,
#' much like a few-shot language model conditions on its prompt. Both
#' classification and regression are supported.
#'
#' @param x A data frame or matrix of predictors, a formula, or a recipe (see
#'   Details).
#' @param y When `x` is a data frame or matrix, a vector of outcomes: a factor
#'   for classification or a numeric vector for regression.
#' @param data A data frame for the formula and recipe methods.
#' @param formula A formula specifying the outcome and predictors.
#' @param path Path to a converted TabICL checkpoint directory containing
#'   `config.json` and `model.safetensors`. Use the classifier checkpoint for a
#'   factor outcome and the regressor checkpoint for a numeric outcome (the
#'   outcome type is checked against the checkpoint). See the **Pretrained
#'   weights** section.
#' @param n_estimators An integer for the number of ensemble members (default
#'   `8`). Each member preprocesses, permutes features, and (for classification)
#'   shuffles class labels differently; their predictions are averaged. Use `1`
#'   for a single, fully deterministic member.
#' @param norm_methods A character vector of per-member normalization methods.
#'   Currently `"none"` (standardize only) and `"power"` (Yeo-Johnson) are
#'   supported.
#' @param softmax_temperature A number for the temperature applied to the
#'   classification softmax. Only used for classification.
#' @param device A character string for the compute device: `"cpu"` (the
#'   default) or `"cuda"`. See the **Device support** section.
#' @param ... Not currently used.
#'
#' @details
#'
#' ## In-context learning
#'
#' Unlike the other \pkg{brulee} models, `brulee_tab_icl()` does not train any
#' parameters. The pretrained network is fixed; "fitting" simply validates and
#' stores the (encoded) training predictors and outcomes together with a
#' reference to the checkpoint. At [predict()] time the model is given the
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
#' Predictors are made numeric (factors and characters are ordinal-encoded;
#' numeric columns are mean-imputed). For each ensemble member the predictors are
#' then standardized, optionally transformed (`norm_methods`), and have outliers
#' clipped, mirroring the reference implementation. Factor outcomes are label
#' encoded; numeric outcomes are standardized internally and predictions are
#' returned on the original scale. There is _no need to pre-encode factors as
#' indicators_.
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
#' ## Pretrained weights
#'
#' The released TabICL checkpoints are distributed as a Python `.ckpt`. They must
#' first be converted to a directory holding `config.json` and
#' `model.safetensors`, which `path` then points at. Automatic download and
#' conversion of the weights is not yet available, so `path` is required.
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
#'  * `path`: the checkpoint directory the weights are loaded from.
#'  * `config`: the parsed model configuration.
#'  * `task`: `"classification"` or `"regression"`.
#'  * `levels`: the outcome factor levels (classification only).
#'  * `encoders`: the fitted per-column predictor encoders.
#'  * `train_x`, `train_y`: the encoded training context.
#'  * `n_estimators`, `norm_methods`, `softmax_temperature`: ensemble settings.
#'  * `device`: the resolved compute device.
#'  * `blueprint`: the `hardhat` blueprint.
#'
#' @examples
#' \dontrun{
#' # `path` points at a converted TabICL checkpoint directory containing
#' # `config.json` and `model.safetensors`.
#'
#' if (torch::torch_is_installed() & rlang::is_installed("modeldata")) {
#'   data(penguins, package = "modeldata")
#'   penguins <- na.omit(penguins)
#'
#'   in_train <- sample(seq_len(nrow(penguins)), 250)
#'   tr <- penguins[in_train, ]
#'   te <- penguins[-in_train, ]
#'
#'   # Classification
#'   cls_fit <- brulee_tab_icl(
#'     species ~ .,
#'     data = tr,
#'     path = "path/to/tabicl-classifier"
#'   )
#'   predict(cls_fit, te)
#'   predict(cls_fit, te, type = "prob")
#'
#'   # Regression
#'   reg_fit <- brulee_tab_icl(
#'     body_mass_g ~ .,
#'     data = tr,
#'     path = "path/to/tabicl-regressor"
#'   )
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
  path = NULL,
  n_estimators = 8L,
  norm_methods = c("none", "power"),
  softmax_temperature = 0.9,
  device = NULL,
  ...
) {
  processed <- hardhat::mold(x, y)
  tabicl_bridge(
    processed,
    path,
    n_estimators,
    norm_methods,
    softmax_temperature,
    device
  )
}

#' @export
#' @rdname brulee_tab_icl
brulee_tab_icl.matrix <- function(
  x,
  y,
  path = NULL,
  n_estimators = 8L,
  norm_methods = c("none", "power"),
  softmax_temperature = 0.9,
  device = NULL,
  ...
) {
  processed <- hardhat::mold(x, y)
  tabicl_bridge(
    processed,
    path,
    n_estimators,
    norm_methods,
    softmax_temperature,
    device
  )
}

#' @export
#' @rdname brulee_tab_icl
brulee_tab_icl.formula <- function(
  formula,
  data,
  path = NULL,
  n_estimators = 8L,
  norm_methods = c("none", "power"),
  softmax_temperature = 0.9,
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
    path,
    n_estimators,
    norm_methods,
    softmax_temperature,
    device
  )
}

#' @export
#' @rdname brulee_tab_icl
brulee_tab_icl.recipe <- function(
  x,
  data,
  path = NULL,
  n_estimators = 8L,
  norm_methods = c("none", "power"),
  softmax_temperature = 0.9,
  device = NULL,
  ...
) {
  processed <- hardhat::mold(x, data)
  tabicl_bridge(
    processed,
    path,
    n_estimators,
    norm_methods,
    softmax_temperature,
    device
  )
}

# ------------------------------------------------------------------------------
# Bridge

tabicl_bridge <- function(
  processed,
  path,
  n_estimators,
  norm_methods,
  softmax_temperature,
  device,
  call = rlang::caller_env()
) {
  if (!torch::torch_is_installed()) {
    cli::cli_abort(
      "The torch backend has not been installed; use {.run torch::install_torch()}.",
      call = call
    )
  }

  norm_methods <- rlang::arg_match(
    norm_methods,
    c("none", "power"),
    multiple = TRUE,
    call = call
  )
  device <- tabicl_resolve_device(device, call = call)

  outcome <- validate_mlp_outcome(processed$outcomes[[1]], call = call)
  classification <- is.factor(outcome)

  # Resolve the checkpoint: a user-supplied directory, or download the
  # task-appropriate converted checkpoint.
  if (is.null(path)) {
    path <- tabicl_download(
      checkpoint = if (classification) "classifier" else "regressor",
      call = call
    )
  }
  if (!dir.exists(path)) {
    cli::cli_abort(
      "Checkpoint directory {.path {path}} does not exist.",
      call = call
    )
  }

  # Checkpoint files are task-prefixed, so the file for the wrong task is simply
  # absent: a numeric outcome needs the regression checkpoint, a factor the
  # classification one.
  task <- if (classification) "classification" else "regression"
  files <- tabicl_checkpoint_files(task)
  config_path <- file.path(path, files$config)
  if (!file.exists(config_path)) {
    cli::cli_abort(
      c(
        "No {task} checkpoint found in {.path {path}}.",
        "i" = "Expected {.file {files$config}} and {.file {files$weights}}."
      ),
      call = call
    )
  }
  config <- tabicl_parse_config(config_path)
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
          "Outcome has {length(levels)} classes, exceeding the model's \\
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
    norm_methods = norm_methods,
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
  norm_methods,
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
    norm_methods = norm_methods,
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
    "{nrow(x$train_x)} training rows, {ncol(x$train_x)} predictors, \\
     {x$n_estimators} ensemble member{?s}"
  )
  if (x$task == "classification") {
    cli::cli_text("{length(x$levels)} class{?es}: {.val {x$levels}}")
  }
  cli::cli_text("device: {.val {x$device}}")
  invisible(x)
}
