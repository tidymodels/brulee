#' Fit a TabPFN model
#'
#' `brulee_tab_pfn()` applies data to the pre-trained TabPFN tabular
#' foundation model of Hollmann _et al_ (2025), which emulates Bayesian
#' inference for regression and classification. The model runs in R
#' \pkg{torch}; no Python is needed.
#' The arguments mirror `tab_pfn()` in the \pkg{tabpfn} package, which runs the
#' Python implementation.
#'
#' @param x Depending on the context:
#'
#'   * A __data frame__ of predictors.
#'   * A __matrix__ of predictors.
#'   * A __recipe__ specifying a set of preprocessing steps
#'     created from [recipes::recipe()].
#'
#' @param y When `x` is a __data frame__ or __matrix__, `y` is the outcome
#' specified as:
#'
#'   * A __data frame__ with 1 numeric column.
#'   * A __matrix__ with 1 numeric column.
#'   * A numeric __vector__ for regression or a __factor__ for classification.
#'
#' @param data When a __recipe__ or __formula__ is used, `data` is specified as:
#'
#'   * A __data frame__ containing both the predictors and the outcome.
#'
#' @param formula A formula specifying the outcome terms on the left-hand side,
#' and the predictor terms on the right-hand side.
#'
#' @param num_estimators An integer for the ensemble size. When `NULL` (the
#' default), the model version's recommended size is used (8 for v3.5 and v3,
#' 4 for v3.5-fast; v3 uses more for data with many predictors).
#'
#' @param softmax_temperature An adjustment factor that is a divisor in the
#' exponents of the softmax function; it must be greater than 0. When `NULL`
#' (the default), the model version's recommended value is used (1 for v3.5,
#' 0.9 for v3).
#'
#' @param balance_probabilities A logical to adjust the prior probabilities in
#' cases where there is a class imbalance. Default is `FALSE`. Classification
#' only.
#'
#' @param average_before_softmax A logical. For cases where
#' `num_estimators > 1`, should the average be done before using the softmax
#' function or after? Default is `FALSE`.
#'
#' @param training_set_limit An integer of at least 2, or `Inf` (the default)
#' to use every row. When the training set is larger, it is sampled down to
#' exactly that many rows, stratified by class for classification and by
#' quartile for regression. For classification, it must be at least the number
#' of classes, so that every class keeps a row.
#'
#' @param version The model version, such as `"v3.5"`. A bare number works
#' too: `3.5`, `"3.5"`, and `"v3.5"` are equivalent. See [tab_pfn_versions()]
#' for the currently supported versions. The default is the newest version in
#' that list.
#'
#' @param device The torch device: `NULL` (the default; CUDA when available,
#' otherwise the CPU), `"cpu"`, `"cuda"`, or `"mps"` (Apple GPUs).
#'
#' @param ignore_pretraining_limits A logical. By default, the model refuses
#' data larger than it was trained for (for example, more than 5,000 training
#' rows on the CPU for v3.5). Set to `TRUE` to run it anyway.
#'
#' @param ... Not currently used, but required for extensibility.
#'
#' @details
#'
#' ## Before you start: the model weights and their license
#'
#' TabPFN is a *pre-trained* model. Instead of estimating parameters from your
#' data, `brulee_tab_pfn()` feeds your training data, together with the rows to
#' predict, to a large neural network that was trained in advance on millions
#' of synthetic data sets. That network's parameters (its "weights") are made
#' by the company Prior Labs and are **not** included in brulee: you download
#' them once, and Prior Labs requires that you accept their license first.
#'
#' The license allows free use for non-commercial purposes, such as research,
#' teaching, and evaluation. Commercial or production use requires a separate
#' license from Prior Labs (<sales@priorlabs.ai>). Read the license when you
#' accept it.
#'
#' Each model version has its own license. Accepting the TabPFN-3.5 license
#' covers `"v3.5"` (the default) and `"v3.5-fast"`; `"v3"` needs the TabPFN-3
#' license.
#'
#' ### One-time setup
#'
#' 1. Create a free account at <https://ux.priorlabs.ai>.
#' 2. On the **Licenses** tab, accept the license for each model version you
#'    plan to use (TabPFN-3.5 and/or TabPFN-3).
#' 3. On your account page, copy your **API key**. Treat it like a password.
#' 4. Give R the key by adding a line to your `.Renviron` file (open it with
#'    `usethis::edit_r_environ()`), then restart R:
#'
#'    ```
#'    TABPFN_TOKEN=paste-your-api-key-here
#'    ```
#'
#' 5. Download the weights for the version you will use:
#'
#'    ```
#'    brulee::tab_pfn_download_weights("v3.5")
#'    ```
#'
#'    This checks your key and license with Prior Labs, then downloads the
#'    weights from Hugging Face: about 880 MB for `"v3.5"`, 330 MB for
#'    `"v3.5-fast"`, and 450 MB for `"v3"` (one file each for classification
#'    and regression). `tab_pfn_weights_available()` tells you whether they
#'    are already downloaded.
#'
#' In an interactive session you can skip steps 4 and 5: the first time
#' `brulee_tab_pfn()` needs weights that aren't downloaded, it offers to
#' download them, opens the Prior Labs login page in your browser, and asks you
#' to paste your API key. The key is then saved in `~/.cache/tabpfn/auth_token`
#' for later sessions. Non-interactive sessions (scripts, R Markdown and
#' Quarto documents, continuous integration) need `TABPFN_TOKEN`; if the
#' weights are missing there, `brulee_tab_pfn()` stops with an error that says
#' how to download them.
#'
#' ### After the setup
#'
#' The weights are stored in a per-user cache (see
#' [tab_pfn_download_weights()]) and used from there, with no further contact
#' with Prior Labs or internet access. Remove them with [tab_pfn_clear_cache()],
#' for example to free disk space,
#' or because you stop using the model, as the license requires when it
#' ends.
#'
#' If you have used the Python `tabpfn` package on this computer, it may
#' already have downloaded the weights and saved your API key; brulee uses
#' both, so you may have nothing to do.
#'
#' ## Computing requirements
#'
#' The model runs on the CPU or, when available, on a CUDA GPU (or an Apple
#' GPU with `device = "mps"`). On the CPU it accepts at most 5,000 training
#' rows; see "Data size limits" below.
#'
#' ## Differences from the Python package
#'
#' The computations follow the Python `tabpfn` package (version 9.1.0), but
#' the random choices of the ensemble members use R's random number generator
#' (so [set.seed()] makes fits reproducible), and predictions differ slightly
#' from Python's. For v3, Python computes the SVD features with a randomized
#' SVD when the training data has more than a million cells; brulee always
#' uses the exact SVD.
#'
#' Do not use `brulee_tab_pfn()` and the Python-based \pkg{tabpfn} package in
#' the same R session: R torch and Python torch can't be loaded in one process.
#'
#' @eval tabpfn_limits_table_md()
#' @return A `brulee_tab_pfn` object.
#' @seealso [predict.brulee_tab_pfn()], [tab_pfn_download_weights()]
#'
#' @references
#' Müller, S., Hollmann, N., Pineda Arango, S., Grabocka, J., and Hutter, F.
#' (2022). "Transformers can do Bayesian inference." _International
#' Conference on Learning Representations 2022_.
#' \doi{10.48550/arXiv.2112.10510}
#'
#' Hollmann, N., Müller, S., Eggensperger, K., and Hutter, F. (2023).
#' "TabPFN: A transformer that solves small tabular classification problems
#' in a second." _International Conference on Learning Representations 2023_.
#' \doi{10.48550/arXiv.2207.01848}
#'
#' Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M.,
#' Hoo, S. B., Schirrmeister, R. T., and Hutter, F. (2025). "Accurate
#' predictions on small data with a tabular foundation model." _Nature_,
#' 637(8045), 319-326. \doi{10.1038/s41586-024-08328-6}
#'
#' Grinsztajn, L., Flöge, K., Key, O., et al. (2026). "TabPFN-3: Technical
#' report." _arXiv preprint_. \doi{10.48550/arXiv.2605.13986}
#'
#' Jäger, B., Erickson, N., Grinsztajn, L., et al. (2026). "TabPFN-3.5:
#' Technical report." _arXiv preprint_. \doi{10.48550/arXiv.2609.17895}
#'
#' @examples
#' \dontrun{
#' if (rlang::is_installed("modeldata") && tab_pfn_weights_available()) {
#'
#'   # --------------------------------------------------------------------------
#'   # Regression
#'
#'   set.seed(1)
#'   reg_fit <- brulee_tab_pfn(mpg ~ ., data = mtcars[6:32,])
#'   reg_fit
#'
#'   augment(reg_fit, new_data =  mtcars[1:5, -1])
#'   augment(reg_fit, new_data =  mtcars[1:5, -1], quantile_levels = (1:5)/6)
#'
#'   # --------------------------------------------------------------------------
#'   # Classification
#'
#'   set.seed(1)
#'   cls_fit <- brulee_tab_pfn(Species ~ ., data = modeldata::scat[6:110,])
#'   cls_fit
#'
#'   augment(cls_fit, new_data =  modeldata::scat[1:5, -1])
#' }
#' }
#' @rdname brulee_tab_pfn
#' @export
brulee_tab_pfn <- function(x, ...) {
  UseMethod("brulee_tab_pfn")
}

#' @export
#' @rdname brulee_tab_pfn
brulee_tab_pfn.default <- function(x, ...) {
  cli::cli_abort(
    "{.fn brulee_tab_pfn} is not defined for a {.cls {class(x)[1]}}."
  )
}

#' @export
#' @rdname brulee_tab_pfn
brulee_tab_pfn.data.frame <- function(
  x,
  y,
  num_estimators = NULL,
  softmax_temperature = NULL,
  balance_probabilities = FALSE,
  average_before_softmax = FALSE,
  training_set_limit = Inf,
  version = NULL,
  device = NULL,
  ignore_pretraining_limits = FALSE,
  ...
) {
  options <- tabpfn_fit_options(
    num_estimators,
    softmax_temperature,
    balance_probabilities,
    average_before_softmax,
    training_set_limit,
    device,
    ignore_pretraining_limits
  )
  processed <- hardhat::mold(x, y)
  tabpfn_bridge(processed, options, version = version, ...)
}

#' @export
#' @rdname brulee_tab_pfn
brulee_tab_pfn.matrix <- brulee_tab_pfn.data.frame

#' @export
#' @rdname brulee_tab_pfn
brulee_tab_pfn.formula <- function(
  formula,
  data,
  num_estimators = NULL,
  softmax_temperature = NULL,
  balance_probabilities = FALSE,
  average_before_softmax = FALSE,
  training_set_limit = Inf,
  version = NULL,
  device = NULL,
  ignore_pretraining_limits = FALSE,
  ...
) {
  options <- tabpfn_fit_options(
    num_estimators,
    softmax_temperature,
    balance_probabilities,
    average_before_softmax,
    training_set_limit,
    device,
    ignore_pretraining_limits
  )
  # Keep factors as single columns rather than indicators.
  bp <- hardhat::default_formula_blueprint(
    intercept = FALSE,
    allow_novel_levels = FALSE,
    indicators = "none",
    composition = "tibble"
  )
  processed <- hardhat::mold(formula, data, blueprint = bp)
  tabpfn_bridge(processed, options, version = version, ...)
}

#' @export
#' @rdname brulee_tab_pfn
brulee_tab_pfn.recipe <- function(
  x,
  data,
  num_estimators = NULL,
  softmax_temperature = NULL,
  balance_probabilities = FALSE,
  average_before_softmax = FALSE,
  training_set_limit = Inf,
  version = NULL,
  device = NULL,
  ignore_pretraining_limits = FALSE,
  ...
) {
  options <- tabpfn_fit_options(
    num_estimators,
    softmax_temperature,
    balance_probabilities,
    average_before_softmax,
    training_set_limit,
    device,
    ignore_pretraining_limits
  )
  processed <- hardhat::mold(x, data)
  tabpfn_bridge(processed, options, version = version, ...)
}

# ------------------------------------------------------------------------------
# Bridge

tabpfn_fit_options <- function(
  num_estimators,
  softmax_temperature,
  balance_probabilities,
  average_before_softmax,
  training_set_limit,
  device,
  ignore_pretraining_limits,
  call = caller_env()
) {
  check_number_whole(num_estimators, min = 1, allow_null = TRUE, call = call)
  check_softmax_temperature(softmax_temperature, allow_null = TRUE, call = call)
  check_bool(balance_probabilities, call = call)
  check_bool(average_before_softmax, call = call)
  check_number_whole(
    training_set_limit,
    min = 2,
    allow_infinite = TRUE,
    call = call
  )
  check_string(device, allow_null = TRUE, call = call)
  check_bool(ignore_pretraining_limits, call = call)
  list(
    num_estimators = num_estimators,
    softmax_temperature = softmax_temperature,
    balance_probabilities = balance_probabilities,
    average_before_softmax = average_before_softmax,
    training_set_limit = training_set_limit,
    device = device,
    ignore_pretraining_limits = ignore_pretraining_limits
  )
}

tabpfn_bridge <- function(
  processed,
  options,
  version = NULL,
  ...,
  call = caller_env()
) {
  check_dots_empty(call = call)
  if (!torch::torch_is_installed()) {
    cli::cli_abort(
      "The torch backend has not been installed; use {.run torch::install_torch()}.",
      call = call
    )
  }
  version <- tabpfn_resolve_version(version, arg = "version", call = call)
  data <- brulee_foundation_data(processed, call = call)
  outcome <- data$outcome
  # The ensemble members' random choices use their own seed, drawn from R's
  # random number generator so that `set.seed()` makes a fit reproducible.
  options$seed <- sample.int(.Machine$integer.max, 1)
  predictors <- as.data.frame(data$predictors)
  keep <- tabpfn_with_seed(
    options$seed,
    brulee_subsample_rows(outcome, options$training_set_limit, call = call)
  )
  predictors <- predictors[keep, , drop = FALSE]
  outcome <- outcome[keep]

  fit <- tabpfn_impl(predictors, outcome, options, version, call = call)
  new_brulee_tab_pfn(fit, processed$blueprint)
}

tabpfn_impl <- function(x, y, options, version, call = caller_env()) {
  # `task` picks the checkpoint; `task_type` is the model's name for it.
  if (is.factor(y)) {
    task <- "classification"
    task_type <- "multiclass"
  } else {
    task <- "regression"
    task_type <- "regression"
  }
  info <- tabpfn_version_info(version, task, call = call)
  device <- tabpfn_resolve_device(options$device, call = call)
  loaded <- tabpfn_load(info, device, call = call)
  ic <- tabpfn_resolve_inference(
    loaded$inference_config,
    task,
    version,
    call = call
  )
  tabpfn_check_limits(
    x,
    y,
    ic,
    version,
    device,
    options$ignore_pretraining_limits,
    call = call
  )

  encoder <- tabpfn_fit_encoder(
    x,
    min_samples = ic$min_samples_categorical,
    min_unique = ic$min_unique_numeric,
    call = call
  )
  x_train <- tabpfn_encode(encoder, x, call = call)
  categorical <- tabpfn_categorical_columns(encoder)
  n_estimators <- options$num_estimators %||%
    tabpfn_resolve_n_estimators(
      ic$n_estimators,
      ncol(x_train),
      ic$preprocessors
    )

  fit <- list(
    version = version,
    file = info$file,
    sha256 = info$sha256,
    architecture = info$architecture,
    task = task,
    task_type = task_type,
    device = device,
    encoder = encoder,
    categorical = categorical,
    x_train = x_train,
    n_estimators = n_estimators,
    softmax_temperature = options$softmax_temperature %||%
      ic$softmax_temperature,
    average_before_softmax = options$average_before_softmax,
    balance_probabilities = options$balance_probabilities,
    outlier_std = ic$outlier_std,
    preprocessors = ic$preprocessors,
    seed = options$seed
  )

  if (task == "classification") {
    present <- levels(droplevels(y))
    fit$levels <- levels(y)
    fit$classes <- present
    fit$y_train <- match(as.character(y), present) - 1L
    fit$class_counts <- as.numeric(table(factor(
      as.character(y),
      levels = present
    )))
    fit$members <- tabpfn_with_seed(
      fit$seed,
      tabpfn_make_members(
        x_train,
        categorical,
        n_estimators,
        task_type,
        preprocessors = ic$preprocessors,
        n_classes = length(present),
        subsampling = ic$feature_subsampling
      )
    )
  } else {
    y <- as.double(y)
    fit$y_scale <- tabpfn_znorm_fit(y)
    fit$y_train <- (y - fit$y_scale$mean) / fit$y_scale$std
    fit$members <- tabpfn_with_seed(
      fit$seed,
      tabpfn_make_members(
        x_train,
        categorical,
        n_estimators,
        task_type,
        preprocessors = ic$preprocessors,
        y_train = fit$y_train,
        subsampling = ic$feature_subsampling
      )
    )
  }
  fit
}

tabpfn_check_limits <- function(
  x,
  y,
  ic,
  version,
  device,
  ignore_pretraining_limits,
  call
) {
  n <- nrow(x)
  cpu_limit <- tabpfn_cpu_sample_limit(version)
  problems <- character()
  if (ncol(x) > ic$max_features) {
    problems <- c(
      problems,
      "{ncol(x)} predictors (the model supports {ic$max_features})"
    )
  }
  if (n > ic$max_samples) {
    problems <- c(
      problems,
      "{n} training rows (the model supports {ic$max_samples})"
    )
  }
  if (device == "cpu" && n > cpu_limit) {
    problems <- c(
      problems,
      "{n} training rows on the CPU (the model supports {cpu_limit})"
    )
  } else if (device == "cpu" && n > cpu_limit %/% 5) {
    cli::cli_warn(
      c(
        "Running on the CPU with more than {cpu_limit %/% 5} training rows may
         be slow.",
        i = "A CUDA GPU is much faster; see {.arg device}."
      ),
      call = call
    )
  }
  if (length(problems) > 0 && !ignore_pretraining_limits) {
    cli::cli_abort(
      c(
        "The data is larger than the model was trained for:",
        set_names(problems, rep("*", length(problems))),
        i = "Use {.code ignore_pretraining_limits = TRUE} to run it anyway, or
             {.arg training_set_limit} to sample rows."
      ),
      call = call
    )
  }
  if (is.factor(y) && nlevels(droplevels(y)) > ic$max_classes) {
    cli::cli_abort(
      "The model supports at most {ic$max_classes} classes.",
      call = call
    )
  }
  invisible()
}

new_brulee_tab_pfn <- function(fit, blueprint) {
  hardhat::new_model(
    fit = fit,
    levels = fit$levels,
    training = dim(fit$x_train),
    version = fit$version,
    device = fit$device,
    blueprint = blueprint,
    class = "brulee_tab_pfn"
  )
}

#' @export
print.brulee_tab_pfn <- function(x, ...) {
  if (is.null(x$levels)) {
    task <- "regression"
  } else {
    task <- "classification"
  }
  cli::cli_text("TabPFN {x$version} {task} model")
  cli::cli_text(
    "{x$training[1]} training rows, {x$training[2]} predictors,
     {x$fit$n_estimators} ensemble member{?s}"
  )
  if (!is.null(x$levels)) {
    cli::cli_text("{length(x$levels)} class{?es}: {.val {x$levels}}")
  }
  cli::cli_text("device: {.val {x$device}}")
  invisible(x)
}
