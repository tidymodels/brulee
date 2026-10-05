# Reading checkpoints and loading their weights into R modules.
#
# Both formats are read into the same structure:
#   list(tensors, architecture_name, config, inference_config)
# where `tensors` is a named list of torch tensors keyed by the Python
# state-dict names.

tabpfn_read_checkpoint <- function(path, call = caller_env()) {
  if (grepl("\\.safetensors$", path)) {
    tabpfn_read_safetensors(path, call = call)
  } else if (grepl("\\.ckpt$", path)) {
    tabpfn_read_ckpt_checkpoint(path, call = call)
  } else {
    cli::cli_abort(
      "Unknown checkpoint format for {.file {basename(path)}}.",
      call = call
    )
  }
}

# safetensors checkpoints keep the non-tensor checkpoint fields as JSON
# strings in the header metadata (Python `tabpfn.checkpoint.save_as_safetensors`).
tabpfn_read_safetensors <- function(path, call = caller_env()) {
  tensors <- safetensors::safe_load_file(path, framework = "torch")
  meta <- attr(tensors, "metadata")[["__metadata__"]]
  attr(tensors, "metadata") <- NULL
  if (is.null(meta)) {
    cli::cli_abort(
      "{.file {basename(path)}} has no checkpoint metadata.",
      call = call
    )
  }
  fields <- lapply(meta, jsonlite::fromJSON, simplifyVector = FALSE)
  tabpfn_checkpoint(unclass(tensors), fields, path, call)
}

tabpfn_read_ckpt_checkpoint <- function(path, call = caller_env()) {
  obj <- tabpfn_read_ckpt(path, call = call)
  tensors <- obj$state_dict
  obj$state_dict <- NULL
  tabpfn_checkpoint(tensors, obj, path, call)
}

tabpfn_checkpoint <- function(tensors, fields, path, call) {
  required <- c("architecture_name", "config", "inference_config")
  missing <- setdiff(required, names(fields))
  if (length(missing) > 0 || length(tensors) == 0) {
    cli::cli_abort(
      "{.file {basename(path)}} is missing {.field {c(missing, if (!length(tensors)) 'state_dict')}}.",
      call = call
    )
  }
  is_tensor <- vapply(tensors, inherits, logical(1), what = "torch_tensor")
  if (!all(is_tensor)) {
    cli::cli_abort(
      "{.file {basename(path)}} has non-tensor weights: {.val {names(tensors)[!is_tensor]}}.",
      call = call
    )
  }
  list(
    tensors = tensors,
    architecture_name = fields$architecture_name,
    config = fields$config,
    inference_config = fields$inference_config
  )
}

# Copy checkpoint tensors into `model`. R modules use the same attribute
# names as the Python modules, so `model$state_dict()` names match the
# checkpoint keys one to one. Every key must match in both directions, with
# identical shapes; anything else is an error.
tabpfn_load_state <- function(model, tensors, call = caller_env()) {
  state <- model$state_dict()
  missing <- setdiff(names(state), names(tensors))
  unexpected <- setdiff(names(tensors), names(state))
  if (length(missing) > 0) {
    cli::cli_abort(
      c(
        "The checkpoint has no weights for {length(missing)} model
         parameter{?s}.",
        i = "First missing: {.val {utils::head(missing, 5)}}."
      ),
      call = call
    )
  }
  if (length(unexpected) > 0) {
    cli::cli_abort(
      c(
        "The checkpoint has {length(unexpected)} weight{?s} that the model
         doesn't use.",
        i = "First unused: {.val {utils::head(unexpected, 5)}}."
      ),
      call = call
    )
  }
  mismatch <- names(state)[
    !vapply(
      names(state),
      function(k) identical(dim(state[[k]]), dim(tensors[[k]])),
      logical(1)
    )
  ]
  if (length(mismatch) > 0) {
    first <- mismatch[[1]]
    cli::cli_abort(
      c(
        "{length(mismatch)} checkpoint weight{?s} {?has/have} the wrong
         shape.",
        i = "{.val {first}}: expected {dim(state[[first]])}, got
             {dim(tensors[[first]])}."
      ),
      call = call
    )
  }
  model$load_state_dict(tensors[names(state)])
  invisible(model)
}

# ------------------------------------------------------------------------------

# A strict reader for the PyTorch zip checkpoints (`.ckpt`) that TabPFN v3
# ships. R torch can't read them because their top level is a nested dict
# with non-tensor values ("Expected Tensor but got GenericDict").
#
# A checkpoint is a zip archive with `<prefix>/data.pkl` (a pickle, protocol
# 2) and one file per tensor storage, `<prefix>/data/<key>`. The pickle
# interpreter below handles only the opcodes and globals that the v3
# checkpoints use, and errors on anything else, so a checkpoint saved
# differently fails loudly instead of being misread. Adding support for
# something new is a deliberate change to the tables below.

pkl_globals <- c(
  "collections.OrderedDict",
  "torch._utils._rebuild_tensor_v2",
  "torch.FloatStorage"
)

# Storage types and how to read them.
pkl_storage_types <- list(
  "torch.FloatStorage" = list(what = "double", size = 4L, dtype = "float32")
)

tabpfn_read_ckpt <- function(path, call = caller_env()) {
  entries <- utils::unzip(path, list = TRUE)$Name
  pkl_entry <- grep("(^|/)data\\.pkl$", entries, value = TRUE)
  if (length(pkl_entry) != 1L) {
    cli::cli_abort(
      "{.file {basename(path)}} is not a PyTorch zip checkpoint.",
      call = call
    )
  }
  prefix <- sub("data\\.pkl$", "", pkl_entry)
  byteorder_entry <- paste0(prefix, "byteorder")
  if (byteorder_entry %in% entries) {
    byteorder <- tabpfn_read_zip_entry(path, byteorder_entry, "raw")
    if (!identical(rawToChar(byteorder), "little")) {
      cli::cli_abort(
        "{.file {basename(path)}} uses an unsupported byte order.",
        call = call
      )
    }
  }

  storages <- new.env(parent = emptyenv())
  load_storage <- function(key, type, numel) {
    if (is.null(storages[[key]])) {
      spec <- pkl_storage_types[[type]]
      values <- tabpfn_read_zip_entry(
        path,
        paste0(prefix, "data/", key),
        spec$what,
        n = numel,
        size = spec$size
      )
      if (length(values) != numel) {
        cli::cli_abort(
          "Storage {.val {key}} in {.file {basename(path)}} is truncated.",
          call = call
        )
      }
      storages[[key]] <- torch::torch_tensor(values, dtype = spec$dtype)
    }
    storages[[key]]
  }

  bytes <- tabpfn_read_zip_entry(path, pkl_entry, "raw")
  obj <- pkl_unpickle(bytes, load_storage, path = path, call = call)
  pkl_to_r(obj, call = call)
}

tabpfn_read_zip_entry <- function(
  path,
  entry,
  what,
  n = NULL,
  size = NA_integer_
) {
  con <- unz(path, entry, open = "rb")
  on.exit(close(con))
  if (what == "raw") {
    info <- utils::unzip(path, list = TRUE)
    n <- info$Length[info$Name == entry]
    return(readBin(con, "raw", n = n))
  }
  readBin(con, what, n = n, size = size, endian = "little")
}

# Mutable pickle containers. Dicts and lists are environments while
# unpickling, so that a container stored in the memo and later filled by
# SETITEMS / APPENDS is the same object wherever it's referenced.
pkl_dict <- function() {
  env <- new.env(parent = emptyenv())
  env$keys <- list()
  env$values <- list()
  structure(list(env = env), class = "pkl_dict")
}

pkl_list <- function() {
  env <- new.env(parent = emptyenv())
  env$values <- list()
  structure(list(env = env), class = "pkl_list")
}

pkl_tuple <- function(values) structure(values, class = "pkl_tuple")

pkl_none <- structure(list(), class = "pkl_none")

pkl_global <- function(module, name) {
  structure(list(name = paste0(module, ".", name)), class = "pkl_global")
}

pkl_unpickle <- function(bytes, load_storage, path, call) {
  pos <- 1L
  n_bytes <- length(bytes)
  stack <- list()
  marks <- integer()
  memo <- list()

  bad <- function(msg) {
    cli::cli_abort(
      c(
        "Can't read {.file {basename(path)}}: {msg}",
        i = "brulee reads only the checkpoint format used by the TabPFN v3
             releases."
      ),
      call = call
    )
  }
  take <- function(n) {
    if (pos + n - 1L > n_bytes) {
      bad("the pickle ends unexpectedly.")
    }
    out <- bytes[pos:(pos + n - 1L)]
    pos <<- pos + n
    out
  }
  read_uint <- function(n) {
    b <- as.integer(take(n))
    sum(b * 256^(seq_along(b) - 1L))
  }
  read_line <- function() {
    start <- pos
    while (pos <= n_bytes && bytes[pos] != as.raw(0x0a)) {
      pos <<- pos + 1L
    }
    if (pos > n_bytes) {
      bad("the pickle ends unexpectedly.")
    }
    out <- rawToChar(bytes[start:(pos - 1L)])
    pos <<- pos + 1L
    out
  }
  push <- function(x) {
    stack[[length(stack) + 1L]] <<- list(x)
  }
  pop <- function() {
    x <- stack[[length(stack)]][[1]]
    stack[[length(stack)]] <<- NULL
    x
  }
  pop_mark <- function() {
    m <- marks[length(marks)]
    marks <<- marks[-length(marks)]
    items <- if (length(stack) > m) {
      lapply(stack[(m + 1L):length(stack)], `[[`, 1)
    } else {
      list()
    }
    stack <<- if (m > 0L) stack[seq_len(m)] else list()
    items
  }
  memo_put <- function(i) {
    memo[[as.character(i)]] <<- stack[[length(stack)]]
  }
  memo_get <- function(i) {
    x <- memo[[as.character(i)]]
    if (is.null(x)) {
      bad(paste0("memo entry ", i, " is missing."))
    }
    stack[[length(stack) + 1L]] <<- x
  }

  repeat {
    op <- take(1L)
    switch(
      as.character(op),
      "80" = {
        # PROTO
        proto <- read_uint(1L)
        if (proto > 2L) {
          bad(paste("pickle protocol", proto, "is not supported."))
        }
      },
      "2e" = break, # STOP
      "7d" = push(pkl_dict()), # EMPTY_DICT
      "5d" = push(pkl_list()), # EMPTY_LIST
      "29" = push(pkl_tuple(list())), # EMPTY_TUPLE
      "28" = marks <- c(marks, length(stack)), # MARK
      "4e" = push(pkl_none), # NONE
      "88" = push(TRUE), # NEWTRUE
      "89" = push(FALSE), # NEWFALSE
      "4b" = push(as.integer(read_uint(1L))), # BININT1
      "4d" = push(as.integer(read_uint(2L))), # BININT2
      "4a" = {
        # BININT: 4-byte signed
        v <- read_uint(4L)
        push(as.integer(if (v >= 2^31) v - 2^32 else v))
      },
      "47" = push(readBin(take(8L), "double", size = 8L, endian = "big")), # BINFLOAT
      "58" = {
        # BINUNICODE
        n <- read_uint(4L)
        s <- if (n > 0) rawToChar(take(n)) else ""
        Encoding(s) <- "UTF-8"
        push(s)
      },
      "71" = memo_put(read_uint(1L)), # BINPUT
      "72" = memo_put(read_uint(4L)), # LONG_BINPUT
      "68" = memo_get(read_uint(1L)), # BINGET
      "6a" = memo_get(read_uint(4L)), # LONG_BINGET
      "63" = {
        # GLOBAL
        g <- pkl_global(read_line(), read_line())
        if (!g$name %in% pkl_globals) {
          bad(paste0("the object type \"", g$name, "\" is not supported."))
        }
        push(g)
      },
      "74" = push(pkl_tuple(pop_mark())), # TUPLE
      "85" = push(pkl_tuple(list(pop()))), # TUPLE1
      "86" = {
        # TUPLE2
        b <- pop()
        a <- pop()
        push(pkl_tuple(list(a, b)))
      },
      "75" = {
        # SETITEMS
        items <- pop_mark()
        d <- stack[[length(stack)]][[1]]
        if (!inherits(d, "pkl_dict") || length(items) %% 2L != 0L) {
          bad("SETITEMS on a non-dict.")
        }
        idx <- seq(1L, length(items), by = 2L)
        d$env$keys <- c(d$env$keys, items[idx])
        d$env$values <- c(d$env$values, lapply(items[idx + 1L], list))
      },
      "73" = {
        # SETITEM
        value <- pop()
        key <- pop()
        d <- stack[[length(stack)]][[1]]
        if (!inherits(d, "pkl_dict")) {
          bad("SETITEM on a non-dict.")
        }
        d$env$keys <- c(d$env$keys, list(key))
        d$env$values <- c(d$env$values, list(list(value)))
      },
      "61" = {
        # APPEND
        value <- pop()
        l <- stack[[length(stack)]][[1]]
        if (!inherits(l, "pkl_list")) {
          bad("APPEND on a non-list.")
        }
        l$env$values <- c(l$env$values, list(list(value)))
      },
      "65" = {
        # APPENDS
        items <- pop_mark()
        l <- stack[[length(stack)]][[1]]
        if (!inherits(l, "pkl_list")) {
          bad("APPENDS on a non-list.")
        }
        l$env$values <- c(l$env$values, lapply(items, list))
      },
      "51" = {
        # BINPERSID: ('storage', storage type, key, location, numel)
        pid <- pop()
        if (
          !inherits(pid, "pkl_tuple") ||
            length(pid) != 5L ||
            !identical(pid[[1]], "storage") ||
            !inherits(pid[[2]], "pkl_global")
        ) {
          bad("unsupported persistent id.")
        }
        type <- pid[[2]]$name
        if (is.null(pkl_storage_types[[type]])) {
          bad(paste0("the storage type \"", type, "\" is not supported."))
        }
        push(list(storage = load_storage(pid[[3]], type, pid[[5]])))
      },
      "52" = {
        # REDUCE
        args <- pop()
        fn <- pop()
        if (!inherits(fn, "pkl_global")) {
          bad("REDUCE on a non-global.")
        }
        push(pkl_reduce(fn$name, args, bad))
      },
      bad(paste0("pickle opcode 0x", as.character(op), " is not supported."))
    )
  }
  if (length(stack) != 1L) {
    bad("the pickle left an unexpected stack.")
  }
  stack[[1]][[1]]
}

pkl_reduce <- function(name, args, bad) {
  if (name == "collections.OrderedDict") {
    if (length(args) != 0L) {
      bad("OrderedDict with arguments.")
    }
    return(pkl_dict())
  }
  # torch._utils._rebuild_tensor_v2(storage, storage_offset, size, stride,
  #   requires_grad, backward_hooks[, metadata])
  if (length(args) < 6L || !is.list(args[[1]]) || is.null(args[[1]]$storage)) {
    bad("unexpected tensor arguments.")
  }
  backward_hooks <- args[[6]]
  if (
    !inherits(backward_hooks, "pkl_dict") || length(backward_hooks$env$keys)
  ) {
    bad("tensors with backward hooks.")
  }
  storage <- args[[1]]$storage
  size <- as.integer(unlist(args[[3]]))
  stride <- as.integer(unlist(args[[4]]))
  out <- torch::torch_as_strided(
    storage,
    size = size,
    stride = stride,
    storage_offset = as.integer(args[[2]])
  )
  out$clone()
}

pkl_to_r <- function(x, call) {
  if (inherits(x, "pkl_dict")) {
    keys <- x$env$keys
    if (!all(vapply(keys, is_string, logical(1)))) {
      cli::cli_abort("Checkpoint dict with non-string keys.", call = call)
    }
    out <- lapply(x$env$values, function(v) pkl_to_r(v[[1]], call))
    names(out) <- unlist(keys)
    return(out)
  }
  if (inherits(x, "pkl_list")) {
    return(lapply(x$env$values, function(v) pkl_to_r(v[[1]], call)))
  }
  if (inherits(x, "pkl_tuple")) {
    return(lapply(unclass(x), pkl_to_r, call = call))
  }
  if (inherits(x, "pkl_none")) {
    return(NULL)
  }
  x
}

# ------------------------------------------------------------------------------

# Architecture backends, keyed by the checkpoint's `architecture_name`. The
# backend for architecture `<name>` is a pair of functions in
# `R/tabpfn-<version>.R`:
#
#   * `<name>_parse_config(config, call)` strictly parses the checkpoint's
#     architecture config;
#   * `<name>_model(config)` builds the R module, whose
#     `forward(x, y, task_type)` returns the raw model output and whose
#     `borders()` method returns the regression bar-distribution borders.
#
# They are found by name, so a new architecture is only a new file; nothing
# shared lists the architectures.

tabpfn_architecture <- function(name, call = caller_env()) {
  ns <- asNamespace("brulee")
  parse_config <- get0(
    paste0(name, "_parse_config"),
    envir = ns,
    inherits = FALSE
  )
  build <- get0(paste0(name, "_model"), envir = ns, inherits = FALSE)
  if (!is.function(parse_config) || !is.function(build)) {
    cli::cli_abort(
      c(
        "The TabPFN architecture {.val {name}} is not supported by brulee.",
        i = "The checkpoint may be newer than this version of brulee."
      ),
      call = call
    )
  }
  list(parse_config = parse_config, build = build)
}

# Build the model for a checkpoint read by `tabpfn_read_checkpoint()`, load its
# weights, and put it in evaluation mode. `expected` is the registry's
# architecture for the file, which must agree with the checkpoint.
tabpfn_build_model <- function(
  checkpoint,
  expected = NULL,
  call = caller_env()
) {
  name <- checkpoint$architecture_name
  if (!is.null(expected) && !identical(name, expected)) {
    cli::cli_abort(
      "The checkpoint uses the {.val {name}} architecture, but the registry
       lists {.val {expected}}.",
      call = call
    )
  }
  arch <- tabpfn_architecture(name, call = call)
  config <- arch$parse_config(checkpoint$config, call = call)
  model <- arch$build(config)
  tabpfn_load_state(model, checkpoint$tensors, call = call)
  model$eval()
  model
}

# ------------------------------------------------------------------------------

# Strict parsing of a checkpoint's architecture config.
#
# Python fills absent keys from the architecture's config dataclass and
# ignores keys it doesn't know. Here:
#   * `defaults` are the dataclass defaults (absent keys get them);
#   * `fixed` are keys the Python architecture ignores because the behavior
#     is hard-coded; they are accepted only with the hard-coded value, so a
#     checkpoint that changes one fails instead of being misread;
#   * `free` are keys that only tune performance and may take any value;
#   * any other key is an error.

tabpfn_parse_config <- function(
  config,
  defaults,
  fixed = list(),
  free = character(),
  architecture,
  call = caller_env()
) {
  unknown <- setdiff(names(config), c(names(defaults), names(fixed), free))
  if (length(unknown) > 0) {
    cli::cli_abort(
      c(
        "The checkpoint config has {length(unknown)} setting{?s} that brulee
         doesn't know for {.val {architecture}}: {.val {unknown}}.",
        i = "The checkpoint may be newer than this version of brulee."
      ),
      call = call
    )
  }
  for (key in intersect(names(fixed), names(config))) {
    if (!identical(config[[key]], fixed[[key]])) {
      cli::cli_abort(
        c(
          "The checkpoint config sets {.field {key}} to {.val {format(config[[key]])}},
           but brulee only supports {.val {format(fixed[[key]])}} for
           {.val {architecture}}.",
          i = "The checkpoint may be newer than this version of brulee."
        ),
        call = call
      )
    }
  }
  out <- defaults
  for (key in intersect(names(defaults), names(config))) {
    # `[<-` with list() keeps a NULL value instead of dropping the entry.
    out[key] <- list(config[[key]])
  }
  out
}

# ------------------------------------------------------------------------------

# The checkpoint's inference config says how data is prepared for the model
# (Python `tabpfn/inference_config.py`). Absent keys get Python's
# `InferenceConfig` defaults; unknown keys, and values the R pipeline doesn't
# implement, are errors, so a newer checkpoint fails loudly instead of being
# preprocessed differently than Python would.

# `PREPROCESS_TRANSFORMS` has no default; every checkpoint sets it.
tabpfn_inference_defaults <- list(
  PREPROCESS_TRANSFORMS = NULL,
  MAX_UNIQUE_FOR_CATEGORICAL_FEATURES = 30L,
  MIN_UNIQUE_FOR_NUMERICAL_FEATURES = 4L,
  MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE = 100L,
  MIN_CARDINALITY_FOR_TEXT = 30L,
  SOFTMAX_TEMPERATURE = 0.9,
  N_ESTIMATORS = "auto",
  TRANSFORM_DATES = FALSE,
  TRANSFORM_TEXT = FALSE,
  TEXT_N_COMPONENTS = 30L,
  OUTLIER_REMOVAL_STD = "auto",
  FEATURE_SHIFT_METHOD = "shuffle",
  CLASS_SHIFT_METHOD = "shuffle",
  FINGERPRINT_FEATURE = TRUE,
  POLYNOMIAL_FEATURES = "no",
  SUBSAMPLE_SAMPLES = NULL,
  SAMPLE_SUBSAMPLING_METHOD = "auto",
  ENABLE_GPU_PREPROCESSING = FALSE,
  FEATURE_SUBSAMPLING_METHOD = "balanced",
  FEATURE_SUBSAMPLING_CONSTANT_FEATURE_COUNT = 50L,
  FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT = "auto",
  REGRESSION_Y_PREPROCESS_TRANSFORMS = list(NULL, "safepower"),
  USE_SKLEARN_16_DECIMAL_PRECISION = FALSE,
  MAX_NUMBER_OF_CLASSES = 10L,
  MAX_NUMBER_OF_FEATURES = 500L,
  MAX_NUMBER_OF_SAMPLES = 10000L,
  MAX_CPU_SAMPLES = 1000L,
  FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM = TRUE,
  PASSTHROUGH_INF = FALSE,
  `_REGRESSION_DEFAULT_OUTLIER_REMOVAL_STD` = NULL,
  `_CLASSIFICATION_DEFAULT_OUTLIER_REMOVAL_STD` = 12
)

# Settings that select pipeline behavior, and the values the R pipeline
# implements.
tabpfn_inference_supported <- list(
  FEATURE_SHIFT_METHOD = "shuffle",
  CLASS_SHIFT_METHOD = "shuffle",
  FINGERPRINT_FEATURE = TRUE,
  POLYNOMIAL_FEATURES = "no",
  SUBSAMPLE_SAMPLES = NULL,
  ENABLE_GPU_PREPROCESSING = TRUE,
  USE_SKLEARN_16_DECIMAL_PRECISION = FALSE,
  FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM = TRUE,
  PASSTHROUGH_INF = FALSE
)

# Feature transforms the R pipeline implements. With GPU preprocessing on,
# Python runs these in torch (`preprocessing/torch/`).
tabpfn_quantile_transforms <- c(
  quantile_uni = "uni",
  quantile_uni_coarse = "coarse",
  quantile_uni_extrapolate = "extrapolate"
)
tabpfn_squashing_transforms <- c(
  squashing_scaler_default = 3,
  squashing_scaler_max10 = 10
)
tabpfn_categorical_encodings <- c(
  "ordinal_shuffled",
  "ordinal_very_common_categories_shuffled",
  "numeric"
)

# One entry of PREPROCESS_TRANSFORMS as the R pipeline uses it, or NULL when
# it isn't supported.
tabpfn_preprocessor <- function(tr) {
  name <- tr$name
  gpu <- if (name %in% names(tabpfn_quantile_transforms)) {
    "quantile"
  } else if (name %in% names(tabpfn_squashing_transforms)) {
    "squash"
  } else if (identical(name, "none")) {
    "none"
  }
  append <- tr$append_original
  ok <- !is.null(gpu) &&
    isTRUE(tr$categorical_name %in% tabpfn_categorical_encodings) &&
    (is.null(tr$global_transformer_name) ||
      identical(tr$global_transformer_name, "svd_quarter_components")) &&
    (isFALSE(append) || isTRUE(append) || identical(append, "auto")) &&
    is.null(tr$max_onehot_cardinality) &&
    isFALSE(tr$differentiable %||% FALSE)
  if (!ok) {
    return(NULL)
  }
  list(
    name = name,
    gpu = gpu,
    quantile = if (gpu == "quantile") tabpfn_quantile_transforms[[name]],
    squash_max = if (gpu == "squash") tabpfn_squashing_transforms[[name]],
    categorical = tr$categorical_name,
    svd = identical(tr$global_transformer_name, "svd_quarter_components"),
    append = append,
    max_features = as.integer(tr$max_features_per_estimator)
  )
}

tabpfn_resolve_inference <- function(
  inference_config,
  task,
  version,
  call = caller_env()
) {
  unknown <- setdiff(names(inference_config), names(tabpfn_inference_defaults))
  if (length(unknown) > 0) {
    cli::cli_abort(
      c(
        "The checkpoint's inference config has {length(unknown)} setting{?s}
         that brulee doesn't know: {.val {unknown}}.",
        i = "The checkpoint may be newer than this version of brulee."
      ),
      call = call
    )
  }
  ic <- tabpfn_inference_defaults
  for (key in names(inference_config)) {
    ic[key] <- list(inference_config[[key]])
  }

  unsupported <- character()
  for (key in names(tabpfn_inference_supported)) {
    if (!identical(ic[[key]], tabpfn_inference_supported[[key]])) {
      unsupported <- c(unsupported, key)
    }
  }
  preprocessors <- lapply(ic$PREPROCESS_TRANSFORMS, tabpfn_preprocessor)
  if (
    length(preprocessors) == 0 ||
      any(vapply(preprocessors, is.null, logical(1)))
  ) {
    unsupported <- c(unsupported, "PREPROCESS_TRANSFORMS")
  }
  y_transforms <- ic$REGRESSION_Y_PREPROCESS_TRANSFORMS
  if (
    !identical(
      lapply(y_transforms, function(x) x %||% "none"),
      list("none", "safepower")
    )
  ) {
    unsupported <- c(unsupported, "REGRESSION_Y_PREPROCESS_TRANSFORMS")
  }
  if (!is.numeric(ic$N_ESTIMATORS) && !identical(ic$N_ESTIMATORS, "auto")) {
    unsupported <- c(unsupported, "N_ESTIMATORS")
  }
  # "auto" means balanced unless there are more than 100,000 rows (beyond
  # the supported sizes), when Python ranks features with LightGBM.
  subsampling <- ic$FEATURE_SUBSAMPLING_METHOD
  if (identical(subsampling, "auto")) {
    subsampling <- "balanced"
  }
  if (!subsampling %in% c("balanced", "random")) {
    unsupported <- c(unsupported, "FEATURE_SUBSAMPLING_METHOD")
  }
  if (length(unsupported) > 0) {
    cli::cli_abort(
      c(
        "TabPFN {version} uses preprocessing that brulee doesn't implement
         yet.",
        i = "Unsupported setting{?s}: {.field {unsupported}}."
      ),
      call = call
    )
  }

  outlier_std <- ic$OUTLIER_REMOVAL_STD
  if (identical(outlier_std, "auto")) {
    outlier_std <- if (task == "classification") {
      ic$`_CLASSIFICATION_DEFAULT_OUTLIER_REMOVAL_STD`
    } else {
      ic$`_REGRESSION_DEFAULT_OUTLIER_REMOVAL_STD`
    }
  }
  list(
    n_estimators = ic$N_ESTIMATORS,
    softmax_temperature = ic$SOFTMAX_TEMPERATURE,
    outlier_std = outlier_std,
    preprocessors = preprocessors,
    feature_subsampling = subsampling,
    min_samples_categorical = ic$MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
    min_unique_numeric = ic$MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
    max_classes = ic$MAX_NUMBER_OF_CLASSES,
    max_features = ic$MAX_NUMBER_OF_FEATURES,
    max_samples = ic$MAX_NUMBER_OF_SAMPLES
  )
}

# Ensemble size: the checkpoint's value, or for "auto" 8 unless more members
# are needed to cover every column (at most 32). Python
# `scale_n_estimators_for_feature_coverage`.
tabpfn_resolve_n_estimators <- function(
  n_estimators,
  n_columns,
  preprocessors
) {
  if (is.numeric(n_estimators)) {
    return(as.integer(n_estimators))
  }
  budget <- min(vapply(preprocessors, function(p) p$max_features, integer(1)))
  needed <- ceiling(n_columns / budget)
  if (needed > 8) as.integer(min(needed, 32)) else 8L
}
