fixture_path <- function(name, ext) {
  test_path("fixtures", "tabpfn", paste0(name, ext))
}

skip_if_no_torch <- function() {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed(), "libtorch is not installed")
}

skip_if_no_fixture <- function(name) {
  skip_if_no_torch()
  skip_if_not(
    file.exists(fixture_path(name, ".safetensors.gz")),
    paste("fixture", name, "is not available")
  )
}

# Decompress `gz` into a temporary file that lives as long as `env`. Binary
# fixtures are gzipped so that R CMD check doesn't flag their headers.
local_gunzip <- function(gz, ext, env = parent.frame()) {
  tmp <- withr::local_tempfile(fileext = ext, .local_envir = env)
  con <- gzfile(gz, "rb")
  on.exit(close(con))
  writeBin(readBin(con, "raw", n = 1e8), tmp)
  tmp
}

load_fixture <- function(name) {
  tmp <- local_gunzip(fixture_path(name, ".safetensors.gz"), ".safetensors")
  list(
    tensors = safetensors::safe_load_file(tmp, framework = "torch"),
    meta = jsonlite::read_json(
      fixture_path(name, ".json"),
      simplifyVector = TRUE
    )
  )
}

max_abs_diff <- function(a, b) {
  torch::torch_max(torch::torch_abs(a - b))$item()
}

# Snapshot transform for messages that name a temporary file.
scrub_tempfile <- function(x) {
  gsub("file[[:alnum:]]+\\.", "<tempfile>.", x)
}

# Build the R model for a parity fixture and load the fixture's weights.
fixture_model <- function(fx) {
  params <- fx$tensors[startsWith(names(fx$tensors), "param.")]
  names(params) <- sub("^param\\.", "", names(params))
  config <- jsonlite::read_json(
    fixture_path(fx$meta$name, ".json"),
    simplifyVector = FALSE
  )$config
  arch <- tabpfn_architecture(fx$meta$architecture)
  model <- arch$build(config)
  tabpfn_load_state(model, params)
  model$eval()
  model
}

# A recorded call's input or output, e.g. call_tensor(fx, "x_embed", "out").
call_tensor <- function(fx, module, part) {
  fx$tensors[[paste0("call.", module, ".", part)]]
}

expect_close <- function(actual, expected, tolerance = 1e-5) {
  expect_identical(dim(actual), dim(expected))
  scale <- max(1, expected$abs()$max()$item())
  expect_lt(max_abs_diff(actual, expected) / scale, tolerance)
}

pipeline_fixtures <- c(
  "pipeline-classification",
  "pipeline-regression",
  "pipeline-v3-classification",
  "pipeline-v3-regression"
)

# Pipeline fixtures (dev/tabpfn/make_pipeline_fixtures.py): the data, Python's
# per-member random choices, the model calls, and the predictions.
load_pipeline_fixture <- function(name) {
  tmp <- local_gunzip(
    fixture_path(name, ".safetensors.gz"),
    ".safetensors",
    env = parent.frame()
  )
  meta <- jsonlite::read_json(fixture_path(name, ".json"))
  data <- as.data.frame(lapply(meta$data, function(v) {
    unlist(lapply(v, function(e) {
      if (is.null(e)) {
        NA
      } else {
        e
      }
    }))
  }))
  for (nm in unlist(meta$categorical_columns)) {
    data[[nm]] <- factor(data[[nm]])
  }
  list(
    tensors = safetensors::safe_load_file(tmp, framework = "torch"),
    meta = meta,
    data = data,
    n_train = meta$n_train
  )
}

# Python's member choices in brulee's member format.
fixture_members <- function(fx, n_cols) {
  lapply(fx$meta$members, function(m) {
    if (is.null(m$features)) {
      features <- seq_len(n_cols)
    } else {
      features <- as.integer(unlist(m$features)) + 1L
    }
    class_permutation <- NULL
    if (!is.null(m$class_permutation)) {
      class_permutation <- as.integer(unlist(m$class_permutation))
    }
    list(
      preprocessor = (m$preprocessor %||% 0L) + 1L,
      features = features,
      class_permutation = class_permutation,
      category_maps = lapply(m$category_maps, function(v) {
        as.integer(unlist(v))
      }),
      shuffle = as.integer(unlist(m$shuffle)),
      target_transform = m$target_transform
    )
  })
}

as_numeric <- function(t) as.numeric(torch::as_array(t))

skip_if_no_weights <- function(version = "v3.5") {
  skip_if_no_torch()
  skip_if_not(
    tab_pfn_weights_available(version),
    paste("TabPFN", version, "weights are not cached")
  )
  # Verify the cached weights now, so that the message about reusing the
  # Python package's weights doesn't appear in the tests' output.
  infos <- tabpfn_checkpoint_infos(
    version,
    c("classification", "regression"),
    "default"
  )
  for (info in infos) {
    suppressMessages(tabpfn_checkpoint_path(info, ask = FALSE))
  }
}

# Use a pipeline fixture's tiny random-weight checkpoint in place of the
# released weights for the version.
local_tiny_checkpoint <- function(fx, env = parent.frame()) {
  name <- fx$meta$checkpoint
  path <- local_gunzip(
    test_path("fixtures", "tabpfn", "checkpoints", paste0(name, ".gz")),
    if (grepl("\\.safetensors$", name)) {
      ".safetensors"
    } else {
      ".ckpt"
    },
    env = env
  )
  local_mocked_bindings(
    tabpfn_checkpoint_path = function(info, ...) path,
    .env = env
  )
  # A model loaded from another checkpoint must not be reused.
  tabpfn_forget_models()
  withr::defer(tabpfn_forget_models(), envir = env)
  path
}

# The preprocessing configs of a pipeline fixture's checkpoint.
fixture_preprocessors <- function(fx) {
  transforms <- fx$meta$preprocess_transforms %||%
    list(list(
      name = "none",
      categorical_name = "ordinal_shuffled",
      append_original = FALSE,
      max_features_per_estimator = 768L
    ))
  lapply(transforms, tabpfn_preprocessor)
}

# Model inputs for a pipeline fixture with Python's member choices.
fixture_inputs <- function(fx, n_sigma) {
  n <- fx$n_train
  encoder <- tabpfn_fit_encoder(fx$data[seq_len(n), ])
  x_all <- tabpfn_encode(encoder, fx$data)
  members <- fixture_members(fx, ncol(x_all))
  y_members <- lapply(seq_along(members), function(i) {
    as_numeric(fx$tensors[[sprintf("member.%d.y_train", i - 1)]])
  })
  list(
    x_all = x_all,
    encoder = encoder,
    members = members,
    inputs = tabpfn_member_inputs(
      members,
      fixture_preprocessors(fx),
      x_all,
      n,
      y_members,
      tabpfn_categorical_columns(encoder),
      n_sigma
    )
  )
}

# Compare model inputs with the fixture's recorded model calls, which batch
# members of equal width in member order. The values can differ from the
# reference platform's by floating-point rounding, so they're compared with a
# tolerance.
expect_model_inputs <- function(inputs, fx, tolerance = 1e-5) {
  calls <- fx$meta$calls
  start <- 1
  for (i in seq_along(calls)) {
    ref_x <- fx$tensors[[sprintf("call.%d.x", i - 1)]]
    b <- ref_x$size(2)
    members <- start:(start + b - 1)
    x <- torch::torch_stack(inputs$x[members], dim = 2)
    expect_identical(dim(x), dim(ref_x))
    expect_identical(
      torch::as_array(x$isnan()),
      torch::as_array(ref_x$isnan())
    )
    expect_equal(
      torch::as_array(x$nan_to_num()),
      torch::as_array(ref_x$nan_to_num()),
      tolerance = tolerance
    )
    y <- inputs$y$index_select(
      2,
      torch::torch_tensor(members, dtype = torch::torch_long())
    )
    expect_equal(max_abs_diff(y, fx$tensors[[sprintf("call.%d.y", i - 1)]]), 0)
    start <- start + b
  }
}

# Python's fingerprint column of each member, in member order, from the
# fixture's recorded model inputs. The fingerprint is the last column before
# the shuffle, and soft clipping leaves its [0, 1] values unchanged.
fixture_fingerprints <- function(fx) {
  out <- list()
  for (i in seq_along(fx$meta$calls)) {
    ref_x <- fx$tensors[[sprintf("call.%d.x", i - 1)]]
    for (b in seq_len(ref_x$size(2))) {
      shuffle <- unlist(fx$meta$members[[length(out) + 1]]$shuffle)
      col <- which(shuffle == max(shuffle))
      out[[length(out) + 1]] <- ref_x[, b, col]
    }
  }
  out
}

# The fingerprint hashes the exact float32 bits of each row, so a one-ulp
# difference in an earlier step (the v3 SVD features, whose LAPACK differs
# between platforms) changes it completely. Tests that compare later steps
# with Python use Python's fingerprints instead; "fingerprints match Python"
# tests the fingerprint itself.
local_python_fingerprints <- function(fx, env = parent.frame()) {
  fingerprints <- fixture_fingerprints(fx)
  i <- 0
  local_mocked_bindings(
    tabpfn_fingerprint = function(x, num_train) {
      # Members come in order, once per prediction.
      i <<- i %% length(fingerprints) + 1
      fingerprints[[i]]
    },
    .env = env
  )
}

# The model outputs of a pipeline fixture's calls, in member order.
fixture_outputs <- function(fx) {
  outs <- lapply(seq_along(fx$meta$calls), function(i) {
    fx$tensors[[sprintf("call.%d.out", i - 1)]]
  })
  torch::torch_cat(outs, dim = 2)
}

# A small local file served over file://, for the download tests.
local_fake_file <- function(env = parent.frame()) {
  dir <- withr::local_tempdir(.local_envir = env)
  src <- file.path(dir, "remote.bin")
  writeBin(as.raw(1:200), src)
  list(
    url = paste0("file://", normalizePath(src)),
    size = file.size(src),
    sha256 = cli::hash_file_sha256(src),
    dir = dir
  )
}
