# Test helpers shared across test-chronos2-*.R files. testthat sources every
# `helper-*.R` before running the test files, so the helpers below are
# available to any test in this directory.

stub_chronos_loaders <- function(also_mock_predict_core = FALSE) {
  fake_dir <- file.path(
    tempdir(check = TRUE),
    paste0("chronos-stub-", as.integer(Sys.time()))
  )
  dir.create(fake_dir, recursive = TRUE, showWarnings = FALSE)

  bindings <- list(
    chronos2_download = function(model_id, revision, cache_dir) {
      list(
        model_dir = fake_dir,
        sha = if (grepl("^[0-9a-f]{40}$", revision)) {
          revision
        } else {
          "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        }
      )
    },
    chronos2_parse_config = function(path) {
      list(
        d_model = 384L,
        num_layers = 12L,
        num_heads = 12L,
        output_patch_size = 16L,
        max_output_patches = 64L,
        quantiles = (1:9) / 10
      )
    },
    chronos2_model = function(config) {
      structure(list(config = config), class = "fake_chronos_module")
    },
    load_chronos2_weights = function(model, path) invisible(NULL)
  )

  if (also_mock_predict_core) {
    # Return a deterministic [n_series, n_model_quantiles, prediction_length]
    # array. Quantiles are fixed at config$quantiles = (1:9)/10 (see above).
    bindings$chronos2_predict_core <- function(
      object,
      context,
      prediction_length = NULL,
      past_covariates = NULL,
      future_covariates = NULL
    ) {
      if (is.list(context)) {
        n_series <- length(context)
      } else {
        n_series <- 1L
      }
      n_q <- length(object$config$quantiles)
      preds <- array(0, dim = c(n_series, n_q, prediction_length))
      for (s in seq_len(n_series)) {
        for (q in seq_len(n_q)) {
          for (t in seq_len(prediction_length)) {
            preds[s, q, t] <- s * 100 + object$config$quantiles[q] * 10 + t
          }
        }
      }
      list(
        predictions = preds,
        quantiles = object$config$quantiles,
        prediction_length = prediction_length
      )
    }
  }

  do.call(
    testthat::local_mocked_bindings,
    c(bindings, list(.package = "brulee", .env = parent.frame()))
  )
  # `model$to(...)` and `model$eval()` need to no-op
  assign(
    "$.fake_chronos_module",
    function(x, name) {
      if (name %in% c("to", "eval")) {
        return(function(...) invisible(NULL))
      }
      unclass(x)[[name]]
    },
    envir = globalenv()
  )
  withr::defer(rm("$.fake_chronos_module", envir = globalenv()), parent.frame())
}

chicago_subset <- function(n = 200) {
  data(Chicago, package = "modeldata", envir = environment())
  Chi <- Chicago[seq_len(n), c("date", "ridership", "Clark_Lake", "Austin")]
  Chi$series_id <- "L"
  Chi[, c("series_id", "date", "ridership", "Clark_Lake", "Austin")]
}
