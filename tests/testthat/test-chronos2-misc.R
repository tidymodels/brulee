# Tests for the download / version-management / config-parsing helpers in
# R/chronos2-misc.R. These cover the non-torch surface of that file (the
# torch model architecture, weight loader, and torch utility helpers
# require real torch operations and are exercised by integration tests).

skip_if_not_installed("jsonlite")

# ------------------------------------------------------------------------------
# chronos2_default_revision

test_that("chronos2_default_revision returns a 40-character hex SHA", {
  sha <- brulee:::chronos2_default_revision()
  expect_type(sha, "character")
  expect_length(sha, 1L)
  expect_match(sha, "^[0-9a-f]{40}$")
})

# ------------------------------------------------------------------------------
# chronos2_resolve_revision

test_that("chronos2_resolve_revision passes 40-char SHAs through unchanged", {
  sha <- "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
  # Provide a curl_fetch_memory that would fail loudly if called, to prove
  # the SHA path doesn't hit the network.
  testthat::local_mocked_bindings(
    curl_fetch_memory = function(...) {
      stop("SHA path should not hit curl")
    },
    .package = "curl"
  )
  expect_equal(
    brulee:::chronos2_resolve_revision("amazon/chronos-2", sha),
    sha
  )
})

test_that("chronos2_resolve_revision calls HF API for non-SHA revisions", {
  fake_sha <- "1111111111111111111111111111111111111111"
  testthat::local_mocked_bindings(
    curl_fetch_memory = function(url, handle = NULL) {
      list(
        status_code = 200L,
        content = charToRaw(jsonlite::toJSON(
          list(sha = fake_sha),
          auto_unbox = TRUE
        ))
      )
    },
    .package = "curl"
  )

  expect_equal(
    brulee:::chronos2_resolve_revision("amazon/chronos-2", "main"),
    fake_sha
  )
})

test_that("chronos2_resolve_revision errors on a non-200 status code", {
  testthat::local_mocked_bindings(
    curl_fetch_memory = function(url, handle = NULL) {
      list(status_code = 404L, content = raw())
    },
    .package = "curl"
  )

  expect_snapshot(error = TRUE, {
    brulee:::chronos2_resolve_revision("amazon/chronos-2", "v0")
  })
})

test_that("chronos2_resolve_revision errors when curl itself fails", {
  testthat::local_mocked_bindings(
    curl_fetch_memory = function(url, handle = NULL) {
      stop("simulated network failure")
    },
    .package = "curl"
  )

  expect_snapshot(error = TRUE, {
    brulee:::chronos2_resolve_revision("amazon/chronos-2", "v0")
  })
})

test_that("chronos2_resolve_revision errors when HF API has no sha field", {
  testthat::local_mocked_bindings(
    curl_fetch_memory = function(url, handle = NULL) {
      list(
        status_code = 200L,
        content = charToRaw(jsonlite::toJSON(
          list(message = "bad"),
          auto_unbox = TRUE
        ))
      )
    },
    .package = "curl"
  )

  expect_snapshot(error = TRUE, {
    brulee:::chronos2_resolve_revision("amazon/chronos-2", "v0")
  })
})

# ------------------------------------------------------------------------------
# chronos2_remote_size

test_that("chronos2_remote_size returns the Content-Length when present", {
  testthat::local_mocked_bindings(
    new_handle = function() list(),
    handle_setopt = function(handle, ...) handle,
    curl_fetch_memory = function(url, handle = NULL) {
      list(
        status_code = 200L,
        headers = charToRaw("HTTP/1.1 200 OK\r\nContent-Length: 12345\r\n\r\n")
      )
    },
    parse_headers = function(raw) {
      strsplit(rawToChar(raw), "\r\n")[[1]]
    },
    .package = "curl"
  )

  expect_equal(brulee:::chronos2_remote_size("http://x"), 12345)
})

test_that("chronos2_remote_size returns NA when the server omits Content-Length", {
  testthat::local_mocked_bindings(
    new_handle = function() list(),
    handle_setopt = function(handle, ...) handle,
    curl_fetch_memory = function(url, handle = NULL) {
      list(
        status_code = 200L,
        headers = charToRaw("HTTP/1.1 200 OK\r\n\r\n")
      )
    },
    parse_headers = function(raw) {
      strsplit(rawToChar(raw), "\r\n")[[1]]
    },
    .package = "curl"
  )

  expect_true(is.na(brulee:::chronos2_remote_size("http://x")))
})

test_that("chronos2_remote_size returns NA on a 4xx status", {
  testthat::local_mocked_bindings(
    new_handle = function() list(),
    handle_setopt = function(handle, ...) handle,
    curl_fetch_memory = function(url, handle = NULL) {
      list(status_code = 404L, headers = raw())
    },
    .package = "curl"
  )

  expect_true(is.na(brulee:::chronos2_remote_size("http://x")))
})

test_that("chronos2_remote_size returns NA when curl errors", {
  testthat::local_mocked_bindings(
    new_handle = function() list(),
    handle_setopt = function(handle, ...) handle,
    curl_fetch_memory = function(url, handle = NULL) stop("boom"),
    .package = "curl"
  )

  expect_true(is.na(brulee:::chronos2_remote_size("http://x")))
})

# ------------------------------------------------------------------------------
# chronos2_download_file

test_that("chronos2_download_file keeps a cached file with matching size", {
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)
  writeBin(as.raw(rep(0, 16)), tmp)

  testthat::local_mocked_bindings(
    chronos2_remote_size = function(url) 16,
    .package = "brulee"
  )
  # Would error if invoked
  testthat::local_mocked_bindings(
    curl_download = function(...) stop("should not download"),
    .package = "curl"
  )

  expect_invisible(brulee:::chronos2_download_file("http://x", tmp, "test"))
  expect_true(file.exists(tmp))
})

test_that("chronos2_download_file redownloads when the cached file is incomplete", {
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)
  writeBin(as.raw(rep(0, 4)), tmp) # only 4 bytes cached

  download_calls <- 0L
  testthat::local_mocked_bindings(
    chronos2_remote_size = function(url) 16,
    .package = "brulee"
  )
  testthat::local_mocked_bindings(
    curl_download = function(url, dest, mode, quiet) {
      download_calls <<- download_calls + 1L
      writeBin(as.raw(rep(0, 16)), dest)
      invisible(dest)
    },
    .package = "curl"
  )

  brulee:::chronos2_download_file("http://x", tmp, "test")
  expect_equal(download_calls, 1L)
  expect_equal(file.size(tmp), 16)
})

test_that("chronos2_download_file downloads when the cache is empty", {
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)
  expect_false(file.exists(tmp))

  testthat::local_mocked_bindings(
    chronos2_remote_size = function(url) 16,
    .package = "brulee"
  )
  testthat::local_mocked_bindings(
    curl_download = function(url, dest, mode, quiet) {
      writeBin(as.raw(rep(0, 16)), dest)
      invisible(dest)
    },
    .package = "curl"
  )

  brulee:::chronos2_download_file("http://x", tmp, "test")
  expect_true(file.exists(tmp))
})

test_that("chronos2_download_file errors after exhausting retries", {
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)

  testthat::local_mocked_bindings(
    chronos2_remote_size = function(url) 16,
    .package = "brulee"
  )
  attempts <- 0L
  testthat::local_mocked_bindings(
    curl_download = function(url, dest, mode, quiet) {
      attempts <<- attempts + 1L
      stop("simulated download failure")
    },
    .package = "curl"
  )

  # `cli_progress_step` includes timings (e.g. "[15ms]") that vary per
  # run; strip them so the snapshot is stable.
  expect_snapshot(
    error = TRUE,
    transform = function(lines) {
      gsub("\\[[0-9]+(\\.[0-9]+)?\\s*m?s\\]", "[TIME]", lines)
    },
    {
      brulee:::chronos2_download_file(
        "http://x",
        tmp,
        "test",
        max_attempts = 2L
      )
    }
  )
  expect_equal(attempts, 2L)
})

test_that("chronos2_download_file is happy when HEAD doesn't expose size", {
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)

  testthat::local_mocked_bindings(
    chronos2_remote_size = function(url) NA_real_,
    .package = "brulee"
  )
  download_calls <- 0L
  testthat::local_mocked_bindings(
    curl_download = function(url, dest, mode, quiet) {
      download_calls <<- download_calls + 1L
      writeBin(as.raw(rep(0, 7)), dest) # arbitrary size, no validation
      invisible(dest)
    },
    .package = "curl"
  )

  brulee:::chronos2_download_file("http://x", tmp, "test")
  expect_equal(download_calls, 1L)
  expect_true(file.exists(tmp))
})

# ------------------------------------------------------------------------------
# chronos2_download

test_that("chronos2_download resolves the revision and downloads the files", {
  cache <- file.path(tempdir(), paste0("chr-cache-", as.integer(Sys.time())))
  on.exit(unlink(cache, recursive = TRUE), add = TRUE)

  fake_sha <- "abcdefabcdefabcdefabcdefabcdefabcdefabcd"
  resolved_with <- NULL
  files_downloaded <- character()

  testthat::local_mocked_bindings(
    chronos2_resolve_revision = function(model_id, revision) {
      resolved_with <<- list(model_id = model_id, revision = revision)
      fake_sha
    },
    chronos2_download_file = function(url, dest, label, max_attempts = 3L) {
      files_downloaded <<- c(files_downloaded, basename(dest))
      file.create(dest)
      invisible(dest)
    },
    .package = "brulee"
  )

  res <- brulee:::chronos2_download(
    model_id = "amazon/chronos-2",
    revision = "main",
    cache_dir = cache
  )

  expect_equal(res$sha, fake_sha)
  expect_true(grepl(fake_sha, res$model_dir, fixed = TRUE))
  expect_true(dir.exists(res$model_dir))
  expect_setequal(files_downloaded, c("config.json", "model.safetensors"))
  expect_equal(resolved_with$model_id, "amazon/chronos-2")
  expect_equal(resolved_with$revision, "main")
})

# ------------------------------------------------------------------------------
# chronos2_parse_config

test_that("chronos2_parse_config flattens the JSON layout into a list", {
  cfg_path <- tempfile(fileext = ".json")
  on.exit(unlink(cfg_path), add = TRUE)

  json <- list(
    d_model = 384L,
    d_ff = 1024L,
    d_kv = 64L,
    num_heads = 12L,
    num_layers = 12L,
    dropout_rate = 0.1,
    layer_norm_epsilon = 1e-6,
    rope_theta = 10000,
    vocab_size = 1024L,
    pad_token_id = 0L,
    reg_token_id = 1L,
    chronos_config = list(
      context_length = 1024L,
      input_patch_size = 16L,
      input_patch_stride = 16L,
      output_patch_size = 16L,
      max_output_patches = 64L,
      quantiles = (1:9) / 10,
      use_arcsinh = TRUE,
      use_reg_token = TRUE,
      time_encoding_scale = 1.0
    )
  )
  writeLines(jsonlite::toJSON(json, auto_unbox = TRUE), cfg_path)

  cfg <- brulee:::chronos2_parse_config(cfg_path)

  expect_equal(cfg$d_model, 384L)
  expect_equal(cfg$num_layers, 12L)
  expect_equal(cfg$output_patch_size, 16L)
  expect_equal(cfg$max_output_patches, 64L)
  expect_equal(cfg$quantiles, (1:9) / 10)
  expect_true(cfg$use_arcsinh)
})
