# Tests for helpers in R/chronos2-misc.R.
#
# Covers: download / version-management / config-parsing helpers, torch
# utility functions (nan_to_num, torch_nanmean, torch_nansum,
# left_pad_and_stack, left_pad_and_cat_2D), preprocessing modules
# (instance_norm, patch, prepare_patched_context, prepare_patched_future),
# layers (layer_norm, rotate_half, apply_rotary_pos_emb, MLP, MHA),
# and the weight loader.

# ------------------------------------------------------------------------------
# chronos2_default_revision

test_that("chronos2_default_revision returns a 40-character hex SHA", {
  sha <- brulee:::chronos2_default_revision()
  expect_type(sha, "character")
  expect_length(sha, 1L)

  expect_match(sha, "^[0-9a-f]{40}$")
})

test_that("chronos2_default_revision returns consistent value across calls", {
  sha1 <- brulee:::chronos2_default_revision()
  sha2 <- brulee:::chronos2_default_revision()
  expect_identical(sha1, sha2)
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
  skip_if_not_installed("jsonlite")
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
  skip_if_not_installed("jsonlite")
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

test_that("chronos2_resolve_revision errors when sha is empty string", {
  skip_if_not_installed("jsonlite")
  testthat::local_mocked_bindings(
    curl_fetch_memory = function(url, handle = NULL) {
      list(
        status_code = 200L,
        content = charToRaw(jsonlite::toJSON(
          list(sha = ""),
          auto_unbox = TRUE
        ))
      )
    },
    .package = "curl"
  )

  expect_snapshot(error = TRUE, {
    brulee:::chronos2_resolve_revision("amazon/chronos-2", "main")
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

test_that("chronos2_remote_size handles lowercase content-length header", {
  testthat::local_mocked_bindings(
    new_handle = function() list(),
    handle_setopt = function(handle, ...) handle,
    curl_fetch_memory = function(url, handle = NULL) {
      list(
        status_code = 200L,
        headers = charToRaw("HTTP/1.1 200 OK\r\ncontent-length: 99999\r\n\r\n")
      )
    },
    parse_headers = function(raw) {
      strsplit(rawToChar(raw), "\r\n")[[1]]
    },
    .package = "curl"
  )

  expect_equal(brulee:::chronos2_remote_size("http://x"), 99999)
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

test_that("chronos2_download_file keeps any cached file when remote size is NA", {
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)
  writeBin(as.raw(rep(0, 42)), tmp)

  testthat::local_mocked_bindings(
    chronos2_remote_size = function(url) NA_real_,
    .package = "brulee"
  )
  testthat::local_mocked_bindings(
    curl_download = function(...) stop("should not be called"),
    .package = "curl"
  )

  expect_invisible(brulee:::chronos2_download_file("http://x", tmp, "test"))
  expect_true(file.exists(tmp))
  expect_equal(file.size(tmp), 42)
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

  suppressMessages(
    brulee:::chronos2_download_file("http://x", tmp, "test")
  )
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

  suppressMessages(
    brulee:::chronos2_download_file("http://x", tmp, "test")
  )
  expect_true(file.exists(tmp))
})

test_that("chronos2_download_file succeeds after initial failure", {
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)

  testthat::local_mocked_bindings(
    chronos2_remote_size = function(url) 16,
    .package = "brulee"
  )

  attempt <- 0L
  testthat::local_mocked_bindings(
    curl_download = function(url, dest, mode, quiet) {
      attempt <<- attempt + 1L
      if (attempt == 1L) {
        stop("transient failure")
      }
      writeBin(as.raw(rep(0, 16)), dest)
      invisible(dest)
    },
    .package = "curl"
  )

  suppressMessages(suppressWarnings(
    brulee:::chronos2_download_file("http://x", tmp, "test", max_attempts = 3L)
  ))
  expect_true(file.exists(tmp))
  expect_equal(file.size(tmp), 16)
  expect_equal(attempt, 2L)
})

test_that("chronos2_download_file retries when downloaded file size is wrong", {
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)

  testthat::local_mocked_bindings(
    chronos2_remote_size = function(url) 32,
    .package = "brulee"
  )

  attempt <- 0L
  testthat::local_mocked_bindings(
    curl_download = function(url, dest, mode, quiet) {
      attempt <<- attempt + 1L
      if (attempt < 3L) {
        writeBin(as.raw(rep(0, 10)), dest)
      } else {
        writeBin(as.raw(rep(0, 32)), dest)
      }
      invisible(dest)
    },
    .package = "curl"
  )

  suppressMessages(suppressWarnings(
    brulee:::chronos2_download_file("http://x", tmp, "test", max_attempts = 3L)
  ))
  expect_equal(file.size(tmp), 32)
  expect_equal(attempt, 3L)
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
  skip_if_not_installed("safetensors")
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

  suppressMessages(
    brulee:::chronos2_download_file("http://x", tmp, "test")
  )
  expect_equal(download_calls, 1L)
  expect_true(file.exists(tmp))
})

# ------------------------------------------------------------------------------
# chronos2_download

test_that("chronos2_download resolves the revision and downloads the files", {
  skip_if_not_installed("jsonlite")
  skip_if_not_installed("safetensors")
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
    chronos2_confirm_download = function(model_id, cache_dir, call = NULL) {
      invisible(TRUE)
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

test_that("chronos2_download errors when uncached and non-interactive", {
  cache <- file.path(tempdir(), paste0("chr-cache-", as.integer(Sys.time())))
  on.exit(unlink(cache, recursive = TRUE), add = TRUE)

  testthat::local_mocked_bindings(
    is_interactive = function() FALSE,
    .package = "rlang"
  )

  expect_error(
    brulee:::chronos2_download(cache_dir = cache),
    "No cached .* Chronos weights found"
  )
})

test_that("chronos2_confirm_download aborts when the user declines", {
  testthat::local_mocked_bindings(
    is_interactive = function() TRUE,
    .package = "rlang"
  )
  testthat::local_mocked_bindings(
    menu = function(choices, ...) 2L,
    .package = "utils"
  )

  expect_error(
    suppressMessages(
      brulee:::chronos2_confirm_download("amazon/chronos-2", tempdir())
    ),
    "Download declined"
  )
})

test_that("chronos2_confirm_download returns TRUE when the user accepts", {
  testthat::local_mocked_bindings(
    is_interactive = function() TRUE,
    .package = "rlang"
  )
  testthat::local_mocked_bindings(
    menu = function(choices, ...) 1L,
    .package = "utils"
  )

  expect_true(
    suppressMessages(
      brulee:::chronos2_confirm_download("amazon/chronos-2", tempdir())
    )
  )
})

# ------------------------------------------------------------------------------
# chronos2_parse_config

test_that("chronos2_parse_config flattens the JSON layout into a list", {
  skip_if_not_installed("jsonlite")
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

test_that("chronos2_parse_config handles different field values", {
  skip_if_not(torch::torch_is_installed())
  skip_if_not_installed("jsonlite")
  cfg_path <- tempfile(fileext = ".json")
  on.exit(unlink(cfg_path), add = TRUE)

  json <- list(
    d_model = 256L,
    d_ff = 512L,
    d_kv = 32L,
    num_heads = 8L,
    num_layers = 6L,
    dropout_rate = 0.0,
    layer_norm_epsilon = 1e-8,
    rope_theta = 5000,
    vocab_size = 2L,
    pad_token_id = 0L,
    reg_token_id = 1L,
    chronos_config = list(
      context_length = 512L,
      input_patch_size = 8L,
      input_patch_stride = 8L,
      output_patch_size = 8L,
      max_output_patches = 32L,
      quantiles = c(0.1, 0.5, 0.9),
      use_arcsinh = FALSE,
      use_reg_token = FALSE,
      time_encoding_scale = 2.0
    )
  )
  writeLines(jsonlite::toJSON(json, auto_unbox = TRUE), cfg_path)

  cfg <- brulee:::chronos2_parse_config(cfg_path)

  expect_equal(cfg$d_model, 256L)
  expect_equal(cfg$d_kv, 32L)
  expect_equal(cfg$quantiles, c(0.1, 0.5, 0.9))
  expect_false(cfg$use_arcsinh)
  expect_false(cfg$use_reg_token)
  expect_equal(cfg$time_encoding_scale, 2.0)
})

# ==============================================================================
# Torch utility functions (require torch)
# ==============================================================================

test_that("nan_to_num replaces NaN with the specified value", {
  skip_if_not(torch::torch_is_installed())

  x <- torch::torch_tensor(c(1.0, NaN, 3.0, NaN))
  result <- brulee:::nan_to_num(x, nan = 0.0)
  expect_equal(as.numeric(result), c(1, 0, 3, 0))

  result2 <- brulee:::nan_to_num(x, nan = -1.0)
  expect_equal(as.numeric(result2), c(1, -1, 3, -1))
})

test_that("torch_nanmean computes mean ignoring NaN", {
  skip_if_not(torch::torch_is_installed())

  x <- torch::torch_tensor(matrix(
    c(1, NaN, 3, 4, NaN, 6),
    nrow = 2,
    byrow = TRUE
  ))
  result <- brulee:::torch_nanmean(x, dim = -1L)
  expect_equal(as.numeric(result), c(2.0, 5.0), tolerance = 1e-5)
})

test_that("torch_nanmean handles all-NaN rows", {
  skip_if_not(torch::torch_is_installed())

  x <- torch::torch_tensor(matrix(c(NaN, NaN, 1, 2), nrow = 2, byrow = TRUE))
  result <- brulee:::torch_nanmean(x, dim = -1L)
  # All-NaN row gets 0 from nan_to_num, divided by clamped count of 1
  expect_equal(as.numeric(result)[1], 0.0)
  expect_equal(as.numeric(result)[2], 1.5, tolerance = 1e-5)
})

test_that("torch_nansum sums ignoring NaN", {
  skip_if_not(torch::torch_is_installed())

  x <- torch::torch_tensor(matrix(
    c(1, NaN, 3, 4, 5, NaN),
    nrow = 2,
    byrow = TRUE
  ))
  result <- brulee:::torch_nansum(x, dim = -1L)
  expect_equal(as.numeric(result), c(4.0, 9.0))
})

test_that("left_pad_and_stack pads shorter tensors with NaN on the left", {
  skip_if_not(torch::torch_is_installed())

  t1 <- torch::torch_tensor(c(1, 2, 3))
  t2 <- torch::torch_tensor(c(4, 5))

  result <- brulee:::left_pad_and_stack(list(t1, t2))

  expect_equal(result$size(), c(2L, 3L))
  expect_equal(as.numeric(result[1, ]), c(1, 2, 3))
  # t2 is padded: [NaN, 4, 5]
  expect_true(is.nan(as.numeric(result[2, 1])))
  expect_equal(as.numeric(result[2, 2:3]), c(4, 5))
})

test_that("left_pad_and_stack works when all tensors have same length", {
  skip_if_not(torch::torch_is_installed())

  t1 <- torch::torch_tensor(c(1, 2))
  t2 <- torch::torch_tensor(c(3, 4))

  result <- brulee:::left_pad_and_stack(list(t1, t2))
  expect_equal(result$size(), c(2L, 2L))
  expect_equal(as.numeric(result[1, ]), c(1, 2))
  expect_equal(as.numeric(result[2, ]), c(3, 4))
})

test_that("left_pad_and_cat_2D pads shorter 2D tensors with NaN on the left", {
  skip_if_not(torch::torch_is_installed())

  t1 <- torch::torch_tensor(matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, byrow = TRUE))
  t2 <- torch::torch_tensor(matrix(c(7, 8), nrow = 1))

  result <- brulee:::left_pad_and_cat_2D(list(t1, t2))

  expect_equal(result$size(), c(3L, 3L))
  expect_equal(as.numeric(result[1, ]), c(1, 2, 3))
  expect_equal(as.numeric(result[2, ]), c(4, 5, 6))
  # t2 row is padded: [NaN, 7, 8]
  expect_true(is.nan(as.numeric(result[3, 1])))
  expect_equal(as.numeric(result[3, 2:3]), c(7, 8))
})

test_that("left_pad_and_cat_2D works when all same width", {
  skip_if_not(torch::torch_is_installed())

  t1 <- torch::torch_tensor(matrix(1:4, nrow = 2, byrow = TRUE))
  t2 <- torch::torch_tensor(matrix(5:6, nrow = 1))

  result <- brulee:::left_pad_and_cat_2D(list(t1, t2))
  expect_equal(result$size(), c(3L, 2L))
})

# ------------------------------------------------------------------------------
# Preprocessing modules

test_that("chronos2_instance_norm normalizes and inverts correctly", {
  skip_if_not(torch::torch_is_installed())

  norm <- brulee:::chronos2_instance_norm(use_arcsinh = FALSE)
  x <- torch::torch_tensor(matrix(
    c(2, 4, 6, 8, 10, 12),
    nrow = 2,
    byrow = TRUE
  ))

  result <- norm(x)
  scaled <- result[[1]]
  loc_scale <- result[[2]]

  # Mean should be approximately zero after normalization
  means <- as.numeric(scaled$mean(dim = -1L))
  expect_true(all(abs(means) < 0.1))

  # Inverse should recover original
  recovered <- norm$inverse(scaled, loc_scale)
  expect_equal(as.numeric(recovered), as.numeric(x), tolerance = 1e-4)
})

test_that("chronos2_instance_norm works with arcsinh transform", {
  skip_if_not(torch::torch_is_installed())

  norm <- brulee:::chronos2_instance_norm(use_arcsinh = TRUE)
  x <- torch::torch_tensor(matrix(
    c(1, 10, 100, 2, 20, 200),
    nrow = 2,
    byrow = TRUE
  ))

  result <- norm(x)
  scaled <- result[[1]]
  loc_scale <- result[[2]]

  recovered <- norm$inverse(scaled, loc_scale)
  expect_equal(as.numeric(recovered), as.numeric(x), tolerance = 1e-3)
})

test_that("chronos2_patch splits and pads correctly", {
  skip_if_not(torch::torch_is_installed())

  patch <- brulee:::chronos2_patch(patch_size = 4L, patch_stride = 4L)

  # Input of length 8: should yield 2 patches of size 4
  x <- torch::torch_tensor(
    matrix(as.double(1:8), nrow = 1),
    dtype = torch::torch_float32()
  )
  result <- patch(x)
  expect_equal(result$size(), c(1L, 2L, 4L))

  # Input of length 12: should yield 3 patches of size 4 (no padding needed)
  x2 <- torch::torch_tensor(
    matrix(as.double(1:12), nrow = 1),
    dtype = torch::torch_float32()
  )
  result2 <- patch(x2)
  expect_equal(result2$size(), c(1L, 3L, 4L))
})

test_that("chronos2_layer_norm produces stable output", {
  skip_if_not(torch::torch_is_installed())

  ln <- brulee:::chronos2_layer_norm(hidden_size = 8L)
  x <- torch::torch_randn(2, 4, 8)

  result <- ln(x)
  expect_equal(result$size(), c(2L, 4L, 8L))
  # Output should not be all zeros
  expect_true(as.numeric(result$abs()$sum()) > 0)
})

test_that("rotate_half rotates tensor halves correctly", {
  skip_if_not(torch::torch_is_installed())

  x <- torch::torch_tensor(array(
    c(1, 2, 3, 4, 5, 6, 7, 8),
    dim = c(1, 1, 1, 8)
  ))
  result <- brulee:::rotate_half(x)

  vals <- as.numeric(result)
  # First half should be negated second half of input
  expect_equal(vals[1:4], c(-5, -6, -7, -8))
  # Second half should be first half of input
  expect_equal(vals[5:8], c(1, 2, 3, 4))
})

test_that("apply_rotary_pos_emb returns correctly shaped tensors", {
  skip_if_not(torch::torch_is_installed())

  batch <- 2L
  n_heads <- 4L
  seq_len <- 8L
  head_dim <- 16L

  q <- torch::torch_randn(batch, n_heads, seq_len, head_dim)
  k <- torch::torch_randn(batch, n_heads, seq_len, head_dim)
  cos_val <- torch::torch_randn(batch, seq_len, head_dim)
  sin_val <- torch::torch_randn(batch, seq_len, head_dim)

  result <- brulee:::apply_rotary_pos_emb(q, k, cos_val, sin_val)

  expect_length(result, 2L)
  expect_equal(result[[1]]$size(), c(batch, n_heads, seq_len, head_dim))
  expect_equal(result[[2]]$size(), c(batch, n_heads, seq_len, head_dim))
})

# ------------------------------------------------------------------------------
# prepare_patched_context and prepare_patched_future

test_that("prepare_patched_context truncates when context exceeds context_length", {
  skip_if_not(torch::torch_is_installed())

  config <- list(
    context_length = 8L,
    input_patch_size = 4L,
    input_patch_stride = 4L,
    time_encoding_scale = 1.0
  )

  norm <- brulee:::chronos2_instance_norm(use_arcsinh = FALSE)
  patch <- brulee:::chronos2_patch(patch_size = 4L, patch_stride = 4L)

  # Context of length 16 > context_length of 8
  context <- torch::torch_randn(1, 16)
  context_mask <- torch::torch_ones(1, 16)

  result <- brulee:::prepare_patched_context(
    context,
    context_mask,
    norm,
    patch,
    config
  )

  # After truncation to 8, with patch_size 4 we get 2 patches
  expect_equal(result$num_patches, 2L)
  expect_equal(result$patched_context$size(2), 2L)
})

test_that("prepare_patched_context returns expected structure", {
  skip_if_not(torch::torch_is_installed())

  config <- list(
    context_length = 32L,
    input_patch_size = 4L,
    input_patch_stride = 4L,
    time_encoding_scale = 1.0
  )

  norm <- brulee:::chronos2_instance_norm(use_arcsinh = FALSE)
  patch <- brulee:::chronos2_patch(patch_size = 4L, patch_stride = 4L)

  context <- torch::torch_randn(2, 16)
  context_mask <- torch::torch_ones(2, 16)

  result <- brulee:::prepare_patched_context(
    context,
    context_mask,
    norm,
    patch,
    config
  )

  expect_true("patched_context" %in% names(result))
  expect_true("attention_mask" %in% names(result))
  expect_true("loc_scale" %in% names(result))
  expect_true("num_patches" %in% names(result))
  # patched_context has shape [batch, num_patches, patch_size*3]
  expect_equal(result$patched_context$size(1), 2L)
  expect_equal(result$patched_context$size(3), 4L * 3L)
})

test_that("prepare_patched_future returns expected structure without covariates", {
  skip_if_not(torch::torch_is_installed())

  config <- list(
    output_patch_size = 4L,
    time_encoding_scale = 1.0
  )

  result <- brulee:::prepare_patched_future(
    num_output_patches = 2L,
    config = config,
    batch_size = 3L,
    device = torch::torch_device("cpu"),
    dtype = torch::torch_float32()
  )

  expect_true("patched_future" %in% names(result))
  # Shape: [batch, num_output_patches, patch_size * 3]
  expect_equal(result$patched_future$size(), c(3L, 2L, 4L * 3L))
})

test_that("prepare_patched_future pads short future covariates", {
  skip_if_not(torch::torch_is_installed())

  config <- list(
    output_patch_size = 4L,
    time_encoding_scale = 1.0
  )

  norm <- brulee:::chronos2_instance_norm(use_arcsinh = FALSE)
  # Establish loc/scale
  dummy <- torch::torch_tensor(matrix(rnorm(9), nrow = 3))
  norm_result <- norm(dummy)
  loc_scale <- norm_result[[2]]

  # Future covariates of length 4, but num_output_patches=2 => final_future_length=8
  # So covariates need to be padded from 4 to 8
  future_covariates <- torch::torch_randn(3, 4)

  result <- brulee:::prepare_patched_future(
    num_output_patches = 2L,
    config = config,
    batch_size = 3L,
    device = torch::torch_device("cpu"),
    dtype = torch::torch_float32(),
    future_covariates = future_covariates,
    loc_scale = loc_scale,
    instance_norm = norm
  )

  expect_equal(result$patched_future$size(), c(3L, 2L, 4L * 3L))
})

test_that("prepare_patched_future accepts explicit future_covariates_mask", {
  skip_if_not(torch::torch_is_installed())

  config <- list(
    output_patch_size = 4L,
    time_encoding_scale = 1.0
  )

  norm <- brulee:::chronos2_instance_norm(use_arcsinh = FALSE)
  dummy <- torch::torch_tensor(matrix(rnorm(12), nrow = 3))
  norm_result <- norm(dummy)
  loc_scale <- norm_result[[2]]

  future_covariates <- torch::torch_randn(3, 8)
  mask <- torch::torch_ones(3, 8)

  result <- brulee:::prepare_patched_future(
    num_output_patches = 2L,
    config = config,
    batch_size = 3L,
    device = torch::torch_device("cpu"),
    dtype = torch::torch_float32(),
    future_covariates = future_covariates,
    future_covariates_mask = mask,
    loc_scale = loc_scale,
    instance_norm = norm
  )

  expect_equal(result$patched_future$size(), c(3L, 2L, 4L * 3L))
})

test_that("prepare_patched_future handles future covariates", {
  skip_if_not(torch::torch_is_installed())

  config <- list(
    output_patch_size = 4L,
    time_encoding_scale = 1.0
  )

  norm <- brulee:::chronos2_instance_norm(use_arcsinh = FALSE)
  # Need to run a forward pass first to establish loc/scale
  dummy <- torch::torch_tensor(matrix(rnorm(12), nrow = 3))
  norm_result <- norm(dummy)
  loc_scale <- norm_result[[2]]

  future_covariates <- torch::torch_randn(3, 8)

  result <- brulee:::prepare_patched_future(
    num_output_patches = 2L,
    config = config,
    batch_size = 3L,
    device = torch::torch_device("cpu"),
    dtype = torch::torch_float32(),
    future_covariates = future_covariates,
    loc_scale = loc_scale,
    instance_norm = norm
  )

  expect_equal(result$patched_future$size(), c(3L, 2L, 4L * 3L))
})

# ------------------------------------------------------------------------------
# Full model (tiny) forward pass

test_that("chronos2_model forward produces correctly shaped output", {
  skip_if_not(torch::torch_is_installed())

  config <- list(
    d_model = 32L,
    d_ff = 64L,
    d_kv = 16L,
    num_heads = 2L,
    num_layers = 1L,
    dropout_rate = 0.0,
    layer_norm_epsilon = 1e-6,
    rope_theta = 10000,
    vocab_size = 2L,
    pad_token_id = 0L,
    reg_token_id = 1L,
    context_length = 64L,
    input_patch_size = 4L,
    input_patch_stride = 4L,
    output_patch_size = 4L,
    max_output_patches = 4L,
    quantiles = c(0.1, 0.5, 0.9),
    use_arcsinh = FALSE,
    use_reg_token = TRUE,
    time_encoding_scale = 1.0
  )

  model <- brulee:::chronos2_model(config)
  model$eval()

  context <- torch::torch_randn(2, 20)

  torch::with_no_grad({
    output <- model(context, num_output_patches = 2L)
  })

  # Output shape: [batch, num_quantiles, num_output_patches * output_patch_size]
  expect_equal(output$size(), c(2L, 3L, 8L))
})

test_that("chronos2_model encode produces encoder hidden states", {
  skip_if_not(torch::torch_is_installed())

  config <- list(
    d_model = 32L,
    d_ff = 64L,
    d_kv = 16L,
    num_heads = 2L,
    num_layers = 1L,
    dropout_rate = 0.0,
    layer_norm_epsilon = 1e-6,
    rope_theta = 10000,
    vocab_size = 2L,
    pad_token_id = 0L,
    reg_token_id = 1L,
    context_length = 64L,
    input_patch_size = 4L,
    input_patch_stride = 4L,
    output_patch_size = 4L,
    max_output_patches = 4L,
    quantiles = c(0.1, 0.5, 0.9),
    use_arcsinh = FALSE,
    use_reg_token = TRUE,
    time_encoding_scale = 1.0
  )

  model <- brulee:::chronos2_model(config)
  model$eval()

  context <- torch::torch_randn(1, 16)

  torch::with_no_grad({
    enc <- model$encode(context, num_output_patches = 1L)
  })

  expect_true("hidden_states" %in% names(enc))
  expect_true("loc_scale" %in% names(enc))
  expect_equal(enc$hidden_states$size(1), 1L)
  expect_equal(enc$hidden_states$size(3), 32L)
})

# ------------------------------------------------------------------------------
# load_chronos2_weights (smoke test with tiny model)

test_that("load_chronos2_weights warns on shape mismatch and skips unknown keys", {
  skip_if_not(torch::torch_is_installed())
  skip_if_not_installed("safetensors")

  config <- list(
    d_model = 32L,
    d_ff = 64L,
    d_kv = 16L,
    num_heads = 2L,
    num_layers = 1L,
    dropout_rate = 0.0,
    layer_norm_epsilon = 1e-6,
    rope_theta = 10000,
    vocab_size = 2L,
    pad_token_id = 0L,
    reg_token_id = 1L,
    context_length = 64L,
    input_patch_size = 4L,
    input_patch_stride = 4L,
    output_patch_size = 4L,
    max_output_patches = 4L,
    quantiles = c(0.1, 0.5, 0.9),
    use_arcsinh = FALSE,
    use_reg_token = TRUE,
    time_encoding_scale = 1.0
  )

  model <- brulee:::chronos2_model(config)

  st_path <- tempfile(fileext = ".safetensors")
  on.exit(unlink(st_path), add = TRUE)

  tensors <- list()
  # Wrong shape for shared.weight (should be [2, 32] but we give [5, 5])
  tensors[["shared.weight"]] <- torch::torch_randn(5, 5)
  # Unknown key (not rope_embed.inv_freq, so it gets tracked as skipped)
  tensors[["some_unknown_key.weight"]] <- torch::torch_randn(10, 10)
  # A rope inv_freq key should be silently ignored (not tracked as skipped)
  tensors[[
    "encoder.block.0.layer.0.self_attention.rope_embed.inv_freq"
  ]] <- torch::torch_randn(8)

  safetensors::safe_save_file(tensors, st_path)

  expect_warning(
    brulee:::load_chronos2_weights(model, st_path),
    "Shape mismatch"
  )
})

test_that("load_chronos2_weights assigns parameters from tensor dict", {
  skip_if_not(torch::torch_is_installed())
  skip_if_not_installed("safetensors")

  config <- list(
    d_model = 32L,
    d_ff = 64L,
    d_kv = 16L,
    num_heads = 2L,
    num_layers = 1L,
    dropout_rate = 0.0,
    layer_norm_epsilon = 1e-6,
    rope_theta = 10000,
    vocab_size = 2L,
    pad_token_id = 0L,
    reg_token_id = 1L,
    context_length = 64L,
    input_patch_size = 4L,
    input_patch_stride = 4L,
    output_patch_size = 4L,
    max_output_patches = 4L,
    quantiles = c(0.1, 0.5, 0.9),
    use_arcsinh = FALSE,
    use_reg_token = TRUE,
    time_encoding_scale = 1.0
  )

  model <- brulee:::chronos2_model(config)

  # Build a fake safetensors file with correctly shaped tensors
  st_path <- tempfile(fileext = ".safetensors")
  on.exit(unlink(st_path), add = TRUE)

  # Collect model parameters and create matching tensors
  tensors <- list()
  tensors[["shared.weight"]] <- torch::torch_randn(2, 32)
  tensors[["input_patch_embedding.hidden_layer.weight"]] <- torch::torch_randn(
    64,
    12
  )
  tensors[["input_patch_embedding.hidden_layer.bias"]] <- torch::torch_randn(64)
  tensors[["input_patch_embedding.output_layer.weight"]] <- torch::torch_randn(
    32,
    64
  )
  tensors[["input_patch_embedding.output_layer.bias"]] <- torch::torch_randn(32)
  tensors[[
    "input_patch_embedding.residual_layer.weight"
  ]] <- torch::torch_randn(32, 12)
  tensors[["input_patch_embedding.residual_layer.bias"]] <- torch::torch_randn(
    32
  )
  tensors[["output_patch_embedding.hidden_layer.weight"]] <- torch::torch_randn(
    64,
    32
  )
  tensors[["output_patch_embedding.hidden_layer.bias"]] <- torch::torch_randn(
    64
  )
  tensors[["output_patch_embedding.output_layer.weight"]] <- torch::torch_randn(
    12,
    64
  )
  tensors[["output_patch_embedding.output_layer.bias"]] <- torch::torch_randn(
    12
  )
  tensors[[
    "output_patch_embedding.residual_layer.weight"
  ]] <- torch::torch_randn(12, 32)
  tensors[["output_patch_embedding.residual_layer.bias"]] <- torch::torch_randn(
    12
  )
  tensors[["encoder.final_layer_norm.weight"]] <- torch::torch_randn(32)
  tensors[["encoder.block.0.layer.0.layer_norm.weight"]] <- torch::torch_randn(
    32
  )
  tensors[[
    "encoder.block.0.layer.0.self_attention.q.weight"
  ]] <- torch::torch_randn(32, 32)
  tensors[[
    "encoder.block.0.layer.0.self_attention.k.weight"
  ]] <- torch::torch_randn(32, 32)
  tensors[[
    "encoder.block.0.layer.0.self_attention.v.weight"
  ]] <- torch::torch_randn(32, 32)
  tensors[[
    "encoder.block.0.layer.0.self_attention.o.weight"
  ]] <- torch::torch_randn(32, 32)
  tensors[["encoder.block.0.layer.1.layer_norm.weight"]] <- torch::torch_randn(
    32
  )
  tensors[[
    "encoder.block.0.layer.1.self_attention.q.weight"
  ]] <- torch::torch_randn(32, 32)
  tensors[[
    "encoder.block.0.layer.1.self_attention.k.weight"
  ]] <- torch::torch_randn(32, 32)
  tensors[[
    "encoder.block.0.layer.1.self_attention.v.weight"
  ]] <- torch::torch_randn(32, 32)
  tensors[[
    "encoder.block.0.layer.1.self_attention.o.weight"
  ]] <- torch::torch_randn(32, 32)
  tensors[["encoder.block.0.layer.2.layer_norm.weight"]] <- torch::torch_randn(
    32
  )
  tensors[["encoder.block.0.layer.2.mlp.wi.weight"]] <- torch::torch_randn(
    64,
    32
  )
  tensors[["encoder.block.0.layer.2.mlp.wo.weight"]] <- torch::torch_randn(
    32,
    64
  )

  safetensors::safe_save_file(tensors, st_path)

  # Store original shared weight for comparison
  orig_shared <- as.numeric(model$shared$weight)

  brulee:::load_chronos2_weights(model, st_path)

  # Verify the weight was changed
  new_shared <- as.numeric(model$shared$weight)
  expect_false(identical(orig_shared, new_shared))
  expect_equal(
    new_shared,
    as.numeric(tensors[["shared.weight"]]),
    tolerance = 1e-6
  )
})
