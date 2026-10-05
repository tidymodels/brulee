# Tests of the infrastructure shared by the pretrained models
# (R/pretrained-weights.R).

# ------------------------------------------------------------------------------
# brulee_confirm_download

test_that("brulee_confirm_download errors when non-interactive", {
  testthat::local_mocked_bindings(
    is_interactive = function() FALSE,
    .package = "rlang"
  )

  root <- withr::local_tempdir()
  expect_snapshot(
    error = TRUE,
    transform = function(x) gsub(root, "<root>", x, fixed = TRUE),
    brulee_confirm_download(
      label = "amazon/chronos-2",
      size = "500MB",
      fn = "brulee_chronos",
      root = root,
      hint = "Run {.fn brulee_chronos} in an interactive session to download them."
    )
  )
})

test_that("brulee_confirm_download aborts when the user declines", {
  testthat::local_mocked_bindings(
    is_interactive = function() TRUE,
    .package = "rlang"
  )
  testthat::local_mocked_bindings(
    menu = function(choices, ...) 2L,
    .package = "utils"
  )

  expect_snapshot(
    error = TRUE,
    brulee_confirm_download(
      label = "amazon/chronos-2",
      size = "500MB",
      fn = "brulee_chronos",
      root = tempdir(),
      hint = "Run {.fn brulee_chronos} in an interactive session to download them."
    )
  )
})

test_that("brulee_confirm_download returns TRUE when the user accepts", {
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
      brulee:::brulee_confirm_download(
        label = "amazon/chronos-2",
        size = "500MB",
        fn = "brulee_chronos",
        root = tempdir(),
        hint = "Run {.fn brulee_chronos} in an interactive session to download them."
      )
    )
  )
})

# ------------------------------------------------------------------------------
# brulee_remote_size

test_that("brulee_remote_size returns the Content-Length when present", {
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

  expect_equal(brulee:::brulee_remote_size("http://x"), 12345)
})

test_that("brulee_remote_size handles lowercase content-length header", {
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

  expect_equal(brulee:::brulee_remote_size("http://x"), 99999)
})

test_that("brulee_remote_size returns NA when the server omits Content-Length", {
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

  expect_true(is.na(brulee:::brulee_remote_size("http://x")))
})

test_that("brulee_remote_size returns NA on a 4xx status", {
  testthat::local_mocked_bindings(
    new_handle = function() list(),
    handle_setopt = function(handle, ...) handle,
    curl_fetch_memory = function(url, handle = NULL) {
      list(status_code = 404L, headers = raw())
    },
    .package = "curl"
  )

  expect_true(is.na(brulee:::brulee_remote_size("http://x")))
})

test_that("brulee_remote_size returns NA when curl errors", {
  testthat::local_mocked_bindings(
    new_handle = function() list(),
    handle_setopt = function(handle, ...) handle,
    curl_fetch_memory = function(url, handle = NULL) stop("boom"),
    .package = "curl"
  )

  expect_true(is.na(brulee:::brulee_remote_size("http://x")))
})

# ------------------------------------------------------------------------------
# brulee_download_file

test_that("brulee_download_file keeps a cached file with matching size", {
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)
  writeBin(as.raw(rep(0, 16)), tmp)

  testthat::local_mocked_bindings(
    brulee_remote_size = function(url) 16,
    .package = "brulee"
  )
  # Would error if invoked
  testthat::local_mocked_bindings(
    curl_download = function(...) stop("should not download"),
    .package = "curl"
  )

  expect_invisible(brulee:::brulee_download_file("http://x", tmp, "test"))
  expect_true(file.exists(tmp))
})

test_that("brulee_download_file keeps any cached file when remote size is NA", {
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)
  writeBin(as.raw(rep(0, 42)), tmp)

  testthat::local_mocked_bindings(
    brulee_remote_size = function(url) NA_real_,
    .package = "brulee"
  )
  testthat::local_mocked_bindings(
    curl_download = function(...) stop("should not be called"),
    .package = "curl"
  )

  expect_invisible(brulee:::brulee_download_file("http://x", tmp, "test"))
  expect_true(file.exists(tmp))
  expect_equal(file.size(tmp), 42)
})

test_that("brulee_download_file redownloads when the cached file is incomplete", {
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)
  writeBin(as.raw(rep(0, 4)), tmp) # only 4 bytes cached

  download_calls <- 0L
  testthat::local_mocked_bindings(
    brulee_remote_size = function(url) 16,
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
    brulee:::brulee_download_file("http://x", tmp, "test")
  )
  expect_equal(download_calls, 1L)
  expect_equal(file.size(tmp), 16)
})

test_that("brulee_download_file downloads when the cache is empty", {
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)
  expect_false(file.exists(tmp))

  testthat::local_mocked_bindings(
    brulee_remote_size = function(url) 16,
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
    brulee:::brulee_download_file("http://x", tmp, "test")
  )
  expect_true(file.exists(tmp))
})

test_that("brulee_download_file succeeds after initial failure", {
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)

  testthat::local_mocked_bindings(
    brulee_remote_size = function(url) 16,
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
    brulee:::brulee_download_file("http://x", tmp, "test", max_attempts = 3L)
  ))
  expect_true(file.exists(tmp))
  expect_equal(file.size(tmp), 16)
  expect_equal(attempt, 2L)
})

test_that("brulee_download_file retries when downloaded file size is wrong", {
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)

  testthat::local_mocked_bindings(
    brulee_remote_size = function(url) 32,
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
    brulee:::brulee_download_file("http://x", tmp, "test", max_attempts = 3L)
  ))
  expect_equal(file.size(tmp), 32)
  expect_equal(attempt, 3L)
})

test_that("brulee_download_file errors after exhausting retries", {
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)

  testthat::local_mocked_bindings(
    brulee_remote_size = function(url) 16,
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
      brulee:::brulee_download_file(
        "http://x",
        tmp,
        "test",
        max_attempts = 2L
      )
    }
  )
  expect_equal(attempts, 2L)
})

test_that("brulee_download_file is happy when HEAD doesn't expose size", {
  skip_if_not_installed("safetensors")
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)

  testthat::local_mocked_bindings(
    brulee_remote_size = function(url) NA_real_,
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
    brulee:::brulee_download_file("http://x", tmp, "test")
  )
  expect_equal(download_calls, 1L)
  expect_true(file.exists(tmp))
})

# ------------------------------------------------------------------------------
# brulee_download_file with a checksum

test_that("brulee_download_file checks the size and checksum", {
  remote <- local_fake_file()
  dest <- file.path(remote$dir, "cache", "model.safetensors")
  brulee_download_file(
    remote$url,
    dest,
    label = "model.safetensors",
    size = remote$size,
    sha256 = remote$sha256
  ) |>
    suppressMessages()
  expect_identical(cli::hash_file_sha256(dest), remote$sha256)
  expect_false(file.exists(paste0(dest, ".part")))
})

test_that("brulee_download_file rejects a download with the wrong checksum", {
  remote <- local_fake_file()
  dest <- file.path(remote$dir, "model.safetensors")
  expect_snapshot(
    suppressMessages(
      brulee_download_file(
        remote$url,
        dest,
        label = "model.safetensors",
        size = remote$size,
        sha256 = strrep("0", 64)
      )
    ),
    error = TRUE
  )
  expect_false(file.exists(dest))
  expect_false(file.exists(paste0(dest, ".part")))
})
