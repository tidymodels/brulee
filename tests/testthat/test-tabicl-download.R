# Tests for the TabICL weight downloader (R/tabicl-download.R). The network
# helper is mocked, so these exercise URL construction, the cache layout, and the
# no-URL guard without hitting the network.

test_that("tabicl_download errors when no URL is configured", {
  skip_on_cran()
  # The default base URL is NULL until a host is chosen.
  expect_snapshot(error = TRUE, brulee:::tabicl_download("classification"))
})

test_that("tabicl_download fetches the task files into the cache layout", {
  skip_on_cran()
  cache <- withr::local_tempdir()
  requested <- character()

  testthat::local_mocked_bindings(
    chronos2_download_file = function(url, dest, label, max_attempts = 3L) {
      requested[[length(requested) + 1L]] <<- url
      writeLines("stub", dest)
      invisible(dest)
    }
  )

  dir <- brulee:::tabicl_download(
    "regression",
    base_url = "https://example.com/tabicl",
    version = "v2",
    date = "2026-02-12",
    cache_dir = cache
  )

  # Cache layout: <cache>/<version>/<date>/<TaskLabel>/.
  expect_match(dir, "v2/2026-02-12/Regression$")
  # Files are written with the task-prefixed names brulee reads.
  expect_true(file.exists(file.path(dir, "regression.config.json")))
  expect_true(file.exists(file.path(dir, "regression.model.safetensors")))
  # URLs mirror the layout under the base URL.
  expect_true(any(grepl(
    "https://example.com/tabicl/v2/2026-02-12/Regression/regression.config.json",
    requested,
    fixed = TRUE
  )))
  expect_true(any(grepl("regression.model.safetensors", requested)))

  # A download then satisfies the cache lookup.
  withr::local_options(brulee.tabicl_cache_dir = cache)
  expect_equal(
    normalizePath(brulee:::tabicl_cache_lookup("regression")),
    normalizePath(dir)
  )
})

test_that("tabicl_download rejects unknown tasks", {
  skip_on_cran()
  expect_error(
    brulee:::tabicl_download("bogus", base_url = "https://example.com"),
    class = "rlang_error"
  )
})
