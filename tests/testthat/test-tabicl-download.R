# Tests for the TabICL weight downloader (R/tabicl-download.R). The network
# helper is mocked, so these exercise the release-asset URLs, the cache layout,
# and the availability check without hitting the network.

test_that("tab_icl_download_weights fetches both tasks into the cache layout", {
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

  dir <- suppressMessages(tab_icl_download_weights(
    version = "v2",
    date = "2026-02-12",
    cache_dir = cache
  ))

  # Cache layout: <cache>/<version>/<date>/<TaskLabel>/.
  expect_match(dir, "v2/2026-02-12$")
  for (label in c("Classification", "Regression")) {
    prefix <- tolower(label)
    task_dir <- file.path(dir, label)
    expect_true(file.exists(file.path(
      task_dir,
      paste0(prefix, ".config.json")
    )))
    expect_true(file.exists(
      file.path(task_dir, paste0(prefix, ".model.safetensors"))
    ))
  }

  # Each file is fetched from its release asset on the tag <version>-<date>.
  base <- "https://github.com/tidymodels/tabicl-weights/releases/download/v2-2026-02-12"
  expect_setequal(
    requested,
    file.path(
      base,
      c(
        "classification.config.json",
        "classification.model.safetensors",
        "regression.config.json",
        "regression.model.safetensors"
      )
    )
  )

  # A download then satisfies the cache lookup for both tasks.
  withr::local_options(brulee.tabicl_cache_dir = cache)
  expect_match(
    normalizePath(suppressMessages(brulee:::tabicl_cache_lookup("regression"))),
    normalizePath(file.path(dir, "Regression")),
    fixed = TRUE
  )
})

test_that("tab_icl_download_weights can fetch a single task", {
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

  suppressMessages(tab_icl_download_weights(
    "classification",
    cache_dir = cache
  ))

  expect_true(tab_icl_weights_available("classification", cache_dir = cache))
  expect_false(tab_icl_weights_available("regression", cache_dir = cache))
  expect_true(all(grepl("classification\\.", requested)))
})

test_that("tab_icl_weights_available reflects the cache state", {
  skip_on_cran()
  cache <- withr::local_tempdir()
  expect_false(tab_icl_weights_available(cache_dir = cache))

  testthat::local_mocked_bindings(
    chronos2_download_file = function(url, dest, label, max_attempts = 3L) {
      writeLines("stub", dest)
      invisible(dest)
    }
  )
  suppressMessages(tab_icl_download_weights(cache_dir = cache))

  expect_true(tab_icl_weights_available(cache_dir = cache))
})

test_that("the tab_icl weight helpers reject unknown tasks", {
  skip_on_cran()
  expect_snapshot(error = TRUE, tab_icl_download_weights("bogus"))
  expect_snapshot(error = TRUE, tab_icl_weights_available("bogus"))
})

test_that("tabicl_cache_lookup errors when uncached and non-interactive", {
  cache <- withr::local_tempdir()
  withr::local_options(brulee.tabicl_cache_dir = cache)
  testthat::local_mocked_bindings(
    is_interactive = function() FALSE,
    .package = "rlang"
  )

  expect_error(
    brulee:::tabicl_cache_lookup("regression"),
    "No cached regression TabICL checkpoint found"
  )
})

test_that("tabicl_cache_lookup aborts when the user declines the prompt", {
  cache <- withr::local_tempdir()
  withr::local_options(brulee.tabicl_cache_dir = cache)
  testthat::local_mocked_bindings(
    is_interactive = function() TRUE,
    .package = "rlang"
  )
  testthat::local_mocked_bindings(
    menu = function(choices, ...) 2L,
    .package = "utils"
  )

  expect_error(
    suppressMessages(brulee:::tabicl_cache_lookup("regression")),
    "Download declined"
  )
})

test_that("tabicl_cache_lookup downloads when the user accepts the prompt", {
  cache <- withr::local_tempdir()
  withr::local_options(brulee.tabicl_cache_dir = cache)
  testthat::local_mocked_bindings(
    is_interactive = function() TRUE,
    .package = "rlang"
  )
  testthat::local_mocked_bindings(
    menu = function(choices, ...) 1L,
    .package = "utils"
  )
  testthat::local_mocked_bindings(
    chronos2_download_file = function(url, dest, label, max_attempts = 3L) {
      writeLines("stub", dest)
      invisible(dest)
    }
  )

  path <- suppressMessages(brulee:::tabicl_cache_lookup("regression"))
  expect_true(dir.exists(path))
  expect_match(path, "Regression", fixed = TRUE)
})
