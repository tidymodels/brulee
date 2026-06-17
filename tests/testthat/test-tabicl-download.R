# Tests for the TabICL weight downloader (R/tabicl-download.R). The HF network
# helpers are mocked, so these exercise URL construction, caching layout, and the
# not-yet-hosted guard without hitting the network.

test_that("tabicl_download errors when no repo is configured", {
  skip_on_cran()
  # The default repo is NULL until the converted weights are hosted.
  expect_snapshot(error = TRUE, brulee:::tabicl_download("classifier"))
})

test_that("tabicl_download fetches config + safetensors into a cache dir", {
  skip_on_cran()
  cache <- withr::local_tempdir()
  requested <- character()

  testthat::local_mocked_bindings(
    chronos2_resolve_revision = function(model_id, revision) {
      "0123456789abcdef0123456789abcdef01234567"
    },
    chronos2_download_file = function(url, dest, label, max_attempts = 3L) {
      requested[[length(requested) + 1L]] <<- url
      writeLines("stub", dest)
      invisible(dest)
    }
  )

  dir <- brulee:::tabicl_download(
    "regressor",
    repo_id = "org/repo",
    revision = "main",
    cache_dir = cache
  )

  # Files are written with the task-prefixed names brulee reads.
  expect_true(file.exists(file.path(dir, "regression.config.json")))
  expect_true(file.exists(file.path(dir, "regression.model.safetensors")))
  # Cache layout: <cache>/<repo-slug>/<sha>/<checkpoint>.
  expect_match(dir, "org--repo")
  expect_match(dir, "regressor$")
  # Prefixed files are fetched from the repo root.
  expect_true(any(grepl(
    "huggingface.co/org/repo/resolve/.*/regression.config.json",
    requested
  )))
  expect_true(any(grepl("regression.model.safetensors", requested)))
})

test_that("tabicl_download rejects unknown checkpoints", {
  skip_on_cran()
  expect_error(
    brulee:::tabicl_download("bogus", repo_id = "org/repo"),
    class = "rlang_error"
  )
})
