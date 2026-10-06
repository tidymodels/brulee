test_that("versions come from the registry", {
  expect_identical(tab_pfn_versions(), c("v3", "v3.5", "v3.5-fast"))
  expect_identical(tabpfn_registry()$default_version, "v3.5")
})

test_that("model versions are normalized and checked", {
  expect_null(tabpfn_normalize_version(NULL))
  expect_identical(tabpfn_normalize_version(3.5), "v3.5")
  expect_identical(tabpfn_normalize_version("3"), "v3")
  expect_identical(tabpfn_normalize_version("v3.5-fast"), "v3.5-fast")

  expect_identical(tabpfn_resolve_version(NULL), "v3.5")
  expect_identical(tabpfn_resolve_version(3), "v3")
  expect_snapshot(tabpfn_resolve_version(2.5), error = TRUE)
  expect_snapshot(tabpfn_resolve_version(c("v3", "v3.5")), error = TRUE)
})

test_that("version info names the checkpoint and its architecture", {
  info <- tabpfn_version_info("v3.5-fast", "regression")
  expect_identical(info$file, "tabpfn-v3.5-fast-20260909.safetensors")
  expect_identical(info$architecture, "tabpfn_v3_5")
  expect_identical(info$license_repo, "tabpfn_3_5")
  expect_identical(info$format, "safetensors")

  info <- tabpfn_version_info("v3", "classification")
  expect_identical(info$file, "tabpfn-v3-classifier-v3_default.ckpt")
  expect_identical(info$architecture, "tabpfn_v3")
  expect_identical(info$format, "ckpt")

  expect_snapshot(
    tabpfn_version_info(
      "v3",
      "regression",
      "tabpfn-v3-classifier-v3_default.ckpt"
    ),
    error = TRUE
  )
})

test_that("every registry file has a checksum and size", {
  files <- tabpfn_registry()$files
  expect_all_true(vapply(files, function(x) nchar(x$sha256) == 64, logical(1)))
  expect_all_true(vapply(files, function(x) x$size > 0, logical(1)))
})

test_that("the cache directory is the per-user cache unless set", {
  withr::local_options(brulee.tabpfn_cache_dir = NULL)
  expect_identical(
    tabpfn_cache_dir(),
    file.path(tools::R_user_dir("brulee", which = "cache"), "tabpfn")
  )
  withr::local_options(brulee.tabpfn_cache_dir = "~/some/dir")
  expect_identical(tabpfn_cache_dir(), path.expand("~/some/dir"))

  withr::local_envvar(TABPFN_MODEL_CACHE_DIR = "/tmp/tabpfn-models")
  expect_identical(tabpfn_python_cache_dir(), "/tmp/tabpfn-models")
})

fake_info <- function(remote) {
  list(
    version = "v3.5",
    task = "classification",
    repo_id = "Prior-Labs/tabpfn_3_5",
    license_repo = "tabpfn_3_5",
    architecture = "tabpfn_v3_5",
    file = "model.safetensors",
    sha256 = remote$sha256,
    size = remote$size,
    format = "safetensors"
  )
}

test_that("cached checkpoints are checked, and missing ones need consent", {
  remote <- local_fake_file()
  cache <- withr::local_tempdir()
  withr::local_options(
    brulee.tabpfn_cache_dir = cache,
    rlang_interactive = FALSE
  )
  withr::local_envvar(TABPFN_MODEL_CACHE_DIR = withr::local_tempdir())
  info <- fake_info(remote)

  expect_snapshot(
    tabpfn_checkpoint_path(info),
    error = TRUE,
    transform = function(x) {
      gsub(cache, "<cache>", x, fixed = TRUE)
    }
  )

  file.copy(sub("^file://", "", remote$url), file.path(cache, info$file))
  expect_identical(tabpfn_checkpoint_path(info), file.path(cache, info$file))

  bad_size <- info
  bad_size$size <- 10
  expect_snapshot(
    tabpfn_checkpoint_path(bad_size),
    error = TRUE,
    transform = function(x) {
      gsub(cache, "<cache>", x, fixed = TRUE)
    }
  )

  bad_sha <- info
  bad_sha$sha256 <- strrep("0", 64)
  tabpfn_env$verified <- character()
  expect_snapshot(
    tabpfn_checkpoint_path(bad_sha),
    error = TRUE,
    transform = function(x) {
      gsub(cache, "<cache>", x, fixed = TRUE)
    }
  )
})

test_that("downloading checks the license first", {
  remote <- local_fake_file()
  cache <- withr::local_tempdir()
  withr::local_options(brulee.tabpfn_cache_dir = cache)
  withr::local_envvar(TABPFN_MODEL_CACHE_DIR = withr::local_tempdir())
  info <- fake_info(remote)
  calls <- character()
  local_mocked_bindings(
    tabpfn_ensure_license = function(license_repo, ...) {
      calls <<- c(calls, paste("license", license_repo))
    },
    tabpfn_hf_url = function(repo_id, file) remote$url,
    tabpfn_count_download = function(repo_id) calls <<- c(calls, "count")
  )
  expect_snapshot(
    path <- tabpfn_checkpoint_path(info, ask = FALSE),
    transform = function(x) {
      x <- gsub(cache, "<cache>", x, fixed = TRUE)
      x <- gsub("<file://.*$", "<remote>", x)
      gsub("\\[[0-9.]+m?s\\]", "[<time>]", x)
    }
  )
  expect_identical(path, file.path(cache, info$file))
  expect_identical(calls, c("license tabpfn_3_5", "count"))
})

test_that("weights cached by Python are used in place", {
  remote <- local_fake_file()
  python_cache <- withr::local_tempdir()
  withr::local_envvar(TABPFN_MODEL_CACHE_DIR = python_cache)
  cache <- withr::local_tempdir()
  info <- fake_info(remote)
  file.copy(sub("^file://", "", remote$url), file.path(python_cache, info$file))
  tabpfn_env$verified <- character()
  expect_snapshot(
    path <- tabpfn_checkpoint_path(info, cache_dir = cache),
    transform = function(x) {
      gsub(python_cache, "<python-cache>", x, fixed = TRUE)
    }
  )
  expect_identical(path, file.path(python_cache, info$file))
  expect_length(list.files(cache), 0)
})

test_that("cached weights are found in either cache", {
  remote <- local_fake_file()
  python_cache <- withr::local_tempdir()
  withr::local_envvar(TABPFN_MODEL_CACHE_DIR = python_cache)
  cache <- withr::local_tempdir()
  info <- fake_info(remote)
  local_mocked_bindings(tabpfn_checkpoint_infos = function(...) list(info))
  src <- sub("^file://", "", remote$url)

  expect_false(tab_pfn_weights_available("v3.5", cache_dir = cache))
  file.copy(src, file.path(python_cache, info$file))
  expect_true(tab_pfn_weights_available("v3.5", cache_dir = cache))
  unlink(file.path(python_cache, info$file))
  file.copy(src, file.path(cache, info$file))
  expect_true(tab_pfn_weights_available("v3.5", cache_dir = cache))
  # A partial download doesn't count.
  writeBin(as.raw(1:10), file.path(cache, info$file))
  expect_false(tab_pfn_weights_available("v3.5", cache_dir = cache))
})

test_that("checking for weights needs a known version and task", {
  expect_snapshot(error = TRUE, tab_pfn_weights_available("v2"))
  expect_snapshot(
    error = TRUE,
    tab_pfn_weights_available("v3.5", task = "survival")
  )
})

test_that("the cache can be cleared, by version", {
  cache <- withr::local_tempdir()
  files <- c(
    "tabpfn-v3.5-20260909.safetensors",
    "tabpfn-v3-classifier-v3_default.ckpt",
    "tabpfn-v3-regressor-v3_default.ckpt.part",
    "unrelated.txt"
  )
  for (f in files) {
    writeBin(as.raw(1:10), file.path(cache, f))
  }
  expect_snapshot(
    deleted <- tab_pfn_clear_cache("v3", cache_dir = cache, ask = FALSE),
    transform = function(x) gsub(cache, "<cache>", x, fixed = TRUE)
  )
  expect_setequal(basename(deleted), files[2:3])
  expect_setequal(list.files(cache), files[c(1, 4)])

  expect_snapshot(
    tab_pfn_clear_cache(cache_dir = cache, ask = FALSE),
    transform = function(x) gsub(cache, "<cache>", x, fixed = TRUE)
  )
  expect_identical(list.files(cache), "unrelated.txt")
  expect_snapshot(
    tab_pfn_clear_cache(cache_dir = cache, ask = FALSE),
    transform = function(x) gsub(cache, "<cache>", x, fixed = TRUE)
  )
})

test_that("clearing the cache asks first in interactive sessions", {
  cache <- withr::local_tempdir()
  writeBin(as.raw(1:10), file.path(cache, "tabpfn-v3.5-20260909.safetensors"))
  local_mocked_bindings(menu = function(...) 2L, .package = "utils")
  expect_snapshot(
    tab_pfn_clear_cache(cache_dir = cache, ask = TRUE),
    transform = function(x) gsub(cache, "<cache>", x, fixed = TRUE)
  )
  expect_length(list.files(cache), 1)
})

# A fake Prior Labs / Hugging Face server. `token_ok` and `accepted` set the
# answers; NULL means "unreachable".
local_fake_server <- function(
  token_ok = TRUE,
  accepted = TRUE,
  env = parent.frame()
) {
  local_mocked_bindings(
    tabpfn_http_get = function(url, token = NULL) {
      if (grepl("huggingface.co/api/models", url, fixed = TRUE)) {
        return(list(
          status = 200,
          body = '{"cardData": {"license_name": "tabpfn-3-5-license-v1.0"}}'
        ))
      }
      if (grepl("/protected/", url, fixed = TRUE)) {
        if (is.null(token_ok)) {
          return(NULL)
        }
        if (token_ok) {
          return(list(status = 200, body = ""))
        }
        return(list(status = 401, body = ""))
      }
      if (grepl("/account/license/", url, fixed = TRUE)) {
        if (is.null(accepted)) {
          return(NULL)
        }
        return(list(
          status = 200,
          body = paste0('{"accepted": ', tolower(accepted), "}")
        ))
      }
      stop("unexpected URL: ", url)
    },
    .env = env
  )
}

local_token_files <- function(env = parent.frame()) {
  dir <- withr::local_tempdir(.local_envir = env)
  files <- file.path(dir, c("cache/auth_token", "client/token"))
  local_mocked_bindings(tabpfn_token_files = function() files, .env = env)
  withr::local_envvar(TABPFN_TOKEN = NA, .local_envir = env)
  tabpfn_env$accepted_repos <- character()
  withr::defer(tabpfn_env$accepted_repos <- character(), envir = env)
  files
}

test_that("tokens are found in the environment, then the cache files", {
  files <- local_token_files()
  expect_null(tabpfn_cached_token())

  dir.create(dirname(files[[2]]), recursive = TRUE)
  writeLines("client-token", files[[2]])
  expect_identical(
    tabpfn_cached_token(),
    list(token = "client-token", source = files[[2]])
  )

  tabpfn_save_token("saved-token")
  expect_identical(tabpfn_cached_token()$token, "saved-token")

  withr::local_envvar(TABPFN_TOKEN = " env-token ")
  expect_identical(
    tabpfn_cached_token(),
    list(token = "env-token", source = "env")
  )
})

test_that("an accepted license with a valid token passes", {
  local_token_files()
  local_fake_server()
  withr::local_envvar(TABPFN_TOKEN = "abc")
  expect_true(tabpfn_ensure_license("tabpfn_3_5"))
  expect_identical(tabpfn_env$accepted_repos, "tabpfn_3_5")
})

test_that("license problems give informative errors", {
  local_token_files()
  withr::local_envvar(TABPFN_TOKEN = "abc")
  withr::local_options(rlang_interactive = FALSE)

  local_fake_server(accepted = FALSE)
  expect_snapshot(tabpfn_ensure_license("tabpfn_3_5"), error = TRUE)

  local_fake_server(token_ok = NULL)
  expect_snapshot(tabpfn_ensure_license("tabpfn_3_5"), error = TRUE)

  local_fake_server(token_ok = FALSE)
  expect_snapshot(tabpfn_ensure_license("tabpfn_3_5"), error = TRUE)
})

test_that("a pasted API key is verified and saved", {
  files <- local_token_files()
  local_fake_server()
  local_mocked_bindings(tabpfn_prompt_for_token = function(...) "pasted")
  # The temporary path is written differently on Windows (8.3 names, mixed
  # separators), so replace all of it.
  expect_snapshot(tabpfn_ensure_license("tabpfn_3_5"), transform = function(x) {
    gsub("'[^']*[/\\\\]cache[/\\\\]auth_token'", "'<tmp>/cache/auth_token'", x)
  })
  expect_identical(readLines(files[[1]]), "pasted")
})

test_that("TABPFN_NO_BROWSER disables the prompt", {
  withr::local_envvar(TABPFN_NO_BROWSER = "1")
  expect_true(tabpfn_no_browser())
  withr::local_envvar(TABPFN_NO_BROWSER = "false")
  expect_false(tabpfn_no_browser())
})

test_that("every version has data size limits", {
  for (version in tab_pfn_versions()) {
    expect_named(
      tabpfn_limits(version),
      c("rows", "rows_cpu", "predictors", "classes"),
      ignore.order = TRUE
    )
  }
  expect_identical(tabpfn_abbreviate_count(1e6), "1M")
  expect_identical(tabpfn_abbreviate_count(5000), "5K")
  expect_identical(tabpfn_abbreviate_count(160), "160")
})

test_that("the registry's limits match the checkpoints", {
  for (version in tab_pfn_versions()) {
    skip_if_no_weights(version)
    for (task in c("classification", "regression")) {
      info <- tabpfn_version_info(version, task)
      ckpt <- tabpfn_read_checkpoint(tabpfn_checkpoint_path(info, ask = FALSE))
      ic <- tabpfn_resolve_inference(ckpt$inference_config, task, version)
      limits <- tabpfn_limits(version)
      expect_equal(ic$max_samples, limits$rows)
      expect_equal(ic$max_features, limits$predictors)
      if (task == "classification") {
        expect_equal(ic$max_classes, limits$classes)
      }
    }
  }
})
