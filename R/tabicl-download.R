# Local cache for the converted TabICL weights, and the downloader that
# populates it.
#
# The released checkpoints are Python `.ckpt` pickles, which R cannot read; they
# are converted offline to task-prefixed safetensors + JSON (see
# dev/tabicl/convert_ckpt.py, which writes to dev/tabicl/artifacts) and uploaded
# as assets on a GitHub release of the `tidymodels/tabicl-weights` repo. Each
# release is tagged `<version>-<date>` (for example `v2-2026-02-12`) and carries
# the four files for both tasks as individual assets (the large safetensors are
# release assets, not committed to the repo, so the source-archive tarball does
# not contain them). The files brulee reads are cached under the per-user cache
# directory from `tools::R_user_dir("brulee", "cache")`, in
# `<version>/<date>/<TaskLabel>/`, mirroring the artifacts layout.
#
# `brulee_tab_icl()` reads from this cache; it never downloads on its own, and
# neither does attaching the package. The user populates the cache explicitly
# with `tab_icl_download_weights()`, and `tab_icl_weights_available()` reports
# whether it is populated.

# The GitHub repo hosting the converted weights, and the released checkpoint
# version/date the downloader fetches by default.
tabicl_default_repo <- function() {
  "tidymodels/tabicl-weights"
}
tabicl_default_version <- function() {
  "v2"
}
tabicl_default_date <- function() {
  "2026-02-12"
}

# URL of a single release asset on the tag `<version>-<date>`.
tabicl_asset_url <- function(
  file,
  version,
  date,
  repo = tabicl_default_repo()
) {
  sprintf(
    "https://github.com/%s/releases/download/%s-%s/%s",
    repo,
    version,
    date,
    file
  )
}

# Root of the local TabICL weight cache, mirroring chronos2's caching approach.
# Holds only the files brulee reads, under <version>/<date>/<TaskLabel>/ (the
# same structure as dev/tabicl/artifacts). Overridable via an option, mainly for
# tests.
tabicl_cache_dir <- function() {
  getOption(
    "brulee.tabicl_cache_dir",
    default = tools::R_user_dir("brulee", which = "cache")
  )
}

# Directory label per task for the <Classification|Regression> subfolder.
tabicl_task_label <- function(task) {
  if (identical(task, "classification")) "Classification" else "Regression"
}

# Locate a cached checkpoint for a task, choosing the latest version/date, or
# return NULL if none is cached. The directory must contain both task-prefixed
# files. Non-erroring so it can back the availability check.
tabicl_find_checkpoint <- function(task, cache_dir = tabicl_cache_dir()) {
  files <- tabicl_checkpoint_files(task)

  candidates <- Sys.glob(
    file.path(cache_dir, "*", "*", tabicl_task_label(task))
  )
  has_both <- purrr::map_lgl(
    candidates,
    \(d) {
      file.exists(file.path(d, files$config)) &&
        file.exists(file.path(d, files$weights))
    }
  )
  candidates <- candidates[has_both]

  if (length(candidates) == 0) {
    return(NULL)
  }
  # The parent directory of the task folder is the date (YYYY-MM-DD).
  candidates[order(basename(dirname(candidates)))][length(candidates)]
}

# Like tabicl_find_checkpoint() but errors when nothing is cached; used at fit
# time where a missing checkpoint is fatal.
tabicl_cache_lookup <- function(task, call = rlang::caller_env()) {
  path <- tabicl_find_checkpoint(task)
  if (is.null(path)) {
    root <- tabicl_cache_dir()
    cli::cli_abort(
      c(
        "No cached {task} TabICL checkpoint found in {.path {root}}.",
        "i" = "Download them with {.fn tab_icl_download_weights}."
      ),
      call = call
    )
  }
  path
}

#' Download and cache pretrained TabICL weights
#'
#' [brulee_tab_icl()] needs pretrained weights that are not shipped with the
#' package. `tab_icl_download_weights()` fetches them from a release of the
#' `tidymodels/tabicl-weights` GitHub repository into the local cache.
#' `tab_icl_weights_available()` reports whether the cache already holds usable
#' weights.
#'
#' @param task The task(s) to act on, one or both of `"classification"` and
#'   `"regression"`. Both are fetched (and checked) by default.
#' @param version,date The release to fetch, identifying the tag
#'   `<version>-<date>` (for example `"v2"` and `"2026-02-12"`).
#' @param repo The `owner/name` of the GitHub repository hosting the weights.
#' @param cache_dir The root of the local weight cache. Defaults to the
#'   `brulee.tabicl_cache_dir` option, or a per-user cache directory via
#'   [tools::R_user_dir()]`("brulee", "cache")`.
#' @param call The calling environment, used for error messages.
#'
#' @details
#' Each release carries the two files brulee reads per task (a JSON config and a
#' safetensors weight file) as individual assets. They are downloaded into
#' `<cache_dir>/<version>/<date>/<TaskLabel>/`. A file already present and
#' complete is left in place, so re-running resumes rather than re-downloads.
#'
#' The cache location can be overridden with the `brulee.tabicl_cache_dir`
#' option. Weights are only downloaded when you call
#' `tab_icl_download_weights()`; neither attaching brulee nor calling
#' [brulee_tab_icl()] downloads them automatically. If [brulee_tab_icl()] is
#' run before the weights are cached, it errors and points you here.
#'
#' @return
#' `tab_icl_download_weights()` invisibly returns the populated
#' `<cache_dir>/<version>/<date>` directory. `tab_icl_weights_available()`
#' returns a single logical.
#'
#' @examplesIf FALSE
#' tab_icl_download_weights()
#' tab_icl_weights_available()
#' @export
tab_icl_download_weights <- function(
  task = c("classification", "regression"),
  version = tabicl_default_version(),
  date = tabicl_default_date(),
  repo = tabicl_default_repo(),
  cache_dir = tabicl_cache_dir(),
  call = rlang::caller_env()
) {
  task <- rlang::arg_match(task, multiple = TRUE, error_call = call)

  # The safetensors assets are large; lift download.file()'s default timeout.
  withr::local_options(timeout = max(600L, getOption("timeout")))

  dest_root <- file.path(cache_dir, version, date)
  for (tk in task) {
    files <- tabicl_checkpoint_files(tk)
    dest <- file.path(dest_root, tabicl_task_label(tk))
    dir.create(dest, recursive = TRUE, showWarnings = FALSE)
    for (f in c(files$config, files$weights)) {
      url <- tabicl_asset_url(f, version, date, repo = repo)
      chronos2_download_file(url, file.path(dest, f), label = f)
    }
  }
  cli::cli_inform(
    "All {.pkg TabICL} weight files are locally available at {.file {cache_dir}}.",
    call = NULL
  )

  invisible(dest_root)
}

#' @rdname tab_icl_download_weights
#' @export
tab_icl_weights_available <- function(
  task = c("classification", "regression"),
  cache_dir = tabicl_cache_dir()
) {
  task <- rlang::arg_match(task, multiple = TRUE)
  all(purrr::map_lgl(
    task,
    \(tk) !is.null(tabicl_find_checkpoint(tk, cache_dir = cache_dir))
  ))
}
