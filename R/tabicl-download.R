# Local cache for the converted TabICL weights, and the downloader that
# populates it.
#
# The released checkpoints are Python `.ckpt` pickles, which R cannot read; they
# are converted offline to task-prefixed safetensors + JSON (see
# dev/tabicl/convert_ckpt.py, which writes to dev/tabicl/artifacts) and hosted
# for download. Following chronos2, the files brulee reads are cached under
# `~/.cache/TabICL/<version>/<date>/<TaskLabel>/`, mirroring the artifacts layout.
#
# The download URL is not yet decided, so `tabicl_default_base_url()` returns
# NULL and `tabicl_download()` errors until it is set. `brulee_tab_icl()` reads
# from the cache and errors if nothing is cached (it does not download
# automatically).

# Base URL the converted weights are hosted under (to be named later). Files are
# fetched from <base_url>/<version>/<date>/<TaskLabel>/<task-prefixed file>. Any
# location curl can reach works, including `file:///path/to/dev/tabicl/artifacts`
# for a local copy.
tabicl_default_base_url <- function() {
  NULL
}

# The released checkpoint version and date the downloader fetches by default.
tabicl_default_version <- function() {
  "v2"
}
tabicl_default_date <- function() {
  "2026-02-12"
}

# Root of the local TabICL weight cache, mirroring chronos2's caching approach.
# Holds only the files brulee reads, under <version>/<date>/<TaskLabel>/ (the
# same structure as dev/tabicl/artifacts). Overridable via an option, mainly for
# tests.
tabicl_cache_dir <- function() {
  getOption(
    "brulee.tabicl_cache_dir",
    default = file.path(Sys.getenv("HOME"), ".cache", "TabICL")
  )
}

# Directory label per task for the <Classification|Regression> subfolder.
tabicl_task_label <- function(task) {
  if (identical(task, "classification")) "Classification" else "Regression"
}

# Locate a cached checkpoint for a task, choosing the latest version/date. The
# directory must contain both task-prefixed files. Errors if none is cached.
tabicl_cache_lookup <- function(task, call = rlang::caller_env()) {
  files <- tabicl_checkpoint_files(task)
  root <- tabicl_cache_dir()

  candidates <- Sys.glob(file.path(root, "*", "*", tabicl_task_label(task)))
  has_both <- vapply(
    candidates,
    function(d) {
      file.exists(file.path(d, files$config)) &&
        file.exists(file.path(d, files$weights))
    },
    logical(1)
  )
  candidates <- candidates[has_both]

  if (length(candidates) == 0) {
    cli::cli_abort(
      c(
        "No cached {task} TabICL checkpoint found in {.path {root}}.",
        "i" = "Download one with {.fn tabicl_download} once a URL is configured,
               or convert and cache a checkpoint offline (see {.path dev/tabicl})."
      ),
      call = call
    )
  }
  # The parent directory of the task folder is the date (YYYY-MM-DD).
  candidates[order(basename(dirname(candidates)))][length(candidates)]
}

# Download a converted TabICL checkpoint into the local cache and return its
# directory. The two task-prefixed files are pulled from
#   <base_url>/<version>/<date>/<TaskLabel>/<file>
# into the matching cache subdirectory, reusing chronos2's curl helper for size
# validation and retries.
tabicl_download <- function(
  task = c("classification", "regression"),
  base_url = tabicl_default_base_url(),
  version = tabicl_default_version(),
  date = tabicl_default_date(),
  cache_dir = tabicl_cache_dir(),
  call = rlang::caller_env()
) {
  task <- rlang::arg_match(task, call = call)
  if (is.null(base_url)) {
    cli::cli_abort(
      c(
        "No TabICL download URL is configured yet.",
        "i" = "Pass {.arg base_url}, or convert and cache a checkpoint offline
               (see {.path dev/tabicl})."
      ),
      call = call
    )
  }

  files <- tabicl_checkpoint_files(task)
  rel <- paste(version, date, tabicl_task_label(task), sep = "/")
  dest <- file.path(cache_dir, version, date, tabicl_task_label(task))
  dir.create(dest, recursive = TRUE, showWarnings = FALSE)

  # The safetensors file is large; lift download.file()'s default timeout.
  old_timeout <- getOption("timeout")
  options(timeout = max(600L, old_timeout))
  on.exit(options(timeout = old_timeout), add = TRUE)

  base <- sub("/+$", "", base_url)
  for (f in c(files$config, files$weights)) {
    url <- sprintf("%s/%s/%s", base, rel, f)
    chronos2_download_file(url, file.path(dest, f), label = paste(task, f))
  }

  dest
}
