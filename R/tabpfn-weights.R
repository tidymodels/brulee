# Model versions, the weight cache, the Prior Labs license check, and the
# documented data size limits for `brulee_tab_pfn()`.
#
# Everything here is driven by the registry (inst/tabpfn-registry.json, built
# by dev/tabpfn/make_registry.py), which lists the supported model versions,
# their checkpoint files, the architecture each file uses, and each version's
# data size limits. A model version and an architecture are different things:
# v3.5 and v3.5-fast are separate checkpoints of the same `tabpfn_v3_5`
# architecture. Code dispatches on the architecture (see
# `tabpfn_architecture()`); users choose a version. A new version that reuses
# an architecture is only a registry entry; a new architecture also needs its
# own `R/tabpfn-<architecture>.R` file. Nothing in this file changes.

# ------------------------------------------------------------------------------
# Session state

# State kept for the R session: the parsed registry, licenses confirmed,
# checkpoints whose checksum was verified, and loaded models. It is only read
# and changed through the functions below.
tabpfn_env <- new.env(parent = emptyenv())

tabpfn_license_accepted <- function(license_repo) {
  license_repo %in% tabpfn_env$accepted_repos
}

tabpfn_mark_license_accepted <- function(license_repo) {
  tabpfn_env$accepted_repos <- union(tabpfn_env$accepted_repos, license_repo)
  invisible()
}

tabpfn_forget_licenses <- function() {
  tabpfn_env$accepted_repos <- NULL
  invisible()
}

tabpfn_is_verified <- function(path) {
  path %in% tabpfn_env$verified
}

tabpfn_mark_verified <- function(path) {
  tabpfn_env$verified <- union(tabpfn_env$verified, path)
  invisible()
}

# Forget the given checkpoints, or all of them.
tabpfn_forget_verified <- function(paths = NULL) {
  if (is.null(paths)) {
    tabpfn_env$verified <- NULL
  } else {
    tabpfn_env$verified <- setdiff(tabpfn_env$verified, paths)
  }
  invisible()
}

# Loaded models, keyed by checkpoint path and device.
tabpfn_cached_model <- function(key) {
  tabpfn_env$models[[key]]
}

tabpfn_cache_model <- function(key, entry) {
  if (is.null(tabpfn_env$models)) {
    tabpfn_env$models <- new.env(parent = emptyenv())
  }
  tabpfn_env$models[[key]] <- entry
  invisible(entry)
}

tabpfn_forget_models <- function() {
  tabpfn_env$models <- NULL
  invisible()
}

# ------------------------------------------------------------------------------
# Registry

tabpfn_registry <- function() {
  if (is.null(tabpfn_env$registry)) {
    path <- system.file(
      "tabpfn-registry.json",
      package = "brulee",
      mustWork = TRUE
    )
    tabpfn_env$registry <- jsonlite::read_json(path, simplifyVector = TRUE)
  }
  tabpfn_env$registry
}

#' List available TabPFN model versions
#'
#' Returns the model versions that [brulee_tab_pfn()] can run, which can be
#' passed to its `version` argument and to [tab_pfn_download_weights()].
#'
#' @return A character vector of model version strings.
#' @examples
#' tab_pfn_versions()
#' @export
tab_pfn_versions <- function() {
  names(tabpfn_registry()$versions)
}

# Normalizes a user-supplied model version. Users may pass a bare number
# (e.g. `3.5` or `"3.5"`); we prefix a `v` so it matches the `v`-prefixed
# version names. The prefix is only added when the value does not already
# start with `v`, and matching remains exact, so a bare `3.5` will never match
# something like `v3.5-fast`.
tabpfn_normalize_version <- function(x) {
  if (is.null(x)) {
    return(x)
  }

  if (is.numeric(x)) {
    x <- format(x, trim = TRUE)
  }

  if (is.character(x) && !grepl("^v", x)) {
    x <- paste0("v", x)
  }

  x
}

tabpfn_check_version <- function(
  x,
  arg = caller_arg(x),
  call = caller_env()
) {
  valid_versions <- tab_pfn_versions()
  if (!is_string(x) || !x %in% valid_versions) {
    cli::cli_abort(
      c(
        "{.arg {arg}} must be one of {.or {.val {valid_versions}}}.",
        x = "{.val {x}} is not a supported model version."
      ),
      call = call
    )
  }
  invisible(x)
}

tabpfn_resolve_version <- function(
  version,
  arg = caller_arg(version),
  call = caller_env()
) {
  force(arg)
  if (
    !is.null(version) &&
      !(length(version) == 1L && (is.character(version) || is.numeric(version)))
  ) {
    cli::cli_abort(
      "{.arg {arg}} must be a single string or number, not
       {obj_type_friendly(version)}.",
      call = call
    )
  }
  version <- tabpfn_normalize_version(version) %||%
    tabpfn_registry()$default_version
  tabpfn_check_version(version, arg = arg, call = call)
  version
}

# Information about one version: repo, architecture, and the checkpoint file
# for a task ("classification" or "regression"). `file` picks one of the
# alternative checkpoints; the default is the version's default file.
tabpfn_version_info <- function(
  version,
  task = c("classification", "regression"),
  file = NULL,
  call = caller_env()
) {
  task <- arg_match(task)
  reg <- tabpfn_registry()
  info <- reg$versions[[version]]
  task_files <- info$tasks[[task]]
  file <- file %||% task_files$default
  if (!file %in% task_files$files) {
    cli::cli_abort(
      c(
        "{.val {file}} is not a {task} checkpoint for TabPFN {version}.",
        i = "Available files: {.val {task_files$files}}."
      ),
      call = call
    )
  }
  list(
    version = version,
    task = task,
    repo_id = info$repo_id,
    license_repo = info$license_repo,
    architecture = info$architecture,
    file = file,
    sha256 = reg$files[[file]]$sha256,
    size = reg$files[[file]]$size,
    format = reg$files[[file]]$format
  )
}

# ------------------------------------------------------------------------------
# Weight cache

# Checkpoints are cached in a `tabpfn` directory of brulee's per-user cache
# (overridable with the `brulee.tabpfn_cache_dir` option, mainly for tests).
# Weights that the Python `tabpfn` package already downloaded are reused in
# place, read-only, after their checksum is verified; brulee never writes to
# the Python cache.
#
# Attaching brulee never downloads anything. `brulee_tab_pfn()` prompts to
# download a missing checkpoint in interactive sessions and errors otherwise;
# `tab_pfn_download_weights()` populates the cache explicitly and
# `tab_pfn_weights_available()` reports whether it is populated.

tabpfn_cache_dir <- function() {
  path.expand(getOption(
    "brulee.tabpfn_cache_dir",
    default = file.path(brulee_cache_dir(), "tabpfn")
  ))
}

# The cache directory of the Python `tabpfn` package: the
# `TABPFN_MODEL_CACHE_DIR` environment variable, or `~/Library/Caches/tabpfn`
# on macOS, `$XDG_CACHE_HOME/tabpfn` or `~/.cache/tabpfn` on Linux, and
# `%APPDATA%/tabpfn` on Windows. Only read.
tabpfn_python_cache_dir <- function() {
  env <- trimws(Sys.getenv("TABPFN_MODEL_CACHE_DIR"))
  if (nzchar(env)) {
    return(path.expand(env))
  }
  if (.Platform$OS.type == "windows") {
    appdata <- trimws(Sys.getenv("APPDATA"))
    if (nzchar(appdata)) {
      return(file.path(appdata, "tabpfn"))
    }
    return(NA_character_)
  }
  if (Sys.info()[["sysname"]] == "Darwin") {
    return(file.path(path.expand("~"), "Library", "Caches", "tabpfn"))
  }
  xdg <- trimws(Sys.getenv("XDG_CACHE_HOME"))
  if (nzchar(xdg)) {
    return(file.path(xdg, "tabpfn"))
  }
  file.path(path.expand("~"), ".cache", "tabpfn")
}

# Notes about where the weights come from (downloaded now, already cached, or
# reused from the Python package) are shown only in interactive sessions, and
# at most once a day for each `key`. The date a note was last shown is kept in
# `.notes/` in the cache directory, so the limit holds across sessions.
# Returns whether to show the note now, and records it when so.
tabpfn_note_due <- function(key, cache_dir = tabpfn_cache_dir()) {
  if (!is_interactive()) {
    return(FALSE)
  }
  stamp <- file.path(cache_dir, ".notes", key)
  today <- format(Sys.Date())
  if (file.exists(stamp)) {
    shown <- readLines(stamp, n = 1L, warn = FALSE)
    if (identical(shown, today)) {
      return(FALSE)
    }
  }
  dir.create(dirname(stamp), recursive = TRUE, showWarnings = FALSE)
  try(writeLines(today, stamp), silent = TRUE)
  TRUE
}

tabpfn_hf_url <- function(repo_id, file) {
  paste0("https://huggingface.co/", repo_id, "/resolve/main/", file)
}

# Python fetches the repo's config.json after a download only so that
# Hugging Face counts the download; do the same.
tabpfn_count_download <- function(repo_id) {
  try(tabpfn_http_get(tabpfn_hf_url(repo_id, "config.json")), silent = TRUE)
  invisible()
}

# Check a cached checkpoint against the registry. The sha256 is checked once
# per session per file since it reads the whole file.
tabpfn_verify_cached <- function(path, info, call = caller_env()) {
  if (file.size(path) != info$size) {
    cli::cli_abort(
      c(
        "The cached checkpoint {.file {path}} has an unexpected size
         ({format_bytes(file.size(path))}, expected
         {format_bytes(info$size)}).",
        i = "It may be incomplete or a different release. Delete it and
             download it again."
      ),
      call = call
    )
  }
  if (!tabpfn_is_verified(path)) {
    if (!identical(cli::hash_file_sha256(path), info$sha256)) {
      cli::cli_abort(
        c(
          "The cached checkpoint {.file {path}} does not have the expected
           checksum.",
          i = "It may be corrupted or a different release. Delete it and
               download it again."
        ),
        call = call
      )
    }
    tabpfn_mark_verified(path)
  }
  invisible(path)
}

tabpfn_confirm_download <- function(info, call) {
  brulee_confirm_download(
    label = paste("TabPFN", info$version),
    size = format_bytes(info$size),
    fn = "brulee_tab_pfn",
    root = tabpfn_cache_dir(),
    hint = sprintf(
      "Download them with {.run brulee::tab_pfn_download_weights(\"%s\")}.",
      info$version
    ),
    call = call
  )
}

# A cached copy of the checkpoint described by `info` (from
# `tabpfn_version_info()`) with the expected size, from the brulee cache or the
# Python cache, or NULL. Non-erroring, so it can back
# `tab_pfn_weights_available()`.
tabpfn_find_checkpoint <- function(info, cache_dir = tabpfn_cache_dir()) {
  dirs <- c(cache_dir, tabpfn_python_cache_dir())
  paths <- file.path(dirs[!is.na(dirs)], info$file)
  paths <- paths[file.exists(paths) & file.size(paths) == info$size]
  if (length(paths) == 0) {
    NULL
  } else {
    paths[[1]]
  }
}

# Path to the checkpoint described by `info`, downloading it into the
# brulee cache if needed (after asking, when `ask`).
tabpfn_checkpoint_path <- function(
  info,
  ask = TRUE,
  cache_dir = tabpfn_cache_dir(),
  call = caller_env()
) {
  found <- tabpfn_find_checkpoint(info, cache_dir)
  if (!is.null(found)) {
    from_python <- normalizePath(dirname(found)) !=
      normalizePath(cache_dir, mustWork = FALSE)
    if (
      from_python && tabpfn_note_due(paste0("python-", info$file), cache_dir)
    ) {
      cli::cli_inform(c(
        i = "Using the TabPFN {info$version} weights downloaded by the Python
             {.pkg tabpfn} package: {.file {found}}."
      ))
    }
    return(tabpfn_verify_cached(found, info, call = call))
  }
  path <- file.path(cache_dir, info$file)
  if (file.exists(path)) {
    # Present but incomplete or a different release: report it.
    return(tabpfn_verify_cached(path, info, call = call))
  }
  if (ask) {
    tabpfn_confirm_download(info, call = call)
  }
  tabpfn_ensure_license(info$license_repo, call = call)
  note <- tabpfn_note_due(paste0("download-", info$file), cache_dir)
  if (note) {
    cli::cli_inform(c(
      i = "Downloading the TabPFN {info$version} weights ({.file {info$file}},
           {format_bytes(info$size)}) from {.url https://huggingface.co/{info$repo_id}}
           into {.path {cache_dir}}."
    ))
  }
  brulee_download_file(
    tabpfn_hf_url(info$repo_id, info$file),
    path,
    label = info$file,
    size = info$size,
    sha256 = info$sha256,
    verbose = is_interactive(),
    call = call
  )
  tabpfn_count_download(info$repo_id)
  tabpfn_mark_verified(path)
  if (note) {
    cli::cli_inform(c(
      v = "Saved the TabPFN {info$version} weights to {.file {path}}."
    ))
  }
  path
}

#' Download and cache pretrained TabPFN weights
#'
#' [brulee_tab_pfn()] needs pretrained weights that are not shipped with the
#' package. `tab_pfn_download_weights()` downloads them from Prior Labs'
#' Hugging Face repositories into the local cache.
#' `tab_pfn_weights_available()` reports whether the cache already holds them.
#'
#' Files already cached (by brulee, or by the Python `tabpfn` package) are
#' checked and skipped, so re-running does not download again. Attaching
#' brulee never downloads the weights. If [brulee_tab_pfn()] is run before they
#' are cached, it asks to download them in an interactive session and errors,
#' pointing here, otherwise.
#'
#' `tab_pfn_weights_available()` only looks for files of the expected size;
#' [brulee_tab_pfn()] verifies their checksums when it first loads them.
#'
#' @section Cache location:
#' The weights are stored in a `tabpfn` directory of brulee's per-user cache,
#' [tools::R_user_dir()]`("brulee", "cache")`. Set the `brulee.tabpfn_cache_dir`
#' option to use another directory. Weights that the Python `tabpfn` package
#' has already downloaded (into its own cache, or the directory in the
#' `TABPFN_MODEL_CACHE_DIR` environment variable) are used from there without
#' copying, after their checksum is verified. Use [tab_pfn_clear_cache()] to
#' delete the weights brulee downloaded.
#'
#' Notes about where the weights come from (downloaded, already cached, or
#' reused from the Python package) appear only in interactive sessions, at most
#' once a day for each file.
#'
#' @section License:
#' The weights are released by Prior Labs under a non-commercial license that
#' must be accepted once before the first download. The first download asks
#' you to log in to Prior Labs, accept the license, and paste your API key.
#' The key is stored in `~/.cache/tabpfn/auth_token`, the same file the Python
#' `tabpfn` package uses. In non-interactive sessions, set the `TABPFN_TOKEN`
#' environment variable to your API key instead. See [brulee_tab_pfn()] for
#' the full setup.
#'
#' @param version A model version from [tab_pfn_versions()]. A number such
#'   as `3.5` is also accepted. The default is the newest version.
#' @param task The task(s), one or both of `"classification"` and
#'   `"regression"`. Both by default. (v3.5 uses one checkpoint for both.)
#' @param files `"default"` downloads each task's default checkpoint; `"all"`
#'   also downloads the alternative checkpoints.
#' @param cache_dir The root of the local weight cache.
#' @return `tab_pfn_download_weights()` invisibly returns the paths of the
#'   checkpoints. `tab_pfn_weights_available()` returns a single logical:
#'   `TRUE` when the checkpoints for every task in `task` are cached.
#' @examplesIf FALSE
#' tab_pfn_download_weights("v3.5")
#' tab_pfn_weights_available("v3.5")
#' tab_pfn_weights_available("v3", task = "regression")
#' @export
tab_pfn_download_weights <- function(
  version = NULL,
  task = c("classification", "regression"),
  files = c("default", "all"),
  cache_dir = tabpfn_cache_dir()
) {
  version <- tabpfn_resolve_version(version)
  task <- arg_match(task, multiple = TRUE)
  files <- arg_match(files)
  infos <- tabpfn_checkpoint_infos(version, task, files)
  paths <- purrr::map_chr(
    infos,
    function(info) {
      found <- tabpfn_find_checkpoint(info, cache_dir)
      if (
        !is.null(found) &&
          tabpfn_note_due(paste0("cached-", info$file), cache_dir)
      ) {
        cli::cli_inform(c(
          v = "The TabPFN {info$version} weights ({.file {info$file}}) are
               already cached: {.file {found}}."
        ))
      }
      tabpfn_checkpoint_path(info, ask = FALSE, cache_dir = cache_dir)
    }
  )
  invisible(unname(paths))
}

#' @rdname tab_pfn_download_weights
#' @export
tab_pfn_weights_available <- function(
  version = NULL,
  task = c("classification", "regression"),
  cache_dir = tabpfn_cache_dir()
) {
  version <- tabpfn_resolve_version(version)
  task <- arg_match(task, multiple = TRUE)
  infos <- tabpfn_checkpoint_infos(version, task, "default")
  all(purrr::map_lgl(
    infos,
    function(info) !is.null(tabpfn_find_checkpoint(info, cache_dir))
  ))
}

# `tabpfn_version_info()` for each wanted checkpoint file of a version.
tabpfn_checkpoint_infos <- function(version, task, files) {
  tasks <- tabpfn_registry()$versions[[version]]$tasks[task]
  infos <- list()
  for (t in names(tasks)) {
    if (files == "default") {
      wanted <- tasks[[t]]$default
    } else {
      wanted <- tasks[[t]]$files
    }
    for (file in wanted) {
      infos[[file]] <- tabpfn_version_info(version, t, file)
    }
  }
  unname(infos)
}

#' Remove cached TabPFN weights
#'
#' Deletes checkpoints that brulee downloaded for [brulee_tab_pfn()], for one
#' model version or all of them, along with any partial downloads, and
#' unloads models loaded in the current session. Use it to free disk space
#' (each checkpoint takes 200 MB to 900 MB), to force a fresh download, or to
#' delete the weights when you stop using them, as the Prior Labs license
#' requires when it ends.
#'
#' Weights downloaded by the Python `tabpfn` package are never removed: delete
#' those files yourself if you no longer need them.
#'
#' @inheritParams tab_pfn_download_weights
#' @param version A model version from [tab_pfn_versions()], or `NULL`
#'   (the default) for all versions.
#' @param ask A logical: ask for confirmation before deleting? Defaults to
#'   `TRUE` in interactive sessions.
#' @return The paths of the deleted files, invisibly.
#' @examplesIf FALSE
#' tab_pfn_clear_cache("v3")
#' tab_pfn_clear_cache()
#' @export
tab_pfn_clear_cache <- function(
  version = NULL,
  cache_dir = tabpfn_cache_dir(),
  ask = rlang::is_interactive()
) {
  check_bool(ask)
  if (is.null(version)) {
    versions <- tab_pfn_versions()
  } else {
    versions <- tabpfn_resolve_version(version)
  }
  names <- unique(unlist(purrr::map(versions, function(v) {
    purrr::map(tabpfn_registry()$versions[[v]]$tasks, function(t) t$files)
  })))
  paths <- file.path(cache_dir, c(names, paste0(names, ".part")))
  paths <- paths[file.exists(paths)]
  if (is.null(version)) {
    label <- "TabPFN"
  } else {
    label <- paste("TabPFN", versions)
  }
  if (length(paths) == 0) {
    cli::cli_inform(c(i = "No cached {label} weights in {.path {cache_dir}}."))
    return(invisible(character()))
  }
  size <- format_bytes(sum(file.size(paths)))
  if (ask) {
    cli::cli_inform(c(
      "These cached {label} weight {cli::qty(length(paths))}file{?s}
       ({size}) will be deleted:",
      set_names(basename(paths), rep("*", length(paths)))
    ))
    if (utils::menu(c("Yes", "No"), title = "Delete them?") != 1L) {
      cli::cli_inform("Nothing was deleted.")
      return(invisible(character()))
    }
  }
  unlink(paths)
  tabpfn_forget_models()
  tabpfn_forget_verified(paths)
  cli::cli_inform(c(
    v = "Deleted {length(paths)} cached {label} weight
         {cli::qty(length(paths))}file{?s} ({size}) from {.path {cache_dir}}."
  ))
  invisible(paths)
}

# ------------------------------------------------------------------------------

# License acceptance for the gated TabPFN weights, ported from Python
# `tabpfn/browser_auth.py`. The token and its cache file are shared with the
# Python package, so users who already accepted the license there are not
# asked again.
#
# Differences from Python: there is no local callback server (the user pastes
# the API key instead), a token from `TABPFN_TOKEN` is never written to disk,
# and an invalid cached token is reported rather than deleted.

tabpfn_token_files <- function() {
  home <- path.expand("~")
  c(
    file.path(home, ".cache", "tabpfn", "auth_token"),
    file.path(home, ".tabpfn", "token")
  )
}

tabpfn_cached_token <- function() {
  env_token <- trimws(Sys.getenv("TABPFN_TOKEN"))
  if (nzchar(env_token)) {
    return(list(token = env_token, source = "env"))
  }
  for (path in tabpfn_token_files()) {
    if (file.exists(path)) {
      token <- trimws(paste(readLines(path, warn = FALSE), collapse = ""))
      if (nzchar(token)) {
        return(list(token = token, source = path))
      }
    }
  }
  NULL
}

tabpfn_save_token <- function(token) {
  path <- tabpfn_token_files()[[1]]
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeLines(token, path)
  Sys.chmod(path, mode = "0600")
  invisible(path)
}

tabpfn_auth_urls <- function() {
  auth <- tabpfn_registry()$auth
  list(
    gui = sub("/$", "", Sys.getenv("TABPFN_AUTH_GUI_URL", auth$gui_url)),
    api = sub("/$", "", Sys.getenv("TABPFN_AUTH_API_URL", auth$api_url))
  )
}

# GET `url`; returns `list(status, body)`, or NULL when the server can't be
# reached.
tabpfn_http_get <- function(url, token = NULL) {
  handle <- curl::new_handle(timeout = 10)
  if (!is.null(token)) {
    curl::handle_setheaders(handle, Authorization = paste("Bearer", token))
  }
  res <- tryCatch(
    curl::curl_fetch_memory(url, handle = handle),
    error = function(e) NULL
  )
  if (is.null(res)) {
    return(NULL)
  }
  list(status = res$status_code, body = rawToChar(res$content))
}

# TRUE: valid. FALSE: invalid or expired. NULL: server unreachable.
tabpfn_verify_token <- function(token, api_url) {
  res <- tabpfn_http_get(paste0(api_url, "/protected/"), token)
  if (is.null(res)) {
    return(NULL)
  }
  if (res$status == 200) {
    return(TRUE)
  }
  if (res$status %in% c(401, 403)) {
    return(FALSE)
  }
  NULL
}

# TRUE: accepted. FALSE: not accepted or invalid token. NULL: unreachable.
tabpfn_check_license_accepted <- function(token, api_url, license_name) {
  url <- paste0(
    api_url,
    "/account/license/?version=",
    utils::URLencode(license_name, reserved = TRUE)
  )
  res <- tabpfn_http_get(url, token)
  if (is.null(res)) {
    return(NULL)
  }
  if (res$status %in% c(401, 403)) {
    return(FALSE)
  }
  if (res$status != 200) {
    return(NULL)
  }
  isTRUE(jsonlite::fromJSON(res$body)$accepted)
}

# The license name comes from the Hugging Face model card.
tabpfn_license_name <- function(license_repo, call = caller_env()) {
  url <- paste0("https://huggingface.co/api/models/Prior-Labs/", license_repo)
  hf_token <- Sys.getenv("HF_TOKEN")
  if (!nzchar(hf_token)) {
    hf_token <- NULL
  }
  res <- tabpfn_http_get(url, hf_token)
  if (is.null(res) || res$status != 200) {
    cli::cli_abort(
      c(
        "Could not read the model card for {.val Prior-Labs/{license_repo}}
         from Hugging Face.",
        i = "Check your internet connection and try again."
      ),
      call = call
    )
  }
  name <- jsonlite::fromJSON(res$body)$cardData$license_name
  if (!is_string(name)) {
    cli::cli_abort(
      "The model card for {.val Prior-Labs/{license_repo}} has no license
       name; please report this to {.email support@priorlabs.ai}.",
      call = call
    )
  }
  name
}

tabpfn_abort_unreachable <- function(call) {
  cli::cli_abort(
    c(
      "Could not reach the Prior Labs license server.",
      i = "Check your internet connection and try again."
    ),
    call = call
  )
}

tabpfn_no_browser <- function() {
  value <- tolower(trimws(Sys.getenv("TABPFN_NO_BROWSER")))
  nzchar(value) && !value %in% c("0", "false", "no", "off")
}

# Ask the user to log in, accept the license, and paste their API key.
# Returns the key, or NULL when the session can't prompt.
tabpfn_prompt_for_token <- function(gui_url, license_repo) {
  if (!rlang::is_interactive() || tabpfn_no_browser()) {
    return(NULL)
  }
  login_url <- paste0(
    gui_url,
    "/login?hf_repo_id=",
    utils::URLencode(license_repo, reserved = TRUE)
  )
  cli::cli_inform(c(
    "TabPFN needs a one-time license acceptance before the model weights
     can be downloaded.",
    "*" = "Log in (or register) at {.url {login_url}}.",
    "*" = "Accept the license on the Licenses tab.",
    "*" = "Copy your API key from {.url {gui_url}/account} and paste it below."
  ))
  utils::browseURL(login_url)
  token <- trimws(readline("API key: "))
  if (nzchar(token)) {
    token
  } else {
    NULL
  }
}

tabpfn_abort_no_token <- function(gui_url, call) {
  cli::cli_abort(
    c(
      "TabPFN needs a one-time license acceptance before the model weights
       can be downloaded.",
      i = "In a non-interactive session:",
      "*" = "Log in (or register) at {.url {gui_url}}.",
      "*" = "Accept the license on the Licenses tab.",
      "*" = "Copy your API key from {.url {gui_url}/account}.",
      "*" = "Set the {.envvar TABPFN_TOKEN} environment variable to it,
             e.g. in your {.file .Renviron} file."
    ),
    call = call
  )
}

# Make sure the user has accepted the license for `license_repo` (e.g.
# "tabpfn_3_5"). Called before downloading weights; loading weights that are
# already cached needs no license check, as in Python.
tabpfn_ensure_license <- function(license_repo, call = caller_env()) {
  if (tabpfn_license_accepted(license_repo)) {
    return(invisible(TRUE))
  }
  urls <- tabpfn_auth_urls()
  license_name <- tabpfn_license_name(license_repo, call = call)

  cached <- tabpfn_cached_token()
  if (!is.null(cached)) {
    status <- tabpfn_verify_token(cached$token, urls$api)
    if (is.null(status)) {
      tabpfn_abort_unreachable(call)
    }
    if (isTRUE(status)) {
      accepted <- tabpfn_check_license_accepted(
        cached$token,
        urls$api,
        license_name
      )
      if (is.null(accepted)) {
        tabpfn_abort_unreachable(call)
      }
      if (accepted) {
        tabpfn_mark_license_accepted(license_repo)
        return(invisible(TRUE))
      }
    } else if (cached$source == "env") {
      cli::cli_inform(c(
        "!" = "The API key in {.envvar TABPFN_TOKEN} is invalid or expired."
      ))
    } else {
      cli::cli_inform(c(
        "!" = "The API key in {.file {cached$source}} is invalid or expired."
      ))
    }
  }

  token <- tabpfn_prompt_for_token(urls$gui, license_repo)
  if (is.null(token)) {
    tabpfn_abort_no_token(urls$gui, call)
  }
  if (isFALSE(tabpfn_verify_token(token, urls$api))) {
    cli::cli_abort("The Prior Labs server rejected that API key.", call = call)
  }
  path <- tabpfn_save_token(token)
  cli::cli_inform(c(v = "API key saved to {.file {path}}."))

  accepted <- tabpfn_check_license_accepted(token, urls$api, license_name)
  if (is.null(accepted)) {
    tabpfn_abort_unreachable(call)
  }
  if (!accepted) {
    accept_url <- paste0(
      urls$gui,
      "/accept-license?hf_repo_id=",
      utils::URLencode(license_repo, reserved = TRUE)
    )
    cli::cli_abort(
      c(
        "The TabPFN license is not accepted yet.",
        i = "Accept it at {.url {accept_url}} and try again."
      ),
      call = call
    )
  }
  tabpfn_mark_license_accepted(license_repo)
  invisible(TRUE)
}

# ------------------------------------------------------------------------------
# Data size limits

# At fit time, `brulee_tab_pfn()` enforces the limits that each checkpoint's
# inference config declares (rows, predictors, classes) and, on the CPU, the
# per-version limit that the Python package applies in place of the
# checkpoint's value (`tabpfn.inference_config.cpu_sample_limit()`). The
# registry records both for each version; this table documents them, and a
# test checks it against the checkpoints when the weights are cached.

tabpfn_limits <- function(version) {
  limits <- tabpfn_registry()$versions[[version]]$limits
  if (is.null(limits)) {
    cli::cli_abort(
      "The registry has no data size limits for TabPFN {version}.",
      .internal = TRUE
    )
  }
  limits
}

# Python warns about slow CPU inference above a fifth of the CPU limit.
tabpfn_cpu_sample_limit <- function(version) {
  as.integer(tabpfn_limits(version)$rows_cpu)
}

# 1000 -> "1K", 1e6 -> "1M", for the documentation table.
tabpfn_abbreviate_count <- function(x) {
  if (is.na(x)) {
    return("unknown")
  }
  if (x >= 1e6) {
    return(paste0(x / 1e6, "M"))
  }
  if (x >= 1000) {
    return(paste0(x / 1000, "K"))
  }
  as.character(x)
}

tabpfn_limits_table_md <- function() {
  rows <- purrr::map_chr(
    tab_pfn_versions(),
    function(version) {
      limits <- tabpfn_limits(version)
      counts <- purrr::map_chr(
        limits[c("rows", "rows_cpu", "predictors", "classes")],
        tabpfn_abbreviate_count
      )
      paste0("| `\"", version, "\"` | ", paste(counts, collapse = " | "), " |")
    }
  )
  c(
    "@section Data size limits:",
    "",
    "Each model version was trained for data up to a certain size, and",
    "`brulee_tab_pfn()` refuses larger data with an error that names the count",
    "and the limit. Rows are training rows.",
    "",
    "| Version | Rows (GPU) | Rows (CPU) | Predictors | Classes |",
    "| --- | --- | --- | --- | --- |",
    rows,
    "",
    "The CPU column applies whenever the model runs on the CPU (including",
    "`device = NULL` on a machine without a CUDA GPU). Above a fifth of that",
    "limit, `brulee_tab_pfn()` warns that prediction may be slow. Set",
    "`ignore_pretraining_limits = TRUE` to lift the row and predictor limits;",
    "the model then runs on data larger than it was trained for, more slowly",
    "and possibly less accurately. The class limit can't be lifted. Use",
    "`training_set_limit` to fit on a sample instead.",
    "",
    "Memory grows with the number of rows times the number of predictors and",
    "the ensemble size: on the CPU, 5,000 training rows with 50 predictors and",
    "8 ensemble members need about 12 GB. The row and predictor maxima trade",
    "off against each other, so you cannot always reach both at once. For",
    "`\"v3.5\"`, Prior Labs recommends up to 6,000 predictors even though the",
    "model accepts 20,000. See <https://docs.priorlabs.ai/models>."
  )
}
