# Infrastructure shared by the models with pretrained weights (Chronos,
# TabICL, TabPFN): the cache location, the downloader, and the confirmation
# gate. Each model keeps its own file layout and download sources.

# ------------------------------------------------------------------------------
# Cache

# Root of brulee's per-user weight cache. Each model has its own layout under
# it (and, for TabICL and TabPFN, an option to move it, mainly for tests).
brulee_cache_dir <- function() {
  tools::R_user_dir("brulee", which = "cache")
}

format_bytes <- function(x) {
  format(structure(x, class = "object_size"), units = "auto", standard = "SI")
}

# ------------------------------------------------------------------------------
# Download

# Fetch the Content-Length for a URL via a HEAD request, returning NA when
# the server doesn't expose it (in which case we just skip size checking).
brulee_remote_size <- function(url) {
  handle <- curl::new_handle()
  curl::handle_setopt(handle, nobody = TRUE, followlocation = TRUE)
  res <- tryCatch(
    curl::curl_fetch_memory(url, handle = handle),
    error = function(e) e
  )
  if (inherits(res, "error") || res$status_code >= 400L) {
    return(NA_real_)
  }
  hdrs <- curl::parse_headers(res$headers)
  cl <- grep("^content-length:", hdrs, ignore.case = TRUE, value = TRUE)
  if (length(cl) == 0L) {
    return(NA_real_)
  }
  as.numeric(sub("^[Cc]ontent-[Ll]ength:\\s*([0-9]+).*$", "\\1", cl[[1L]]))
}

# Download a single file with size validation and bounded retries. If the
# destination already holds a complete file (its size matches `size`, or
# `size` is unknown), it is kept. The file is downloaded under a temporary
# name and renamed once complete, so an interrupted download never leaves a
# partial file under the real name. When `sha256` is given, a complete
# download with a different checksum is an error rather than a retry: the
# upstream file has changed. `verbose = FALSE` silences the progress and retry
# messages, but not the errors.
brulee_download_file <- function(
  url,
  dest,
  label,
  max_attempts = 3L,
  size = brulee_remote_size(url),
  sha256 = NULL,
  verbose = TRUE,
  call = rlang::current_env()
) {
  if (file.exists(dest) && (is.na(size) || file.size(dest) == size)) {
    return(invisible(dest))
  }

  if (file.exists(dest)) {
    if (verbose) {
      cli::cli_alert_info(
        "Cached {.file {label}} is incomplete (size {.val {file.size(dest)}} of {.val {size}}); re-downloading."
      )
    }
    file.remove(dest)
  }

  dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
  part <- paste0(dest, ".part")
  on.exit(unlink(part), add = TRUE)
  # `download.file()` defaults to a 60-second timeout, which is too tight for
  # weight files of several hundred MB.
  withr::local_options(timeout = max(3600L, getOption("timeout")))

  for (attempt in seq_len(max_attempts)) {
    if (verbose) {
      cli::cli_progress_step("Downloading {.url {url}}")
    }
    err <- tryCatch(
      {
        curl::curl_download(url, part, mode = "wb", quiet = TRUE)
        NULL
      },
      error = function(e) e
    )

    ok <-
      is.null(err) &&
      file.exists(part) &&
      (is.na(size) || file.size(part) == size)
    if (ok) {
      if (!is.null(sha256) && !identical(cli::hash_file_sha256(part), sha256)) {
        cli::cli_abort(
          c(
            "The downloaded file {.file {label}} does not have the expected
             checksum.",
            "i" = "The upstream file may have changed; please report this at
                   {.url https://github.com/tidymodels/brulee/issues}."
          ),
          call = call
        )
      }
      brulee_install_file(part, dest, size = size, call = call)
      return(invisible(dest))
    }

    if (file.exists(part)) {
      file.remove(part)
    }
    if (verbose && attempt < max_attempts) {
      cli::cli_alert_warning(
        "Attempt {attempt}/{max_attempts} for {.val {label}} failed; retrying."
      )
    }
  }

  cli::cli_abort(
    c(
      "Failed to download {.url {url}} after {max_attempts} attempts.",
      "i" = "If you keep hitting this, try a different network or proxy."
    ),
    call = call
  )
}

# Move a completed download into place. A rename can fail, for example across
# file systems or, on Windows, when the destination exists or is open in
# another program; then the file is copied instead. Either way, the
# destination must end up complete.
brulee_install_file <- function(
  part,
  dest,
  size = NA,
  call = rlang::caller_env()
) {
  # Base R's warnings are replaced by the error below.
  if (!suppressWarnings(file.rename(part, dest))) {
    suppressWarnings(file.copy(part, dest, overwrite = TRUE))
  }
  complete <- file.exists(dest) && (is.na(size) || file.size(dest) == size)
  if (!complete) {
    cli::cli_abort(
      c(
        "The downloaded file could not be saved as {.file {dest}}.",
        "i" = "Check that {.path {dirname(dest)}} is writable and that the
               file isn't open in another program, then try again."
      ),
      call = call
    )
  }
  invisible(dest)
}

# ------------------------------------------------------------------------------
# Confirmation gate

# brulee never downloads large pretrained weights silently. When they are
# missing, prompt for confirmation in an interactive session and error
# otherwise. `label` names what's missing (e.g. "amazon/chronos-2" or
# "Classification weights for TabICL"), highlighted in both messages; `size` is
# a human string like "500MB"; `fn` is the calling user-facing function,
# referenced when the prompt is declined; `root` is the cache directory that
# was checked, reported only in the non-interactive error (not the prompt);
# `hint` is the non-interactive error's "i" bullet, since the callers point
# users at different next steps (an explicit downloader for TabICL and
# TabPFN, simply re-running interactively for Chronos).
brulee_confirm_download <- function(
  label,
  size,
  fn,
  root,
  hint,
  call = rlang::caller_env()
) {
  if (!rlang::is_interactive()) {
    cli::cli_abort(
      c(
        "No cached {.field {label}} weights found in {.path {root}}.",
        "i" = hint
      ),
      call = call
    )
  }

  cli::cli_inform("The weights for {.field {label}} are not found locally.")
  choice <- utils::menu(
    c("Yes", "No"),
    title = paste0("Download now (~", size, ")?")
  )
  if (choice != 1L) {
    cli::cli_abort(
      "Download declined; {.fn {fn}} needs the weights to continue.",
      call = call
    )
  }

  invisible(TRUE)
}
