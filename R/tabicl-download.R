# Download converted TabICL checkpoints (config.json + model.safetensors).
#
# The released checkpoints are Python `.ckpt` pickles, which R cannot read; they
# are converted offline to safetensors + JSON (see dev/tabicl/convert_ckpt.py)
# and hosted for download. This reuses the chronos2 HF helpers
# (`chronos2_resolve_revision()`, `chronos2_download_file()`) for revision
# pinning, caching, size validation, and retries.
#
# Hosting of the converted weights is not yet finalized, so `tabicl_default_repo()`
# returns NULL and the public default asks the user to supply `path`. Once the
# converted artifacts are hosted, set the repo and revision below and automatic
# download is enabled with no other changes.

# HuggingFace repo holding the converted checkpoints, or NULL when not yet
# hosted. Layout expected: `<checkpoint>/config.json` and
# `<checkpoint>/model.safetensors` for `checkpoint` in classifier / regressor.
tabicl_default_repo <- function() {
  NULL
}

# Pinned revision (40-char commit SHA recommended) for reproducibility. Replace
# with a real SHA when the weights are hosted; do not track a moving branch.
tabicl_default_revision <- function() {
  "main"
}

tabicl_default_cache_dir <- function() {
  file.path(Sys.getenv("HOME"), ".cache", "brulee-tabicl")
}

# Download a converted TabICL checkpoint, returning the local directory holding
# its `config.json` and `model.safetensors` (suitable for `tabicl_load_model()`).
tabicl_download <- function(
  checkpoint = c("classifier", "regressor"),
  repo_id = tabicl_default_repo(),
  revision = tabicl_default_revision(),
  cache_dir = tabicl_default_cache_dir(),
  call = rlang::caller_env()
) {
  checkpoint <- rlang::arg_match(checkpoint, call = call)
  if (is.null(repo_id)) {
    cli::cli_abort(
      c(
        "Automatic TabICL weight download is not available yet.",
        "i" = "Convert a released checkpoint and pass its directory via {.arg path}."
      ),
      call = call
    )
  }

  sha <- chronos2_resolve_revision(repo_id, revision)
  model_dir <- file.path(
    cache_dir,
    gsub("/", "--", repo_id),
    sha,
    checkpoint
  )
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

  # The safetensors file is large; lift download.file()'s default timeout.
  old_timeout <- getOption("timeout")
  options(timeout = max(600L, old_timeout))
  on.exit(options(timeout = old_timeout), add = TRUE)

  for (f in c("config.json", "model.safetensors")) {
    url <- sprintf(
      "https://huggingface.co/%s/resolve/%s/%s/%s",
      repo_id,
      sha,
      checkpoint,
      f
    )
    chronos2_download_file(
      url,
      file.path(model_dir, f),
      label = paste(checkpoint, f)
    )
  }

  model_dir
}
