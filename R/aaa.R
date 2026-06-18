#' @import torch
#' @import rlang
#' @importFrom stats complete.cases model.matrix terms
#' @importFrom utils globalVariables
#'

#' @importFrom ggplot2 autoplot
#' @export
ggplot2::autoplot

#' @importFrom generics tunable
#' @export
generics::tunable

#' @importFrom stats coef
#' @export
stats::coef

# ------------------------------------------------------------------------------

utils::globalVariables(
  c(
    "object",
    "iteration",
    "loss"
  )
)

# ------------------------------------------------------------------------------

# nocov start
.onAttach <- function(libname, pkgname) {
  s3_register("ggplot2::autoplot", "brulee_mlp")
  tabicl_attach_weights()
  invisible()
}

# brulee_tab_icl() needs pretrained TabICL weights that are not shipped with the
# package. On attach, populate the cache if it is empty: offer to download in
# interactive sessions, download outright otherwise. Set
# `options(brulee.tabicl_autodownload = FALSE)` to opt out. A failed download
# never blocks attaching the package.
tabicl_attach_weights <- function() {
  if (
    tab_icl_weights_available() ||
      !isTRUE(getOption("brulee.tabicl_autodownload", TRUE))
  ) {
    return(invisible())
  }

  if (interactive()) {
    packageStartupMessage(cli::format_message(c(
      "brulee needs pretrained TabICL weights for {.fn brulee_tab_icl}, which
       are not cached yet.",
      "i" = "They will be downloaded from {.val {tabicl_default_repo()}} into
             {.path {tabicl_cache_dir()}}."
    )))
    answer <- utils::menu(
      c("Yes", "No"),
      title = "Download the TabICL weights now?"
    )
    if (!identical(answer, 1L)) {
      packageStartupMessage(cli::format_message(c(
        "i" = "Skipping download. Run {.fn tab_icl_download_weights} later to
               enable {.fn brulee_tab_icl}."
      )))
      return(invisible())
    }
  }

  tryCatch(
    tab_icl_download_weights(),
    error = function(e) {
      packageStartupMessage(cli::format_message(c(
        "!" = "Could not download TabICL weights: {conditionMessage(e)}",
        "i" = "Run {.fn tab_icl_download_weights} to retry."
      )))
    }
  )
  invisible()
}

# Dynamic reg helper -----------------------------------------------------------

# vctrs/register-s3.R
# https://github.com/r-lib/vctrs/blob/master/R/register-s3.R
s3_register <- function(generic, class, method = NULL) {
  stopifnot(is.character(generic), length(generic) == 1)
  stopifnot(is.character(class), length(class) == 1)

  pieces <- strsplit(generic, "::")[[1]]
  stopifnot(length(pieces) == 2)
  package <- pieces[[1]]
  generic <- pieces[[2]]

  if (is.null(method)) {
    method <- get(paste0(generic, ".", class), envir = parent.frame())
  }
  stopifnot(is.function(method))

  if (package %in% loadedNamespaces()) {
    registerS3method(generic, class, method, envir = asNamespace(package))
  }

  # Always register hook in case package is later unloaded & reloaded
  setHook(
    packageEvent(package, "onLoad"),
    function(...) {
      registerS3method(generic, class, method, envir = asNamespace(package))
    }
  )
}

# nocov end
