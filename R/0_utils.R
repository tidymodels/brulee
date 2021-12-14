
# ------------------------------------------------------------------------------
# used in print methods

brulee_print <- function(x, ...) {
  lvl <- get_levels(x)
  if (is.null(lvl)) {
    chr_y <- "numeric outcome"
  } else {
    chr_y <- paste(length(lvl), "classes")
  }
  cat(
    format(x$dims$n, big.mark = ","), "samples,",
    format(x$dims$p, big.mark = ","), "features,",
    chr_y, "\n"
  )
  if (!is.null(x$dims$levels) && !is.null(x$parameters$class_weights)) {
    cat("class weights",
        paste0(
          names(x$parameters$class_weights),
          "=",
          format(x$parameters$class_weights),
          collapse = ", "
        ),
        "\n")
  }
  if (x$parameters$penalty > 0) {
    cat("weight decay:", x$parameters$penalty, "\n")
  }
  if (any(names(x$parameters) == "dropout")) {
    cat("dropout proportion:", x$parameters$dropout, "\n")
  }
  cat("batch size:", x$parameters$batch_size, "\n")

  if (!is.null(x$loss)) {
    it <- x$best_epoch
    chr_it <- cli::pluralize("{it} epoch{?s}:")
    if(x$parameters$validation > 0) {
      if (is.na(x$y_stats$mean)) {
        cat("validation loss after", chr_it,
            signif(x$loss[it], 5), "\n")
      } else {
        cat("scaled validation loss after", chr_it,
            signif(x$loss[it], 5), "\n")
      }
    } else {
      if (is.na(x$y_stats$mean)) {
        cat("training set loss after", chr_it,
            signif(x$loss[it], 5), "\n")
      } else {
        cat("scaled training set loss after", chr_it,
            signif(x$loss[it], 5), "\n")
      }
    }
  }
  invisible(x)
}

# ------------------------------------------------------------------------------


model_to_raw <- function(model) {
  con <- rawConnection(raw(), open = "w")
  on.exit({close(con)}, add = TRUE)
  torch::torch_save(model, con)
  r <- rawConnectionValue(con)
  r
}

# ------------------------------------------------------------------------------

