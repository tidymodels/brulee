last_param <- function(x) {
 last_epoch <- length(x$loss)
 coef(x, epoch = last_epoch)
}

save_coef <- function(x) {
 cl <- match.call()
 nm <- as.character(cl$x)
 fl <- file.path("saved", paste0(nm, ".rds"))
 if (file.exists(fl)) {
  return(FALSE)
 }
 res <- last_param(x)
 saveRDS(res, file = fl)
 invisible(TRUE)
}

load_coef <- function(x) {
 cl <- match.call()
 nm <- as.character(cl$x)
 fl <- file.path("saved", paste0(nm, ".rds"))
 if (!file.exists(fl)) {
  rlang::abort(paste("Can't find test results:", fl))
 }
 readRDS(fl)
}

