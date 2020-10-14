check_missing_data <- function(x, y, fn = "some function", verbose = FALSE) {
 compl_data <- complete.cases(x, y)
 if (any(!compl_data)) {
  x <- x[compl_data, , drop = FALSE]
  y <- y[compl_data]
  if (verbose) {
   cl_chr <- as.character()
   msg <- paste0(fn, "() removed ", sum(!compl_data), " rows of ",
                 "data due to missing values.")
   rlang::warn(msg)
  }
 }
 list(x = x, y = y)
}

check_data_att <- function(x, y) {
  # check matrices/vectors, matrix type, matrix column names
  if (!is.matrix(x) || !is.numeric(x)) {
    rlang::abort("'x' should be a numeric matrix.")
  }
  nms <- colnames(x)
  if (length(nms) != ncol(x)) {
    rlang::abort("Every column of 'x' should have a name.")
  }
  if (!is.vector(y)) {
    rlang::abort("'y' should be a vector.")
  }
  invisible(NULL)
}


format_msg <- function(fn, arg) {
 if (is.null(fn)) {
  fn <- "The function"
 } else {
  fn <- paste0(fn, "()")
 }
 paste0(fn, " expected '", arg, "'")
}

check_rng <- function(x, x_min, x_max, incl = c(TRUE, TRUE)) {
 if (incl[[1]]) {
  lo <- x <= x_min
 } else {
  lo <- x <  x_min
 }
 if (incl[[2]]) {
  hi <- x >= x_max
 } else {
  hi <- x >  x_max
 }
 lo | hi
}

numeric_loss_values <- c("mse", "poisson", "smooth_l1", "l1")
check_regression_loss <- function(loss_function) {
 check_character(loss_function, single = TRUE, vals = numeric_loss_values)

 # TODO return a different format
 dplyr::case_when(
  loss_function == "poisson" ~ "torch::nnf_poisson_nll_loss",
  loss_function == "smooth_l1" ~ "torch::nnf_smooth_l1_loss",
  loss_function == "l1" ~ "torch::nnf_l1_loss",
  TRUE ~ "torch::nnf_mse_loss"
 )

}

check_classification_loss <- function(x) {

}

check_optimizer <- function(x) {

}


check_integer <-
 function(x,
          single = TRUE,
          x_min = -Inf, x_max = Inf, incl = c(TRUE, TRUE),
          fn = NULL) {
  cl <- match.call()
  arg <- as.character(cl$x)

  if (!is.integer(x)) {
   msg <- paste(format_msg(fn, arg), "to be integer.")
   rlang::abort(msg)
  }

  if (single && length(x) > 1) {
   msg <- paste(format_msg(fn, arg), "to be a single integer.")
   rlang::abort(msg)
  }

  out_of_range <- check_rng(x, x_min, x_max, incl)
  if (any(out_of_range)) {
   msg <- paste0(format_msg(fn, arg),
                 " to be an integer on ",
                 ifelse(incl[[1]], "[", "("), x_min, ", ",
                 x_max, ifelse(incl[[2]], "]", ")"), ".")
   rlang::abort(msg)
  }

 invisible(TRUE)
}

check_double <- function(x,
                          single = TRUE,
                          x_min = -Inf, x_max = Inf, incl = c(TRUE, TRUE),
                          fn = NULL) {
 cl <- match.call()
 arg <- as.character(cl$x)

 if (!is.double(x)) {
  msg <- paste(format_msg(fn, arg), "to be a double.")
  rlang::abort(msg)
 }

 if (single && length(x) > 1) {
  msg <- paste(format_msg(fn, arg), "to be a single double.")
  rlang::abort(msg)
 }

 out_of_range <- check_rng(x, x_min, x_max, incl)
 if (any(out_of_range)) {
  msg <- paste0(format_msg(fn, arg),
                " to be a double on ",
                ifelse(incl[[1]], "[", "("), x_min, ", ",
                x_max, ifelse(incl[[2]], "]", ")"), ".")
  rlang::abort(msg)
 }

 invisible(TRUE)
}

check_character <- function(x, single = TRUE, vals = NULL, fn = NULL) {
 cl <- match.call()
 arg <- as.character(cl$x)

 if (!is.character(x)) {
  msg <- paste(format_msg(fn, arg), "to be character.")
  rlang::abort(msg)
 }

 if (single && length(x) > 1) {
  msg <- paste(format_msg(fn, arg), "to be a single character string.")
  rlang::abort(msg)
 }

 if (any(!(x %in% vals))) {
  msg <- paste0(format_msg(fn, arg),
                "  contains an incorrect value.")
  rlang::abort(msg)
 }

 invisible(TRUE)
}

check_logical <- function(x, single = TRUE, fn = NULL) {
 cl <- match.call()
 arg <- as.character(cl$x)

 if (!is.logical(x)) {
  msg <- paste(format_msg(fn, arg), "to be logical.")
  rlang::abort(msg)
 }

 if (single && length(x) > 1) {
  msg <- paste(format_msg(fn, arg), "to be a single logical.")
  rlang::abort(msg)
 }
 invisible(TRUE)
}


