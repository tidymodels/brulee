check_missing_data <- function(x, y, fn = "some function", verbose = FALSE) {
  compl_data <- complete.cases(x, y)
  if (any(!compl_data)) {
    x <- x[compl_data, , drop = FALSE]
    y <- y[compl_data]
    if (verbose) {
      cl_chr <- as.character()
      msg <- paste0(fn, "() removed ", sum(!compl_data), " rows of ",
                    "data due to missing values.")
      cli::cli_warn(msg)
    }
  }
  list(x = x, y = y)
}

check_data_att <- function(x, y) {
  hardhat::validate_outcomes_are_univariate(y)

  # check matrices/vectors, matrix type, matrix column names
  if (!is.matrix(x) || !is.numeric(x)) {
    cli::cli_abort("'x' should be a numeric matrix.")
  }
  nms <- colnames(x)
  if (length(nms) != ncol(x)) {
    cli::cli_abort("Every column of 'x' should have a name.")
  }
  if (!is.vector(y) & !is.factor(y)) {
    cli::cli_abort("'y' should be a vector.")
  }
  invisible(NULL)
}

numeric_loss_values <- c("mse", "poisson", "smooth_l1", "l1")
check_regression_loss <- function(loss_function) {
 loss_function <- rlang::arg_match0(loss_function, numeric_loss_values) # TODO add call

  # TODO return a different format
  dplyr::case_when(
    loss_function == "poisson" ~ "torch::nnf_poisson_nll_loss",
    loss_function == "smooth_l1" ~ "torch::nnf_smooth_l1_loss",
    loss_function == "l1" ~ "torch::nnf_l1_loss",
    TRUE ~ "torch::nnf_mse_loss"
  )

}

check_single_logical <- function(x, call = rlang::caller_env()) {
 cl <- match.call()
 arg_nm <- as.character(cl$x)
 msg <- "{.arg {arg_nm}} should be a single logical value, not {brulee:::obj_type_friendly(x)}."
  if (!is.logical(x)) {
   cli::cli_abort(msg, call = call)
  }
 if (length(x) > 1 || any(is.na(x))) {
  cli::cli_abort(msg, call = call)
 }
 invisible(x)
}

check_number_whole_vec <- function(x, call = rlang::caller_env(), ...) {
 cl <- match.call()
 arg <- as.character(cl$x)

 for (i in x) {
  rlang:::check_number_whole(i, arg = arg, call = call, ...)
 }
 x <- as.integer(x)
 invisible(x)
}

check_number_decimal_vec <- function(x, call = rlang::caller_env(), ...) {
 cl <- match.call()
 arg <- as.character(cl$x)

 for (i in x) {
  rlang:::check_number_decimal(i, arg = arg, call = call, ...)
 }
 invisible(x)
}

check_class_weights <- function(wts, lvls, xtab, fn) {
  if (length(lvls) == 0) {
    return(NULL)
  }

  if (is.null(wts)) {
    wts <- rep(1, length(lvls))
    return(torch::torch_tensor(wts))
  }
  if (!is.numeric(wts)) {
    cli::cli_abort("Class weights should be a numeric vector") # TODO rewrite
  }

  if (length(wts) == 1) {
    val <- wts
    wts <- rep(1, length(lvls))
    minority <- names(xtab)[which.min(xtab)]
    wts[lvls == minority] <- val
    names(wts) <- lvls
  }

  if (length(lvls) != length(wts)) {
    msg <- paste0("There were ", length(wts), " class weights given but ",
                  length(lvls), " were expected.")
    cli::cli_abort(msg)
  }

  nms <- names(wts)
  if (is.null(nms)) {
    names(wts) <- lvls
  } else {
    if (!identical(sort(nms), sort(lvls))) {
      msg <- paste("Names for class weights should be:",
                   paste0("'", lvls, "'", collapse = ", "))
      cli::cli_abort(msg)
    }
    wts <- wts[lvls]
  }


  torch::torch_tensor(wts)
}
