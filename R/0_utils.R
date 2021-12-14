
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
  if (!is.null(x$parameters$class_weights)) {
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
    if(x$parameters$validation > 0) {
      if (is.na(x$y_stats$mean)) {
        cat("validation loss after", it, "epochs:",
            signif(x$loss[it], 5), "\n")
      } else {
        cat("scaled validation loss after", it, "epochs:",
            signif(x$loss[it], 5), "\n")
      }
    } else {
      if (is.na(x$y_stats$mean)) {
        cat("training set loss after", it, "epochs:",
            signif(x$loss[it], 5), "\n")
      } else {
        cat("scaled training set loss after", it, "epochs:",
            signif(x$loss[it], 5), "\n")
      }
    }
  }
  invisible(x)
}

# ------------------------------------------------------------------------------


brulee_coefs <- function(object, epoch = NULL, ...) {
  if (!is.null(epoch) && length(epoch) != 1) {
    rlang::abort("'epoch' should be a single integer.")
  }
  max_epochs <- length(object$estimates)

  if (is.null(epoch)) {
    epoch <- object$best_epoch
  } else {
    if (epoch > max_epochs) {
      msg <- glue::glue("There were only {max_epochs} epochs fit. Setting 'epochs' to {max_epochs}.")
      rlang::warn(msg)
      epoch <- max_epochs
    }

  }
  object$estimates[[epoch]]
}


#' Extract Model Coefficients
#'
#' @param object A model fit from \pkg{brulee}.
#' @param epoch A single integer for the training iteration. If left `NULL`,
#' the estimates from the best model fit (via internal performance metrics).
#' @param ... Not currently used.
#' @return For logistic/linear regression, a named vector. For neural networks,
#' a list of arrays.
#'
#' @name brulee-coefs
#' @export
coef.brulee_logistic_reg <- function(object, epoch = NULL, ...) {
  network_params <- brulee_coefs(object, epoch)
  slopes <- network_params$fc1.weight[,2] - network_params$fc1.weight[,1]
  int <- network_params$fc1.bias[2] - network_params$fc1.bias[1]
  param <- c(int, slopes)
  names(param) <- c("(Intercept)", object$dims$features)
  param
}

#' @rdname brulee-coefs
#' @export
coef.brulee_linear_reg <- function(object, epoch = NULL, ...) {
  network_params <- brulee_coefs(object, epoch)
  slopes <- network_params$fc1.weight[1,]
  int <- network_params$fc1.bias
  param <- c(int, slopes)
  names(param) <- c("(Intercept)", object$dims$features)
  param
}

#' @rdname brulee-coefs
#' @export
coef.brulee_mlp <- brulee_coefs

#' @rdname brulee-coefs
#' @export
coef.brulee_multinomial_reg <- function(object, epoch = NULL, ...) {
  network_params <- brulee_coefs(object, epoch)
  slopes <- t(network_params$fc1.weight)
  int <- network_params$fc1.bias
  param <- rbind(int, slopes)
  rownames(param) <- c("(Intercept)", object$dims$features)
  colnames(param) <- object$dims$levels
  param
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

