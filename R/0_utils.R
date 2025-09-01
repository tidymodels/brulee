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
    format(x$dims$n, big.mark = ","),
    "samples,",
    format(x$dims$p, big.mark = ","),
    "features,",
    chr_y,
    "\n"
  )
  if (!is.null(x$dims$levels) && !is.null(x$parameters$class_weights)) {
    cat(
      "class weights",
      paste0(
        lvl,
        "=",
        format(x$parameters$class_weights),
        collapse = ", "
      ),
      "\n"
    )
  }
  if (x$parameters$penalty > 0) {
    cat("weight decay:", x$parameters$penalty, "\n")
  }
  if (any(names(x$parameters) == "dropout")) {
    cat("dropout proportion:", x$parameters$dropout, "\n")
  }
  cat("batch size:", x$parameters$batch_size, "\n")

  if (all(c("sched", "sched_opt") %in% names(x$parameters))) {
    cat_schedule(x$parameters)
  }

  if (!is.null(x$loss)) {
    it <- x$best_epoch
    chr_it <- cli::pluralize("{it} epoch{?s}:")
    if (x$parameters$validation > 0) {
      if (is.na(x$y_stats$mean)) {
        cat("validation loss after", chr_it, signif(x$loss[it], 3), "\n")
      } else {
        cat("scaled validation loss after", chr_it, signif(x$loss[it], 3), "\n")
      }
    } else {
      if (is.na(x$y_stats$mean)) {
        cat("training set loss after", chr_it, signif(x$loss[it], 3), "\n")
      } else {
        cat(
          "scaled training set loss after",
          chr_it,
          signif(x$loss[it], 3),
          "\n"
        )
      }
    }
  }
  invisible(x)
}

# ------------------------------------------------------------------------------

cat_schedule <- function(x) {
  if (x$sched == "none") {
    cat("learn rate:", x$learn_rate, "\n")
  } else {
    .fn <- paste0("schedule_", x$sched)
    cl <- rlang::call2(.fn, !!!x$sched_opt)
    chr_cl <- rlang::expr_deparse(cl, width = 200)

    cat(gsub("^schedule_", "schedule: ", chr_cl), "\n")
  }
  invisible(NULL)
}

# ------------------------------------------------------------------------------

#TODO make sure indicies are good with extra result
# update print method for class weights and val loss
# cli update

model_to_raw <- function(model) {
  con <- rawConnection(raw(), open = "w")
  on.exit(
    {
      close(con)
    },
    add = TRUE
  )
  torch::torch_save(model, con)
  r <- rawConnectionValue(con)
  r
}

# ------------------------------------------------------------------------------

lx_term <- function(norm) {
  function(model) {
    is_bias <- grepl("bias", names(model$parameters))
    coefs <- model$parameters[!is_bias]
    l <- lapply(coefs, function(x) {
      torch::torch_sum(norm(x))
    })
    torch::torch_sum(torch::torch_stack(l))
  }
}

l2_term <- lx_term(function(x) torch::torch_pow(x, 2))
l1_term <- lx_term(function(x) torch::torch_abs(x))

# -------------------------------------------------------------------------

make_penalized_loss <- function(loss_fn, model, penalty, mixture, opt) {
  if (!opt_uses_penalty(opt)) {
    return(loss_fn)
  }
  force(loss_fn)
  function(...) {
    loss <- loss_fn(...)
    if (penalty > 0) {
      l_term <- mixture * l1_term(model) + (1 - mixture) / 2 * l2_term(model)
      l_term <- float_64(l_term)
      penalty <- float_64(penalty)
      loss <- loss + penalty * l_term
    }
    loss
  }
}

# adapted from ps:::is_cran_check()
is_cran_check <- function() {
  if (identical(Sys.getenv("NOT_CRAN"), "true")) {
    FALSE
  } else {
    Sys.getenv("_R_CHECK_PACKAGE_NAME_", "") != ""
  }
}

float_64 <- function(x) {
 torch::torch_tensor(x, dtype = torch::torch_float64())
}

