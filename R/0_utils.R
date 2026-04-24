# ------------------------------------------------------------------------------
# used in print methods

brulee_print <- function(x, ...) {
  lvl <- get_levels(x)

  n <- format(x$dims$n, big.mark = ",")
  p <- format(x$dims$p, big.mark = ",")

  data_lst <-
    c(
      " " = "Samples: {n}",
      " " = "Predictors: {p}"
    )
  if (!is.null(lvl)) {
    data_lst <- c(data_lst, " " = "Classes: {.val {lvl}}")
  }
  cli::cli_bullets(data_lst)

  cat("\n")

  param_lst <-
    c(
      " " = "Activation: {.val {x$parameters$activation}}",
      " " = "# Hidden Units: {x$parameters$hidden_units}"
    )
  if (inherits(x, "brulee_resnet")) {
    param_lst <- c(param_lst, " " = "# BatchNorm Outputs: {x$parameters$batch_norm_units}")
  }

  param_lst <-
    c(
      param_lst,
      c(
        " " = "Learning Rate: {signif(x$parameters$learn_rate, 3)}, Schedule:
        {.val {x$parameters$sched}}",
        " " = "Stopping iterations: {x$parameters$stop_iter}"
      )
    )
  if (x$parameters$validation > 0) {
    param_lst <- c(
      param_lst,
      " " = "% Validation: {signif(x$parameters$validation, 3)}"
    )
  }
  if (x$parameters$dropout > 0) {
    param_lst <- c(
      param_lst,
      " " = "Dropout: {signif(x$parameters$penalty, 3)}"
    )
  }
  if (x$parameters$penalty > 0) {
    param_lst <- c(
      param_lst,
      " " = "Penalty: {signif(x$parameters$penalty, 3)},
        {round(x$parameters$mixture * 100, 1)}% L1"
    )
  }
  if (x$parameters$momentum > 0) {
    param_lst <- c(
      param_lst,
      " " = "Momentum: {signif(x$parameters$momentum, 3)}"
    )
  }

  param_lst <- c(param_lst, " " = "Optimizer: {.val {x$parameters$optimizer}}")
  if (x$parameters$optimizer != "LBFGS") {
    param_lst <- c(param_lst, " " = "Batch Size: {x$parameters$batch_size}")
  }

  if (!is.null(x$dims$levels) && !is.null(x$parameters$class_weights)) {
    if (!all(x$parameters$class_weights == 1.0)) {
      # fmt: skip
      weights_str <-
     paste0(lvl, "=", format(x$parameters$class_weights, digits = 3),
            collapse = ", ")
      param_lst <- c(param_lst, " " = "Class Weights: {weights_str}")
    }
  }

  cli::cli_bullets(param_lst)

  n_params <- format(get_num_resnet_coef(x), big.mark = ",")

  res_list <- c(" " = "# Parameters: {n_params}")

  if (!is.null(x$loss)) {
    it <- x$best_epoch
    loss_val <- signif(x$loss[it], 3)
    epoch_str <- cli::pluralize("{it} epoch{?s}")

    if (x$parameters$validation > 0) {
      if (is.na(x$y_stats$mean)) {
        res_list <- c(
          res_list,
          " " = "validation loss after {epoch_str}: {loss_val}"
        )
      } else {
        res_list <- c(
          res_list,
          " " = "scaled validation loss after {epoch_str}: {loss_val}"
        )
      }
    } else {
      if (is.na(x$y_stats$mean)) {
        res_list <- c(
          res_list,
          " " = "training set loss after {epoch_str}: {loss_val}"
        )
      } else {
        res_list <- c(
          res_list,
          " " = "scaled training set loss after {epoch_str}: {loss_val}"
        )
      }
    }
  }

  cat("\n")
  cli::cli_bullets(res_list)

  invisible(x)
}

# ------------------------------------------------------------------------------

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
  # ADAMw always uses weight_decay in the optimizer (not in loss)
  if (opt == "ADAMw") {
    return(loss_fn)
  }
  # Pure L2 (mixture = 0) can be handled by optimizer's weight_decay, except LBFGS
  if (identical(mixture, 0.0) && opt != "LBFGS") {
    return(loss_fn)
  }
  # L1 or elastic net (mixture > 0) must be added to loss
  # LBFGS always needs penalty in loss (no weight_decay support)
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
