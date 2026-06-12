# ------------------------------------------------------------------------------
# Zero-padded epoch labels for verbose output

format_epoch_labels <- function(x) {
  gsub(" ", "0", format(x))
}

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

  param_lst <- c()

  # Neural network specific parameters
  if (!is.null(x$parameters$activation)) {
    param_lst <- c(
      param_lst,
      " " = "Activation: {.val {x$parameters$activation}}",
      " " = "# Hidden Units: {x$parameters$hidden_units}"
    )
  }

  if (inherits(x, "brulee_resnet")) {
    param_lst <- c(
      param_lst,
      " " = "# Bottleneck Units: {x$parameters$bottleneck_units}"
    )
  }

  # Common parameters
  if (!is.null(x$parameters$sched)) {
    param_lst <- c(
      param_lst,
      " " = "Learning Rate: {signif(x$parameters$learn_rate, 3)}, Schedule:
        {.val {x$parameters$sched}}"
    )
  } else {
    param_lst <- c(
      param_lst,
      " " = "Learning Rate: {signif(x$parameters$learn_rate, 3)}"
    )
  }
  if (!is.null(x$parameters$stop_iter)) {
    param_lst <- c(
      param_lst,
      " " = "Stopping iterations: {x$parameters$stop_iter}"
    )
  }

  if (x$parameters$validation > 0) {
    param_lst <- c(
      param_lst,
      " " = "% Validation: {signif(x$parameters$validation, 3)}"
    )
  }

  if (!is.null(x$parameters$dropout) && x$parameters$dropout > 0) {
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

  if (!is.null(x$parameters$optimizer)) {
    param_lst <- c(
      param_lst,
      " " = "Optimizer: {.val {x$parameters$optimizer}}"
    )
    if (x$parameters$optimizer != "LBFGS") {
      param_lst <- c(param_lst, " " = "Batch Size: {x$parameters$batch_size}")
    }
  }

  if (!is.null(x$device)) {
    param_lst <- c(param_lst, " " = "Device: {.val {x$device}}")
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
      # Create penalty tensor on the same device as l_term
      # l_term is already float64 from model parameters, on the correct device
      penalty_tensor <- torch::torch_tensor(
        penalty,
        dtype = torch::torch_float64(),
        device = l_term$device
      )
      loss <- loss + penalty_tensor * l_term
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

float_64 <- function(x, device = NULL) {
  if (is.null(device)) {
    # Let torch_tensor use the current device context (from with_device)
    torch::torch_tensor(x, dtype = torch::torch_float64())
  } else {
    # Explicitly specify device when provided
    torch::torch_tensor(x, dtype = torch::torch_float64(), device = device)
  }
}

# Convert class weights to tensor on current device
# Returns NULL if weights are NULL, otherwise converts to float64 tensor
weights_to_tensor <- function(wts) {
  if (is.null(wts)) {
    return(NULL)
  }
  float_64(wts)
}

# ------------------------------------------------------------------------------
# Architecture summary helpers (shared by summary.brulee_resnet / .brulee_mlp)

arch_activation_label <- c(
  nn_relu = "ReLU",
  nn_elu = "ELU",
  nn_tanh = "Tanh",
  nn_log_sigmoid = "LogSigmoid",
  nn_relu6 = "ReLU6",
  nn_leaky_relu = "LeakyReLU",
  nn_gelu = "GELU",
  nn_celu = "CELU",
  nn_selu = "SELU"
)

arch_param_count <- function(m) {
  params <- m$parameters
  if (length(params) == 0) {
    return(0L)
  }
  sum(vapply(params, function(p) p$numel(), integer(1)))
}

arch_fmt_module <- function(m) {
  cls <- class(m)[1]
  if (cls %in% names(arch_activation_label)) {
    return(arch_activation_label[[cls]])
  }
  switch(
    cls,
    nn_batch_norm1d = paste0("BatchNorm1d(", m$num_features, ")"),
    nn_linear = paste0("Linear(", m$in_features, " -> ", m$out_features, ")"),
    nn_dropout = paste0("Dropout(p = ", format(m$p), ")"),
    nn_softmax = "Softmax",
    sub("^nn_", "", cls)
  )
}

arch_fmt_row <- function(label, n_par, indent = "    ") {
  sprintf(
    "%s%-32s %6s params\n",
    indent,
    label,
    format(n_par, big.mark = ",")
  )
}

arch_is_noop <- function(m) {
  inherits(m, "nn_dropout") && isTRUE(m$p == 0)
}

last_epoch_note <- function(epoch, max_epoch, call = rlang::caller_env()) {
  if (epoch > max_epoch) {
    cli::cli_warn(
      "The model fit only {max_epoch} epoch{?s}; predictions cannot be made at
   epoch {epoch}, so last epoch is used.",
      call = call
    )
  }
  invisible(NULL)
}
