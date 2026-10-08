# ------------------------------------------------------------------------------
# Zero-padded epoch labels for verbose output

format_epoch_labels <- function(x) {
  gsub(" ", "0", format(x))
}

# ------------------------------------------------------------------------------
# Training-set subsampling for the foundation models (TabICL, TabPFN)

# The number of rows to keep from each stratum so that exactly `limit` rows
# are kept in total: proportional to the stratum sizes `counts`, at least one
# per stratum, with the rounding error moved one row at a time to or from the
# largest strata. Needs `limit` >= the number of strata with rows.
brulee_stratum_sizes <- function(counts, limit) {
  alloc <- pmax(1L, as.integer(round(limit * counts / sum(counts))))
  drift <- limit - sum(alloc)
  while (drift != 0L) {
    if (drift > 0L) {
      cand <- which(alloc < counts)
      if (length(cand) == 0L) {
        break
      }
      j <- cand[which.max(counts[cand])]
      alloc[j] <- alloc[j] + 1L
      drift <- drift - 1L
    } else {
      cand <- which(alloc > 1L)
      j <- cand[which.max(counts[cand])]
      alloc[j] <- alloc[j] - 1L
      drift <- drift + 1L
    }
  }
  alloc
}

# The training data of a foundation model fit, after checking the outcome:
# one numeric or factor column. Rows with a missing outcome are dropped with a
# warning; the models handle missing predictors but not missing outcomes.
brulee_foundation_data <- function(processed, call = rlang::caller_env()) {
  outcomes <- processed$outcomes
  if (ncol(outcomes) != 1) {
    cli::cli_abort(
      "The outcome must be a single column, not {ncol(outcomes)} columns.",
      call = call
    )
  }
  outcome <- outcomes[[1]]
  if (!is.factor(outcome) && !is.numeric(outcome)) {
    cli::cli_abort(
      "The outcome must be a factor (classification) or numeric
       (regression), not {obj_type_friendly(outcome)}.",
      call = call
    )
  }
  predictors <- processed$predictors
  missing <- is.na(outcome)
  if (all(missing)) {
    cli::cli_abort("Every value of the outcome is missing.", call = call)
  }
  if (any(missing)) {
    cli::cli_warn(
      "Removed {sum(missing)} row{?s} with a missing outcome.",
      call = call
    )
    outcome <- outcome[!missing]
    predictors <- predictors[!missing, , drop = FALSE]
  }
  check_outcome_varies(outcome, call = call)
  list(predictors = predictors, outcome = outcome)
}

# Rows kept by `training_set_limit`: all of them when there are at most
# `limit`, otherwise exactly `limit`, stratified by class (classification) or
# by quartile (regression). Uses R's random number generator; callers that
# need a particular seed set it around the call.
brulee_subsample_rows <- function(outcome, limit, call = rlang::caller_env()) {
  n <- length(outcome)
  if (n <= limit) {
    return(seq_len(n))
  }
  limit <- as.integer(limit)
  if (is.factor(outcome)) {
    strata <- droplevels(outcome)
    if (limit < nlevels(strata)) {
      cli::cli_abort(
        "{.arg training_set_limit} ({limit}) is smaller than the number of
         outcome classes ({nlevels(strata)}); cannot keep at least one row
         per class.",
        call = call
      )
    }
  } else {
    # Quartiles, or fewer strata when the limit is below 4. A constant or
    # nearly constant outcome can have fewer distinct breaks.
    n_strata <- min(4L, limit)
    probs <- seq(0, 1, length.out = n_strata + 1)
    breaks <- unique(stats::quantile(outcome, probs))
    if (length(breaks) < 2) {
      strata <- factor(rep(1L, n))
    } else {
      strata <- cut(outcome, breaks, include.lowest = TRUE)
    }
  }
  idx <- split(seq_len(n), strata, drop = TRUE)
  sizes <- brulee_stratum_sizes(lengths(idx), limit)
  taken <- purrr::map2(idx, sizes, \(i, size) i[sample.int(length(i), size)])
  sort(unlist(taken, use.names = FALSE))
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
    # `loss` includes epoch zero at position 1, so the best epoch's loss is at
    # `it + 1` (matching the predict()/coef() indexing).
    loss_val <- signif(x$loss[it + 1], 3)
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
      # l_term is already float32 from model parameters, on the correct device
      penalty_tensor <- torch::torch_tensor(
        penalty,
        dtype = torch::torch_float32(),
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

# --- Device-handling notes (read once, applied throughout brulee) ------------
#
# brulee runs on three torch backends: CPU, CUDA, and MPS (Apple Metal). Two
# subtleties with how torch (R) and the MPS backend behave shape the code in
# this file and in every *-fit.R / *-predict.R helper:
#
# 1. `torch::with_device(device = X, { ... })` does NOT make `torch_tensor()`
#    calls inside the block default to device X. New tensors created with
#    `torch_tensor()` land on the CPU regardless of the surrounding
#    `with_device` block. The fit/predict code therefore passes
#    `device = device` explicitly to every `torch_tensor()` / `float_32()` /
#    `weights_to_tensor()` call. Without this, tensors land on the CPU while
#    the model is on MPS/CUDA, producing "Placeholder storage" errors at
#    forward time on MPS or device-mismatch errors on CUDA.
#
# 2. The MPS RNG is not reliably reset by `torch::torch_manual_seed()`.
#    Inside a `with_device(mps, ...)` block, `nn_linear()` and related
#    constructors *do* allocate parameters directly on MPS, and the
#    subsequent `nn_init_*` calls then draw from the MPS RNG. Even after
#    calling `set.seed(s)` + `torch_manual_seed(s)`, two fits on MPS will
#    produce different initial weights. The fix is to construct modules
#    OUTSIDE `with_device` so initialization runs on the CPU (whose RNG IS
#    properly seeded), then move the module to the target device with
#    `model$to(device = device)`. CUDA's RNG is well-behaved with
#    `torch_manual_seed()`, but the same pattern is applied uniformly so
#    that all three backends produce reproducible inits from the same seed.
#
# Pattern in every fit function:
#   torch::torch_manual_seed(start_seed + 1)  # seed CPU RNG
#   model <- foo_module(...)                  # build params on CPU
#   model$to(device = device)                 # move to target device
#   training_output <- torch::with_device(device = device, {
#     ...                                     # data loaders + training loop
#   })
# -----------------------------------------------------------------------------

# Create a float32 tensor on the given device. `device = NULL` is *not* a
# request for "the current with_device context" -- torch does not propagate
# that to torch_tensor(). NULL just means "let torch pick", which in practice
# is the CPU. Callers from inside training loops or dataloader closures must
# pass `device` explicitly.
float_32 <- function(x, device = NULL) {
  if (is.null(device)) {
    torch::torch_tensor(x, dtype = torch::torch_float32())
  } else {
    torch::torch_tensor(x, dtype = torch::torch_float32(), device = device)
  }
}

# Convert class weights to a float32 tensor on the given device.
# Returns NULL if weights are NULL.
#
# `device` must be passed explicitly when called from inside a training-loop
# closure: `with_device` context does not propagate through coro/dataloader
# closure execution, so without an explicit device the weight tensor lands on
# the CPU while inputs/targets live on the GPU. The per-call signature
# `weights_to_tensor(wts, device = input$device)` ensures co-location with
# whatever the loss function received.
weights_to_tensor <- function(wts, device = NULL) {
  if (is.null(wts)) {
    return(NULL)
  }
  float_32(wts, device = device)
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
  sum(purrr::map_int(params, \(p) p$numel()))
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

# Convert classification model output to a row-stochastic probability tensor.
# Modules fit on brulee >= the cross-entropy refactor emit raw logits and carry
# `output_type = "logits"` on the result; older fits emitted softmax-normalized
# probabilities and have no `output_type`, so we leave them alone.
to_probs <- function(x, model) {
  if (
    isTRUE(model$dims$y > 1L) &&
      identical(model$output_type, "logits")
  ) {
    x <- torch::nnf_softmax(x, dim = 2L)
  }
  x
}

# Reduce an out-of-range `epoch` to the last one that was fit, warning when that
# happens. Callers pass the `estimates` list itself rather than a length so that
# the off-by-one below lives in exactly one place.
clamp_epoch <- function(epoch, estimates, call = rlang::caller_env()) {
  # All models prepend the initial pre-training parameters as the first element
  # of `estimates` (epoch zero), so an epoch number is one less than its
  # position in the list and the largest epoch that can be indexed is
  # `length(estimates) - 1L`.
  max_epoch <- length(estimates) - 1L
  if (epoch > max_epoch) {
    cli::cli_warn(
      "The model was fit for {max_epoch} epoch{?s}; the last epoch is used
       instead of epoch {epoch}.",
      call = call
    )
    epoch <- max_epoch
  }
  epoch
}
