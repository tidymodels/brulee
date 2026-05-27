#' Summarize the architecture of a brulee model
#'
#' `summary()` methods \pkg{brulee} neural network models print a
#' layer-by-layer description of the fitted torch module: each component's
#' type, shape, and parameter count, followed by the total parameter count.
#' For `brulee_resnet`, residual (skip) connections and their projection
#' layers are shown at the block boundaries where they apply.
#'
#' @param object A `brulee_resnet`, `brulee_mlp`, `brulee_rln`, or
#'   `brulee_auto_int` object.
#' @param ... Not used.
#'
#' @return The model object, invisibly. Called for its side effect of
#' printing the architecture.
#'
#' @examplesIf !brulee:::is_cran_check()
#' \donttest{
#' if (torch::torch_is_installed() & rlang::is_installed("modeldata")) {
#'   data(ames, package = "modeldata")
#'   ames$Sale_Price <- log10(ames$Sale_Price)
#'
#'   set.seed(1)
#'   fit <- brulee_resnet(Sale_Price ~ Longitude + Latitude, data = ames,
#'                        hidden_units = c(8, 4), bottleneck_units = c(6, 3),
#'                        residual_at = 2, epochs = 3)
#'   summary(fit)
#' }
#' }
#' @name summary.brulee
NULL

#' @rdname summary.brulee
#' @export
summary.brulee_resnet <- function(object, ...) {
  module <- revive_model(object$model_obj)
  num_pred <- length(object$dims$features)
  y_dim <- as.integer(module$y_dim)
  residual_at <- as.integer(module$residual_at)
  num_layers <- as.integer(module$num_layers)

  block_starts <- if (length(residual_at) > 0) {
    c(1L, residual_at[seq_len(length(residual_at) - 1L)] + 1L)
  } else {
    integer(0)
  }
  block_ends <- residual_at

  total <- 0L
  cat(cli::style_bold("Residual network architecture"), "\n", sep = "")
  cat(
    "inputs: ",
    num_pred,
    " | output dim: ",
    y_dim,
    " | layers: ",
    num_layers,
    "\n\n",
    sep = ""
  )

  for (i in seq_len(num_layers)) {
    is_block_start <- i %in% block_starts
    is_block_end <- i %in% block_ends

    if (is_block_start) {
      grp <- which(block_starts == i)
      start_idx <- block_starts[grp]
      end_idx <- block_ends[grp]
      header <- if (start_idx == end_idx) {
        paste0("Residual group ", grp, " (block ", start_idx, ", + skip)")
      } else {
        paste0(
          "Residual group ",
          grp,
          " (blocks ",
          start_idx,
          "-",
          end_idx,
          ", + skip)"
        )
      }
      cat(cli::style_bold(header), "\n", sep = "")
    } else if (length(residual_at) == 0 && i == 1L) {
      cat(cli::style_bold("Blocks (no residual connections)"), "\n", sep = "")
    }

    layer <- module$layers[[i]]
    cat("  Block ", i, ":\n", sep = "")
    for (nm in names(layer$children)) {
      mod <- layer[[nm]]
      if (arch_is_noop(mod)) {
        next
      }
      n_par <- arch_param_count(mod)
      total <- total + n_par
      cat(arch_fmt_row(arch_fmt_module(mod), n_par))
    }

    if (is_block_end) {
      proj_key <- as.character(i)
      if (proj_key %in% names(module$projection_layers)) {
        proj_name <- module$projection_layers[[proj_key]]
        proj <- module[[proj_name]]
        n_par <- arch_param_count(proj)
        total <- total + n_par
        cat(sprintf(
          "  + skip: %-26s %6s params\n",
          arch_fmt_module(proj),
          format(n_par, big.mark = ",")
        ))
      } else {
        cat("  + skip: identity (no parameters)\n")
      }
      cat("\n")
    }
  }

  cat(cli::style_bold("Output head"), "\n", sep = "")
  for (nm in c("bn_out", "linear_out")) {
    mod <- module[[nm]]
    n_par <- arch_param_count(mod)
    total <- total + n_par
    cat(arch_fmt_row(arch_fmt_module(mod), n_par))
  }
  if (y_dim > 1L) {
    cat(arch_fmt_row("Softmax", 0L))
  }

  cat(
    "\n",
    cli::style_bold("Total parameters: "),
    format(total, big.mark = ","),
    "\n",
    sep = ""
  )
  invisible(object)
}

#' @rdname summary.brulee
#' @export
summary.brulee_rln <- function(object, ...) {
  module <- revive_model(object$model_obj)
  num_pred <- length(object$dims$features)

  total <- 0L
  cat(
    cli::style_bold("Regularization Learning Network architecture"),
    "\n",
    sep = ""
  )
  cat(
    "inputs: ",
    num_pred,
    " | hidden units: ",
    object$parameters$hidden_units,
    " | activation: ",
    object$parameters$activation,
    "\n\n",
    sep = ""
  )

  for (nm in c("linear1", "act", "linear2")) {
    mod <- module[[nm]]
    n_par <- arch_param_count(mod)
    total <- total + n_par
    cat(arch_fmt_row(arch_fmt_module(mod), n_par, indent = "  "))
  }

  cat(
    "\n",
    cli::style_bold("Total parameters: "),
    format(total, big.mark = ","),
    "\n",
    sep = ""
  )
  invisible(object)
}

#' @rdname summary.brulee
#' @export
summary.brulee_auto_int <- function(object, ...) {
  module <- revive_model(object$model_obj)
  num_features <- object$dims$p_cat + object$dims$p_cont
  y_dim <- as.integer(module$y_dim)
  num_embedding <- object$parameters$num_embedding
  num_attn_heads <- object$parameters$num_attn_heads
  num_attn_feat <- object$parameters$num_attn_feat
  num_attn_blocks <- object$parameters$num_attn_blocks
  num_attn <- num_attn_feat * num_attn_heads

  total <- 0L
  cat(cli::style_bold("AutoInt architecture"), "\n", sep = "")
  cat(
    "inputs: ",
    num_features,
    " (",
    object$dims$p_cat,
    " categorical, ",
    object$dims$p_cont,
    " numeric)",
    " | output dim: ",
    y_dim,
    "\n\n",
    sep = ""
  )

  # --- Embedding layer ---
  cat(cli::style_bold("Embedding layer"), "\n", sep = "")
  emb <- module$embedding
  if (emb$n_cat > 0) {
    for (i in seq_len(emb$n_cat)) {
      mod <- emb$cat_embeddings[[i]]
      n_par <- arch_param_count(mod)
      total <- total + n_par
      label <- paste0(
        "Embedding(",
        mod$num_embeddings,
        " -> ",
        mod$embedding_dim,
        ")"
      )
      cat(arch_fmt_row(label, n_par, indent = "  "))
    }
  }
  if (emb$n_cont > 0) {
    n_par <- emb$cont_weights$numel()
    total <- total + n_par
    label <- paste0(
      "ContWeights(",
      emb$n_cont,
      " x ",
      num_embedding,
      ")"
    )
    cat(arch_fmt_row(label, n_par, indent = "  "))
  }

  emb_drop <- module$embedding_drop
  if (!arch_is_noop(emb_drop)) {
    cat(arch_fmt_row(arch_fmt_module(emb_drop), 0L, indent = "  "))
  }
  cat("\n")

  # --- Self-attention backbone ---
  cat(
    cli::style_bold("Self-attention backbone"),
    " (",
    num_attn_blocks,
    " block",
    if (num_attn_blocks > 1) "s",
    ", ",
    num_attn_heads,
    " head",
    if (num_attn_heads > 1) "s",
    ")\n",
    sep = ""
  )

  backbone <- module$backbone
  n_par <- arch_param_count(backbone$input_proj)
  total <- total + n_par
  cat(arch_fmt_row(arch_fmt_module(backbone$input_proj), n_par, indent = "  "))

  for (i in seq_len(num_attn_blocks)) {
    attn <- backbone$attention_layers[[i]]
    n_par <- arch_param_count(attn)
    total <- total + n_par
    label <- paste0(
      "MultiheadAttention(",
      num_attn,
      ", heads=",
      num_attn_heads,
      ")"
    )
    cat(arch_fmt_row(label, n_par, indent = "  "))
  }

  n_par <- arch_param_count(backbone$V_res)
  total <- total + n_par
  cat(sprintf(
    "  + skip: %-24s %6s params\n",
    arch_fmt_module(backbone$V_res),
    format(n_par, big.mark = ",")
  ))

  act_label <- arch_fmt_module(backbone$act)
  cat(arch_fmt_row(act_label, 0L, indent = "  "))
  cat(
    "  output: ",
    num_features,
    " embeddings of dim ",
    num_attn,
    " (flattened: ",
    num_features * num_attn,
    ")\n\n",
    sep = ""
  )

  # --- Hidden layers (optional) ---
  if (!is.null(module$hidden)) {
    cat(cli::style_bold("Hidden layers"), "\n", sep = "")
    child_names <- names(module$hidden$children)
    for (nm in child_names) {
      mod <- module$hidden[[nm]]
      if (arch_is_noop(mod)) {
        next
      }
      n_par <- arch_param_count(mod)
      total <- total + n_par
      cat(arch_fmt_row(arch_fmt_module(mod), n_par, indent = "  "))
    }

    if (!is.null(module$hidden_drop) && !arch_is_noop(module$hidden_drop)) {
      cat(arch_fmt_row(arch_fmt_module(module$hidden_drop), 0L, indent = "  "))
    }
    cat("\n")
  }

  # --- Output head ---
  cat(cli::style_bold("Output head"), "\n", sep = "")
  n_par <- arch_param_count(module$output_head)
  total <- total + n_par
  cat(arch_fmt_row(arch_fmt_module(module$output_head), n_par, indent = "  "))
  if (y_dim > 1L) {
    cat(arch_fmt_row("Softmax", 0L, indent = "  "))
  }

  cat(
    "\n",
    cli::style_bold("Total parameters: "),
    format(total, big.mark = ","),
    "\n",
    sep = ""
  )
  invisible(object)
}
