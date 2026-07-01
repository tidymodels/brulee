#' Summarize the architecture of a brulee model
#'
#' `summary()` methods \pkg{brulee} neural network models print a
#' layer-by-layer description of the fitted torch module: each component's
#' type, shape, and parameter count, followed by the total parameter count.
#' For `brulee_resnet`, residual (skip) connections and their projection
#' layers are shown at the block boundaries where they apply.
#'
#' @param object A `brulee_resnet`, `brulee_mlp`, `brulee_rln`,
#'   `brulee_auto_int`, or `brulee_saint` object.
#' @param ... Not used.
#'
#' @return The model object, invisibly. Called for its side effect of
#' printing the architecture.
#'
#' @examplesIf !brulee:::is_cran_check()
#' \donttest{
#' if (torch::torch_is_installed() && rlang::is_installed("modeldata")) {
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

  if (length(residual_at) > 0) {
    block_starts <- c(1L, residual_at[seq_len(length(residual_at) - 1L)] + 1L)
  } else {
    block_starts <- integer(0)
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
      if (start_idx == end_idx) {
        header <- paste0(
          "Residual group ",
          grp,
          " (block ",
          start_idx,
          ", + skip)"
        )
      } else {
        header <- paste0(
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

#' @rdname summary.brulee
#' @export
summary.brulee_saint <- function(object, ...) {
  module <- revive_model(object$model_obj)
  num_features <- object$dims$p_cat + object$dims$p_cont
  y_dim <- as.integer(module$y_dim)
  num_embedding <- object$parameters$num_embedding
  num_attn_heads <- object$parameters$num_attn_heads
  num_attn_blocks <- object$parameters$num_attn_blocks
  attention_type <- object$parameters$attention_type
  target_token <- isTRUE(object$parameters$target_token)

  total <- 0L
  cat(cli::style_bold("SAINT architecture"), "\n", sep = "")
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
    "\n",
    "attention: ",
    attention_type,
    " | embedding dim: ",
    num_embedding,
    " | target token: ",
    target_token,
    "\n\n",
    sep = ""
  )

  fmt_row <- function(label, n_par, indent = "  ") {
    sprintf(
      "%s%-36s %9s params\n",
      indent,
      label,
      format(n_par, big.mark = ",")
    )
  }

  # --- Embedding layer ---
  cat(cli::style_bold("Embedding layer"), "\n", sep = "")
  emb <- module$embedding

  if (isTRUE(emb$target_token)) {
    n_par_target <- as.integer(prod(emb$target_token_weight$shape))
    total <- total + n_par_target
    cat(fmt_row(
      paste0("Target token (1 x ", num_embedding, ")"),
      n_par_target
    ))
  }

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
      cat(fmt_row(label, n_par))
    }
  }

  if (emb$n_cont > 0) {
    n_par_each <- arch_param_count(emb$cont_mlps[[1]])
    n_par_total <- n_par_each * emb$n_cont
    total <- total + n_par_total
    label <- paste0(
      emb$n_cont,
      " x MLP(1 -> 100 -> ",
      num_embedding,
      ")"
    )
    cat(fmt_row(label, n_par_total))
  }
  cat("\n")

  # --- Transformer backbone ---
  type_label <- switch(
    attention_type,
    column = "column attention",
    row = "row attention",
    both = "column + row attention"
  )
  cat(
    cli::style_bold("Transformer backbone"),
    " (",
    num_attn_blocks,
    " block",
    if (num_attn_blocks > 1) "s",
    ", ",
    type_label,
    ")\n",
    sep = ""
  )

  backbone <- module$backbone

  if (attention_type == "both") {
    saint_print_colrow_blocks(backbone, num_attn_blocks, fmt_row)
  } else if (attention_type == "column") {
    saint_print_col_blocks(backbone, num_attn_blocks, fmt_row)
  } else {
    saint_print_row_blocks(backbone, num_attn_blocks, fmt_row)
  }

  total <- total + saint_backbone_params(backbone)
  cat("\n")

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
      cat(fmt_row(arch_fmt_module(mod), n_par))
    }

    if (!is.null(module$hidden_drop) && !arch_is_noop(module$hidden_drop)) {
      cat(fmt_row(arch_fmt_module(module$hidden_drop), 0L))
    }
    cat("\n")
  }

  # --- Output head ---
  cat(cli::style_bold("Output head"), "\n", sep = "")
  n_par <- arch_param_count(module$output_head)
  total <- total + n_par
  cat(fmt_row(arch_fmt_module(module$output_head), n_par))
  if (y_dim > 1L) {
    cat(fmt_row("Softmax", 0L))
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

saint_backbone_params <- function(backbone) {
  total <- 0L
  for (i in seq_along(backbone$layers)) {
    layer <- backbone$layers[[i]]
    child_names <- names(layer$children)
    for (nm in child_names) {
      total <- total + arch_param_count(layer[[nm]])
    }
  }
  total
}

saint_print_colrow_blocks <- function(backbone, num_blocks, fmt_row) {
  for (i in seq_len(num_blocks)) {
    layer <- backbone$layers[[i]]
    children <- names(layer$children)
    cat("  Block ", i, ":\n", sep = "")

    cat("    Column attention:\n")
    for (j in 1:4) {
      mod <- layer[[children[j]]]
      if (arch_is_noop(mod)) {
        next
      }
      cat(fmt_row(
        saint_module_label(mod),
        arch_param_count(mod),
        indent = "      "
      ))
    }

    cat("    Row attention:\n")
    for (j in 5:8) {
      mod <- layer[[children[j]]]
      if (arch_is_noop(mod)) {
        next
      }
      cat(fmt_row(
        saint_module_label(mod),
        arch_param_count(mod),
        indent = "      "
      ))
    }
  }
}

saint_print_col_blocks <- function(backbone, num_blocks, fmt_row) {
  for (i in seq_len(num_blocks)) {
    layer <- backbone$layers[[i]]
    children <- names(layer$children)
    cat("  Block ", i, ":\n", sep = "")
    for (nm in children) {
      mod <- layer[[nm]]
      if (arch_is_noop(mod)) {
        next
      }
      cat(fmt_row(
        saint_module_label(mod),
        arch_param_count(mod),
        indent = "    "
      ))
    }
  }
}

saint_print_row_blocks <- function(backbone, num_blocks, fmt_row) {
  for (i in seq_len(num_blocks)) {
    layer <- backbone$layers[[i]]
    children <- names(layer$children)
    cat("  Block ", i, ":\n", sep = "")
    for (nm in children) {
      mod <- layer[[nm]]
      if (arch_is_noop(mod)) {
        next
      }
      cat(fmt_row(
        saint_module_label(mod),
        arch_param_count(mod),
        indent = "    "
      ))
    }
  }
}

saint_module_label <- function(mod) {
  cls <- class(mod)[1]
  if (cls == "saint_attention") {
    heads <- as.integer(mod$heads)
    in_feat <- mod$to_qkv$in_features
    paste0("Attention(dim=", in_feat, ", heads=", heads, ")")
  } else if (cls == "saint_feedforward") {
    layers <- mod$net$children
    lin1 <- layers[[names(layers)[1]]]
    paste0("FeedForward(", lin1$in_features, ", GEGLU)")
  } else if (cls == "nn_layer_norm") {
    norm_shape <- as.integer(mod$normalized_shape)
    paste0("LayerNorm(", norm_shape, ")")
  } else {
    arch_fmt_module(mod)
  }
}
