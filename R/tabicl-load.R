# Load released TabICL weights into a `tabicl_model`.
#
# The released `.ckpt` is converted offline to task-prefixed `<task>.config.json`
# + `<task>.model.safetensors` (see dev/tabicl/convert_ckpt.py). This file parses
# that config, builds the R
# module tree, and copies every checkpoint tensor into the matching parameter,
# verifying that the mapping is exact in both directions (no unmatched file keys,
# no unfilled parameters). The map mirrors the Python state_dict dot-paths.

# ------------------------------------------------------------------------------
# state_dict key -> R parameter map

# Each builder returns a named list mapping Python state_dict keys (relative to
# `prefix`) to the corresponding R parameter tensors.

tabicl_map_linear <- function(prefix, layer) {
  list2 <- list()
  list2[[paste0(prefix, "weight")]] <- layer$weight
  list2[[paste0(prefix, "bias")]] <- layer$bias
  list2
}

# qassmax-mlp-elementwise SSMax: nn.Sequential indices 0/2 are Linear layers,
# which are R 1-based [[1]] / [[3]].
tabicl_map_ssmax <- function(prefix, layer) {
  c(
    tabicl_map_linear(paste0(prefix, "base_mlp.0."), layer$base_mlp[[1]]),
    tabicl_map_linear(paste0(prefix, "base_mlp.2."), layer$base_mlp[[3]]),
    tabicl_map_linear(paste0(prefix, "query_mlp.0."), layer$query_mlp[[1]]),
    tabicl_map_linear(paste0(prefix, "query_mlp.2."), layer$query_mlp[[3]])
  )
}

tabicl_map_mha <- function(prefix, mha) {
  m <- list()
  m[[paste0(prefix, "in_proj_weight")]] <- mha$in_proj_weight
  m[[paste0(prefix, "in_proj_bias")]] <- mha$in_proj_bias
  m <- c(m, tabicl_map_linear(paste0(prefix, "out_proj."), mha$out_proj))
  if (!is.null(mha$ssmax_layer)) {
    m <- c(m, tabicl_map_ssmax(paste0(prefix, "ssmax_layer."), mha$ssmax_layer))
  }
  m
}

tabicl_map_block <- function(prefix, block) {
  m <- list()
  m[[paste0(prefix, "norm1.weight")]] <- block$norm1$weight
  m[[paste0(prefix, "norm2.weight")]] <- block$norm2$weight
  if (!is.null(block$norm1$bias)) {
    m[[paste0(prefix, "norm1.bias")]] <- block$norm1$bias
    m[[paste0(prefix, "norm2.bias")]] <- block$norm2$bias
  }
  m <- c(
    m,
    tabicl_map_linear(paste0(prefix, "linear1."), block$linear1),
    tabicl_map_linear(paste0(prefix, "linear2."), block$linear2),
    tabicl_map_mha(paste0(prefix, "attn."), block$attn)
  )
  m
}

tabicl_map_isab <- function(prefix, isab) {
  m <- list()
  m[[paste0(prefix, "ind_vectors")]] <- isab$ind_vectors
  c(
    m,
    tabicl_map_block(paste0(prefix, "multihead_attn1."), isab$multihead_attn1),
    tabicl_map_block(paste0(prefix, "multihead_attn2."), isab$multihead_attn2)
  )
}

# Full model: col_embedder -> row_interactor -> icl_predictor.
tabicl_state_map <- function(model) {
  m <- list()

  # Column embedder
  ce <- model$col_embedder
  m <- c(m, tabicl_map_linear("col_embedder.in_linear.", ce$in_linear))
  m <- c(m, tabicl_map_linear("col_embedder.y_encoder.", ce$y_encoder))
  for (i in seq_along(ce$tf_col$blocks)) {
    m <- c(
      m,
      tabicl_map_isab(
        sprintf("col_embedder.tf_col.blocks.%d.", i - 1L),
        ce$tf_col$blocks[[i]]
      )
    )
  }

  # Row interactor
  ri <- model$row_interactor
  m[["row_interactor.cls_tokens"]] <- ri$cls_tokens
  m[["row_interactor.out_ln.weight"]] <- ri$out_ln$weight
  if (!is.null(ri$out_ln$bias)) {
    m[["row_interactor.out_ln.bias"]] <- ri$out_ln$bias
  }
  m[["row_interactor.tf_row.rope.freqs"]] <- ri$tf_row$rope$freqs
  for (i in seq_along(ri$tf_row$blocks)) {
    m <- c(
      m,
      tabicl_map_block(
        sprintf("row_interactor.tf_row.blocks.%d.", i - 1L),
        ri$tf_row$blocks[[i]]
      )
    )
  }

  # ICL predictor
  icl <- model$icl_predictor
  if (!is.null(icl$ln)) {
    m[["icl_predictor.ln.weight"]] <- icl$ln$weight
    if (!is.null(icl$ln$bias)) {
      m[["icl_predictor.ln.bias"]] <- icl$ln$bias
    }
  }
  m <- c(m, tabicl_map_linear("icl_predictor.y_encoder.", icl$y_encoder))
  m <- c(m, tabicl_map_linear("icl_predictor.decoder.0.", icl$decoder[[1]]))
  m <- c(m, tabicl_map_linear("icl_predictor.decoder.2.", icl$decoder[[3]]))
  for (i in seq_along(icl$tf_icl$blocks)) {
    m <- c(
      m,
      tabicl_map_block(
        sprintf("icl_predictor.tf_icl.blocks.%d.", i - 1L),
        icl$tf_icl$blocks[[i]]
      )
    )
  }

  m
}

# ------------------------------------------------------------------------------
# Config + model construction

# Parse the converter's config.json into the fields tabicl_model() needs.
tabicl_parse_config <- function(path) {
  cfg <- jsonlite::fromJSON(path)
  required <- c(
    "max_classes",
    "num_quantiles",
    "embed_dim",
    "col_num_blocks",
    "col_nhead",
    "col_num_inds",
    "col_feature_group_size",
    "col_target_aware",
    "col_ssmax",
    "row_num_blocks",
    "row_nhead",
    "row_num_cls",
    "row_rope_base",
    "icl_num_blocks",
    "icl_nhead",
    "icl_ssmax",
    "ff_factor",
    "activation",
    "norm_first",
    "bias_free_ln"
  )
  missing <- setdiff(required, names(cfg))
  if (length(missing) > 0) {
    cli::cli_abort("Config is missing required field{?s}: {.val {missing}}.")
  }
  cfg
}

# ------------------------------------------------------------------------------
# Weight loading

# Copy every checkpoint tensor into the matching model parameter, erroring if the
# mapping is not exact in both directions.
load_tabicl_weights <- function(model, tensors, verbose = FALSE) {
  assignments <- tabicl_state_map(model)
  file_keys <- names(tensors)
  map_keys <- names(assignments)

  missing_in_file <- setdiff(map_keys, file_keys)
  unmatched_in_file <- setdiff(file_keys, map_keys)
  if (length(missing_in_file) > 0) {
    cli::cli_abort(c(
      "Checkpoint is missing {length(missing_in_file)} expected parameter{?s}.",
      "x" = "First missing: {.val {utils::head(missing_in_file, 5)}}"
    ))
  }
  if (length(unmatched_in_file) > 0) {
    cli::cli_abort(c(
      "Checkpoint has {length(unmatched_in_file)} parameter{?s} with no model slot.",
      "x" = "First unmatched: {.val {utils::head(unmatched_in_file, 5)}}"
    ))
  }

  shape_mismatch <- character()
  torch::with_no_grad({
    for (key in map_keys) {
      param <- assignments[[key]]
      tensor <- tensors[[key]]
      if (!identical(dim(param), dim(tensor))) {
        shape_mismatch <- c(shape_mismatch, key)
        next
      }
      param$copy_(tensor)
    }
  })
  if (length(shape_mismatch) > 0) {
    cli::cli_abort(c(
      "Shape mismatch for {length(shape_mismatch)} parameter{?s}.",
      "x" = "First mismatch: {.val {utils::head(shape_mismatch, 5)}}"
    ))
  }

  if (verbose) {
    cli::cli_inform("Loaded {length(map_keys)} TabICL parameters.")
  }
  invisible(model)
}

# The two files brulee reads for a task, prefixed so a loose file is
# self-identifying: "classification.config.json" / "classification.model.safetensors"
# (and the regression equivalents).
tabicl_checkpoint_files <- function(task) {
  prefix <- if (identical(task, "classification")) {
    "classification"
  } else {
    "regression"
  }
  list(
    config = paste0(prefix, ".config.json"),
    weights = paste0(prefix, ".model.safetensors")
  )
}

# Build a tabicl_model from a converted checkpoint directory and load its
# weights. `task` selects the task-prefixed filenames. Returns the model in eval
# mode.
tabicl_load_model <- function(
  model_dir,
  task,
  device = "cpu",
  verbose = FALSE
) {
  files <- tabicl_checkpoint_files(task)
  config <- tabicl_parse_config(file.path(model_dir, files$config))
  tensors <- safetensors::safe_load_file(
    file.path(model_dir, files$weights),
    framework = "torch"
  )
  model <- tabicl_model(config)
  load_tabicl_weights(model, tensors, verbose = verbose)
  model$eval()
  model$to(device = device)
  list(model = model, config = config)
}
