# Real-weight stage parity checks against the released checkpoints.
#
# Unlike the committed testthat fixtures (small, random weights), this loads the
# actual converted checkpoint (dev/tabicl/artifacts/<kind>/model.safetensors,
# git-ignored) and the step-1 golden stage outputs, builds each R stage, copies
# the real weights in, and compares to the golden. Run from the package root:
#
#   Rscript dev/tabicl/check_stages.R
#
# Requires convert_ckpt.py + dump_golden.py to have been run for both kinds.

suppressPackageStartupMessages({
  library(torch)
  library(safetensors)
})

for (f in c(
  "tabicl-rope",
  "tabicl-attention",
  "tabicl-layers",
  "tabicl-interaction",
  "tabicl-learning",
  "tabicl-model",
  "tabicl-embedding",
  "tabicl-quantile"
)) {
  source(file.path("R", paste0(f, ".R")))
}

max_abs_diff <- function(a, b) as.numeric(torch_max(torch_abs(a - b)))

# Relative max error: absolute max diff divided by the golden's max magnitude.
# Stage outputs can have RMS in the tens/hundreds, so an absolute float32
# tolerance is misleading; this reports the meaningful relative agreement.
rel_max_diff <- function(out, golden) {
  max_abs_diff(out, golden) /
    max(as.numeric(torch_max(torch_abs(golden))), 1e-8)
}

report_stage <- function(kind, stage, out, golden, tol = 1e-4) {
  diff <- rel_max_diff(out, golden)
  cat(sprintf(
    "[%s] %-12s rel max|Δ| = %.3e  %s\n",
    kind,
    stage,
    diff,
    if (diff < tol) "OK" else "FAIL"
  ))
}

copy_ssmax <- function(layer, sd, pre) {
  cp <- function(param, key) with_no_grad(param$copy_(sd[[key]]))
  cp(layer$base_mlp[[1]]$weight, paste0(pre, "base_mlp.0.weight"))
  cp(layer$base_mlp[[1]]$bias, paste0(pre, "base_mlp.0.bias"))
  cp(layer$base_mlp[[3]]$weight, paste0(pre, "base_mlp.2.weight"))
  cp(layer$base_mlp[[3]]$bias, paste0(pre, "base_mlp.2.bias"))
  cp(layer$query_mlp[[1]]$weight, paste0(pre, "query_mlp.0.weight"))
  cp(layer$query_mlp[[1]]$bias, paste0(pre, "query_mlp.0.bias"))
  cp(layer$query_mlp[[3]]$weight, paste0(pre, "query_mlp.2.weight"))
  cp(layer$query_mlp[[3]]$bias, paste0(pre, "query_mlp.2.bias"))
}

copy_block <- function(block, sd, pre, has_ln_bias) {
  cp <- function(param, key) {
    with_no_grad(param$copy_(sd[[key]]))
  }
  cp(block$norm1$weight, paste0(pre, "norm1.weight"))
  cp(block$norm2$weight, paste0(pre, "norm2.weight"))
  if (has_ln_bias) {
    cp(block$norm1$bias, paste0(pre, "norm1.bias"))
    cp(block$norm2$bias, paste0(pre, "norm2.bias"))
  }
  cp(block$linear1$weight, paste0(pre, "linear1.weight"))
  cp(block$linear1$bias, paste0(pre, "linear1.bias"))
  cp(block$linear2$weight, paste0(pre, "linear2.weight"))
  cp(block$linear2$bias, paste0(pre, "linear2.bias"))
  cp(block$attn$in_proj_weight, paste0(pre, "attn.in_proj_weight"))
  cp(block$attn$in_proj_bias, paste0(pre, "attn.in_proj_bias"))
  cp(block$attn$out_proj$weight, paste0(pre, "attn.out_proj.weight"))
  cp(block$attn$out_proj$bias, paste0(pre, "attn.out_proj.bias"))
  if (!is.null(block$attn$ssmax_layer)) {
    copy_ssmax(block$attn$ssmax_layer, sd, paste0(pre, "attn.ssmax_layer."))
  }
}

copy_isab <- function(isab, sd, pre, has_ln_bias) {
  with_no_grad(isab$ind_vectors$copy_(sd[[paste0(pre, "ind_vectors")]]))
  copy_block(
    isab$multihead_attn1,
    sd,
    paste0(pre, "multihead_attn1."),
    has_ln_bias
  )
  copy_block(
    isab$multihead_attn2,
    sd,
    paste0(pre, "multihead_attn2."),
    has_ln_bias
  )
}

# Module-level weight copiers (prefix = the module's path in the state dict).
# These mirror the eventual production weight loader.

copy_col_module <- function(mod, sd, cfg, prefix = "col_embedder.") {
  has_ln_bias <- !cfg$bias_free_ln
  cp <- function(param, key) with_no_grad(param$copy_(sd[[key]]))
  cp(mod$in_linear$weight, paste0(prefix, "in_linear.weight"))
  cp(mod$in_linear$bias, paste0(prefix, "in_linear.bias"))
  cp(mod$y_encoder$weight, paste0(prefix, "y_encoder.weight"))
  cp(mod$y_encoder$bias, paste0(prefix, "y_encoder.bias"))
  for (i in seq_len(cfg$col_num_blocks)) {
    copy_isab(
      mod$tf_col$blocks[[i]],
      sd,
      sprintf("%stf_col.blocks.%d.", prefix, i - 1L),
      has_ln_bias
    )
  }
}

copy_row_module <- function(mod, sd, cfg, prefix = "row_interactor.") {
  has_ln_bias <- !cfg$bias_free_ln
  with_no_grad({
    mod$cls_tokens$copy_(sd[[paste0(prefix, "cls_tokens")]])
    mod$out_ln$weight$copy_(sd[[paste0(prefix, "out_ln.weight")]])
    if (has_ln_bias) {
      mod$out_ln$bias$copy_(sd[[paste0(prefix, "out_ln.bias")]])
    }
    mod$tf_row$rope$freqs$copy_(sd[[paste0(prefix, "tf_row.rope.freqs")]])
  })
  for (i in seq_len(cfg$row_num_blocks)) {
    copy_block(
      mod$tf_row$blocks[[i]],
      sd,
      sprintf("%stf_row.blocks.%d.", prefix, i - 1L),
      has_ln_bias
    )
  }
}

copy_icl_module <- function(mod, sd, cfg, prefix = "icl_predictor.") {
  has_ln_bias <- !cfg$bias_free_ln
  cp <- function(param, key) with_no_grad(param$copy_(sd[[key]]))
  if (cfg$norm_first) {
    cp(mod$ln$weight, paste0(prefix, "ln.weight"))
    if (has_ln_bias) {
      cp(mod$ln$bias, paste0(prefix, "ln.bias"))
    }
  }
  cp(mod$y_encoder$weight, paste0(prefix, "y_encoder.weight"))
  cp(mod$y_encoder$bias, paste0(prefix, "y_encoder.bias"))
  cp(mod$decoder[[1]]$weight, paste0(prefix, "decoder.0.weight"))
  cp(mod$decoder[[1]]$bias, paste0(prefix, "decoder.0.bias"))
  cp(mod$decoder[[3]]$weight, paste0(prefix, "decoder.2.weight"))
  cp(mod$decoder[[3]]$bias, paste0(prefix, "decoder.2.bias"))
  for (i in seq_len(cfg$icl_num_blocks)) {
    copy_block(
      mod$tf_icl$blocks[[i]],
      sd,
      sprintf("%stf_icl.blocks.%d.", prefix, i - 1L),
      has_ln_bias
    )
  }
}

load_artifacts <- function(kind) {
  art <- file.path("dev/tabicl/artifacts", kind)
  list(
    cfg = jsonlite::fromJSON(file.path(art, "config.json")),
    sd = safe_load_file(
      file.path(art, "model.safetensors"),
      framework = "torch"
    ),
    golden = safe_load_file(
      file.path(art, "golden/stage_outputs.safetensors"),
      framework = "torch"
    ),
    inputs = safe_load_file(
      file.path(art, "golden/inputs.safetensors"),
      framework = "torch"
    )
  )
}

build_col <- function(cfg) {
  tabicl_col_embedding(
    embed_dim = cfg$embed_dim,
    num_blocks = cfg$col_num_blocks,
    nhead = cfg$col_nhead,
    dim_feedforward = cfg$ff_factor * cfg$embed_dim,
    num_inds = cfg$col_num_inds,
    feature_group_size = cfg$col_feature_group_size,
    target_aware = cfg$col_target_aware,
    max_classes = cfg$max_classes,
    reserve_cls_tokens = cfg$row_num_cls,
    activation = cfg$activation,
    norm_first = cfg$norm_first,
    bias_free_ln = cfg$bias_free_ln,
    ssmax = cfg$col_ssmax
  )
}

build_row <- function(cfg) {
  tabicl_row_interaction(
    embed_dim = cfg$embed_dim,
    num_blocks = cfg$row_num_blocks,
    nhead = cfg$row_nhead,
    dim_feedforward = cfg$ff_factor * cfg$embed_dim,
    num_cls = cfg$row_num_cls,
    rope_base = cfg$row_rope_base,
    activation = cfg$activation,
    norm_first = cfg$norm_first,
    bias_free_ln = cfg$bias_free_ln
  )
}

build_icl <- function(cfg) {
  icl_dim <- cfg$embed_dim * cfg$row_num_cls
  out_dim <- if (cfg$max_classes == 0) cfg$num_quantiles else cfg$max_classes
  tabicl_icl_learning(
    max_classes = cfg$max_classes,
    out_dim = out_dim,
    d_model = icl_dim,
    num_blocks = cfg$icl_num_blocks,
    nhead = cfg$icl_nhead,
    dim_feedforward = icl_dim * cfg$ff_factor,
    activation = cfg$activation,
    norm_first = cfg$norm_first,
    bias_free_ln = cfg$bias_free_ln,
    ssmax = cfg$icl_ssmax
  )
}

check_col_embedding <- function(kind) {
  a <- load_artifacts(kind)
  mod <- build_col(a$cfg)$eval()
  copy_col_module(mod, a$sd, a$cfg)
  out <- with_no_grad(mod(a$inputs$X, a$inputs$y_train))
  report_stage(kind, "col_embed", out, a$golden$col_embed)
}

check_row_interaction <- function(kind) {
  a <- load_artifacts(kind)
  mod <- build_row(a$cfg)$eval()
  copy_row_module(mod, a$sd, a$cfg)
  out <- with_no_grad(mod(a$golden$col_embed))
  report_stage(kind, "row_interact", out, a$golden$row_interact)
}

check_icl <- function(kind) {
  a <- load_artifacts(kind)
  mod <- build_icl(a$cfg)$eval()
  copy_icl_module(mod, a$sd, a$cfg)
  out <- with_no_grad(mod(a$golden$row_interact, a$inputs$y_train))
  report_stage(kind, "icl_out", out, a$golden$icl_out)
}

check_full <- function(kind) {
  a <- load_artifacts(kind)
  mod <- tabicl_model(a$cfg)$eval()
  copy_col_module(mod$col_embedder, a$sd, a$cfg)
  copy_row_module(mod$row_interactor, a$sd, a$cfg)
  copy_icl_module(mod$icl_predictor, a$sd, a$cfg)
  out <- with_no_grad(mod(a$inputs$X, a$inputs$y_train))
  report_stage(kind, "full_forward", out, a$golden$icl_out)
}

# Regression head: raw model quantiles -> distribution stats.
check_regression_stats <- function() {
  a <- load_artifacts("regressor")
  dist <- tabicl_quantile_dist(a$golden$icl_out)
  report_stage(
    "regressor",
    "qd_mean",
    tabicl_qdist_mean(dist),
    a$golden$qd_mean
  )
  report_stage(
    "regressor",
    "qd_median",
    tabicl_qdist_median(dist),
    a$golden$qd_median
  )
  report_stage(
    "regressor",
    "qd_quantiles",
    tabicl_qdist_quantiles(dist, c(0.1, 0.5, 0.9)),
    a$golden$qd_quantiles
  )
}

for (kind in c("classifier", "regressor")) {
  check_col_embedding(kind)
  check_row_interaction(kind)
  check_icl(kind)
  check_full(kind)
}
check_regression_stats()
