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
  "tabicl-embedding"
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

check_col_embedding <- function(kind) {
  art <- file.path("dev/tabicl/artifacts", kind)
  cfg <- jsonlite::fromJSON(file.path(art, "config.json"))
  sd <- safe_load_file(file.path(art, "model.safetensors"), framework = "torch")
  golden <- safe_load_file(
    file.path(art, "golden/stage_outputs.safetensors"),
    framework = "torch"
  )
  inputs <- safe_load_file(
    file.path(art, "golden/inputs.safetensors"),
    framework = "torch"
  )

  mod <- tabicl_col_embedding(
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
  mod$eval()

  has_ln_bias <- !cfg$bias_free_ln
  with_no_grad({
    mod$in_linear$weight$copy_(sd[["col_embedder.in_linear.weight"]])
    mod$in_linear$bias$copy_(sd[["col_embedder.in_linear.bias"]])
    mod$y_encoder$weight$copy_(sd[["col_embedder.y_encoder.weight"]])
    mod$y_encoder$bias$copy_(sd[["col_embedder.y_encoder.bias"]])
  })
  for (i in seq_len(cfg$col_num_blocks)) {
    copy_isab(
      mod$tf_col$blocks[[i]],
      sd,
      sprintf("col_embedder.tf_col.blocks.%d.", i - 1L),
      has_ln_bias
    )
  }

  # y_train must be the (B, train_size) labels; regression labels are floats.
  y_train <- inputs$y_train
  out <- with_no_grad(mod(inputs$X, y_train))
  report_stage(kind, "col_embed", out, golden$col_embed)
}

check_row_interaction <- function(kind) {
  art <- file.path("dev/tabicl/artifacts", kind)
  cfg <- jsonlite::fromJSON(file.path(art, "config.json"))
  sd <- safe_load_file(file.path(art, "model.safetensors"), framework = "torch")
  golden <- safe_load_file(
    file.path(art, "golden/stage_outputs.safetensors"),
    framework = "torch"
  )

  mod <- tabicl_row_interaction(
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
  mod$eval()

  has_ln_bias <- !cfg$bias_free_ln
  with_no_grad({
    mod$cls_tokens$copy_(sd[["row_interactor.cls_tokens"]])
    mod$out_ln$weight$copy_(sd[["row_interactor.out_ln.weight"]])
    if (has_ln_bias) {
      mod$out_ln$bias$copy_(sd[["row_interactor.out_ln.bias"]])
    }
    mod$tf_row$rope$freqs$copy_(sd[["row_interactor.tf_row.rope.freqs"]])
  })
  for (i in seq_len(cfg$row_num_blocks)) {
    copy_block(
      mod$tf_row$blocks[[i]],
      sd,
      sprintf("row_interactor.tf_row.blocks.%d.", i - 1L),
      has_ln_bias
    )
  }

  out <- with_no_grad(mod(golden$col_embed))
  report_stage(kind, "row_interact", out, golden$row_interact)
}

for (kind in c("classifier", "regressor")) {
  check_col_embedding(kind)
  check_row_interaction(kind)
}
