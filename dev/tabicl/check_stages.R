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
  "tabicl-interaction"
)) {
  source(file.path("R", paste0(f, ".R")))
}

max_abs_diff <- function(a, b) as.numeric(torch_max(torch_abs(a - b)))

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
  diff <- max_abs_diff(out, golden$row_interact)
  cat(sprintf(
    "[%s] row_interact max|Δ| = %.3e  %s\n",
    kind,
    diff,
    if (diff < 1e-4) "OK" else "FAIL"
  ))
}

for (kind in c("classifier", "regressor")) {
  check_row_interaction(kind)
}
