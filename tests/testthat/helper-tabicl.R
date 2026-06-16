# Helpers for TabICL primitive parity tests.
#
# Fixtures under tests/testthat/fixtures/tabicl/ are generated from the Python
# reference implementation by dev/tabicl/dump_primitives.py: each holds a
# module's weights, the inputs, and the golden output. The tests build the R
# module, copy the weights in, run a forward pass, and compare to the golden
# output. See dev/tabicl/README.md.

# Skip a test unless torch + safetensors are usable and the fixtures exist.
skip_if_no_tabicl_fixtures <- function(name) {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip_if_not_installed("safetensors")
  skip_if_not_installed("jsonlite")
  if (!torch::torch_is_installed()) {
    skip("libtorch not installed")
  }
  if (!file.exists(tabicl_fixture_path(name, "safetensors.gz"))) {
    skip(paste0("fixture not found: ", name))
  }
}

tabicl_fixture_path <- function(name, ext) {
  testthat::test_path("fixtures", "tabicl", paste0(name, ".", ext))
}

# Fixtures are stored gzip-compressed (see dev/tabicl/dump_primitives.py) so the
# raw safetensors header never trips R CMD check's executable-magic heuristic.
# safe_load_file needs a real path, so decompress to a temp file first.
tabicl_load_fixture <- function(name) {
  gz <- tabicl_fixture_path(name, "safetensors.gz")
  con <- gzfile(gz, "rb")
  on.exit(close(con))
  chunks <- list()
  repeat {
    chunk <- readBin(con, "raw", n = 1024L^2)
    if (length(chunk) == 0L) {
      break
    }
    chunks[[length(chunks) + 1L]] <- chunk
  }
  tmp <- tempfile(fileext = ".safetensors")
  writeBin(do.call(c, chunks), tmp)
  safetensors::safe_load_file(tmp, framework = "torch")
}

tabicl_fixture_meta <- function(name) {
  jsonlite::fromJSON(tabicl_fixture_path(name, "json"))
}

# Largest absolute elementwise difference between two tensors.
tabicl_max_abs_diff <- function(a, b) {
  as.numeric(torch::torch_max(torch::torch_abs(a - b)))
}

# Relative max error: max abs diff scaled by the reference's max magnitude. Used
# for stage outputs that legitimately contain large values (e.g. the column
# embedder's CLS slots keep the -100 skip value through residual connections),
# where an absolute float32 tolerance would be misleading.
tabicl_rel_max_diff <- function(actual, expected) {
  scale <- max(as.numeric(torch::torch_max(torch::torch_abs(expected))), 1e-8)
  tabicl_max_abs_diff(actual, expected) / scale
}

# Copy a qassmax-mlp-elementwise SSMax layer's weights from a fixture. Python
# nn.Sequential indices 0/2 (Linear, GELU, Linear) map to R 1-based [[1]]/[[3]].
tabicl_copy_ssmax <- function(layer, f, prefix = "ssmax.") {
  torch::with_no_grad({
    layer$base_mlp[[1]]$weight$copy_(f[[paste0(prefix, "base_mlp.0.weight")]])
    layer$base_mlp[[1]]$bias$copy_(f[[paste0(prefix, "base_mlp.0.bias")]])
    layer$base_mlp[[3]]$weight$copy_(f[[paste0(prefix, "base_mlp.2.weight")]])
    layer$base_mlp[[3]]$bias$copy_(f[[paste0(prefix, "base_mlp.2.bias")]])
    layer$query_mlp[[1]]$weight$copy_(f[[paste0(prefix, "query_mlp.0.weight")]])
    layer$query_mlp[[1]]$bias$copy_(f[[paste0(prefix, "query_mlp.0.bias")]])
    layer$query_mlp[[3]]$weight$copy_(f[[paste0(prefix, "query_mlp.2.weight")]])
    layer$query_mlp[[3]]$bias$copy_(f[[paste0(prefix, "query_mlp.2.bias")]])
  })
  invisible(layer)
}

# Copy the packed projection + output projection of a tabicl_mha from a fixture.
tabicl_copy_mha <- function(mha, f) {
  torch::with_no_grad({
    mha$in_proj_weight$copy_(f$in_proj_weight)
    mha$in_proj_bias$copy_(f$in_proj_bias)
    mha$out_proj$weight$copy_(f[["out_proj.weight"]])
    mha$out_proj$bias$copy_(f[["out_proj.bias"]])
  })
  if (!is.null(mha$ssmax_layer)) {
    tabicl_copy_ssmax(mha$ssmax_layer, f)
  }
  invisible(mha)
}

# Copy a tabicl_mha_block's parameters from fixture tensors keyed by `prefix`
# (e.g. "ri.tf_row.blocks.0.").
tabicl_copy_mha_block <- function(block, f, prefix) {
  torch::with_no_grad({
    block$norm1$weight$copy_(f[[paste0(prefix, "norm1.weight")]])
    block$norm2$weight$copy_(f[[paste0(prefix, "norm2.weight")]])
    if (!is.null(block$norm1$bias)) {
      block$norm1$bias$copy_(f[[paste0(prefix, "norm1.bias")]])
      block$norm2$bias$copy_(f[[paste0(prefix, "norm2.bias")]])
    }
    block$linear1$weight$copy_(f[[paste0(prefix, "linear1.weight")]])
    block$linear1$bias$copy_(f[[paste0(prefix, "linear1.bias")]])
    block$linear2$weight$copy_(f[[paste0(prefix, "linear2.weight")]])
    block$linear2$bias$copy_(f[[paste0(prefix, "linear2.bias")]])
    block$attn$in_proj_weight$copy_(f[[paste0(prefix, "attn.in_proj_weight")]])
    block$attn$in_proj_bias$copy_(f[[paste0(prefix, "attn.in_proj_bias")]])
    block$attn$out_proj$weight$copy_(f[[paste0(
      prefix,
      "attn.out_proj.weight"
    )]])
    block$attn$out_proj$bias$copy_(f[[paste0(prefix, "attn.out_proj.bias")]])
  })
  if (!is.null(block$attn$ssmax_layer)) {
    tabicl_copy_ssmax(
      block$attn$ssmax_layer,
      f,
      prefix = paste0(prefix, "attn.ssmax_layer.")
    )
  }
  invisible(block)
}

# Copy an induced-self-attention block (ISAB) from fixture tensors keyed by
# `prefix` (e.g. "ce.tf_col.blocks.0.").
tabicl_copy_isab <- function(isab, f, prefix) {
  torch::with_no_grad(
    isab$ind_vectors$copy_(f[[paste0(prefix, "ind_vectors")]])
  )
  tabicl_copy_mha_block(
    isab$multihead_attn1,
    f,
    paste0(prefix, "multihead_attn1.")
  )
  tabicl_copy_mha_block(
    isab$multihead_attn2,
    f,
    paste0(prefix, "multihead_attn2.")
  )
  invisible(isab)
}

# Copy a tabicl_col_embedding from fixture tensors keyed under "ce.".
tabicl_copy_col_embedding <- function(ce, f) {
  torch::with_no_grad({
    ce$in_linear$weight$copy_(f[["ce.in_linear.weight"]])
    ce$in_linear$bias$copy_(f[["ce.in_linear.bias"]])
    ce$y_encoder$weight$copy_(f[["ce.y_encoder.weight"]])
    ce$y_encoder$bias$copy_(f[["ce.y_encoder.bias"]])
  })
  for (i in seq_along(ce$tf_col$blocks)) {
    tabicl_copy_isab(
      ce$tf_col$blocks[[i]],
      f,
      prefix = sprintf("ce.tf_col.blocks.%d.", i - 1L)
    )
  }
  invisible(ce)
}

# Copy a tabicl_row_interaction from fixture tensors keyed under "ri.".
tabicl_copy_row_interaction <- function(ri, f) {
  torch::with_no_grad({
    ri$cls_tokens$copy_(f[["ri.cls_tokens"]])
    ri$out_ln$weight$copy_(f[["ri.out_ln.weight"]])
    if (!is.null(ri$out_ln$bias)) {
      ri$out_ln$bias$copy_(f[["ri.out_ln.bias"]])
    }
    ri$tf_row$rope$freqs$copy_(f[["ri.tf_row.rope.freqs"]])
  })
  for (i in seq_along(ri$tf_row$blocks)) {
    tabicl_copy_mha_block(
      ri$tf_row$blocks[[i]],
      f,
      prefix = sprintf("ri.tf_row.blocks.%d.", i - 1L)
    )
  }
  invisible(ri)
}
