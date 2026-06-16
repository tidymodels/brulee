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
  if (!file.exists(tabicl_fixture_path(name, "safetensors"))) {
    skip(paste0("fixture not found: ", name))
  }
}

tabicl_fixture_path <- function(name, ext) {
  testthat::test_path("fixtures", "tabicl", paste0(name, ".", ext))
}

tabicl_load_fixture <- function(name) {
  safetensors::safe_load_file(
    tabicl_fixture_path(name, "safetensors"),
    framework = "torch"
  )
}

tabicl_fixture_meta <- function(name) {
  jsonlite::fromJSON(tabicl_fixture_path(name, "json"))
}

# Largest absolute elementwise difference between two tensors.
tabicl_max_abs_diff <- function(a, b) {
  as.numeric(torch::torch_max(torch::torch_abs(a - b)))
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
