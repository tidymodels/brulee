# Parity tests for the TabICL RoPE module (R/tabicl-rope.R) against the Python
# reference (non-interleaved rotation, theta = 100000). Fixtures are generated
# by dev/tabicl/dump_primitives.py.

test_that("tabicl_rotate_half splits and rotates the last dimension", {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip_if_not_installed("safetensors")
  if (!torch::torch_is_installed()) {
    skip("libtorch not installed")
  }

  x <- torch::torch_tensor(matrix(c(1, 2, 3, 4), nrow = 1))
  rotated <- brulee:::tabicl_rotate_half(x)
  # [x1, x2] -> [-x2, x1] with x1 = (1, 2), x2 = (3, 4).
  expect_equal(as.numeric(rotated), c(-3, -4, 1, 2))
})

test_that("tabicl_rope matches the Python reference", {
  skip_if_no_tabicl_fixtures("rope")

  f <- tabicl_load_fixture("rope")
  meta <- tabicl_fixture_meta("rope")

  rope <- brulee:::tabicl_rope(dim = meta$head_dim, theta = meta$theta)
  torch::with_no_grad(rope$freqs$copy_(f$freqs))

  out <- rope(f$q)

  expect_equal(dim(out), dim(f$out))
  expect_lt(tabicl_max_abs_diff(out, f$out), 1e-5)
})

test_that("tabicl_rope preserves the L2 norm of each position", {
  skip_on_cran()
  skip_if_not_installed("torch")
  if (!torch::torch_is_installed()) {
    skip("libtorch not installed")
  }

  # Rotation is orthogonal, so the per-token norm is unchanged.
  rope <- brulee:::tabicl_rope(dim = 16, theta = 100000)
  q <- torch::torch_randn(2, 4, 5, 16)
  out <- rope(q)

  norm_in <- torch::torch_sqrt(torch::torch_sum(q^2, dim = -1))
  norm_out <- torch::torch_sqrt(torch::torch_sum(out^2, dim = -1))
  expect_lt(tabicl_max_abs_diff(norm_in, norm_out), 1e-5)
})
