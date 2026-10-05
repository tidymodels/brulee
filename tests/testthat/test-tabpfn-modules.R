test_that("attention in query chunks matches attention in one pass", {
  skip_if_no_torch()
  torch::torch_manual_seed(40961)
  q <- torch::torch_randn(2, 4, 37, 8)
  k <- torch::torch_randn(2, 4, 23, 8)
  v <- torch::torch_randn(2, 4, 23, 8)
  withr::local_options(brulee.tabpfn_mps_attention_cells = 2 * 4 * 23 * 5)
  expect_lt(
    max_abs_diff(
      tabpfn_sdpa_chunked(q, k, v),
      torch::torch_scaled_dot_product_attention(q, k, v)
    ),
    1e-6
  )
})
