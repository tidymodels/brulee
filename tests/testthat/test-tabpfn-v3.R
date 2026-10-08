test_that("unsupported v3 settings are rejected", {
  expect_snapshot(
    tabpfn_v3_parse_config(list(use_nan_indicators = FALSE)),
    error = TRUE
  )
})

v3_fixtures <- c("v3-multiclass", "v3-regression")

test_that("the v3 forward pass matches Python", {
  for (name in v3_fixtures) {
    skip_if_no_fixture(name)
    fx <- load_fixture(name)
    model <- fixture_model(fx)
    expect_identical(model$task_type, fx$meta$task_type)
    out <- torch::with_no_grad(
      model(fx$tensors$input.x, fx$tensors$input.y)
    )
    expect_close(out, fx$tensors$output)
  }
})

test_that("v3 preprocessing and stages match Python", {
  for (name in v3_fixtures) {
    skip_if_no_fixture(name)
    fx <- load_fixture(name)
    model <- fixture_model(fx)
    num_train <- fx$meta$n_train

    torch::with_no_grad({
      emb <- call_tensor(fx, "feature_distribution_embedder", "kwargs.x_BRiCE")
      expect_close(
        model$feature_distribution_embedder(emb, num_train),
        call_tensor(fx, "feature_distribution_embedder", "out.0")
      )
      emb <- call_tensor(fx, "column_aggregator", "kwargs.x_BRiCE")
      expect_close(
        model$column_aggregator(emb),
        call_tensor(fx, "column_aggregator", "out")
      )
      for (i in seq_along(model$icl_blocks)) {
        module <- paste0("icl_blocks.", i - 1)
        expect_close(
          model$icl_blocks[[i]](call_tensor(fx, module, "args.0"), num_train),
          call_tensor(fx, module, "out.0")
        )
      }
    })
  }
})

test_that("a v3 checkpoint serves only its own task", {
  skip_if_no_fixture("v3-regression")
  fx <- load_fixture("v3-regression")
  model <- fixture_model(fx)
  expect_snapshot(
    model(fx$tensors$input.x, fx$tensors$input.y, "multiclass"),
    error = TRUE
  )
})

test_that("processing rows in chunks gives the same output", {
  for (name in v3_fixtures) {
    skip_if_no_fixture(name)
    fx <- load_fixture(name)
    model <- fixture_model(fx)
    model$config$inference_chunk_cells <- 20
    out <- torch::with_no_grad(model(fx$tensors$input.x, fx$tensors$input.y))
    expect_close(out, fx$tensors$output)
  }
})
