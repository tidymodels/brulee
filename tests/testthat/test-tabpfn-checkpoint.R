test_that("both checkpoint formats give the same result", {
  skip_if_no_torch()
  ckpt <- local_gunzip(
    test_path("fixtures/tabpfn/checkpoints/tiny.ckpt.gz"),
    ".ckpt"
  )
  st <- local_gunzip(
    test_path("fixtures/tabpfn/checkpoints/tiny.safetensors.gz"),
    ".safetensors"
  )

  from_ckpt <- tabpfn_read_checkpoint(ckpt)
  from_st <- tabpfn_read_checkpoint(st)

  expect_named(
    from_ckpt,
    c("tensors", "architecture_name", "config", "inference_config")
  )
  expect_identical(from_ckpt$architecture_name, "tabpfn_tiny")
  expect_identical(from_st$architecture_name, from_ckpt$architecture_name)
  expect_identical(
    from_st$config[order(names(from_st$config))],
    from_ckpt$config[order(names(from_ckpt$config))]
  )
  expect_setequal(names(from_st$tensors), names(from_ckpt$tensors))
  for (key in names(from_ckpt$tensors)) {
    expect_equal(
      max_abs_diff(from_st$tensors[[key]], from_ckpt$tensors[[key]]),
      0
    )
  }
})

test_that("unknown checkpoint formats are rejected", {
  expect_snapshot(tabpfn_read_checkpoint("model.bin"), error = TRUE)
})

tiny_module <- function() {
  torch::nn_module(
    initialize = function() {
      self$layer <- torch::nn_linear(2, 3)
      self$blocks <- torch::nn_module_list(list(torch::nn_linear(3, 1)))
    }
  )()
}

tiny_tensors <- function() {
  list(
    layer.weight = torch::torch_ones(3, 2),
    layer.bias = torch::torch_zeros(3),
    blocks.0.weight = torch::torch_full(c(1, 3), 2),
    blocks.0.bias = torch::torch_ones(1)
  )
}

test_that("weights load when names and shapes match", {
  skip_if_no_torch()
  model <- tiny_module()
  tabpfn_load_state(model, tiny_tensors())
  expect_equal(max_abs_diff(model$layer$weight, torch::torch_ones(3, 2)), 0)
  expect_equal(model$blocks[[1]]$weight$sum()$item(), 6)
})

test_that("weight loading is strict", {
  skip_if_no_torch()
  model <- tiny_module()

  missing <- tiny_tensors()
  missing$layer.bias <- NULL
  expect_snapshot(tabpfn_load_state(model, missing), error = TRUE)

  extra <- c(tiny_tensors(), list(other.weight = torch::torch_ones(1)))
  expect_snapshot(tabpfn_load_state(model, extra), error = TRUE)

  wrong_shape <- tiny_tensors()
  wrong_shape$layer.weight <- torch::torch_ones(2, 3)
  expect_snapshot(tabpfn_load_state(model, wrong_shape), error = TRUE)
})

test_that("torch zip checkpoints are read with values, offsets, and strides", {
  skip_if_no_torch()
  path <- local_gunzip(
    test_path("fixtures/tabpfn/checkpoints/tiny.ckpt.gz"),
    ".ckpt"
  )
  expected <- jsonlite::read_json(
    test_path("fixtures/tabpfn/checkpoints/tiny-values.json"),
    simplifyVector = FALSE
  )

  obj <- tabpfn_read_ckpt(path)
  expect_named(
    obj,
    c(
      "state_dict",
      "config",
      "inference_config",
      "architecture_name",
      "optimizer_state",
      "trained_epochs_until_now"
    )
  )
  expect_named(obj$state_dict, names(expected$tensors))
  for (key in names(expected$tensors)) {
    tensor <- obj$state_dict[[key]]
    expect_identical(
      dim(tensor),
      as.integer(unlist(expected$tensors[[key]]$shape))
    )
    expect_equal(
      as.numeric(torch::as_array(tensor$flatten())),
      unlist(expected$tensors[[key]]$values),
      tolerance = 1e-7
    )
  }
  expect_identical(obj$config$embed_dim, 16L)
  expect_identical(obj$config$rope_base, 1e5)
  expect_null(obj$config$kv)
  expect_contains(names(obj$config), "kv")
  expect_identical(obj$inference_config$PREPROCESS_TRANSFORMS[[1]]$name, "none")
  expect_false(obj$inference_config$PREPROCESS_TRANSFORMS[[1]]$append_original)
  expect_null(obj$optimizer_state)
})

test_that("unsupported storage types are rejected", {
  skip_if_no_torch()
  path <- local_gunzip(
    test_path("fixtures/tabpfn/checkpoints/int64.ckpt.gz"),
    ".ckpt"
  )
  expect_snapshot(
    tabpfn_read_ckpt(path),
    error = TRUE,
    transform = scrub_tempfile
  )
})

test_that("files that aren't torch zip checkpoints are rejected", {
  path <- withr::local_tempfile(fileext = ".zip")
  withr::with_dir(tempdir(), {
    writeLines("x", "not-a-checkpoint.txt")
    utils::zip(path, "not-a-checkpoint.txt", flags = "-q")
  })
  expect_snapshot(
    tabpfn_read_ckpt(path),
    error = TRUE,
    transform = scrub_tempfile
  )
})

test_that("the model's architecture must match the registry", {
  skip_if_no_torch()
  checkpoint <- list(architecture_name = "tabpfn_v3_5")
  expect_snapshot(tabpfn_build_model(checkpoint, "tabpfn_v3"), error = TRUE)
  checkpoint <- list(architecture_name = "tabpfn_v9")
  expect_snapshot(tabpfn_build_model(checkpoint), error = TRUE)
})
