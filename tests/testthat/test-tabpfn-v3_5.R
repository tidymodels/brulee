test_that("v3.5 config parsing is strict", {
  config <- list(max_num_classes = 160L, num_buckets = 5000L, use_rope = TRUE)
  parsed <- tabpfn_v3_5_parse_config(config)
  expect_identical(parsed$max_num_classes, 160L)
  expect_identical(parsed$embed_dim, 128L)
  expect_contains(names(parsed), "icl_num_kv_heads")
  expect_null(parsed$icl_num_kv_heads)

  expect_snapshot(
    tabpfn_v3_5_parse_config(c(config, list(new_setting = 1))),
    error = TRUE
  )
  expect_snapshot(
    tabpfn_v3_5_parse_config(list(use_rope = FALSE)),
    error = TRUE
  )
})

v35_fixtures <- c(
  "v3_5-multiclass",
  "v3_5-regression",
  "v3_5-multiclass-ecdf-batch"
)

test_that("the v3.5 forward pass matches Python", {
  for (name in v35_fixtures) {
    skip_if_no_fixture(name)
    fx <- load_fixture(name)
    model <- fixture_model(fx)
    out <- torch::with_no_grad(
      model(fx$tensors$input.x, fx$tensors$input.y, fx$meta$task_type)
    )
    expect_close(out, fx$tensors$output)
  }
})

test_that("v3.5 preprocessing and stages match Python", {
  for (name in v35_fixtures) {
    skip_if_no_fixture(name)
    fx <- load_fixture(name)
    model <- fixture_model(fx)
    x <- fx$tensors$input.x
    num_train <- fx$meta$n_train

    torch::with_no_grad({
      # Preprocessing, ECDF ranks, and grouping give the cell embedder's input.
      prep <- model$preprocess(x, num_train)
      grouped <- tabpfn_group_features(
        list(prep$x, prep$nan_ind, prep$ecdf),
        model$feature_group_size
      )
      expect_close(grouped, call_tensor(fx, "x_embed", "args.0"))
      expect_close(
        model$x_embed(grouped),
        call_tensor(fx, "x_embed", "out")
      )

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

test_that("v3.5 submodules match Python", {
  skip_if_no_fixture("v3_5-multiclass")
  fx <- load_fixture("v3_5-multiclass")
  model <- fixture_model(fx)

  torch::with_no_grad({
    norm <- model$icl_blocks[[1]]$icl_attention$q_norm
    module <- "icl_blocks.0.icl_attention.q_norm"
    expect_close(
      norm(call_tensor(fx, module, "args.0")),
      call_tensor(fx, module, "out")
    )

    scaling <- model$icl_blocks[[1]]$icl_attention$softmax_scaling_layer
    module <- "icl_blocks.0.icl_attention.softmax_scaling_layer"
    expect_close(
      scaling(
        call_tensor(fx, module, "args.0"),
        fx$meta$call_meta[[paste0("call.", module, ".args.1")]]
      ),
      call_tensor(fx, module, "out")
    )

    isab <- model$feature_distribution_embedder$layers[[1]]
    module <- "feature_distribution_embedder.layers.0.cross_attn_block1"
    expect_close(
      isab$cross_attn_block1(
        call_tensor(fx, module, "args.0"),
        call_tensor(fx, module, "args.1")
      ),
      call_tensor(fx, module, "out")
    )

    module <- "heads"
    expect_close(
      model$heads(
        call_tensor(fx, module, "args.0"),
        call_tensor(fx, module, "args.1"),
        call_tensor(fx, module, "args.2"),
        "multiclass",
        3L
      ),
      call_tensor(fx, module, "out")
    )
  })
})

test_that("processing rows in chunks gives the same output", {
  for (name in v35_fixtures) {
    skip_if_no_fixture(name)
    fx <- load_fixture(name)
    model <- fixture_model(fx)
    model$config$inference_chunk_cells <- 20
    out <- torch::with_no_grad(
      model(fx$tensors$input.x, fx$tensors$input.y, fx$meta$task_type)
    )
    expect_close(out, fx$tensors$output)
  }
})
