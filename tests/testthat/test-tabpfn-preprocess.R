test_that("the encoder codes categories in sorted order", {
  x <- data.frame(
    num = c(1.5, NA, 3, 4),
    chr = c("b", "a", NA, "B"),
    con = 2
  )
  encoder <- tabpfn_fit_encoder(x)
  expect_identical(encoder$columns$chr$levels, c("B", "a", "b"))
  expect_identical(encoder$columns$con$type, "constant")
  encoded <- tabpfn_encode(encoder, x)
  expect_identical(unname(encoded[, "chr"]), c(2, 1, NaN, 0))
  expect_identical(unname(encoded[, "num"]), c(1.5, NaN, 3, 4))

  new <- data.frame(num = 1, chr = "z", con = 5)
  expect_identical(unname(tabpfn_encode(encoder, new)[1, ]), c(1, -1, 2))
})

test_that("low-cardinality numeric columns are categorical in large data", {
  x <- data.frame(v = rep(c(10, 20, 30), length.out = 101))
  expect_identical(tabpfn_fit_encoder(x)$columns$v$type, "categorical")
  expect_identical(
    tabpfn_fit_encoder(x[1:100, , drop = FALSE])$columns$v$type,
    "numeric"
  )
})

test_that("infinite and unsupported predictors are rejected", {
  encoder <- tabpfn_fit_encoder(data.frame(v = c(1, 2)))
  expect_snapshot(tabpfn_encode(encoder, data.frame(v = Inf)), error = TRUE)
  expect_snapshot(tabpfn_fit_encoder(data.frame(d = Sys.Date())), error = TRUE)
})

test_that("generated members are reproducible and balanced", {
  x <- matrix(c(1:20, rep(0:1, 10)), ncol = 2)
  v35_pre <- list(tabpfn_preprocessor(list(
    name = "none",
    categorical_name = "ordinal_shuffled",
    append_original = FALSE,
    max_features_per_estimator = 768L
  )))
  members <- tabpfn_with_seed(
    81723,
    tabpfn_make_members(
      x,
      c(FALSE, TRUE),
      8,
      "multiclass",
      v35_pre,
      n_classes = 2
    )
  )
  again <- tabpfn_with_seed(
    81723,
    tabpfn_make_members(
      x,
      c(FALSE, TRUE),
      8,
      "multiclass",
      v35_pre,
      n_classes = 2
    )
  )
  expect_identical(members, again)
  perms <- vapply(
    members,
    function(m) paste(m$class_permutation, collapse = ""),
    ""
  )
  expect_identical(perms, rep(c("01", "10"), each = 4))
  expect_all_true(vapply(
    members,
    function(m) length(m$shuffle) == 3,
    logical(1)
  ))

  y <- as.numeric(scale(1:20))
  members <- tabpfn_with_seed(
    81723,
    tabpfn_make_members(
      x,
      c(FALSE, TRUE),
      4,
      "regression",
      v35_pre,
      y_train = y
    )
  )
  has_transform <- vapply(
    members,
    function(m) !is.null(m$target_transform),
    logical(1)
  )
  expect_identical(has_transform, c(FALSE, FALSE, TRUE, TRUE))
})

test_that("feature subsets cover the columns evenly", {
  subsets <- tabpfn_with_seed(81723, tabpfn_feature_subsets(10, rep(5, 4)))
  expect_all_true(vapply(subsets, length, integer(1)) == 5)
  expect_identical(sort(unique(unlist(subsets))), 1:10)
  expect_identical(tabpfn_feature_subsets(3, c(5, 5)), list(1:3, 1:3))
  random <- tabpfn_with_seed(
    81723,
    tabpfn_feature_subsets(10, rep(4, 3), "random")
  )
  expect_all_true(vapply(random, length, integer(1)) == 4)
})

test_that("the same seed gives the same ensemble", {
  x <- data.frame(a = c(1:19, NA), b = factor(rep(c("u", "v"), 10)), c = 20:1)
  y <- factor(rep(c("p", "q"), 10))
  fit_members <- function(seed) {
    local_mocked_bindings(
      tabpfn_load = function(...) list(inference_config = list())
    )
    ic <- list(
      n_estimators = 4L,
      softmax_temperature = 1,
      outlier_std = 12,
      preprocessors = list(tabpfn_preprocessor(list(
        name = "none",
        categorical_name = "ordinal_shuffled",
        append_original = FALSE,
        max_features_per_estimator = 768L
      ))),
      feature_subsampling = "balanced",
      min_samples_categorical = 100L,
      min_unique_numeric = 4L,
      max_classes = 10L,
      max_features = 500L,
      max_samples = 1e4
    )
    local_mocked_bindings(tabpfn_resolve_inference = function(...) ic)
    options <- tabpfn_fit_options(NULL, NULL, FALSE, FALSE, Inf, "cpu", FALSE)
    options$seed <- seed
    tabpfn_impl(x, y, options, "v3.5")$members
  }
  expect_identical(fit_members(81723), fit_members(81723))
  expect_false(identical(fit_members(81723), fit_members(40961)))
})

test_that("model inputs match Python given Python's member choices", {
  for (name in pipeline_fixtures) {
    skip_if_no_fixture(name)
    fx <- load_pipeline_fixture(name)
    n <- fx$n_train
    local_python_fingerprints(fx)
    res <- fixture_inputs(fx, fx$meta$outlier_std)
    categorical <- tabpfn_categorical_columns(res$encoder)
    pre <- fixture_preprocessors(fx)

    # The CPU steps.
    for (i in seq_along(res$members)) {
      m <- res$members[[i]]
      layout <- tabpfn_member_layout(
        res$x_all[seq_len(n), m$features, drop = FALSE],
        categorical[m$features],
        pre[[m$preprocessor]]
      )
      cpu <- unname(tabpfn_member_cpu(m, res$x_all, layout)[seq_len(n), ])
      ref <- torch::as_array(fx$tensors[[sprintf("member.%d.X_train", i - 1)]])
      expect_identical(is.nan(cpu), is.nan(ref))
      expect_identical(cpu[!is.nan(cpu)], ref[!is.nan(ref)])
    }

    # Then the torch steps give the model's input.
    expect_model_inputs(res$inputs, fx)
  }
})

test_that("fingerprints match Python", {
  # The v3 fixtures are left out: their fingerprints hash the SVD features,
  # which are only bit-identical to Python's on the reference platform.
  for (name in c("pipeline-classification", "pipeline-regression")) {
    skip_if_no_fixture(name)
    fx <- load_pipeline_fixture(name)
    res <- fixture_inputs(fx, fx$meta$outlier_std)
    fingerprints <- fixture_fingerprints(fx)
    for (i in seq_along(res$inputs$x)) {
      shuffle <- unlist(fx$meta$members[[i]]$shuffle)
      expect_identical(
        as_numeric(res$inputs$x[[i]][, which(shuffle == max(shuffle))]),
        as_numeric(fingerprints[[i]])
      )
    }
  }
})

test_that("duplicate training rows get distinct fingerprints", {
  skip_if_no_torch()
  x <- torch::torch_tensor(
    matrix(c(1, 1, 2, 1, 5, 5, 6, 5), ncol = 2),
    dtype = torch::torch_float32()
  )
  fp <- as_numeric(tabpfn_fingerprint(x, num_train = 3))
  expect_false(identical(fp[1], fp[2]))
  expect_identical(fp[4], fp[1])
  expect_all_true(fp >= 0 & fp <= 1)
})

test_that("member borders through the target transform match Python exactly", {
  skip_if_no_fixture("pipeline-regression")
  fx <- load_pipeline_fixture("pipeline-regression")
  transform <- fx$meta$members[[2]]$target_transform
  borders <- tabpfn_inverse_target_borders(fx$tensors$borders, transform)
  expect_null(borders$cancel)
  expect_identical(borders$borders, as_numeric(fx$tensors$member.1.borders))
})

test_that("the safepower fit matches Python", {
  skip_if_no_fixture("pipeline-regression")
  fx <- load_pipeline_fixture("pipeline-regression")
  y <- unlist(fx$meta$y)[seq_len(fx$n_train)]
  y_scale <- tabpfn_znorm_fit(y)
  expect_equal(y_scale$mean, fx$meta$y_train_mean)
  expect_equal(y_scale$std, fx$meta$y_train_std)
  z <- (y - y_scale$mean) / y_scale$std
  expect_equal(z, as_numeric(fx$tensors$member.0.y_train))

  fit <- tabpfn_fit_safepower(z)
  python <- fx$meta$members[[2]]$target_transform
  expect_equal(fit$lambda, python$lambda, tolerance = 1e-6)
  expect_equal(
    tabpfn_apply_target_transform(z, python),
    as_numeric(fx$tensors$member.1.y_train)
  )
})

test_that("broken transformed borders are pinned and their buckets cancelled", {
  b <- c(-Inf, -2e3, -1, 0, 1, 2, 5e3, NaN)
  fixed <- tabpfn_fix_borders(b)
  expect_identical(fixed$borders, c(-2, -1, -1, 0, 1, 2, 2, 3))
  expect_identical(fixed$cancel, c(TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, TRUE))
})
