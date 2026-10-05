# End-to-end parity with Python uses the pipeline fixtures' tiny
# random-weight checkpoints. Tests of the interface with the released weights
# run only when those are cached (tab_pfn_download_weights()).

# The x/y interface keeps the column order of the data, which Python's
# preprocessing depends on (the formula interface moves factors last).
fit_with_python_members <- function(fx, env = parent.frame()) {
  local_tiny_checkpoint(fx, env)
  local_python_fingerprints(fx, env)
  train <- fx$data[seq_len(fx$n_train), ]
  outcome <- unlist(fx$meta$y)[seq_len(fx$n_train)]
  if (is.character(outcome)) {
    outcome <- factor(outcome)
  }
  fit <- brulee_tab_pfn(
    train,
    outcome,
    num_estimators = length(fx$meta$members),
    version = fx$meta$version,
    device = "cpu"
  )
  fit$fit$members <- fixture_members(fx, ncol(fit$fit$x_train))
  fit
}

test_that("classification matches Python end to end", {
  for (name in c("pipeline-classification", "pipeline-v3-classification")) {
    skip_if_no_fixture(name)
    fx <- load_pipeline_fixture(name)
    pred <- predict(
      fit_with_python_members(fx),
      fx$data[-seq_len(fx$n_train), ],
      type = "prob"
    )
    expect_equal(
      unname(as.matrix(pred)),
      torch::as_array(fx$tensors$pred.prob),
      tolerance = 1e-5
    )
  }
})

test_that("regression matches Python end to end", {
  for (name in c("pipeline-regression", "pipeline-v3-regression")) {
    skip_if_no_fixture(name)
    fx <- load_pipeline_fixture(name)
    fit <- fit_with_python_members(fx)
    new_data <- fx$data[-seq_len(fx$n_train), ]
    pred <- predict(fit, new_data)
    expect_equal(pred$.pred, as_numeric(fx$tensors$pred.mean), tolerance = 1e-5)
    pred <- predict(
      fit,
      new_data,
      type = "quantile",
      quantile_levels = c(0.1, 0.5, 0.9)
    )
    expect_equal(
      as.matrix(pred$.pred_quantile),
      torch::as_array(fx$tensors$pred.quantiles),
      tolerance = 1e-5,
      ignore_attr = TRUE
    )
  }
})

test_that("regression predictions have the tidymodels format", {
  skip_if_no_weights()
  set.seed(81723)
  fit <- brulee_tab_pfn(mpg ~ ., data = mtcars[-(1:4), ], num_estimators = 2)
  pred <- predict(
    fit,
    mtcars[1:4, ],
    type = "quantile",
    quantile_levels = c(0.1, 0.9)
  )
  expect_named(pred, ".pred_quantile")
  expect_s3_class(pred$.pred_quantile, "quantile_pred")
  q <- as.matrix(pred$.pred_quantile)
  expect_all_true(q[, 1] < q[, 2])
  expect_named(predict(fit, mtcars[1:4, ]), ".pred")
  expect_snapshot(
    predict(fit, mtcars[1:4, ], type = "quantile", quantile_levels = 2),
    error = TRUE
  )
  expect_snapshot(predict(fit, mtcars[1:4, ], type = "prob"), error = TRUE)

  aug <- augment(fit, mtcars[1:4, ], quantile_levels = c(0.1, 0.9))
  expect_named(aug, c(".pred", ".resid", ".pred_quantile", names(mtcars)))
  expect_equal(aug$.resid, mtcars$mpg[1:4] - aug$.pred)
})

test_that("edge cases: one new row, one member, one predictor", {
  skip_if_no_weights()
  set.seed(81723)
  fit <- brulee_tab_pfn(Species ~ Petal.Length, data = iris, num_estimators = 1)
  pred <- predict(fit, iris[51, ], type = "prob")
  expect_identical(nrow(pred), 1L)
  expect_equal(sum(unlist(pred[1, 1:3])), 1, tolerance = 1e-6)

  fit <- brulee_tab_pfn(mpg ~ wt, data = mtcars, num_estimators = 3)
  pred <- predict(fit, mtcars[1, ])
  expect_identical(nrow(pred), 1L)
  expect_true(is.finite(pred$.pred))
})

test_that("MPS predictions match the CPU", {
  skip_if_no_weights()
  skip_if_not(torch::backends_mps_is_available(), "MPS is not available")
  fit_on <- function(device) {
    set.seed(81723)
    brulee_tab_pfn(
      Species ~ .,
      data = iris[-(1:5 * 10), ],
      num_estimators = 2,
      device = device
    )
  }
  expect_equal(
    predict(fit_on("mps"), iris[1:5 * 10, ], type = "prob"),
    predict(fit_on("cpu"), iris[1:5 * 10, ], type = "prob"),
    tolerance = 1e-5
  )
})

test_that("class probabilities match Python from the same model output", {
  for (name in c("pipeline-classification", "pipeline-v3-classification")) {
    skip_if_no_fixture(name)
    fx <- load_pipeline_fixture(name)
    probs <- tabpfn_postprocess_classification(
      fixture_outputs(fx),
      fixture_members(fx, 1),
      length(fx$meta$classes),
      temperature = fx$meta$softmax_temperature
    )
    expect_equal(
      probs,
      torch::as_array(fx$tensors$pred.prob),
      tolerance = 1e-6
    )
  }
})

test_that("regression predictions match Python from the same model output", {
  for (name in c("pipeline-regression", "pipeline-v3-regression")) {
    skip_if_no_fixture(name)
    fx <- load_pipeline_fixture(name)
    borders <- fx$tensors$borders
    log_probs <- tabpfn_postprocess_regression(
      fixture_outputs(fx),
      fixture_members(fx, 1),
      borders,
      temperature = fx$meta$softmax_temperature
    )
    y_scale <- list(mean = fx$meta$y_train_mean, std = fx$meta$y_train_std)
    res <- tabpfn_decode_regression(
      log_probs,
      borders,
      y_scale,
      c(0.1, 0.5, 0.9)
    )
    expect_equal(
      res$mean,
      as_numeric(fx$tensors$pred.mean),
      tolerance = 1e-6
    )
    expect_equal(
      res$median,
      as_numeric(fx$tensors$pred.median),
      tolerance = 1e-6
    )
    # Tail quantiles fall in the wide outer bins of the bar distribution,
    # which magnify rounding differences in the probabilities.
    expect_equal(
      res$quantiles,
      torch::as_array(fx$tensors$pred.quantiles),
      tolerance = 1e-5
    )
  }
})

test_that("translating onto the same grid is a softmax", {
  skip_if_no_torch()
  logits <- torch::torch_randn(3, 6)
  grid <- c(-3, -2, -1, 0, 1, 2, 3)
  expect_equal(
    max_abs_diff(
      tabpfn_translate_probs(logits, grid, grid),
      torch::nnf_softmax(logits, 2)
    ),
    0
  )
  moved <- tabpfn_translate_probs(logits, grid + 0.25, grid)
  expect_equal(as_numeric(moved$sum(dim = 2)), rep(1, 3), tolerance = 1e-6)
})
