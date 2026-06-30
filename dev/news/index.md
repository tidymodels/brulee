# Changelog

## brulee (development version)

- The
  [`brulee_saint()`](https://brulee.tidymodels.org/dev/reference/brulee_saint.md)
  argument `use_target_token` was renamed to `target_token`.

- [`brulee_saint()`](https://brulee.tidymodels.org/dev/reference/brulee_saint.md)
  and
  [`brulee_auto_int()`](https://brulee.tidymodels.org/dev/reference/brulee_auto_int.md)
  now support gradient clipping via the `grad_value_clip` and
  `grad_norm_clip` arguments (both default to `5`), matching
  [`brulee_mlp()`](https://brulee.tidymodels.org/dev/reference/brulee_mlp.md)
  and
  [`brulee_resnet()`](https://brulee.tidymodels.org/dev/reference/brulee_resnet.md).
  This prevents the loss from overflowing to `NaN` during training with
  aggressive learning rates.

- There is now a `type` argument to
  [`predict.brulee_chronos()`](https://brulee.tidymodels.org/dev/reference/predict.brulee_chronos.md):
  `"all"` returns `.pred` and `.pred_quantile` (unchanged default),
  `"numeric"` returns only `.pred`, `"quantile"` returns only
  `.pred_quantile`. The id column is still prepended for multi-series
  models regardless of type.

- Fixed a bug where torch’s L-BFGS optimizer’’’s internal convergence
  flag is NA, throwing an unhelpful error.

### Breaking Changes

- [`predict()`](https://rdrr.io/r/stats/predict.html) for
  [`brulee_chronos()`](https://brulee.tidymodels.org/dev/reference/brulee_chronos.md)
  models was reworked. The historical context is always the data
  supplied to
  [`brulee_chronos()`](https://brulee.tidymodels.org/dev/reference/brulee_chronos.md)
  (the model is pretrained and does no training), so the former
  `new_data` context-override was removed. The argument previously
  called `future_df` is now `new_data`: it describes the future window
  to forecast for and may have at most `prediction_length` rows per
  series (previously exactly `prediction_length`). When fewer rows are
  supplied, the forecast is truncated to those rows.
  [`predict()`](https://rdrr.io/r/stats/predict.html) also gained a
  `type` argument (`"all"`, `"numeric"`, or `"quantile"`) to select
  which prediction columns are returned.

## brulee 1.0.0

CRAN release: 2026-06-17

New models for tabular data:

- Regularization Learning Networks
  ([`brulee_rln()`](https://brulee.tidymodels.org/dev/reference/brulee_rln.md))
  use a conventional MLP architecture but each weight learns its own
  adaptive regularization coefficient.

- ResNet
  ([`brulee_resnet()`](https://brulee.tidymodels.org/dev/reference/brulee_resnet.md))
  can fit a multilayer neural network with skip (i.e. residual)
  connections and batch normalization.

- AutoInt
  ([`brulee_auto_int()`](https://brulee.tidymodels.org/dev/reference/brulee_auto_int.md))
  uses residual connections and columnwise attention mechanisms to
  create embeddings that encourage in-context learning of features.

- Saint
  ([`brulee_saint()`](https://brulee.tidymodels.org/dev/reference/brulee_saint.md))
  uses column and/or row attention mechanisms.

- Chronos2
  ([`brulee_chronos()`](https://brulee.tidymodels.org/dev/reference/brulee_chronos.md))
  is a foundational model for forecasting.

- All modeling functions now support GPU acceleration via the `device`
  parameter. Users can specify `device = "cpu"`, `device = "cuda"`, or
  `device = "mps"` (Apple Silicon). When `device = NULL` (default), the
  package automatically selects CUDA if available, otherwise defaults to
  CPU. Note: MPS is not auto-selected because it doesn’t support float64
  dtype required by brulee.
  See[`?training_efficiency`](https://brulee.tidymodels.org/dev/reference/training_efficiency.md)
  for some related notes.

### Breaking Changes

- Float tensors were changed from 64-bit floats to 32-bit. This is to
  enable GPU usage on MPS devices.

- Parameters are initialized on CPU devices and then converted to the
  chosen device. In some cases, the RNG initialization code is
  independent of the seed.

- For classification, the softmax was moved out of every model’s forward
  pass so the loss can use
  [`torch::nnf_cross_entropy()`](https://torch.mlverse.org/docs/reference/nnf_cross_entropy.html)
  (which applies the log-sum-exp trick internally) instead of
  `nll_loss(log(softmax(x)))`. This avoids `log(0)` underflow that
  produced `NaN` losses and “numerical overflow” early stopping on
  overspecified
  [`brulee_saint()`](https://brulee.tidymodels.org/dev/reference/brulee_saint.md)
  /
  [`brulee_auto_int()`](https://brulee.tidymodels.org/dev/reference/brulee_auto_int.md)
  fits. Affects
  [`brulee_mlp()`](https://brulee.tidymodels.org/dev/reference/brulee_mlp.md),
  [`brulee_logistic_reg()`](https://brulee.tidymodels.org/dev/reference/brulee_logistic_reg.md),
  [`brulee_multinomial_reg()`](https://brulee.tidymodels.org/dev/reference/brulee_multinomial_reg.md),
  [`brulee_resnet()`](https://brulee.tidymodels.org/dev/reference/brulee_resnet.md),
  [`brulee_auto_int()`](https://brulee.tidymodels.org/dev/reference/brulee_auto_int.md),
  and
  [`brulee_saint()`](https://brulee.tidymodels.org/dev/reference/brulee_saint.md).
  New fits carry `output_type = "logits"` so the predict path applies
  softmax; serialized fits from earlier versions of brulee continue to
  predict correctly.

## brulee 0.6.0

CRAN release: 2025-09-02

- Transition from the magrittr pipe to the base R pipe.

- To try to help avoiding numeric overflow in the loss functions:

  - Tensors are stored as a 64-bit float instead of 32-bit.

  - Starting values were transitioned to using Gaussian distribution
    (instead of uniform) with a smaller standard deviation.

  - The results always contain the initial results to use as a fallback
    if there is overflow during the first epoch.

  - [`brulee_mlp()`](https://brulee.tidymodels.org/dev/reference/brulee_mlp.md)
    has two additional parameters, `grad_value_clip` and
    `grad_value_clip`, that prevent issues.

  - The warning was changed to “Early stopping occurred at epoch {X} due
    to numerical overflow of the loss function.”

- Several new SGD optimizers were added: `"ADAMw"`, `"Adadelta"`,
  `"Adagrad"`, and `"RMSprop"`.

- Mixture parameter values different than zero cannot be used for
  several optimizers since they require L2 penalties.

## brulee 0.5.0

CRAN release: 2025-04-07

- Removed a unit test for numerical overflow since it occurs less
  frequently and has become increasingly more challenging to reproduce.

## brulee 0.4.0

CRAN release: 2025-01-30

- Added a convenience function,
  [`brulee_mlp_two_layer()`](https://brulee.tidymodels.org/dev/reference/brulee_mlp.md),
  to more easily fit two-layer networks with parsnip.

- Various changes and improvements to error and warning messages.

- Fixed a bug that occurred when linear activation was used for neural
  networks ([\#68](https://github.com/tidymodels/brulee/issues/68)).

## brulee 0.3.0

CRAN release: 2024-02-14

- Fixed bug where [`coef()`](https://rdrr.io/r/stats/coef.html) didn’t
  would error if used on a
  [`brulee_logistic_reg()`](https://brulee.tidymodels.org/dev/reference/brulee_logistic_reg.md)
  that was trained with a recipe.
  ([\#66](https://github.com/tidymodels/brulee/issues/66))

- Fixed a bug where SGD always being used as the optimizer
  ([\#61](https://github.com/tidymodels/brulee/issues/61)).

- Additional activation functions were added
  ([\#74](https://github.com/tidymodels/brulee/issues/74)).

## brulee 0.2.0

CRAN release: 2022-09-19

- Several learning rate schedulers were added to the modeling functions
  ([\#12](https://github.com/tidymodels/brulee/issues/12)).

- An `optimizer` was added to \[brulee_mlp()\], with a new default being
  LBFGS instead of stochastic gradient descent.

## brulee 0.1.0

CRAN release: 2022-02-02

- Modeling functions gained a `mixture` argument for the proportion of
  L1 penalty that is used.
  ([\#50](https://github.com/tidymodels/brulee/issues/50))

- Penalization was not occurring when quasi-Newton optimization was
  chosen. ([\#50](https://github.com/tidymodels/brulee/issues/50))

## brulee 0.0.1

CRAN release: 2021-12-15

First CRAN release.
