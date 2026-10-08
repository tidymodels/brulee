# Changelog

## brulee (development version)

- [`brulee_tab_pfn()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_pfn.md)
  makes the TabPFN tabular foundation models (versions 3, 3.5, and
  3.5-fast) available, running them in R torch without Python. The model
  weights are released by Prior Labs under non-commercial licenses:
  [`?brulee_tab_pfn`](https://brulee.tidymodels.org/dev/reference/brulee_tab_pfn.md)
  explains the one-time setup,
  [`tab_pfn_download_weights()`](https://brulee.tidymodels.org/dev/reference/tab_pfn_download_weights.md)
  downloads them,
  [`tab_pfn_weights_available()`](https://brulee.tidymodels.org/dev/reference/tab_pfn_download_weights.md)
  checks for them,
  [`tab_pfn_clear_cache()`](https://brulee.tidymodels.org/dev/reference/tab_pfn_clear_cache.md)
  removes them, and
  [`tab_pfn_versions()`](https://brulee.tidymodels.org/dev/reference/tab_pfn_versions.md)
  lists the supported versions. Weights already downloaded by the Python
  `tabpfn` package are reused.

- [`predict()`](https://rdrr.io/r/stats/predict.html) and
  [`augment()`](https://generics.r-lib.org/reference/augment.html) for
  [`brulee_tab_icl()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_icl.md)
  and
  [`brulee_tab_pfn()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_pfn.md)
  return zero-row results for zero-row `new_data` without running the
  model;
  [`brulee_tab_icl()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_icl.md)
  classification previously failed.

- [`brulee_tab_icl()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_icl.md)
  and
  [`brulee_tab_pfn()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_pfn.md)
  now require `softmax_temperature` to be a finite number greater than
  0; a temperature of 0 used to produce `NaN` predictions.

- Models with a numeric outcome
  ([`brulee_linear_reg()`](https://brulee.tidymodels.org/dev/reference/brulee_linear_reg.md),
  [`brulee_mlp()`](https://brulee.tidymodels.org/dev/reference/brulee_mlp.md),
  [`brulee_resnet()`](https://brulee.tidymodels.org/dev/reference/brulee_resnet.md),
  [`brulee_rln()`](https://brulee.tidymodels.org/dev/reference/brulee_rln.md),
  [`brulee_saint()`](https://brulee.tidymodels.org/dev/reference/brulee_saint.md),
  [`brulee_auto_int()`](https://brulee.tidymodels.org/dev/reference/brulee_auto_int.md),
  [`brulee_tab_icl()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_icl.md),
  and
  [`brulee_tab_pfn()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_pfn.md))
  now error informatively when the outcome has a single distinct value.

- [`brulee_tab_icl()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_icl.md)
  and
  [`brulee_tab_pfn()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_pfn.md)
  now check that the outcome is a single numeric or factor column, and
  drop rows with a missing outcome with a warning; previously a missing
  outcome could make the fit fail or produce invalid predictions.

- [`brulee_tab_icl()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_icl.md)
  and
  [`brulee_tab_pfn()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_pfn.md)
  share one way of sampling the training set down to
  `training_set_limit` rows. For
  [`brulee_tab_icl()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_icl.md),
  numeric outcomes are now sampled within quartiles rather than at
  random, and outcome levels with no rows no longer count as classes, so
  exactly `training_set_limit` rows are kept. The rows kept for a given
  [`set.seed()`](https://rdrr.io/r/base/Random.html) differ from earlier
  versions.

- The weight downloads of
  [`brulee_chronos()`](https://brulee.tidymodels.org/dev/reference/brulee_chronos.md),
  [`brulee_tab_icl()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_icl.md),
  and
  [`brulee_tab_pfn()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_pfn.md)
  share one downloader, which now writes to a temporary file and renames
  it when complete, so an interrupted download no longer leaves a
  partial file under the real name. If the completed file can’t be moved
  into place, it is copied, and failing that, the download errors with
  advice instead of reporting success.

## brulee 1.2.0

CRAN release: 2026-09-02

- [`predict()`](https://rdrr.io/r/stats/predict.html) for regression
  [`brulee_tab_icl()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_icl.md)
  models gained two new `type` values. `"quantile"` returns a
  `.pred_quantile` column (a
  [`hardhat::quantile_pred()`](https://hardhat.tidymodels.org/reference/quantile_pred.html)
  vector) at the levels given by the new predict-time `quantile_levels`
  argument, which defaults to `(1:9) / 10`. `"variance"` returns the
  variance of the predictive distribution in a `.pred_variance` column.
  The TabICL regression head was already a quantile regression head
  internally, so this exposes a distribution the model was always
  computing; `type = "numeric"` is unchanged and remains the default.
  Unlike
  [`brulee_chronos()`](https://brulee.tidymodels.org/dev/reference/brulee_chronos.md),
  the levels are not fixed when the model is created and any value in
  the open interval (0, 1) can be requested. See
  [`?predict.brulee_tab_icl`](https://brulee.tidymodels.org/dev/reference/predict.brulee_tab_icl.md)
  for how ensemble members are pooled for each type.

- Added [`augment()`](https://generics.r-lib.org/reference/augment.html)
  methods for all brulee model fits.

  - For regression models, `.pred` and `.resid` are added. For the
    latter, it is computed when the outcome is present in the data. See
    the exceptions below for foundational models.
  - For classification, the hard class predictions and class probability
    estimates are added.
  - For
    [`brulee_chronos()`](https://brulee.tidymodels.org/dev/reference/brulee_chronos.md)
    models, the forecast columns are aligned to the rows of `new_data`
    rather than to the internal per-series prediction order.
  - For
    [`brulee_tab_icl()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_icl.md)
    regression models, quantile regression estimates and the prediction
    variance can be obtained by setting the `quantile_levels` argument
    to a non-null value. `.resid` is then measured against the median of
    the predictive distribution.

- The error thrown when
  [`predict()`](https://rdrr.io/r/stats/predict.html) is given an
  unsupported `type` is now attributed to
  [`predict()`](https://rdrr.io/r/stats/predict.html) rather than to
  brulee’s internal helper.

- [`predict()`](https://rdrr.io/r/stats/predict.html) now clamps an
  `epoch` larger than the number of epochs actually fit, which is what
  its documentation and its warning have always promised. Previously it
  warned and then failed with a `subscript out of bounds` error, and an
  `epoch` exactly equal to `length(fit$estimates)` failed with no
  warning at all. Because early stopping makes the number of epochs fit
  vary by platform, a fixed `epoch` could work on one machine and fail
  on another. [`coef()`](https://rdrr.io/r/stats/coef.html) was already
  correct and now shares the same check
  ([\#138](https://github.com/tidymodels/brulee/issues/138)).

## brulee 1.1.1

CRAN release: 2026-07-13

- Pretrained model weights (for
  [`brulee_tab_icl()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_icl.md)
  and
  [`brulee_chronos()`](https://brulee.tidymodels.org/dev/reference/brulee_chronos.md))
  are no longer downloaded automatically when the package is attached.
  When the weights are missing, both
  [`brulee_tab_icl()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_icl.md)
  and
  [`brulee_chronos()`](https://brulee.tidymodels.org/dev/reference/brulee_chronos.md)
  now prompt to download them in an interactive session and error
  otherwise. TabICL weights can also be downloaded explicitly with
  [`tab_icl_download_weights()`](https://brulee.tidymodels.org/dev/reference/tab_icl_download_weights.md)
  ([\#130](https://github.com/tidymodels/brulee/issues/130)).

- Downloaded model weights are now cached in the per-user cache
  directory returned by `tools::R_user_dir("brulee", "cache")`, which
  respects platform conventions, rather than under `~/.cache`
  ([\#130](https://github.com/tidymodels/brulee/issues/130)).

- [`brulee_resnet()`](https://brulee.tidymodels.org/dev/reference/brulee_resnet.md)
  no longer returns all-`NA` predictions when the training-set size
  leaves a single-row trailing batch (e.g. `mtcars` with default
  settings). Such a batch made batch-normalization compute a variance
  over one sample, corrupting its buffers with `NaN`
  ([\#122](https://github.com/tidymodels/brulee/issues/122)).

## brulee 1.1.0

CRAN release: 2026-07-02

- [`brulee_tab_icl()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_icl.md)
  makes the open-source foundational model TabICL available. On first
  use, there is a substantial download (~ 400MB) for the model weights
  that is cached locally.

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

- Fixed a bug where torch’s L-BFGS optimizers internal convergence flag
  is NA, throwing an unhelpful error.

### Breaking Changes

- The
  [`brulee_saint()`](https://brulee.tidymodels.org/dev/reference/brulee_saint.md)
  argument `use_target_token` was renamed to `target_token`.

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

- All estimated models now include epoch zero (the randomly initialized
  parameters, before any training) as the first element of `loss` and
  `estimates`, matching the neural-network models. These vectors are now
  length `epochs + 1`, `epoch = 0` is a valid argument to
  [`predict()`](https://rdrr.io/r/stats/predict.html) and
  [`coef()`](https://rdrr.io/r/stats/coef.html), and the entry for
  `best_epoch` is at position `best_epoch + 1`. Predictions and
  coefficients for a given (positive) epoch are unchanged. Note: objects
  serialized by earlier versions of these three functions predict off by
  one epoch under the new indexing, so refit any stored models.

  - The [`print()`](https://rdrr.io/r/base/print.html) methods now
    report the loss from the best epoch. Previously the displayed loss
    was taken one epoch too early (it ignored the prepended epoch-zero
    entry in `loss`).

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
