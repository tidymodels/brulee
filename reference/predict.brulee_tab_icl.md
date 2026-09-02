# Predict from a `brulee_tab_icl`

Predict from a `brulee_tab_icl`

## Usage

``` r
# S3 method for class 'brulee_tab_icl'
predict(object, new_data, type = NULL, quantile_levels = (1:9)/10, ...)
```

## Arguments

- object:

  A `brulee_tab_icl` object from
  [`brulee_tab_icl()`](https://brulee.tidymodels.org/reference/brulee_tab_icl.md).

- new_data:

  A data frame or matrix of new predictors.

- type:

  A single character string for the type of prediction. Valid options
  are:

  - `"class"` for hard class predictions (classification).

  - `"prob"` for class probabilities (classification).

  - `"numeric"` for the mean of the predictive distribution
    (regression).

  - `"quantile"` for quantiles of the predictive distribution
    (regression).

  - `"variance"` for its variance (regression).

  If `NULL` (the default), the natural type for the outcome is used:
  `"class"` for a factor outcome and `"numeric"` for a numeric one.

- quantile_levels:

  A numeric vector of quantile levels, each in the open interval
  `(0, 1)`, sorted and unique. Only used when `type = "quantile"`.
  Defaults to `(1:9) / 10`.

- ...:

  Not used, but required for extensibility.

## Value

A tibble of predictions. The number of rows is guaranteed to match
`new_data`. For `type = "prob"` there is one column per outcome class;
otherwise there is a single prediction column: `.pred_class` for
`"class"`, `.pred` for `"numeric"`, `.pred_variance` for `"variance"`,
and `.pred_quantile` for `"quantile"` (a
[`hardhat::quantile_pred()`](https://hardhat.tidymodels.org/reference/quantile_pred.html)
vector packing all requested levels into one column).

## Details

Because TabICL is an in-context learner, prediction reloads the
pretrained weights from the checkpoint directory stored on `object` and
conditions on the training rows captured at fit time. The same
preprocessing and ensembling used for `object` are applied to
`new_data`; see
[`brulee_tab_icl()`](https://brulee.tidymodels.org/reference/brulee_tab_icl.md)
for details. For classification, `"prob"` returns one column per class
(named `.pred_<level>`) and `"class"` returns the highest-probability
class.

### Ensembling and aggregation

The regression checkpoint is a quantile regression head: it emits a
fixed grid of quantile values for every row, which are monotonized and
turned into a continuous distribution (piecewise-linear between the
knots, exponential tails outside them). Every returned statistic is a
readout from that distribution, so `quantile_levels` does not change
what the model computes, only which values are read off afterwards. Any
level in the open interval `(0, 1)` is available, including levels far
enough into the tails to be extrapolated.

With `num_estimators > 1`, each ensemble member yields its own
distribution and the members are combined on the outcome's scale. The
combination rule depends on the statistic:

- `.pred` is the arithmetic mean across members of each member's
  distribution mean.

- `.pred_quantile` averages the members' quantile curves level by level
  (Vincentization). Averaging monotone curves preserves monotonicity, so
  the pooled result is still a valid quantile function.

- `.pred_variance` is the geometric mean across members. A variance is a
  positive scale parameter, so members are pooled multiplicatively.

Two consequences are worth stating plainly. First, `.pred` is the mean
of the predictive distribution, so it is not the same as the 0.5 entry
of `.pred_quantile`; this differs from
[`predict.brulee_chronos()`](https://brulee.tidymodels.org/reference/predict.brulee_chronos.md),
where `.pred` is defined to be the median. Second, `.pred_variance`
pools each member's own variance and so does not include the spread
between members' central estimates; it will not equal the variance
implied by `.pred_quantile`.

`.pred_variance` is on the squared scale of the outcome. The target is
standardized internally and the variance is scaled back by the square of
the target's standard deviation. Note that the reference Python
implementation instead applies the full location-scale inverse to the
variance; brulee's values differ from it deliberately.

## See also

[`brulee_tab_icl()`](https://brulee.tidymodels.org/reference/brulee_tab_icl.md)

## Examples

``` r
if (FALSE) { # \dontrun{
if (rlang::is_installed(c("MASS", "ggplot2")) &
    tab_icl_weights_available() &
     interactive()) {
  library(ggplot2)

  motorcycles <- MASS::mcycle
  in_tr <- seq(1, nrow(motorcycles), by = 2)
  mcycle_tr <- motorcycles[in_tr, ]
  mcycle_te <- motorcycles[-in_tr, ]

  mcycle_grid <-
   dplyr::tibble(
     times = seq(min(motorcycles$times), max(motorcycles$times), length.out = 200)
   )
  mcycle_grid$.row <- seq_len(nrow(mcycle_grid))

  fit <- brulee_tab_icl(accel ~ times, data = mcycle_tr)

  # ------------------------------------------------------------------------------
  # Predict mean acceleration

  mean_pred <- predict(fit, mcycle_grid) |> dplyr::bind_cols(mcycle_grid)

  mean_p <-
   mean_pred |>
   ggplot(aes(times)) +
   geom_point(data = mcycle_te, aes(y = accel), alpha = 1 / 2) +
   geom_line(aes(y = .pred))

  #------------------------------------------------------------------------------Predict 5 %, 50%
  # Predict 5%, 50%, and 90% quantiles of acceleration

  q_pred <-
   predict(fit,
           mcycle_grid,
           type = "quantile",
           quantile_levels = c(0.1, 0.5, 0.9))
  q_pred$.row <- seq_len(nrow(q_pred))

  q_pred_longer <-
   q_pred$.pred_quantile |>
   dplyr::as_tibble() |>
   dplyr::full_join(mcycle_grid, by = ".row") |>
   dplyr::mutate(level = format(.quantile_levels))

  mean_p +
   geom_line(
     data = q_pred_longer,
     aes(y = .pred_quantile, col = level, group = level)
   )
}
} # }
```
