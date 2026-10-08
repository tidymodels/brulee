# Predict from a `brulee_tab_pfn`

Predict from a `brulee_tab_pfn`

## Usage

``` r
# S3 method for class 'brulee_tab_pfn'
predict(object, new_data, type = NULL, quantile_levels = (1:9)/10, ...)
```

## Arguments

- object:

  A `brulee_tab_pfn` object.

- new_data:

  A data frame or matrix of new predictors.

- type:

  A single character. The type of predictions to generate. Valid options
  are:

  - `"numeric"` for numeric predictions (the mean of the predictive
    distribution).

  - `"quantile"` for quantiles of the predictive distribution.

  - `"class"` for hard class predictions.

  - `"prob"` for soft class predictions (i.e., class probabilities).

  The default (`NULL`) is `"class"` for classification and `"numeric"`
  for regression.

- quantile_levels:

  A numeric vector of probabilities in (0, 1) for `type = "quantile"`.

- ...:

  Not used, but required for extensibility.

## Value

A tibble of predictions with one row per row of `new_data`:
`.pred_class`, `.pred_{level}` columns, `.pred`, or `.pred_quantile` (a
[`hardhat::quantile_pred()`](https://hardhat.tidymodels.org/reference/quantile_pred.html)
column), depending on `type`.

## Details

TabPFN runs the network when predicting: each call feeds the stored
training rows and `new_data` through the model.

R torch currently lacks a memory-efficient attention kernel for Apple
GPUs that PyTorch (\>= 2.13) has, so with `device = "mps"` attention is
computed in chunks of rows to keep it within the GPU's memory.

## Examples

``` r
if (FALSE) {
fit <- brulee_tab_pfn(mpg ~ ., data = mtcars)
predict(fit, mtcars[1:3, ])
predict(fit, mtcars[1:3, ], type = "quantile", quantile_levels = c(0.1, 0.9))
}
```
