# Predict from a `brulee_tab_icl`

Predict from a `brulee_tab_icl`

## Usage

``` r
# S3 method for class 'brulee_tab_icl'
predict(object, new_data, type = NULL, ...)
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

  - `"numeric"` for numeric predictions (regression).

  If `NULL` (the default), the natural type for the outcome is used:
  `"class"` for a factor outcome and `"numeric"` for a numeric one.

- ...:

  Not used, but required for extensibility.

## Value

A tibble of predictions. The number of rows is guaranteed to match
`new_data`. For `type = "prob"` there is one column per outcome class;
for `"class"` and `"numeric"` there is a single prediction column.

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

## See also

[`brulee_tab_icl()`](https://brulee.tidymodels.org/reference/brulee_tab_icl.md)

## Examples

``` r
if (FALSE) { # \dontrun{
if (torch::torch_is_installed() && rlang::is_installed("modeldata")) {
  data(penguins, package = "modeldata")
  penguins <- na.omit(penguins)

  fit <- brulee_tab_icl(
    species ~ .,
    data = penguins,
    path = "path/to/tabicl-classifier"
  )
  predict(fit, penguins)
  predict(fit, penguins, type = "prob")
}
} # }
```
