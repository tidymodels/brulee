# Predict from a `brulee_saint`

Predict from a `brulee_saint`

## Usage

``` r
# S3 method for class 'brulee_saint'
predict(object, new_data, type = NULL, epoch = NULL, ...)
```

## Arguments

- object:

  A `brulee_saint` object.

- new_data:

  A data frame or matrix of new predictors.

- type:

  A single character. The type of predictions to generate. Valid options
  are:

  - `"numeric"` for numeric predictions.

  - `"class"` for hard class predictions

  - `"prob"` for soft class predictions (i.e., class probabilities)

- epoch:

  An integer for the epoch to make predictions. If this value is larger
  than the maximum number that was fit, a warning is issued and the
  parameters from the last epoch are used. If left `NULL`, the epoch
  associated with the smallest loss is used.

- ...:

  Not used, but required for extensibility.

## Value

A tibble of predictions. The number of rows in the tibble is guaranteed
to be the same as the number of rows in `new_data`.

## Examples

``` r
# \donttest{
if (torch::torch_is_installed() & rlang::is_installed(c("recipes", "modeldata"))) {
  set.seed(87261)
  tr_data <- modeldata::sim_classification(500)
  te_data <- modeldata::sim_classification(50)

  set.seed(2)
  fit <- brulee_saint(class ~ ., data = tr_data,
                      epochs = 50L, batch_size = 64L, stop_iter = 10L,
                      learn_rate = 0.001)
  fit

  autoplot(fit)

 predict(fit, te_data)
 predict(fit, te_data, type = "prob")
}
#> # A tibble: 50 × 2
#>    .pred_class_1 .pred_class_2
#>            <dbl>         <dbl>
#>  1         0.446         0.554
#>  2         0.454         0.546
#>  3         0.432         0.568
#>  4         0.454         0.546
#>  5         0.454         0.546
#>  6         0.454         0.546
#>  7         0.442         0.558
#>  8         0.446         0.554
#>  9         0.454         0.546
#> 10         0.459         0.541
#> # ℹ 40 more rows
# }
```
