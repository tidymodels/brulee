# Predict from a `brulee_auto_int`

Predict from a `brulee_auto_int`

## Usage

``` r
# S3 method for class 'brulee_auto_int'
predict(object, new_data, type = NULL, epoch = NULL, ...)
```

## Arguments

- object:

  A `brulee_auto_int` object.

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
  fit <- brulee_auto_int(class ~ ., data = tr_data,
                         epochs = 50L, batch_size = 64L, stop_iter = 10L,
                         hidden_units = 5, hidden_activations = "relu",
                         learn_rate = 0.01, penalty = 0.01)
  fit

  autoplot(fit)

 predict(fit, te_data)
 predict(fit, te_data, type = "prob")
}
#> # A tibble: 50 × 2
#>    .pred_class_1 .pred_class_2
#>            <dbl>         <dbl>
#>  1        0.830        0.170  
#>  2        0.528        0.472  
#>  3        0.438        0.562  
#>  4        0.0112       0.989  
#>  5        0.196        0.804  
#>  6        0.0213       0.979  
#>  7        0.0701       0.930  
#>  8        0.0594       0.941  
#>  9        0.993        0.00749
#> 10        0.762        0.238  
#> # ℹ 40 more rows
# }
```
