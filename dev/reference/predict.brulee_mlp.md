# Predict from a `brulee_mlp`

Predict from a `brulee_mlp`

## Usage

``` r
# S3 method for class 'brulee_mlp'
predict(object, new_data, type = NULL, epoch = NULL, ...)
```

## Arguments

- object:

  A `brulee_mlp` object.

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
if (torch::torch_is_installed() && rlang::is_installed(c("recipes", "modeldata"))) {
 # regression example:

 data(ames, package = "modeldata")

 ames$Sale_Price <- log10(ames$Sale_Price)

 set.seed(1)
 in_train <- sample(seq_len(nrow(ames)), 2000)
 ames_train <- ames[ in_train,]
 ames_test  <- ames[-in_train,]

 # Using recipe
 library(recipes)

 ames_rec <-
  recipe(Sale_Price ~ Longitude + Latitude, data = ames_train) |>
    step_normalize(all_numeric_predictors())

 set.seed(2)
 fit <- brulee_mlp(ames_rec, data = ames_train, epochs = 50, batch_size = 32)

 predict(fit, ames_test)
}
#> # A tibble: 930 × 1
#>    .pred
#>    <dbl>
#>  1  5.28
#>  2  5.27
#>  3  5.27
#>  4  5.34
#>  5  5.26
#>  6  5.24
#>  7  5.26
#>  8  5.24
#>  9  5.23
#> 10  5.19
#> # ℹ 920 more rows
# }
```
