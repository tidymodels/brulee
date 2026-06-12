# Predict from a `brulee_resnet`

Predict from a `brulee_resnet`

## Usage

``` r
# S3 method for class 'brulee_resnet'
predict(object, new_data, type = NULL, epoch = NULL, ...)
```

## Arguments

- object:

  A `brulee_resnet` object.

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
 # regression example:

 data(ames, package = "modeldata")

 ames$Sale_Price <- log10(ames$Sale_Price)

 set.seed(1)
 in_train <- sample(1:nrow(ames), 2000)
 ames_train <- ames[ in_train,]
 ames_test  <- ames[-in_train,]

 # Using recipe
 library(recipes)

 ames_rec <-
  recipe(Sale_Price ~ Longitude + Latitude, data = ames_train) |>
    step_normalize(all_numeric_predictors())

 set.seed(2)
 fit <- brulee_resnet(ames_rec, data = ames_train,
                      hidden_units = 2, num_layers = 2, bottleneck_units = 10,
                      epochs = 50, batch_size = 32)

 predict(fit, ames_test)
}
#> # A tibble: 930 × 1
#>    .pred
#>    <dbl>
#>  1  5.21
#>  2  5.30
#>  3  5.29
#>  4  5.23
#>  5  5.25
#>  6  5.21
#>  7  5.22
#>  8  5.22
#>  9  5.22
#> 10  5.21
#> # ℹ 920 more rows
# }
```
