# Predict from a `brulee_rln`

Predict from a `brulee_rln`

## Usage

``` r
# S3 method for class 'brulee_rln'
predict(object, new_data, type = NULL, epoch = NULL, ...)
```

## Arguments

- object:

  A `brulee_rln` object.

- new_data:

  A data frame or matrix of new predictors.

- type:

  A single character. The only valid option is `"numeric"` for numeric
  predictions.

- epoch:

  An integer for the epoch to make predictions. If larger than the
  number of epochs fit, a warning is issued and the last epoch is used.
  If `NULL` (default), the epoch with the smallest loss is used.

- ...:

  Not used, but required for extensibility.

## Value

A tibble of predictions with the same number of rows as `new_data`.

## Examples

``` r
# \donttest{
if (torch::torch_is_installed() && rlang::is_installed(c("recipes", "modeldata"))) {

 data(ames, package = "modeldata")
 ames$Sale_Price <- log10(ames$Sale_Price)

 set.seed(1)
 in_train <- sample(seq_len(nrow(ames)), 2000)
 ames_train <- ames[ in_train,]
 ames_test  <- ames[-in_train,]

 library(recipes)

 ames_rec <-
  recipe(Sale_Price ~ Longitude + Latitude, data = ames_train) |>
    step_normalize(all_numeric_predictors())

 set.seed(2)
 fit <- brulee_rln(ames_rec, data = ames_train, hidden_units = 20L, epochs = 30L)

 predict(fit, ames_test)
}
#> # A tibble: 930 × 1
#>    .pred
#>    <dbl>
#>  1  5.19
#>  2  5.36
#>  3  5.36
#>  4  5.27
#>  5  5.29
#>  6  5.24
#>  7  5.22
#>  8  5.22
#>  9  5.24
#> 10  5.21
#> # ℹ 920 more rows
# }
```
