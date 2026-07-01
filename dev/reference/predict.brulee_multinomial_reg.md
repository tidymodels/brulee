# Predict from a `brulee_multinomial_reg`

Predict from a `brulee_multinomial_reg`

## Usage

``` r
# S3 method for class 'brulee_multinomial_reg'
predict(object, new_data, type = NULL, epoch = NULL, ...)
```

## Arguments

- object:

  A `brulee_multinomial_reg` object.

- new_data:

  A data frame or matrix of new predictors.

- type:

  A single character. The type of predictions to generate. Valid options
  are:

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
if (torch::torch_is_installed() && rlang::is_installed(c("recipes", "yardstick", "modeldata"))) {

  library(recipes)
  library(yardstick)

  data(penguins, package = "modeldata")

  penguins <- penguins |> na.omit()

  set.seed(122)
  in_train <- sample(seq_len(nrow(penguins)), 200)
  penguins_train <- penguins[ in_train,]
  penguins_test  <- penguins[-in_train,]

  rec <- recipe(island ~ ., data = penguins_train) |>
    step_dummy(species, sex) |>
    step_normalize(all_numeric_predictors())

  set.seed(3)
  fit <- brulee_multinomial_reg(rec, data = penguins_train, epochs = 5)
  fit

  predict(fit, penguins_test) |>
    bind_cols(penguins_test) |>
    conf_mat(island, .pred_class)
}
#>            Truth
#> Prediction  Biscoe Dream Torgersen
#>   Biscoe        49     2         3
#>   Dream         11    38         6
#>   Torgersen      9     8         7
# }
```
