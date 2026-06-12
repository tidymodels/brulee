# Predict from a `brulee_chronos` model

Predict from a `brulee_chronos` model

## Usage

``` r
# S3 method for class 'brulee_chronos'
predict(
  object,
  new_data = NULL,
  future_df = NULL,
  prediction_length = NULL,
  quantile_levels = NULL,
  ...
)
```

## Arguments

- object:

  A `brulee_chronos` object returned by
  [`brulee_chronos()`](https://brulee.tidymodels.org/reference/brulee_chronos.md).

- new_data:

  Optional data frame in the same long format as the data used to build
  `object`. It should contain the target and covariate columns named in
  `object`, plus the id and timestamp columns when those were supplied
  at construction. (If the model was built without an id column, every
  row of `new_data` is treated as part of the same single series;
  similarly, if the model was built without a timestamp column, row
  order is used as the time order.) If `NULL` (the default), the context
  stored in `object` is used.

- future_df:

  Optional data frame with future covariate values. Must contain the id
  and timestamp columns (when present in the original model) plus any
  covariate columns to provide for the future window (a subset of the
  past covariates). Each series must have exactly `prediction_length`
  rows.

- prediction_length:

  Number of future time steps to forecast. Defaults to the value stored
  in `object`.

- quantile_levels:

  Numeric vector of quantile levels. Defaults to the value stored in
  `object`.

- ...:

  Not used.

## Value

A [tibble](https://tibble.tidyverse.org/reference/tibble.html) with one
row per future time step per series, in the same order as the rows of
`new_data` (or the stored context). Columns:

- `<id_column>`:

  The time series identifier. Omitted when the context contains a single
  series.

- `.pred`:

  Point forecast, i.e. the median of `.pred_quantile`.

- `.pred_quantile`:

  A
  [`hardhat::quantile_pred()`](https://hardhat.tidymodels.org/reference/quantile_pred.html)
  vector packing all requested quantile levels into a single column.

## Examples

``` r
pkgs <- c("recipes", "lubridate", "modeldata", "ggplot2")

if (FALSE) { # \dontrun{
if (torch::torch_is_installed() & rlang::is_installed(pkgs)) {
 library(dplyr)
 library(ggplot2)

 n <- nrow(modeldata::Chicago)

 prior_data <- modeldata::Chicago[-((n-13):n),]
 test_data <-
  modeldata::Chicago[(n-13):n,] |>
  mutate(day = lubridate::wday(date, label = TRUE))

 # ------------------------------------------------------------------------------
 # Simple, no covariate model

 mod_1 <-
  brulee_chronos(
   ridership ~ 1,
   data = prior_data,
   # Removing `timestamp_column` does not affect the fit
   timestamp_column = c(date),
   prediction_length = 14)

 pred_1 <- predict(mod_1, test_data)
 pred_1

 pred_1 |>
  bind_cols(test_data) |>
  ggplot(aes(date)) +
  geom_point(aes(y = ridership, col = day)) +
  geom_line(aes(y = .pred)) +
  labs(title = "No covariates: Meh") +
  theme_bw()

 # ------------------------------------------------------------------------------
 # Some covariates via the formula method

mod_2 <-
  brulee_chronos(
   ridership ~ Clark_Lake + Belmont + Harlem + Monroe,
   data = prior_data,
   timestamp_column = c(date),
   prediction_length = 14)

 pred_2 <- predict(mod_2, future_df = test_data)

 pred_2 |>
  bind_cols(test_data) |>
  ggplot(aes(date)) +
  geom_point(aes(y = ridership, col = day)) +
  geom_line(aes(y = .pred)) +
  labs(title = "Four covariates: Pretty good") +
  theme_bw()

 # ------------------------------------------------------------------------------
 # Covariates using recipes

 rec <-
  recipe(ridership ~ ., data = prior_data) |>
  update_role(date, new_role = "time")

 mod_3 <- brulee_chronos(rec, data = prior_data, prediction_length = 14)

 pred_3 <- predict(mod_3, future_df = test_data)

 pred_3 |>
  bind_cols(test_data) |>
  ggplot(aes(date)) +
  geom_point(aes(y = ridership, col = day)) +
  geom_line(aes(y = .pred)) +
  labs(title = "All covariates: Better Saturdays") +
  theme_bw()
}
} # }
```
