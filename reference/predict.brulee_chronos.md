# Predict from a `brulee_chronos` model

Predict from a `brulee_chronos` model

## Usage

``` r
# S3 method for class 'brulee_chronos'
predict(
  object,
  new_data = NULL,
  type = "all",
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

  Optional data frame describing the future window to forecast for. It
  should contain the id and timestamp columns (when those were supplied
  at construction) plus any known future covariate values (a subset of
  the past covariates). The number of rows per series is the number of
  future time steps to return and may be at most `prediction_length`;
  supplying more is an error. When a series has fewer rows than
  `prediction_length`, the missing future covariates are treated as
  unknown and the forecast is truncated to the rows provided. If `NULL`
  (the default), the full `prediction_length` horizon is forecast from
  the context stored in `object`. The model is pretrained, so the
  historical context is always the data passed to
  [`brulee_chronos()`](https://brulee.tidymodels.org/reference/brulee_chronos.md)
  and is never overridden here.

- type:

  A single string for the type of prediction to return. The default
  `"all"` returns both the point forecast (`.pred`) and the quantile
  forecast (`.pred_quantile`). Use `"numeric"` for only `.pred` or
  `"quantile"` for only `.pred_quantile`.

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
row per forecast time step per series (up to `nrow(new_data)` rows per
series, or `prediction_length` rows when `new_data` is `NULL`). Columns
depend on `type`:

- `<id_column>`:

  The time series identifier. Omitted when the context contains a single
  series.

- `.pred`:

  Point forecast, i.e. the median of `.pred_quantile`. Returned when
  `type` is `"all"` or `"numeric"`.

- `.pred_quantile`:

  A
  [`hardhat::quantile_pred()`](https://hardhat.tidymodels.org/reference/quantile_pred.html)
  vector packing all requested quantile levels into a single column.
  Returned when `type` is `"all"` or `"quantile"`.

## Examples

``` r
pkgs <- c("recipes", "lubridate", "modeldata", "ggplot2")

if (FALSE) { # \dontrun{
if (torch::torch_is_installed() && rlang::is_installed(pkgs)) {
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

 pred_1 <- predict(mod_1, new_data = test_data)
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

 pred_2 <- predict(mod_2, new_data = test_data)

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

 pred_3 <- predict(mod_3, new_data = test_data)

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
