# Chronos-2 pretrained forecasting model

`brulee_chronos()` loads a pretrained Chronos-2 time series forecasting
quantile regression model from HuggingFace and ingests historical
("context") data so that the returned object is ready to forecast.
Unlike other brulee models, no training is performed; the network has
fixed pretrained weights.

## Usage

``` r
brulee_chronos(x, ...)

# Default S3 method
brulee_chronos(x, ...)

# S3 method for class 'data.frame'
brulee_chronos(
  x,
  y,
  item_id = NULL,
  timestamp = NULL,
  id_column = ".id_column",
  timestamp_column = ".timestamp_column",
  model_id = "amazon/chronos-2",
  revision = chronos2_default_revision(),
  prediction_length = NULL,
  quantile_levels = (1:9)/10,
  device = NULL,
  cache_dir = file.path(Sys.getenv("HOME"), ".cache", "chronos-r"),
  ...
)

# S3 method for class 'formula'
brulee_chronos(
  formula,
  data,
  id_column = NULL,
  timestamp_column = NULL,
  model_id = "amazon/chronos-2",
  revision = chronos2_default_revision(),
  prediction_length = NULL,
  quantile_levels = (1:9)/10,
  device = NULL,
  cache_dir = file.path(Sys.getenv("HOME"), ".cache", "chronos-r"),
  ...
)

# S3 method for class 'recipe'
brulee_chronos(
  x,
  data,
  model_id = "amazon/chronos-2",
  revision = chronos2_default_revision(),
  prediction_length = NULL,
  quantile_levels = (1:9)/10,
  device = NULL,
  cache_dir = file.path(Sys.getenv("HOME"), ".cache", "chronos-r"),
  ...
)
```

## Arguments

- x:

  Depending on the context:

  - A **data frame** of past covariates.

  - A **recipe** specifying preprocessing and roles for `target`, `id`,
    and `time` columns.

  Pass an empty data frame when there are no covariates.

- ...:

  Currently unused.

- y:

  A numeric vector of target values, of length `nrow(x)`.

- item_id:

  Optional vector of time series identifiers, of length `nrow(x)`.
  Default: `NULL`, which treats all rows as a single series.

- timestamp:

  Optional vector of timestamps (Date, POSIXct, or numeric), of length
  `nrow(x)`. Default: `NULL`, which uses row order within each series.

- id_column:

  For the formula method, a tidyselect expression selecting the id
  column in `data` (e.g. `c(series_id)`, `series_id`, or `"series_id"`).
  For the data frame `x_y` method, a character string is used as the
  output label only (the actual id values come from `item_id`). Default:
  `NULL` for the formula method and `".id_column"` for the `x_y` method.
  When omitted, all rows are treated as one series. For the recipe
  method, identify the id column with
  `recipes::update_role(..., new_role = "id")`.

- timestamp_column:

  For the formula method, a tidyselect expression selecting the
  timestamp column in `data`. For the data frame `x_y` method, a
  character string is used as the output label only. Default: `NULL` for
  the formula method and `".timestamp_column"` for the `x_y` method.
  When omitted, row order is used as the time order. For the recipe
  method, identify the timestamp column with
  `recipes::update_role(..., new_role = "time")`.

- model_id:

  A character string identifying the HuggingFace model repository to
  download. Default: `"amazon/chronos-2"` (120M parameters).

- revision:

  A character string identifying which version of the weights to load.
  May be a 40-character commit SHA, a tag, or a branch name on the
  HuggingFace repo (e.g. `"main"`). Default: a commit SHA pinned by
  brulee so the weights cannot change without you opting in. The
  resolved SHA is recorded on the returned object as `object$revision`
  and printed by [`print()`](https://rdrr.io/r/base/print.html).

- prediction_length:

  An integer for the number of future time steps to forecast. Default:
  `NULL` (uses the model maximum). Must not exceed the model maximum.
  Can be overridden at
  [`predict()`](https://rdrr.io/r/stats/predict.html) time.

- quantile_levels:

  A numeric vector of quantile levels to produce in predictions. Must be
  a subset of the model's trained quantiles. Default: `(1:9) / 10`. Can
  be overridden at [`predict()`](https://rdrr.io/r/stats/predict.html)
  time.

- device:

  A character string for the computation device: `"cpu"`, `"cuda"`, or
  `"mps"`. Default: `NULL` (auto-detects best available).

- cache_dir:

  Path to a directory for caching downloaded model files. Default:
  `"~/.cache/chronos-r"`.

- formula:

  A formula of the form `target ~ cov1 + cov2`. Use `target ~ .` when
  there are no covariates. The id and timestamp columns (if named) are
  dropped before the formula is evaluated.

- data:

  When a **recipe** or **formula** is used, `data` is the training set
  with columns for the id, timestamp, target, and any covariates.

## Value

A `brulee_chronos` object with elements:

- `model`: The torch `nn_module` (in eval mode, on the specified
  device).

- `config`: Parsed model configuration list.

- `device`: The torch device in use.

- `prediction_length`: Validated prediction length.

- `quantile_levels`: Validated quantile levels.

- `model_id`: The HuggingFace repository the weights came from.

- `revision`: The 40-character commit SHA of the weights actually
  loaded.

- `blueprint`: The hardhat blueprint for processing new data.

- `context`: A list with the per-series target, covariates, timestamps,
  and column-name metadata that
  [`predict()`](https://rdrr.io/r/stats/predict.html) uses by default.

## Details

### Computing Requirements

This model can be used with or without a graphics processing unit (GPU).
However, it may be computationally slow when used with a CPU (and no
GPU).

### Model Weight File Download

Keep in mind that, on the first usage of the fitting function, the
package will attempt to download the model weights file. This file can
require about 500MB and is locally cached.

### Interface Overview

Every Chronos-2 forecast needs at most four pieces of information about
the historical (context) data:

- A **target** column with the values to forecast (always required),

- An optional **id** column that distinguishes one time series from
  another (e.g. a city, store, or sensor); when omitted, all rows are
  treated as a single series,

- An optional **timestamp** column with the time index of each
  observation; when omitted, rows are read in their existing order,

- Any number of **past covariates**, additional numeric columns measured
  alongside the target.

`brulee_chronos()` is a generic with three interfaces for supplying that
information; this intended to add flexibility in how you declare the
model as well as what data are given as inputs. All three produce an
object that behaves the same way at predict time.

To contrast these approaches, consider the `Chicago` data contained in
the modeldata package. The goal is to predict daily train `ridership`.
There is a `date` column, as well as a set of 14-day lagged ridership
data from our station of interest and from others in the Chicago system.

You could use Chronos in the simplest way by just passing in the column
containing past ridership values. It assumes that there are no gaps in
the data and that the data are arranged/sorted in the proper order (past
to present). The simplest interfaces to use in this case are the formula
and matrix ones.

We could add the `date` column, but this is primarily used to label the
data. Here, we would want the formula or recipe interface.

In these data, only one station's ridership is modeled. Suppose we did
this for all stations. In that case, we would *stack* the ridership data
and use the `id` argument to specify which station corresponds to each
row. In this implementation, that is equivalent to running the function
separately for each station; it is just a simpler interface with some
small computational gains.

If we wanted to use covariates in our model, such as lagged ridership
data, we can do so with the formula or recipe interfaces (see below).

### Formula interface

Use a formula when your data is a single tidy data frame and you want to
name the covariates inline. The `id_column` and `timestamp_column`
arguments use tidyselect, so bare column names,
[`c()`](https://rdrr.io/r/base/c.html) selections, and character strings
all work:

    brulee_chronos(target ~ cov1 + cov2, data = df,
                   id_column = c(series_id), timestamp_column = c(date))

    # bare names also work
    brulee_chronos(target ~ cov1 + cov2, data = df,
                   id_column = series_id, timestamp_column = date)

    # character strings still work for back compatibility
    brulee_chronos(target ~ cov1 + cov2, data = df,
                   id_column = "series_id", timestamp_column = "date")

If you have no covariates, use `target ~ .`. The id and timestamp
columns are excluded automatically. Categorical covariates on the right
hand side are converted to numeric dummy variables (just like
[`lm()`](https://rdrr.io/r/stats/lm.html)).

If you have a single series and no useful timestamp, you can omit both
columns entirely:

    brulee_chronos(target ~ ., data = df_single_series)

### Recipe interface

Use a
[`recipes::recipe()`](https://recipes.tidymodels.org/reference/recipe.html)
when you want to apply preprocessing steps (e.g. normalizing or encoding
columns) before the data reaches the model. With the recipe interface,
the id and timestamp columns are identified by their **role**, not by
name:

    rec <- recipe(target ~ ., data = df) |>
      update_role(item_id,   new_role = "id") |>
      update_role(timestamp, new_role = "time") |>
      step_normalize(all_numeric_predictors())

    brulee_chronos(rec, data = df)

Both the `id` and `time` roles are optional. If neither role is set,
`brulee_chronos()` treats the recipe data as a single series in row
order. All non numeric covariates must be encoded numerically by the
recipe (e.g. with
[`recipes::step_dummy()`](https://recipes.tidymodels.org/reference/step_dummy.html)).

### Data-frame (`x` and `y`) interface

Use the `x_y` interface when you already have your covariates and target
separated. `x` is a data frame of past covariates (zero columns is
allowed when there are no covariates), `y` is the numeric target vector,
and `item_id` / `timestamp` are optional vectors of length `nrow(x)`:

    brulee_chronos(x = df[, c("cov1", "cov2")], y = df$target,
                   item_id = df$item_id, timestamp = df$timestamp)

    # single series, no timestamp:
    brulee_chronos(x = df[, c("cov1", "cov2")], y = df$target)

### Multiple time series

All three interfaces support multiple series in one call. Stack the
series end to end in a single long format data frame and let the id
column distinguish them. `brulee_chronos()` sorts each series by
timestamp before forecasting. When you omit the id column, every row is
treated as part of one series called `"default"`.

### Pre-sorted input

When you omit the timestamp, `brulee_chronos()` uses each series' row
order as its time order. Pre-sort each series before calling
`brulee_chronos()` if you take this shortcut.

### What happens at [`predict()`](https://rdrr.io/r/stats/predict.html) time

By default,
[`predict.brulee_chronos()`](https://brulee.tidymodels.org/dev/reference/predict.brulee_chronos.md)
forecasts from the context data that was supplied at construction. The
use of `new_data` should be determined by how the `brulee_chronos()`
call passed the data. For example, if no covariates were originally
given to the model, there is no need to pass in values when calling
[`predict()`](https://rdrr.io/r/stats/predict.html) and so on. Pass
`future_df` to supply known future values of any covariate (e.g.,
holiday flags, planned promotions). To forecast a **different** series
with the same schema, pass it as `new_data`. It will be processed
through the same blueprint as the original context.

## References

Ansari, A. F., Shchur, O., Küken, J., Auer, A., Han, B., Mercado, P.,
... & Bohlke-Schneider, M. (2025). "Chronos-2: From univariate to
universal forecasting." *arXiv preprint arXiv:2510.15821*.

Ansari, A. F., Shchur, O., Küken, J., Auer, A., Han, B., Mercado, P.,
... & Bohlke-Schneider, M. (2026). "A foundation model for multivariate
time series forecasting.", https://doi.org/10.21203/rs.3.rs-9096522/v1

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

 # ------------------------------------------------------------------------------
 # Some covariates via the formula method

mod_2 <-
  brulee_chronos(
   ridership ~ Clark_Lake + Belmont + Harlem + Monroe,
   data = prior_data,
   timestamp_column = c(date),
   prediction_length = 14)

 # ------------------------------------------------------------------------------
 # Covariates using recipes

 rec <-
  recipe(ridership ~ ., data = prior_data) |>
  update_role(date, new_role = "time")

 mod_3 <- brulee_chronos(rec, data = prior_data, prediction_length = 14)
}
} # }
```
