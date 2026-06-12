# default method errors for unsupported types

    Code
      brulee_chronos(42L)
    Condition
      Error:
      ! `brulee_chronos()` is not defined for a 'integer'.

---

    Code
      brulee_chronos("oops")
    Condition
      Error:
      ! `brulee_chronos()` is not defined for a 'character'.

# matrix input dispatches to the default method and errors

    Code
      brulee_chronos(matrix(rnorm(20), nrow = 10, ncol = 2), y = rnorm(10))
    Condition
      Error:
      ! `brulee_chronos()` is not defined for a 'matrix'.

# non-numeric predictors are rejected when mold doesn't encode them

    Code
      brulee_chronos(rec, data = Chi)
    Condition
      Error in `brulee_chronos_bridge()`:
      ! All past covariates must be numeric. Non-numeric column: "cat".
      i Use a recipe (e.g. `recipes::step_dummy()`) to encode them as numeric.

# length mismatch between y and item_id errors

    Code
      brulee_chronos(x = Chi[, "Clark_Lake", drop = FALSE], y = Chi$ridership,
      item_id = Chi$series_id[-1], timestamp = Chi$date, id_column = "series_id",
      timestamp_column = "date")
    Condition
      Error in `brulee_chronos_bridge()`:
      ! `item_id` has length 199 but `y` has length 200.

# length mismatch between y and timestamp errors

    Code
      brulee_chronos(x = Chi[, "Clark_Lake", drop = FALSE], y = Chi$ridership,
      item_id = Chi$series_id, timestamp = Chi$date[-1], id_column = "series_id",
      timestamp_column = "date")
    Condition
      Error in `brulee_chronos_bridge()`:
      ! `timestamp` has length 199 but `y` has length 200.

# quantile_levels outside (0,1) errors

    Code
      brulee_chronos(ridership ~ ., data = chi, id_column = "series_id",
      timestamp_column = "date", quantile_levels = c(0, 0.5))
    Condition
      Error in `brulee_chronos_bridge()`:
      ! `quantile_levels` must be in the open interval (0, 1).

# negative prediction_length errors

    Code
      brulee_chronos(ridership ~ ., data = chi, id_column = "series_id",
      timestamp_column = "date", prediction_length = -1)
    Condition
      Error in `brulee_chronos_bridge()`:
      ! `prediction_length` must be in the range [1, Inf].

# quantile_levels not available in model errors

    Code
      brulee_chronos(ridership ~ ., data = chi, id_column = "series_id",
      timestamp_column = "date", quantile_levels = 0.123)
    Condition
      Error in `brulee_chronos_bridge()`:
      ! Requested quantile levels not available in model: 0.123.
      i Available: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, and 0.9

# formula errors when id_column tidyselect picks more than one column

    Code
      brulee_chronos(ridership ~ Clark_Lake, data = Chi, id_column = c(series_id,
        date))
    Condition
      Error in `chronos2_resolve_column()`:
      ! `id_column` must select exactly one column, got 2.

# formula errors when id_column tidyselect picks no column

    Code
      brulee_chronos(ridership ~ Clark_Lake, data = Chi, id_column = tidyselect::starts_with(
        "nope_"))
    Condition
      Error in `chronos2_resolve_column()`:
      ! `id_column` must select exactly one column, got 0.

# formula errors when string id_column is not a column in data

    Code
      brulee_chronos(ridership ~ Clark_Lake, data = Chi, id_column = "nope")
    Condition
      Error in `chronos2_resolve_column()`:
      ! Column "nope" (from `id_column`) not found in `data`.

# prediction_length above the model maximum errors

    Code
      brulee_chronos(ridership ~ ., data = chi, id_column = "series_id",
      timestamp_column = "date", prediction_length = 2000L)
    Condition
      Error in `brulee_chronos_bridge()`:
      ! `prediction_length` (2000) exceeds model maximum (1024).

# non-numeric model_id errors

    Code
      brulee_chronos(ridership ~ ., data = chi, id_column = "series_id",
      timestamp_column = "date", model_id = 42)
    Condition
      Error in `brulee_chronos_bridge()`:
      ! `model_id` must be a single string, not the number 42.

# NA values in item_id error

    Code
      brulee_chronos(x = Chi[, "Clark_Lake", drop = FALSE], y = Chi$ridership,
      item_id = na_id, timestamp = Chi$date)
    Condition
      Error in `brulee_chronos_bridge()`:
      ! `item_id` must not contain `NA`.

# NA values in timestamp error

    Code
      brulee_chronos(x = Chi[, "Clark_Lake", drop = FALSE], y = Chi$ridership,
      item_id = Chi$series_id, timestamp = na_ts)
    Condition
      Error in `brulee_chronos_bridge()`:
      ! `timestamp` must not contain `NA`.

# non-numeric target errors

    Code
      brulee_chronos(x = Chi[, "Clark_Lake", drop = FALSE], y = as.character(Chi$
        ridership), item_id = Chi$series_id, timestamp = Chi$date)
    Condition
      Error in `standardize()`:
      ! No `standardize()` method provided for a character vector.

# quantile_levels of length zero errors

    Code
      brulee_chronos(ridership ~ ., data = chi, id_column = "series_id",
      timestamp_column = "date", quantile_levels = numeric(0))
    Condition
      Error in `brulee_chronos_bridge()`:
      ! `quantile_levels` must be a non-empty numeric vector.

# recipe with more than one id role errors

    Code
      brulee_chronos(rec, data = Chi)
    Condition
      Error in `brulee_chronos()`:
      ! The recipe must have at most one variable with role "id".

# recipe with more than one time role errors

    Code
      brulee_chronos(rec, data = Chi)
    Condition
      Error in `brulee_chronos()`:
      ! The recipe must have at most one variable with role "time".

# bridge errors when torch is not installed

    Code
      brulee_chronos(ridership ~ ., data = chi, id_column = "series_id",
      timestamp_column = "date")
    Condition
      Error in `brulee_chronos_bridge()`:
      ! The torch backend has not been installed; use `torch::install_torch()`.

# non-character device argument errors

    Code
      brulee_chronos(ridership ~ ., data = chi, id_column = "series_id",
      timestamp_column = "date", device = 123)
    Condition
      Error in `brulee_chronos_bridge()`:
      ! `device` must be a single string, not the number 123.

# quantile_levels at 1.0 errors

    Code
      brulee_chronos(ridership ~ ., data = chi, id_column = "series_id",
      timestamp_column = "date", quantile_levels = c(0.5, 1))
    Condition
      Error in `brulee_chronos_bridge()`:
      ! `quantile_levels` must be in the open interval (0, 1).

# non-numeric quantile_levels errors

    Code
      brulee_chronos(ridership ~ ., data = chi, id_column = "series_id",
      timestamp_column = "date", quantile_levels = "half")
    Condition
      Error in `brulee_chronos_bridge()`:
      ! `quantile_levels` must be a non-empty numeric vector.

# formula with multiple LHS variables errors

    Code
      brulee_chronos(cbind(ridership, Clark_Lake) ~ Austin, data = Chi, id_column = "series_id",
      timestamp_column = "date")
    Condition
      Error in `brulee_chronos()`:
      ! `formula` must have exactly one variable on the left-hand side.

# formula with target not in data errors (no-covariate path)

    Code
      brulee_chronos(not_a_column ~ 1, data = Chi, id_column = "series_id",
      timestamp_column = "date")
    Condition
      Error in `brulee_chronos()`:
      ! Target column "not_a_column" not found in `data`.

# chronos2_resolve_column errors for string not in data

    Code
      brulee:::chronos2_resolve_column(quo, df, "test_arg")
    Condition
      Error in `brulee:::chronos2_resolve_column()`:
      ! Column "missing_col" (from `test_arg`) not found in `data`.

# chronos2_resolve_column errors on invalid tidyselect expression

    Code
      brulee:::chronos2_resolve_column(quo, df, "test_arg")
    Condition
      Error:
      ! Couldn't resolve `test_arg`: Can't select columns that don't exist. x Column `not_a_col` doesn't exist.

# non-character revision argument errors

    Code
      brulee_chronos(ridership ~ ., data = chi, id_column = "series_id",
      timestamp_column = "date", revision = 123)
    Condition
      Error in `brulee_chronos_bridge()`:
      ! `revision` must be a single string, not the number 123.

# non-character cache_dir argument errors

    Code
      brulee_chronos(ridership ~ ., data = chi, id_column = "series_id",
      timestamp_column = "date", cache_dir = 999)
    Condition
      Error in `brulee_chronos_bridge()`:
      ! `cache_dir` must be a single string, not the number 999.

