# matrix input dispatches to the default method and errors

    Code
      brulee_chronos(matrix(rnorm(20), nrow = 10, ncol = 2), y = rnorm(10))
    Condition
      Error:
      ! `brulee_chronos()` is not defined for a 'matrix'.

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

# non-numeric model_id / cache_dir error

    Code
      brulee_chronos(ridership ~ ., data = chi, id_column = "series_id",
      timestamp_column = "date", model_id = 42)
    Condition
      Error in `brulee_chronos_bridge()`:
      ! `model_id` must be a character vector, not the number 42.

# NA values in item_id or timestamp error

    Code
      brulee_chronos(x = Chi[, "Clark_Lake", drop = FALSE], y = Chi$ridership,
      item_id = na_id, timestamp = Chi$date)
    Condition
      Error in `brulee_chronos_bridge()`:
      ! `item_id` must not contain `NA`.

---

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

