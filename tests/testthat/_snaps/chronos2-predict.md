# predict with new_data missing the id column errors

    Code
      predict(mod, new_data = bad, prediction_length = 3L)
    Condition
      Error in `chronos2_pull_column()`:
      ! Column "series_id" (from `id_column`) not found in `new_data`.

# predict with new_data containing a non-numeric target errors

    Code
      predict(mod, new_data = bad, prediction_length = 3L)
    Condition
      Error in `predict()`:
      ! The target column must be numeric.

# predict with new_data missing the timestamp column errors

    Code
      predict(mod, new_data = bad, prediction_length = 3L)
    Condition
      Error in `chronos2_pull_column()`:
      ! Column "date" (from `timestamp_column`) not found in `new_data`.

# predict errors when predictors-only new_data and future_df are both supplied

    Code
      predict(mod, new_data = future, future_df = future, prediction_length = 5L)
    Condition
      Error in `predict()`:
      ! Cannot use both a predictors-only `new_data` and `future_df`.
      i Both supply future covariate values; pass only one.

# future_df missing the id column errors

    Code
      predict(mod, future_df = future_df, prediction_length = 3L)
    Condition
      Error in `predict()`:
      ! Column "series_id" not found in `future_df`.

# future_df missing the timestamp column errors

    Code
      predict(mod, future_df = future_df, prediction_length = 3L)
    Condition
      Error in `predict()`:
      ! Column "date" not found in `future_df`.

# future_df with wrong row count per series errors

    Code
      predict(mod, future_df = future_df, prediction_length = 5L)
    Condition
      Error in `FUN()`:
      ! Series "L": `future_df` has 3 rows, expected 5.

# new_data still errors when a required column is missing

    Code
      predict(mod, new_data = bad, prediction_length = 3L)
    Condition
      Error in `hardhat::forge()`:
      ! The required column "Austin" is missing.

# future_df still errors when the id column is missing

    Code
      predict(mod, future_df = future_df, prediction_length = 3L)
    Condition
      Error in `predict()`:
      ! Column "series_id" not found in `future_df`.

# chronos2_pull_column errors when the column is missing

    Code
      brulee:::chronos2_pull_column(data.frame(a = 1:3), "missing_col", "id_column")
    Condition
      Error in `brulee:::chronos2_pull_column()`:
      ! Column "missing_col" (from `id_column`) not found in `new_data`.

