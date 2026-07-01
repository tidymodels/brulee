# new_data missing the id column errors

    Code
      predict(mod, new_data = new_df, prediction_length = 3L)
    Condition
      Error in `predict()`:
      ! Column "series_id" not found in `new_data`.

# new_data missing the timestamp column errors

    Code
      predict(mod, new_data = new_df, prediction_length = 3L)
    Condition
      Error in `predict()`:
      ! Column "date" not found in `new_data`.

# new_data longer than prediction_length errors

    Code
      predict(mod, new_data = new_df, prediction_length = 3L)
    Condition
      Error in `predict()`:
      ! Series "L": `new_data` has 5 rows, more than the prediction length (3).

# an unknown type errors

    Code
      predict(mod, type = "bogus", prediction_length = 3L)
    Condition
      Error in `predict()`:
      ! `type` must be one of "all", "numeric", or "quantile", not "bogus".

