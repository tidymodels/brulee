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

# augment() rejects new_data it cannot align

    Code
      augment(mod)
    Condition
      Error in `augment()`:
      ! `new_data` is required for `augment()`.
      i Use `predict(x)` to forecast the full horizon from the stored context.

---

    Code
      augment(mod, new_data = "nope")
    Condition
      Error in `augment()`:
      ! `new_data` should be a data frame, not a string.

---

    Code
      augment(mod, new_data = unknown)
    Condition
      Error in `augment()`:
      ! 3 rows of `new_data` do not belong to a series that `brulee_chronos()` was given.
      x Unknown "series_id" value: "not_a_series".
      i The forecast context is fixed when the model is created.

---

    Code
      augment(mod, new_data = missing_id)
    Condition
      Error in `augment()`:
      ! Column "series_id" in `new_data` should not contain missing values.

