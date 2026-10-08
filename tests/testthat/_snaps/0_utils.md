# clamp_epoch() clamps an out-of-range epoch and warns

    Code
      epoch <- brulee:::clamp_epoch(8L, estimates)
    Condition
      Warning:
      The model was fit for 7 epochs; the last epoch is used instead of epoch 8.

---

    Code
      epoch <- brulee:::clamp_epoch(10L, estimates)
    Condition
      Warning:
      The model was fit for 7 epochs; the last epoch is used instead of epoch 10.

# clamp_epoch() pluralizes the epoch count

    Code
      epoch <- brulee:::clamp_epoch(5L, vector("list", 2L))
    Condition
      Warning:
      The model was fit for 1 epoch; the last epoch is used instead of epoch 5.

# brulee_subsample_rows() can't drop a class

    Code
      brulee_subsample_rows(four, 3, call = NULL)
    Condition
      Error:
      ! `training_set_limit` (3) is smaller than the number of outcome classes (4); cannot keep at least one row per class.

# brulee_foundation_data() needs one numeric or factor outcome

    Code
      brulee_foundation_data(two, call = NULL)
    Condition
      Error:
      ! The outcome must be a single column, not 2 columns.

---

    Code
      brulee_foundation_data(dates, call = NULL)
    Condition
      Error:
      ! The outcome must be a factor (classification) or numeric (regression), not a <Date> object.

# brulee_foundation_data() drops rows with a missing outcome

    Code
      res <- brulee_foundation_data(processed, call = NULL)
    Condition
      Warning:
      Removed 2 rows with a missing outcome.

---

    Code
      brulee_foundation_data(none, call = NULL)
    Condition
      Error:
      ! Every value of the outcome is missing.

