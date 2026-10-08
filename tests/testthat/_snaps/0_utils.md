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

