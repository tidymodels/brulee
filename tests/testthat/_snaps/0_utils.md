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

