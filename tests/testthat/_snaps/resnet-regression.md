# resnet argument validation

    Code
      brulee_resnet(x, y, hidden_units = c(5, 10), batch_norm_units = c(1, 1),
      epochs = 2)
    Condition
      Error in `check_integer()`:
      ! brulee_resnet() expected 'batch_norm_units' to be an integer on [2, Inf].

---

    Code
      brulee_resnet(x, y, hidden_units = c(5, 10), batch_norm_units = c(3, 4, 5),
      epochs = 2)
    Condition
      Error in `validate_resnet_args()`:
      ! The length of `batch_norm_units` (3) must match the length of `hidden_units` (2).

---

    Code
      brulee_resnet(x, y, hidden_units = c(5, 10), batch_norm_units = c(3, 4),
      residual_at = 5, epochs = 2)
    Condition
      Error in `validate_resnet_args()`:
      ! All values in `residual_at` must be between 1 and 2 (the number of layers).

