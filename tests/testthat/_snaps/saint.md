# saint gradient clipping argument validation

    Code
      brulee_saint(y ~ ., data = df, grad_norm_clip = -1)
    Condition
      Error in `brulee_saint()`:
      ! `grad_norm_clip` must be in the range (0, Inf].

---

    Code
      brulee_saint(y ~ ., data = df, grad_value_clip = -1)
    Condition
      Error in `brulee_saint()`:
      ! `grad_value_clip` must be in the range (0, Inf].

# saint gradient clipping prevents loss overflow

    Early stopping occurred at epoch 1 due to numerical overflow of the loss function.

# saint target_token argument is validated

    Code
      brulee_saint(y ~ ., data = single_series, epochs = 2, target_token = "nope",
      verbose = FALSE, device = "cpu")
    Condition
      Error in `brulee_saint()`:
      ! `target_token` must be `TRUE` or `FALSE`, not the string "nope".

