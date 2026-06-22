# saint target_token argument is validated

    Code
      brulee_saint(y ~ ., data = single_series, epochs = 2, target_token = "nope",
      verbose = FALSE, device = "cpu")
    Condition
      Error in `brulee_saint()`:
      ! `target_token` must be `TRUE` or `FALSE`, not the string "nope".

