# saint use_target_token argument is validated

    Code
      brulee_saint(y ~ ., data = single_series, epochs = 2, use_target_token = "nope",
      verbose = FALSE, device = "cpu")
    Condition
      Error in `brulee_saint()`:
      ! `use_target_token` must be `TRUE` or `FALSE`, not the string "nope".

