# tabicl_regressor_stats validates its arguments

    Code
      brulee:::tabicl_regressor_stats(loaded = NULL, x_train = NULL, y_train = NULL,
        x_test = NULL, members = NULL, output_type = "quantiles")
    Condition
      Error:
      ! `alphas` is required when "quantiles" is requested.

---

    Code
      brulee:::tabicl_regressor_stats(loaded = NULL, x_train = NULL, y_train = NULL,
        x_test = NULL, members = NULL, output_type = "bogus")
    Condition
      Error in `brulee:::tabicl_regressor_stats()`:
      ! `output_type` must be one of "mean", "variance", or "quantiles", not "bogus".

