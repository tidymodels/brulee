# augment() errors when the outcome column is not numeric

    Code
      augment(reg_fit, bad)
    Condition
      Error in `augment()`:
      ! Column outcome of `new_data` should be numeric to compute residuals, not a character vector.

# augment() points `type` users at `quantile_levels` for tab_icl

    Code
      augment(fit, x_test, type = "quantile")
    Condition
      Error in `augment()`:
      ! `type` is not an argument of `augment()`.
      i Set `quantile_levels` to add the quantile and variance columns for a regression fit.

---

    Code
      augment(fit, x_test, quantile_levels = 1.5)
    Condition
      Error in `augment()`:
      ! `quantile_levels` must be in the open interval (0, 1).

# augment() rejects quantile_levels for a tab_icl classifier

    Code
      augment(fit, x_test, quantile_levels = 0.5)
    Condition
      Error in `augment()`:
      ! `quantile_levels` is only used for regression fits.

