# predict() rejects bad types and quantile levels

    Code
      predict(fit, x_test, type = "prob")
    Condition
      Error in `predict()`:
      ! Outcome is numeric and the prediction type is "prob".

---

    Code
      predict(fit, x_test, type = "bogus")
    Condition
      Error in `predict()`:
      ! `type` must be one of "numeric", "prob", "class", "quantile", or "variance", not "bogus".

---

    Code
      predict(fit, x_test, type = "quantile", quantile_levels = c(0.5, 0.1))
    Condition
      Error in `predict()`:
      ! `quantile_levels` must be sorted in increasing order.

---

    Code
      predict(fit, x_test, type = "quantile", quantile_levels = c(0, 0.5))
    Condition
      Error in `predict()`:
      ! `quantile_levels` must be in the open interval (0, 1).

---

    Code
      predict(fit, x_test, type = "quantile", quantile_levels = c(0.5, 1))
    Condition
      Error in `predict()`:
      ! `quantile_levels` must be in the open interval (0, 1).

---

    Code
      predict(fit, x_test, type = "quantile", quantile_levels = c(0.1, 0.1))
    Condition
      Error in `predict()`:
      ! `quantile_levels` must be unique.

---

    Code
      predict(fit, x_test, type = "quantile", quantile_levels = numeric(0))
    Condition
      Error in `predict()`:
      ! `quantile_levels` must be a non-empty numeric vector.

---

    Code
      predict(fit, x_test, type = "quantile", quantile_levels = "a")
    Condition
      Error in `predict()`:
      ! `quantile_levels` must be a non-empty numeric vector.

---

    Code
      predict(fit, x_test, type = "quantile", quantile_levels = c(0.1, NA))
    Condition
      Error in `predict()`:
      ! `quantile_levels` cannot contain missing values.

# a classification fit rejects the regression-only types

    Code
      predict(fit, x_test, type = "quantile")
    Condition
      Error in `predict()`:
      ! Outcome is factor and the prediction type is "quantile".

---

    Code
      predict(fit, x_test, type = "variance")
    Condition
      Error in `predict()`:
      ! Outcome is factor and the prediction type is "variance".

