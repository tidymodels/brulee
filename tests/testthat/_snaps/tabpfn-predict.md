# regression predictions have the tidymodels format

    Code
      predict(fit, mtcars[1:4, ], type = "quantile", quantile_levels = 2)
    Condition
      Error in `predict()`:
      ! `quantile_levels` must be in the open interval (0, 1).

---

    Code
      predict(fit, mtcars[1:4, ], type = "prob")
    Condition
      Error in `predict()`:
      ! Outcome is numeric and the prediction type is "prob".

