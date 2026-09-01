# check_type defaults still reject distributional types

    Code
      brulee:::check_type(num_model, "quantile")
    Condition
      Error:
      ! `type` must be one of "numeric", "prob", or "class", not "quantile".

---

    Code
      brulee:::check_type(num_model, "variance")
    Condition
      Error:
      ! `type` must be one of "numeric", "prob", or "class", not "variance".

---

    Code
      brulee:::check_type(num_model, "prob")
    Condition
      Error:
      ! Outcome is numeric and the prediction type is "prob".

---

    Code
      brulee:::check_type(fct_model, "numeric")
    Condition
      Error:
      ! Outcome is factor and the prediction type is "numeric".

# a widened numeric_types still rejects bad input

    Code
      brulee:::check_type(fct_model, "quantile", numeric_types = wide)
    Condition
      Error:
      ! Outcome is factor and the prediction type is "quantile".

---

    Code
      brulee:::check_type(num_model, "bogus", numeric_types = wide)
    Condition
      Error:
      ! `type` must be one of "numeric", "prob", "class", "quantile", or "variance", not "bogus".

