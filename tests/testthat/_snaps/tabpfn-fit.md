# fitting and predicting work through every interface

    Code
      print(fit_f)
    Message
      TabPFN v3.5 classification model
      145 training rows, 4 predictors, 2 ensemble members
      3 classes: "setosa", "versicolor", and "virginica"
      device: "cpu"

# bad arguments are rejected

    Code
      brulee_tab_pfn(mpg ~ ., data = mtcars, num_estimators = 0)
    Condition
      Error in `brulee_tab_pfn()`:
      ! `num_estimators` must be a whole number larger than or equal to 1 or `NULL`, not the number 0.

---

    Code
      brulee_tab_pfn(mpg ~ ., data = mtcars, version = "v2")
    Condition
      Error in `brulee_tab_pfn()`:
      ! `version` must be one of "v3", "v3.5", or "v3.5-fast".
      x "v2" is not a supported model version.

---

    Code
      brulee_tab_pfn(mpg ~ ., data = mtcars, ignore_pretraining_limits = "yes")
    Condition
      Error in `brulee_tab_pfn()`:
      ! `ignore_pretraining_limits` must be `TRUE` or `FALSE`, not the string "yes".

---

    Code
      brulee_tab_pfn(mpg ~ ., data = mtcars, control = list())
    Condition
      Error in `brulee_tab_pfn()`:
      ! `...` must be empty.
      x Problematic argument:
      * control = list()

---

    Code
      brulee_tab_pfn(letters)
    Condition
      Error in `brulee_tab_pfn()`:
      ! `brulee_tab_pfn()` is not defined for a <character>.

---

    Code
      brulee_tab_pfn(mpg ~ ., data = mtcars, device = "tpu")
    Condition
      Error in `brulee_tab_pfn()`:
      ! `device` must be one of "cpu", "cuda", or "mps", not "tpu".
      i Did you mean "cpu"?

# data larger than the model was trained for is refused

    Code
      tabpfn_check_limits(x, y, ic, "v3.5", "cuda", FALSE, call = NULL)
    Condition
      Error:
      ! The data is larger than the model was trained for:
      * 4 predictors (the model supports 3)
      * 60 training rows (the model supports 50)
      i Use `ignore_pretraining_limits = TRUE` to run it anyway, or `training_set_limit` to sample rows.

---

    Code
      tabpfn_check_limits(x, y, ic, "v3.5", "cuda", TRUE, call = NULL)
    Condition
      Error:
      ! The model supports at most 2 classes.

# the CPU row limit applies on the CPU only, with a warning

    Code
      tabpfn_check_limits(big, y, ic, "v3", "cpu", FALSE, call = NULL)
    Condition
      Error:
      ! The data is larger than the model was trained for:
      * 5001 training rows on the CPU (the model supports 5000)
      i Use `ignore_pretraining_limits = TRUE` to run it anyway, or `training_set_limit` to sample rows.

---

    Code
      tabpfn_check_limits(medium, y[1:2000], ic, "v3", "cpu", FALSE, call = NULL)
    Condition
      Warning:
      Running on the CPU with more than 1000 training rows may be slow.
      i A CUDA GPU is much faster; see `device`.

