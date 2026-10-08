# checking double vectors

    Code
      check_number_decimal_vec(letters)
    Condition
      Error:
      ! `letters` should be a double vector.

---

    Code
      check_number_decimal_vec(variable)
    Condition
      Error:
      ! `variable` should not contain missing values.

---

    Code
      check_number_decimal_vec(variable)
    Condition
      Error:
      ! `variable` should be a double vector.

# checking whole number vectors

    Code
      check_number_whole_vec(variable)
    Condition
      Error:
      ! `variable` must be a whole number, not the number 0.5.

---

    Code
      check_number_whole_vec(variable)
    Condition
      Error:
      ! `variable` must be a whole number, not an integer `NA`.

# check_optimizer validates optimizer names

    Code
      brulee:::check_optimizer(x)
    Condition
      Error:
      ! `x` must be one of "SGD", "ADAMw", "Adadelta", "Adagrad", "RMSprop", and "LBFGS", not "adam".

---

    Code
      brulee:::check_optimizer(x)
    Condition
      Error:
      ! `x` must be one of "SGD", "ADAMw", "Adadelta", "Adagrad", "RMSprop", and "LBFGS", not "nope".

---

    Code
      brulee:::check_optimizer(x)
    Condition
      Error:
      ! `x` must be a single string, not the number 123.

---

    Code
      brulee:::check_optimizer(x)
    Condition
      Error:
      ! `x` must be a single string, not a character vector.

# check_classification_loss validates loss names

    Code
      brulee:::check_classification_loss(x)
    Condition
      Error:
      ! `x` must be one of "nll" and "focal", not "mse".

---

    Code
      brulee:::check_classification_loss(x)
    Condition
      Error:
      ! `x` must be a single string, not the number 123.

# check_integer validates integers with bounds

    Code
      brulee:::check_integer(x, single = TRUE, x_min = 1)
    Condition
      Error:
      ! `x` must be in the range [1, Inf].

---

    Code
      brulee:::check_integer(x, single = TRUE)
    Condition
      Error:
      ! `x` must be a whole number, not the string "a".

# check_double validates doubles with exclusive bounds

    Code
      brulee:::check_double(x, single = TRUE, x_min = 0, incl = c(FALSE, TRUE))
    Condition
      Error:
      ! `x` must be in the range (0, Inf].

---

    Code
      brulee:::check_double(x, single = TRUE, x_min = 0, x_max = 1, incl = c(TRUE,
        FALSE))
    Condition
      Error:
      ! `x` must be in the range [0, 1).

# check_outcome_varies() rejects a numeric outcome with one value

    Code
      check_outcome_varies(c(3, 3, NA, 3), call = NULL)
    Condition
      Error:
      ! The outcome has a single value (3), so there is nothing to predict.
      i A numeric outcome needs at least two distinct values.

# models with a numeric outcome reject a constant one

    Code
      brulee_linear_reg(y ~ x, data = d)
    Condition
      Error in `brulee_linear_reg()`:
      ! The outcome has a single value (3), so there is nothing to predict.
      i A numeric outcome needs at least two distinct values.

---

    Code
      brulee_mlp(y ~ x, data = d)
    Condition
      Error in `brulee_mlp()`:
      ! The outcome has a single value (3), so there is nothing to predict.
      i A numeric outcome needs at least two distinct values.

---

    Code
      brulee_resnet(y ~ x, data = d)
    Condition
      Error in `brulee_resnet()`:
      ! The outcome has a single value (3), so there is nothing to predict.
      i A numeric outcome needs at least two distinct values.

---

    Code
      brulee_rln(y ~ x, data = d)
    Condition
      Error in `brulee_rln()`:
      ! The outcome has a single value (3), so there is nothing to predict.
      i A numeric outcome needs at least two distinct values.

---

    Code
      brulee_saint(y ~ x, data = d)
    Condition
      Error in `brulee_saint()`:
      ! The outcome has a single value (3), so there is nothing to predict.
      i A numeric outcome needs at least two distinct values.

---

    Code
      brulee_auto_int(y ~ x, data = d)
    Condition
      Error in `brulee_auto_int()`:
      ! The outcome has a single value (3), so there is nothing to predict.
      i A numeric outcome needs at least two distinct values.

---

    Code
      brulee_tab_icl(y ~ x, data = d)
    Condition
      Error in `brulee_tab_icl()`:
      ! The outcome has a single value (3), so there is nothing to predict.
      i A numeric outcome needs at least two distinct values.

---

    Code
      brulee_tab_pfn(y ~ x, data = d)
    Condition
      Error in `brulee_tab_pfn()`:
      ! The outcome has a single value (3), so there is nothing to predict.
      i A numeric outcome needs at least two distinct values.

# softmax temperatures must be positive and finite

    Code
      check_softmax_temperature(temp, call = NULL)
    Condition
      Error:
      ! `temp` must be greater than 0, not 0.

---

    Code
      check_softmax_temperature(temp, call = NULL)
    Condition
      Error:
      ! `temp` must be greater than 0, not -1.

---

    Code
      check_softmax_temperature(temp, call = NULL)
    Condition
      Error:
      ! `temp` must be a number, not `Inf`.

---

    Code
      check_softmax_temperature(NULL, call = NULL)
    Condition
      Error:
      ! `NULL` must be a number, not `NULL`.

# the foundation models check the softmax temperature

    Code
      brulee_tab_icl(mpg ~ ., data = mtcars, softmax_temperature = 0)
    Condition
      Error in `brulee_tab_icl()`:
      ! `softmax_temperature` must be greater than 0, not 0.

---

    Code
      brulee_tab_pfn(mpg ~ ., data = mtcars, softmax_temperature = 0)
    Condition
      Error in `brulee_tab_pfn()`:
      ! `softmax_temperature` must be greater than 0, not 0.

