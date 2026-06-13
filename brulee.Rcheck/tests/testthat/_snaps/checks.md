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

