# checking double vectors

    Code
      check_number_decimal_vec(letters)
    Condition
      Error in `check_number_decimal_vec()`:
      ! `letters` should be a double vector.

---

    Code
      check_number_decimal_vec(variable)
    Condition
      Error in `check_number_decimal_vec()`:
      ! `variable` should not contain missing values.

---

    Code
      check_number_decimal_vec(variable)
    Condition
      Error in `check_number_decimal_vec()`:
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

