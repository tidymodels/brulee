# checking double vectors

    Code
      brulee:::check_number_decimal_vec(letters)
    Condition
      Error:
      ! `letters` must be a number, not the string "a".

---

    Code
      brulee:::check_number_decimal_vec(variable)
    Condition
      Error:
      ! `variable` must be a number, not a numeric `NA`.

# checking whole number vectors

    Code
      brulee:::check_number_whole_vec(variable)
    Condition
      Error:
      ! `variable` must be a whole number, not the number 0.5.

---

    Code
      brulee:::check_number_whole_vec(variable)
    Condition
      Error:
      ! `variable` must be a whole number, not an integer `NA`.

