# checking single logicals

    Code
      brulee:::check_single_logical(variable)
    Condition
      Error:
      ! `variable` should be a single logical value, not the string "pie".

---

    Code
      brulee:::check_single_logical(variable)
    Condition
      Error:
      ! `variable` should be a single logical value, not `NA`.

---

    Code
      brulee:::check_single_logical(variable)
    Condition
      Error:
      ! `variable` should be a single logical value, not a logical vector.

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

