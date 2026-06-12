# setting class weights

    Code
      brulee:::check_class_weights("a", lvls, cls_xtab, "fabulous")
    Condition
      Error in `rlang::abort()`:
      ! `call` must be a call or environment, not the string "fabulous".

---

    Code
      brulee:::check_class_weights(c(1, 6.25), lvls, cls_xtab, "fabulous")
    Condition
      Error in `rlang::abort()`:
      ! `call` must be a call or environment, not the string "fabulous".

---

    Code
      brulee:::check_class_weights(bad_wts, lvls, cls_xtab, "fabulous")
    Condition
      Error in `rlang::abort()`:
      ! `call` must be a call or environment, not the string "fabulous".

