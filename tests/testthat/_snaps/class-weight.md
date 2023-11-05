# setting class weights

    Code
      brulee:::check_class_weights("a", lvls, cls_xtab, "fabulous")
    Condition
      Error in `brulee:::check_class_weights()`:
      ! fabulous() expected 'class_weights' to a numeric vector

---

    Code
      brulee:::check_class_weights(c(1, 6.25), lvls, cls_xtab, "fabulous")
    Condition
      Error in `brulee:::check_class_weights()`:
      ! There were 2 class weights given but 3 were expected.

---

    Code
      brulee:::check_class_weights(bad_wts, lvls, cls_xtab, "fabulous")
    Condition
      Error in `brulee:::check_class_weights()`:
      ! Names for class weights should be: 'one', 'two', 'three'

