# the tab_icl weight helpers reject unknown tasks

    Code
      tab_icl_download_weights("bogus")
    Condition
      Error:
      ! `task` must be one of "classification" or "regression", not "bogus".

---

    Code
      tab_icl_weights_available("bogus")
    Condition
      Error in `tab_icl_weights_available()`:
      ! `task` must be one of "classification" or "regression", not "bogus".

