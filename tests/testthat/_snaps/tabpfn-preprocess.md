# infinite and unsupported predictors are rejected

    Code
      tabpfn_encode(encoder, data.frame(v = Inf))
    Condition
      Error:
      ! Column v has infinite values.

---

    Code
      tabpfn_fit_encoder(data.frame(d = Sys.Date()))
    Condition
      Error:
      ! Column d has an unsupported type (<Date>).

