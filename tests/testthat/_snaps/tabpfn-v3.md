# unsupported v3 settings are rejected

    Code
      tabpfn_v3_parse_config(list(use_nan_indicators = FALSE))
    Condition
      Error:
      ! brulee doesn't support use_nan_indicators = "FALSE" for "tabpfn_v3".

# a v3 checkpoint serves only its own task

    Code
      model(fx$tensors$input.x, fx$tensors$input.y, "multiclass")
    Condition
      Error in `model()`:
      ! This TabPFN v3 checkpoint is for regression, not multiclass.

