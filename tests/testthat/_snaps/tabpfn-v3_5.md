# v3.5 config parsing is strict

    Code
      tabpfn_v3_5_parse_config(c(config, list(new_setting = 1)))
    Condition
      Error:
      ! The checkpoint config has 1 setting that brulee doesn't know for "tabpfn_v3_5": "new_setting".
      i The checkpoint may be newer than this version of brulee.

---

    Code
      tabpfn_v3_5_parse_config(list(use_rope = FALSE))
    Condition
      Error:
      ! The checkpoint config sets use_rope to "FALSE", but brulee only supports "TRUE" for "tabpfn_v3_5".
      i The checkpoint may be newer than this version of brulee.

