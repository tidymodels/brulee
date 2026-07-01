# load_tabicl_weights errors on unmatched and missing keys

    Code
      brulee:::load_tabicl_weights(model, extra)
    Condition
      Error in `brulee:::load_tabicl_weights()`:
      ! Checkpoint has 1 parameter with no model slot.
      x First unmatched: "col_embedder.bogus.weight"

---

    Code
      brulee:::load_tabicl_weights(model, short)
    Condition
      Error in `brulee:::load_tabicl_weights()`:
      ! Checkpoint is missing 1 expected parameter.
      x First missing: "col_embedder.in_linear.bias"

# tabicl_parse_config errors when fields are missing

    Code
      brulee:::tabicl_parse_config(path)
    Condition
      Error in `brulee:::tabicl_parse_config()`:
      ! Config is missing required fields: "max_classes", "num_quantiles", "col_num_blocks", "col_nhead", "col_num_inds", "col_feature_group_size", "col_target_aware", "col_ssmax", "row_num_blocks", "row_nhead", "row_num_cls", "row_rope_base", "icl_num_blocks", "icl_nhead", "icl_ssmax", "ff_factor", "activation", "norm_first", and "bias_free_ln".

