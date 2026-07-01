# Full TabICL model: column embedding -> row interaction -> in-context learning.
# Mirrors `TabICL.forward` / `_inference_forward` in `src/tabicl/_model/tabicl.py`
# for a single forward pass (one ensemble member). Ensembling, preprocessing,
# and the user-facing fit/predict wrappers are layered on top at integration.
#
# `config` is the parsed checkpoint config (see the converter's config.json):
# embed_dim, col_*/row_*/icl_* sizes, ff_factor, max_classes, num_quantiles,
# activation, norm_first, bias_free_ln.

tabicl_model <- nn_module(
  "tabicl_model",
  initialize = function(config) {
    embed_dim <- config$embed_dim
    ff <- config$ff_factor * embed_dim
    icl_dim <- embed_dim * config$row_num_cls # CLS tokens are concatenated
    if (config$max_classes == 0) {
      out_dim <- config$num_quantiles
    } else {
      out_dim <- config$max_classes
    }
    self$max_classes <- config$max_classes

    # The checkpoint config names the activation as a string; resolve it to the
    # torch function once here so the modules take a function directly.
    act <- switch(
      config$activation,
      gelu = nnf_gelu,
      relu = nnf_relu,
      cli::cli_abort("Unknown {.field activation} {.val {config$activation}}.")
    )

    self$col_embedder <- tabicl_col_embedding(
      embed_dim = embed_dim,
      num_blocks = config$col_num_blocks,
      nhead = config$col_nhead,
      dim_feedforward = ff,
      num_inds = config$col_num_inds,
      feature_group_size = config$col_feature_group_size,
      target_aware = config$col_target_aware,
      max_classes = config$max_classes,
      reserve_cls_tokens = config$row_num_cls,
      activation = act,
      norm_first = config$norm_first,
      bias_free_ln = config$bias_free_ln,
      ssmax = config$col_ssmax
    )

    self$row_interactor <- tabicl_row_interaction(
      embed_dim = embed_dim,
      num_blocks = config$row_num_blocks,
      nhead = config$row_nhead,
      dim_feedforward = ff,
      num_cls = config$row_num_cls,
      rope_base = config$row_rope_base,
      activation = act,
      norm_first = config$norm_first,
      bias_free_ln = config$bias_free_ln
    )

    self$icl_predictor <- tabicl_icl_learning(
      max_classes = config$max_classes,
      out_dim = out_dim,
      d_model = icl_dim,
      num_blocks = config$icl_num_blocks,
      nhead = config$icl_nhead,
      dim_feedforward = icl_dim * config$ff_factor,
      activation = act,
      norm_first = config$norm_first,
      bias_free_ln = config$bias_free_ln,
      ssmax = config$icl_ssmax
    )
  },
  forward = function(
    x,
    y_train,
    return_logits = TRUE,
    softmax_temperature = 0.9
  ) {
    embeddings <- self$col_embedder(x, y_train)
    representations <- self$row_interactor(embeddings)
    self$icl_predictor(
      representations,
      y_train,
      return_logits = return_logits,
      softmax_temperature = softmax_temperature
    )
  }
)
