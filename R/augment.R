# `check_type()` (R/mlp-predict.R) maps the outcome ptype stored in the
# blueprint to the default prediction type: "class" for a factor outcome and
# "numeric" for a numeric one, erroring for anything else. That is exactly the
# mode discriminator that `augment()` needs, so reuse it rather than repeat the
# `is.factor()` / `is.numeric()` branches here.
brulee_mode <- function(x, call = rlang::caller_env()) {
  switch(
    check_type(x, type = NULL, call = call),
    class = "classification",
    numeric = "regression"
  )
}

# The blueprint's outcome ptype is a zero-row tibble whose single column is
# named for the outcome. Formula and recipe fits give the user's variable name;
# `x`/`y` and matrix fits give hardhat's ".outcome", which will not be a column
# of `new_data`, so those fits get no residuals.
brulee_outcome_name <- function(x) {
  nms <- names(x$blueprint$ptypes$outcomes)
  if (length(nms) == 1L) {
    nms
  } else {
    NULL
  }
}

# `new_data` may be a matrix, where `[[` does not take a column name and
# `names()` is NULL.
brulee_outcome_column <- function(new_data, y_nm) {
  if (is.null(y_nm) || !y_nm %in% colnames(new_data)) {
    return(NULL)
  }
  if (is.matrix(new_data)) {
    new_data[, y_nm]
  } else {
    new_data[[y_nm]]
  }
}

# The outcome column of `new_data` to measure residuals against, or NULL when
# it is not there.
brulee_resid_outcome <- function(x, new_data, call = rlang::caller_env()) {
  y <- brulee_outcome_column(new_data, brulee_outcome_name(x))
  if (!is.null(y) && !is.numeric(y)) {
    cli::cli_abort(
      "Column {.field {brulee_outcome_name(x)}} of {.arg new_data} should be
       numeric to compute residuals, not {.obj_type_friendly {y}}.",
      call = call
    )
  }
  y
}

# Residuals are taken against `.pred`, so this runs before the predictions are
# bound to `new_data`: a `.pred` column already in `new_data` would be renamed
# by `bind_cols()` and could otherwise be picked up by mistake.
brulee_add_resid <- function(res, x, new_data, call = rlang::caller_env()) {
  y <- brulee_resid_outcome(x, new_data, call = call)
  if (is.null(y)) {
    return(res)
  }
  res$.resid <- y - res$.pred
  res
}

# used for augment methods
brulee_augment <- function(x, new_data, ...) {
  call <- rlang::current_env()

  if (brulee_mode(x, call = call) == "regression") {
    res <- predict(x, new_data, type = "numeric", ...)
    res <- brulee_add_resid(res, x, new_data, call = call)
  } else {
    res <- dplyr::bind_cols(
      predict(x, new_data, type = "class", ...),
      predict(x, new_data, type = "prob", ...)
    )
  }

  dplyr::bind_cols(res, new_data)
}

## -----------------------------------------------------------------------------

#' Add model predictions to data
#'
#' `augment()` adds every prediction that a model can make for its mode to
#' `new_data`. It is a convenience wrapper around [predict()] and is a common
#' first step before computing performance metrics with the \pkg{yardstick}
#' package.
#'
#' @param x A model fit from \pkg{brulee}.
#' @param new_data A data frame or matrix of predictors. The outcome column may
#' also be present.
#' @param quantile_levels A numeric vector of quantile levels, each in the open
#' interval `(0, 1)`, sorted and unique. Only used by [brulee_tab_icl()]
#' regression fits. The default of `NULL` gives an ordinary numeric prediction;
#' supplying levels additionally returns the predictive distribution. See the
#' details.
#' @param ... Options to pass to [predict()], such as `epoch`.
#'
#' @details
#'
#' For regression models, a `.pred` column is added. When the outcome column is
#' also in `new_data`, a `.resid` column of residuals (i.e., outcome minus
#' prediction) is added as well.
#'
#' For classification models, a `.pred_class` column of hard class predictions
#' is added along with one `.pred_{level}` column of class probabilities per
#' outcome level. No residuals are computed.
#'
#' For [brulee_chronos()] models, `new_data` describes the future window to
#' forecast for and is required. The columns added are those that [predict()]
#' would return for the given `type`: `.pred`, `.pred_quantile`, or both. A
#' `.resid` column is added when the outcome column is in `new_data`, as it is
#' when the forecast is being compared against a known future.
#'
#' In all cases the new columns are added to the front of `new_data`.
#' `new_data` is validated by [predict()], so the same columns are required as
#' for prediction.
#'
#' ## Distributional predictions for `brulee_tab_icl()` regression fits
#'
#' The regression head of a [brulee_tab_icl()] model is a quantile regression
#' head, so it always has a full predictive distribution available rather than
#' just a point estimate. By default `augment()` ignores that and returns an
#' ordinary numeric prediction, exactly like every other regression model:
#' `.pred` and `.resid`.
#'
#' Setting `quantile_levels` asks for the distribution as well. The result then
#' also has a `.pred_quantile` column (a [hardhat::quantile_pred()] vector at
#' the requested levels) and a `.pred_variance` column. These are further
#' readouts of the same distribution as the mean, so they are added *alongside*
#' `.pred` rather than in place of it.
#'
#' Requesting the distribution also changes what `.resid` is measured against.
#' `.pred` is still the mean, but the residual is taken against the
#' distribution's **median**: the head can return a skewed distribution, and
#' for a skewed one the mean is pulled toward the long tail while the median
#' stays with the bulk of the mass, which makes the median the more
#' representative point estimate to measure residuals against. So `.resid` is
#' `outcome - .pred` when `quantile_levels` is `NULL` and `outcome - median`
#' when it is set. ([brulee_chronos()] defines `.pred` to be the median
#' already, so its residuals are against the median in either case.)
#'
#' The median is read off `.pred_quantile` when `quantile_levels` includes
#' `0.5`, and is otherwise requested on its own. That extra readout only
#' happens when `new_data` actually carries the outcome column, so scoring data
#' without an outcome costs nothing more.
#'
#' @return
#'
#' A tibble with the same number of rows as `new_data`, with the prediction
#' columns described above prepended to the columns of `new_data`.
#'
#' @seealso [predict.brulee_mlp()]
#'
#' @examplesIf !brulee:::is_cran_check()
#' \donttest{
#' if (torch::torch_is_installed() && rlang::is_installed(c("recipes", "modeldata"))) {
#'
#'  library(recipes)
#'
#'  # ---------------------------------------------------------------------------
#'  # regression
#'
#'  data(ames, package = "modeldata")
#'
#'  ames$Sale_Price <- log10(ames$Sale_Price)
#'
#'  set.seed(1)
#'  in_train <- sample(seq_len(nrow(ames)), 2000)
#'  ames_train <- ames[ in_train,]
#'  ames_test  <- ames[-in_train,]
#'
#'  ames_rec <-
#'   recipe(Sale_Price ~ Longitude + Latitude, data = ames_train) |>
#'     step_normalize(all_numeric_predictors())
#'
#'  set.seed(2)
#'  reg_fit <- brulee_mlp(ames_rec, data = ames_train, epochs = 50, batch_size = 32)
#'
#'  # `.pred` and `.resid` are added to the front of the data:
#'  augment(reg_fit, ames_test)
#'
#'  # Without the outcome column, there are no residuals:
#'  augment(reg_fit, ames_test[, c("Longitude", "Latitude")])
#'
#'  # Predictions from a specific epoch:
#'  augment(reg_fit, ames_test, epoch = 10)
#'
#'  # ---------------------------------------------------------------------------
#'  # classification
#'
#'  data(penguins, package = "modeldata")
#'
#'  penguins <- penguins |> na.omit()
#'
#'  set.seed(3)
#'  in_train <- sample(seq_len(nrow(penguins)), 200)
#'  penguins_train <- penguins[ in_train,]
#'  penguins_test  <- penguins[-in_train,]
#'
#'  penguins_rec <-
#'   recipe(species ~ ., data = penguins_train) |>
#'     step_dummy(all_nominal_predictors()) |>
#'     step_normalize(all_numeric_predictors())
#'
#'  set.seed(4)
#'  cls_fit <- brulee_mlp(penguins_rec, data = penguins_train, epochs = 20)
#'
#'  # `.pred_class` plus one probability column per class:
#'  augment(cls_fit, penguins_test)
#' }
#' }
#' @name brulee-augment
#' @export
augment.brulee_mlp <- brulee_augment

#' @rdname brulee-augment
#' @export
augment.brulee_linear_reg <- brulee_augment

#' @rdname brulee-augment
#' @export
augment.brulee_logistic_reg <- brulee_augment

#' @rdname brulee-augment
#' @export
augment.brulee_multinomial_reg <- brulee_augment

#' @rdname brulee-augment
#' @export
augment.brulee_resnet <- brulee_augment

#' @rdname brulee-augment
#' @export
augment.brulee_rln <- brulee_augment

#' @rdname brulee-augment
#' @export
augment.brulee_saint <- brulee_augment

#' @rdname brulee-augment
#' @export
augment.brulee_auto_int <- brulee_augment

# The median of the predictive distribution, read off `.pred_quantile` when the
# 0.5 level is already there and otherwise requested on its own. This is only
# reached when `new_data` carries the outcome column, so the extra readout costs
# nothing for the usual case of scoring data without an outcome.
tab_icl_median <- function(res, x, new_data, ...) {
  if (".pred_quantile" %in% names(res)) {
    at_median <- which(
      hardhat::extract_quantile_levels(res$.pred_quantile) == 0.5
    )
    if (length(at_median) == 1L) {
      return(as.matrix(res$.pred_quantile)[, at_median])
    }
  }
  med <- predict(
    x,
    new_data,
    type = "quantile",
    quantile_levels = 0.5,
    ...
  )
  as.matrix(med$.pred_quantile)[, 1L]
}

#' @rdname brulee-augment
#' @export
augment.brulee_tab_icl <- function(x, new_data, quantile_levels = NULL, ...) {
  call <- rlang::current_env()

  # `augment()` picks the prediction types itself, so a `type` in `...` would
  # otherwise reach `predict()` twice and fail on a duplicated formal.
  if ("type" %in% rlang::names2(rlang::enquos(...))) {
    cli::cli_abort(
      c(
        "{.arg type} is not an argument of {.fn augment}.",
        i = "Set {.arg quantile_levels} to add the quantile and variance
             columns for a regression fit."
      ),
      call = call
    )
  }

  if (brulee_mode(x, call = call) == "classification") {
    if (!is.null(quantile_levels)) {
      cli::cli_abort(
        "{.arg quantile_levels} is only used for regression fits.",
        call = call
      )
    }
    res <- dplyr::bind_cols(
      predict(x, new_data, type = "class", ...),
      predict(x, new_data, type = "prob", ...)
    )
    return(dplyr::bind_cols(res, new_data))
  }

  # Checked before any prediction runs, so a bad level fails immediately rather
  # than after the checkpoint has been reloaded and the ensemble has run.
  if (!is.null(quantile_levels)) {
    check_quantile_levels(quantile_levels, call = call)
  }

  res <- predict(x, new_data, type = "numeric", ...)

  if (is.null(quantile_levels)) {
    # No quantiles asked for, so this behaves like any other regression model.
    res <- brulee_add_resid(res, x, new_data, call = call)
    return(dplyr::bind_cols(res, new_data))
  }

  # The regression head is a quantile regression head, so the quantiles and the
  # variance are further readouts of the same predictive distribution as the
  # mean. They are added alongside `.pred` rather than in place of it.
  res <- dplyr::bind_cols(
    res,
    predict(
      x,
      new_data,
      type = "quantile",
      quantile_levels = quantile_levels,
      ...
    ),
    predict(x, new_data, type = "variance", ...)
  )

  # With the distribution in hand, `.resid` is measured against its median
  # rather than against `.pred`, which is the mean. See the details.
  y <- brulee_resid_outcome(x, new_data, call = call)
  if (!is.null(y)) {
    res$.resid <- y - tab_icl_median(res, x, new_data, ...)
    res <- dplyr::relocate(res, ".resid", .after = ".pred")
  }

  dplyr::bind_cols(res, new_data)
}

## -----------------------------------------------------------------------------

#' @rdname brulee-augment
#' @export
augment.brulee_chronos <- function(x, new_data = NULL, ...) {
  call <- rlang::current_env()

  if (is.null(new_data)) {
    cli::cli_abort(
      c(
        "{.arg new_data} is required for {.fn augment}.",
        i = "Use {.code predict(x)} to forecast the full horizon from the
             stored context."
      ),
      call = call
    )
  }
  if (!is.data.frame(new_data)) {
    cli::cli_abort(
      "{.arg new_data} should be a data frame, not {.obj_type_friendly
       {new_data}}.",
      call = call
    )
  }
  # `predict()` treats a zero-column data frame as NULL and would forecast the
  # full horizon instead of one row per row of `new_data`.
  if (ncol(new_data) == 0L) {
    cli::cli_abort(
      "{.arg new_data} should have at least one column.",
      call = call
    )
  }

  ctx <- x$context
  id_column <- ctx$id_column

  if (!isTRUE(ctx$id_synthetic)) {
    if (!id_column %in% names(new_data)) {
      cli::cli_abort(
        "Column {.val {id_column}} not found in {.arg new_data}.",
        call = call
      )
    }
    if (anyNA(new_data[[id_column]])) {
      cli::cli_abort(
        "Column {.val {id_column}} in {.arg new_data} should not contain
         missing values.",
        call = call
      )
    }
  }
  if (
    !isTRUE(ctx$timestamp_synthetic) &&
      !ctx$timestamp_column %in% names(new_data)
  ) {
    cli::cli_abort(
      "Column {.val {ctx$timestamp_column}} not found in {.arg new_data}.",
      call = call
    )
  }

  # `predict()` returns rows grouped by the series order recorded at fit time
  # and sorted by timestamp within each series, not in the row order of
  # `new_data`. This is that order.
  idx <- as.integer(unlist(chronos2_future_rows(ctx, new_data)))

  # Rows for a series the model was never given are silently dropped by
  # `predict()`; catch that here rather than returning a shorter tibble.
  unmatched <- setdiff(seq_len(nrow(new_data)), idx)
  if (length(unmatched) > 0) {
    bad <- unique(as.character(new_data[[id_column]][unmatched]))
    cli::cli_abort(
      c(
        "{length(unmatched)} row{?s} of {.arg new_data} {?does/do} not belong
         to a series that {.fn brulee_chronos} was given.",
        x = "Unknown {.val {id_column}} value{?s}: {.val {bad}}.",
        i = "The forecast context is fixed when the model is created."
      ),
      call = call
    )
  }

  preds <- predict(x, new_data = new_data, ...)

  # `predict()` prepends the id column when the context holds more than one
  # series, duplicating the one already in `new_data`. Use it to check the
  # alignment, then drop it so `bind_cols()` does not rename both copies.
  if (id_column %in% names(preds)) {
    aligned <- identical(
      as.character(preds[[id_column]]),
      as.character(new_data[[id_column]][idx])
    )
    if (!aligned) {
      cli::cli_abort(
        "Forecasts could not be aligned to {.arg new_data}.", # nocov
        call = call # nocov
      ) # nocov
    }
    preds[[id_column]] <- NULL
  }

  # `order(idx)` inverts the permutation, so row i of `preds` lines up with
  # row i of `new_data`.
  preds <- preds[order(idx), , drop = FALSE]

  y <- brulee_outcome_column(new_data, ctx$target_column)
  if (".pred" %in% names(preds) && !is.null(y) && is.numeric(y)) {
    preds$.resid <- y - preds$.pred
  }

  dplyr::bind_cols(preds, new_data)
}
