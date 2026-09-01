# Add `augment()` methods to brulee

## Context

brulee currently exposes only `predict()`, which returns one prediction type per call. Getting the full set of predictions for a model requires the user to know which types their model supports and to bind the results together by hand:

```r
predict(fit, test_data) |>
  bind_cols(predict(fit, test_data, type = "prob")) |>
  bind_cols(test_data)
```

That idiom appears verbatim in brulee's own test suite (`tests/testthat/test-logistic_reg-fit.R`) and in the `predict.brulee_logistic_reg()` example. Every other tidymodels package solves this with `augment()`, the `generics` verb that returns *all* predictions appropriate to the model's mode, bound onto `new_data`. brulee has no `augment` method today — a grep across `R/`, `man/`, `tests/`, `NEWS.md`, and `NAMESPACE` returns zero hits.

The goal is an `augment()` method for all ten fitted model classes that dispatches on the model's mode and returns every prediction column that mode supports, matching `parsnip::augment.model_fit` semantics so brulee behaves the way tidymodels users already expect.

## Design

### Mode → columns

brulee models have no `mode` field. Mode is inferred from the outcome prototype by the existing shared helper `check_type()` (`R/mlp-predict.R:186-221`), which reads `model$blueprint$ptypes$outcomes[[1]]` and branches on `is.factor()` / `is.numeric()`. Called with `type = NULL` it returns `"class"` for a factor outcome and `"numeric"` for a numeric one — it already *is* the mode discriminator, so `augment` wraps it rather than re-deriving the branches.

| Mode | Models | Columns added |
|---|---|---|
| regression | `linear_reg`, `rln`, and `mlp`/`resnet`/`saint`/`auto_int`/`tab_icl` with numeric outcomes | `.pred`, plus `.resid` when the outcome column is present in `new_data` |
| classification | `logistic_reg`, `multinomial_reg`, and `mlp`/`resnet`/`saint`/`auto_int`/`tab_icl` with factor outcomes | `.pred_class`, `.pred_<level>` (one per level) |
| forecasting | `chronos` | `.pred`, `.pred_quantile` (per `type`), plus `.resid` when the target column is present |

Prediction columns come first, then `new_data` — byte-for-byte the order `parsnip:::augment_regression` / `augment_classification` produce.

### Points that constrain the implementation

These were verified in a live R session against real fits; each one changes the code.

- **The generic is `function(x, ...)`**, so every method's first formal must be named `x`, not `object`. Adding `new_data` before `...` is legal S3 and is what broom and parsnip do.
- **Compute `.resid` on the prediction tibble *before* binding**, not with parsnip's `mutate(.resid = !!sym(y_nm) - .pred)` after. Doing it first avoids needing a `utils::globalVariables(".pred")` entry, avoids the `relocate()` round trip, and stays correct when `new_data` already has a `.pred` column (`bind_cols` renames it to `.pred...1`, which breaks the `mutate` form).
- **Matrix `new_data` needs a branch.** `m[["mpg"]]` on a matrix is `subscript out of bounds`, and `names()` on a matrix is `NULL` — so the outcome lookup must use `colnames()` for the membership test and `new_data[, y_nm]` for extraction. Otherwise `.resid` is silently skipped or the call errors.
- **Outcome name is `names(x$blueprint$ptypes$outcomes)`** — the same expression already used at `R/chronos2-fit.R:501`. Formula/recipe fits give the user's variable name; xy and matrix fits give hardhat's `.outcome`, which will essentially never appear in `new_data`, so those fits legitimately get no `.resid`.
- **Do not `forge()` in the worker.** Every `predict()` method already forges (`R/mlp-predict.R:60` and the eight equivalents), so a second forge would re-bake recipes two or three times per call. The worker also cannot use `forged$predictors` anyway, since it must bind onto the *user's* `new_data` in the user's column order. Delegating also means `tab_icl`'s extra `tabicl_encode_transform()` step stays invisible to the worker, and errors surface from one place with hardhat's messages verbatim.
- **Row counts are already guaranteed** by `hardhat::validate_prediction_size()` in every bridge, so `bind_cols()` can never recycle.
- **No method for `brulee_mlp_two_layer`** — `new_brulee_mlp()` sets `class = "brulee_mlp"` (`R/mlp-fit.R:689`) for both.

### The chronos alignment problem

`predict.brulee_chronos()` does **not** return rows in `new_data` order. Verified by reading `R/chronos2-predict.R`:

- Line 193: `future_list <- purrr::map(ctx$item_ids, sub_data)` groups rows by the **fit-time** series order (`ctx$item_ids` is `unique(item_id)` from the training frame, `R/chronos2-fit.R:752`).
- Line 189: within each series, rows are re-sorted by timestamp — `sub[order(sub[[timestamp_column]]), , drop = FALSE]`.
- Lines 262-292: output blocks are assembled in that same `ctx$item_ids` order and `bind_rows()`-ed.

So for a `new_data` whose rows interleave series, or whose timestamps are not already ascending, a naive `dplyr::bind_cols(predict(x, new_data), new_data)` pairs each forecast with the **wrong** input row — silently, with no error or warning. `augment.brulee_chronos()` must reconstruct that permutation and invert it.

The timestamp column is never returned (comments at lines 208-210 and 256 say so explicitly), so a join cannot disambiguate rows within a series — joining on the id alone is many-to-many. Reconstructing the index is the only correct approach.

Three further traps in the same path the method must guard:

- Line 184 uses `new_data[[id_column]] == id`, so an `NA` id injects a phantom `NA` row into *every* series subset.
- Rows whose series is absent from `ctx$item_ids` are never selected and are silently dropped, giving `nrow(preds) < nrow(new_data)`.
- Lines 132-134 coerce a **zero-column** data frame to `NULL` (`length()` on a data frame is `ncol()`), which would forecast the full horizon instead of one row per input row.

## Work Items

### Implementation

- [x] Re-export the generic in `R/aaa.R`, alongside the existing `generics::tunable` block at lines 11-13:
  ```r
  #' @importFrom generics augment
  #' @export
  generics::augment
  ```
  `generics` is already a hard dependency in `Imports`, so no DESCRIPTION change. No `s3_register()` shim either — that exists only because `ggplot2::autoplot` comes from a plotting package; roxygen's `S3method()` lines suffice, exactly as for `tunable`.
- [x] Create `R/augment.R` with two small helpers and the worker:
  - `brulee_mode(x, call)` — `switch(check_type(x, type = NULL, call = call), class = "classification", numeric = "regression")`.
  - `brulee_outcome_name(x)` — `names(x$blueprint$ptypes$outcomes)`, returning `NULL` unless length 1.
  - `brulee_augment(x, new_data, ...)` — for classification, `bind_cols()` of `predict(type = "class")` and `predict(type = "prob")`; for regression, one `predict()` call plus `.resid` when the outcome is present and numeric; then `bind_cols(res, new_data)`. Forwards `...` to `predict()` so `epoch` reaches the eight models that accept it and is harmlessly absorbed by `predict.brulee_tab_icl()`, which has `...` but no `epoch`.
- [x] Add the nine methods in `R/augment.R` following the `R/autoplot.R:65-96` shared-worker pattern — assign the worker directly (`augment.brulee_mlp <- brulee_augment`) rather than wrapping it, so errors render as ``Error in `augment()`:``. One method carries `@name brulee-augment` with the full docs; the other eight are one-liners with `@rdname brulee-augment` + `@export`:
  `augment.brulee_mlp`, `.brulee_linear_reg`, `.brulee_logistic_reg`, `.brulee_multinomial_reg`, `.brulee_resnet`, `.brulee_rln`, `.brulee_saint`, `.brulee_auto_int`, `.brulee_tab_icl`.
- [x] Extract a shared internal `chronos2_future_rows(ctx, new_data)` in `R/chronos2-predict.R` returning the per-series `new_data` row indices, and refactor line 193 to use it so `predict()` and `augment()` cannot drift apart. Use `which()` instead of `==` to fix the `NA`-id phantom-row bug. `order()` is documented stable and `which()` preserves order, so behavior is otherwise identical.
- [x] Add `augment.brulee_chronos()` (in `R/augment.R`, under the same `@rdname`) that: errors on `NULL`, non-data-frame, or zero-column `new_data`; errors on missing id/timestamp columns, `NA` ids, and rows whose series is not in `ctx$item_ids`; computes `idx <- unlist(chronos2_future_rows(ctx, new_data))`; calls `predict()`; asserts the returned id column matches `new_data[[id_column]][idx]`; drops that id column to avoid a `bind_cols` name collision; reorders with `preds[order(idx), ]`; binds; and adds `.resid` when `ctx$target_column` is present and numeric in `new_data`. (That last case is the documented backtesting workflow — the example at `R/chronos2-predict.R:49-71` passes a `test_data` that still carries `ridership`.)
- [x] Run `devtools::document()` to regenerate `NAMESPACE`, `man/brulee-augment.Rd`, and `man/reexports.Rd`.

### Tests

- [x] `tests/testthat/test-augment.R`, guarded with `skip_on_cran()`, `skip_if(!torch::torch_is_installed())`, and the relevant `skip_if_not_installed()` calls. Cases: regression with the outcome present (assert `.resid == outcome - .pred`); regression with it absent (no `.resid`); an xy/matrix fit (outcome is `.outcome`, so no `.resid` even with the full frame); binary classification; multiclass; agreement with two separate `predict()` calls; `epoch` pass-through for both modes, asserting both that it matches `predict(..., epoch =)` and that it *differs* from the default; a `brulee_tab_icl` case reusing the fixture harness from `test-tabicl-fit.R:36-88`; and a 0-row `new_data`. Use the existing `expect_equal(out[0, ], exp_str)` zero-row-prototype style from `test-linear_reg-fit.R`.
- [x] `tests/testthat/test-chronos2-predict.R` — add the alignment test. `stub_chronos_loaders(also_mock_predict_core = TRUE)` in `helper-chronos2.R` makes forecasts deterministic (`s * 100 + q * 10 + t`), so an interleaved multi-series `new_data` (like the frame at `test-chronos2-predict.R:245-250`) can assert exact per-row values — series `A` rows carry `1xx`, series `B` rows carry `2xx`. This test fails under a naive `bind_cols`, so it is the one that proves the fix.
- [x] Snapshot the new error messages with `expect_snapshot(error = TRUE)` into `_snaps/augment.md`: non-numeric outcome column, and the chronos guards (`new_data = NULL`, unknown series id, `NA` id). These are raised by brulee with `call = rlang::current_env()`, so they render cleanly as ``Error in `augment()`:``.
- [x] For any snapshot of an error that brulee *delegates* (e.g. a missing predictor column reaching `hardhat::forge()`), add `transform = \(x) sub(" at [^ ]+:[0-9]+:[0-9]+:", ":", x)`. Under `pkgload` these errors pick up a srcref (``Error in `hardhat::forge()` at brulee/R/linear_reg-predict.R:50:3:``) that disappears once the package is installed without srcrefs, so an untransformed snapshot passes under `devtools::test()` and fails under `R CMD check`. Existing snapshots avoid this only because they are all top-level calls.
- [x] Do **not** add an out-of-range `epoch` test. `last_epoch_note()` (`R/0_utils.R:417-426`) warns without clamping and the bridge then errors with `subscript out of bounds`; through `augment()` on a classification model the warning fires twice first. That is a pre-existing bug worth its own issue, not something to pin here.

### Docs

- [x] Roxygen block on `augment.brulee_mlp` with `@param x`/`@param new_data`/`@param ...`, a `@details` section describing both modes, `@return`, `@seealso [predict.brulee_mlp()]`, and an `@examplesIf !brulee:::is_cran_check()` example wrapped in `\donttest{ if (torch::torch_is_installed() && rlang::is_installed(...)) { ... } }`, following `R/linear_reg-predict.R:15-40`. Show a regression fit (with and without the outcome column, plus `epoch =`) and a classification fit.
- [x] Add a `# brulee (development version)` bullet to `NEWS.md` (the section currently has no bullets).
- [x] No `_pkgdown.yml` change — it has no `reference:` section, so the topic is picked up automatically.

## Verification

```r
devtools::document()
devtools::load_all()

# regression: .pred + .resid, both at the front
fit <- brulee_linear_reg(mpg ~ ., mtcars, epochs = 20)
augment(fit, mtcars)
augment(fit, mtcars[, -1])          # no .resid

# classification: .pred_class + one .pred_<level> per class
data(penguins, package = "modeldata")
penguins <- na.omit(penguins)
cfit <- brulee_logistic_reg(sex ~ ., penguins, epochs = 5)
augment(cfit, penguins)

# epoch reaches predict()
identical(augment(fit, mtcars, epoch = 3)$.pred,
          predict(fit, mtcars, epoch = 3)$.pred)
```

Then the full suite and a clean check:

```r
devtools::test()
devtools::check()
```

Two checks are worth doing deliberately rather than trusting green output:

1. Confirm the chronos alignment test **fails** if `augment.brulee_chronos()` is reduced to a plain `bind_cols()`. If it still passes, the test is not exercising the reordering and needs a more interleaved `new_data`.
2. Run `devtools::check()`, not just `devtools::test()` — it is the only way to catch the srcref-in-snapshot problem described above, since the two disagree by construction.

## Known follow-ups (out of scope)

- Out-of-range `epoch` warns and then errors with `subscript out of bounds` instead of clamping (`R/0_utils.R:417-426`). Pre-existing; file separately.
- Classification `augment()` runs the network twice. Negligible for the small networks, but `brulee_tab_icl` reloads its checkpoint and re-runs the full in-context ensemble on each call (`R/tabicl-predict.R:60`). The class column could be derived from the probabilities with no second forward pass — every classification path computes it as `apply(raw, 1, which.max2)` over the same array — but ship the two-call version first, since the "agrees with `predict()`" test pins the equivalence and makes the optimization safe to land later.
