# Constant predictors after `training_set_limit` subsampling (#145: TabICL + TabPFN)

## Overview

Issue #145: `predict()` on a `brulee_tab_icl()` fit fails with `subscript out of bounds` when a predictor is constant in the stored training rows. `training_set_limit` makes this likely because a rare indicator drops out of the sample. `predict.brulee_tab_icl()` built the ensemble's feature permutations from `ncol(object$train_x)`, but the ensemble engine first drops constant columns (`tabicl_unique_filter()`), so the permutation indexes columns that no longer exist. This fails for any constant column, with or without `training_set_limit`, and also with `num_estimators = 1`.

`brulee_tab_pfn()` already drops constant columns for each member, so the rare-column case works. When every predictor is constant, both models failed with poor errors: TabICL gave `subscript out of bounds`, and TabPFN gave an error from inside `purrr::map()` that was attributed to an internal function.

The fix goes through helpers that both models share, so neither model has its own copy of the logic.

## Checklist

- [x] Add `brulee_varying_columns()` and `brulee_subsample_data()` to `R/0_utils.R`
- [x] Add `check_predictors_vary()` to `R/checks.R`
- [x] Make `tabicl_unique_filter()` and `tabpfn_member_layout()` delegate to `brulee_varying_columns()`
- [x] Use `brulee_subsample_data()` in `tabicl_bridge()` and `tabpfn_bridge()`
- [x] Call `check_predictors_vary()` at fit time in both models
- [x] Fix the feature count in `predict.brulee_tab_icl()`
- [x] Tests: shared helpers (`test-0_utils.R`, `test-checks.R`)
- [x] Tests: TabICL with a constant column, the #145 scenario, and all-constant snapshots
- [x] Tests: TabPFN all-constant snapshots and the #145 scenario
- [x] NEWS bullet
- [x] Run the targeted tests, then the full suite, and check the reprex script

## Details

### Shared helpers

- `brulee_varying_columns(x)`: a per-column logical that is TRUE when an encoded numeric matrix column has more than one distinct value. NaN counts as a value. This rule gives the same results as TabICL's `length(unique(col)) > 1` and as TabPFN's "not all NaN and not all equal to the first value", both of which it replaces: constant columns are dropped, all-NaN columns are dropped, and a value mixed with NaN is kept.
- `brulee_subsample_data(predictors, outcome, limit, call)`: wraps `brulee_subsample_rows()` and the row subsetting that both bridges did inline. It returns `list(predictors, outcome, subsampled)`.
- `check_predictors_vary(x, subsampled, call)`: aborts when no column varies. When the rows came from a `training_set_limit` sample, it adds a hint that the sample may have removed the variation.

### Notes

- `tabicl_unique_filter()` keeps its name (tests use it) and its sklearn `nrow <= threshold` guard.
- Only the count passed to `tabicl_make_members()` changes. When no column is constant the count is the same, so RNG draws and predictions do not change. Saved fits keep working, because nothing new is stored on the object.
