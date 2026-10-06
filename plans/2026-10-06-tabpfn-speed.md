# Faster `brulee_tab_pfn()` on small data

## Context

On `mtcars`, `brulee_tab_pfn()` plus one `augment()` takes about 10.8 s on the CPU, against 4.4 s for `brulee_tab_icl()`. Profiling a fresh session (v3.5, 8 members, CPU) shows where the time goes:

| Step | Time |
| --- | --- |
| sha256 of the 876 MB checkpoint (first fit of each session) | 3.7 s |
| Read the checkpoint, build the model, load the weights | 1.5 s |
| A later fit in the same session | 0.07 s |
| `predict()` or `augment()` | 4.7 s |
| The same prediction without the between-layer garbage collection | 0.4 s |

Two costs dominate, and neither is model computation:

1. **Garbage collection between layers.** R torch frees tensor memory only when R's garbage collector runs, so `tabpfn_release()` (`R/tabpfn-modules.R`) calls `gc(full = TRUE)` after each layer and attention chunk: 11 call sites, 57 calls per prediction. A full collection walks the whole R heap (about 0.075 s each), so this is about 90% of a small prediction. It was added so that large data (for example, the `forested` data on MPS) doesn't run out of memory, where it is needed.
2. **Re-verifying the checkpoint in every session.** `tabpfn_verify_cached()` (`R/tabpfn-weights.R`) hashes the whole checkpoint once per R session (`tabpfn_env$verified`). For v3.5 that is 3.7 s on every first fit.

The goal is a small-data fit and prediction in well under a second after the first session, with no change to predictions and no loss of the memory protection on large data.

## 1. Collect garbage only when it pays

### Measure first

Before choosing a rule, measure on a few sizes (`mtcars`, `iris`, a 1,000-row and a 5,000-row subset of `modeldata::forested`) with CPU and MPS:

- Elapsed time and peak memory (the process's resident size, from `ps::ps_memory_info()` in the measurement script) for three variants of `tabpfn_release()`:
  - the current `gc(full = TRUE)`;
  - `gc(full = FALSE)`, a minor collection: the released tensors are young objects, so this may free most of the memory at a fraction of the cost;
  - no collection.
- Record the results in this plan before changing the code.

### Change

Based on the measurements, make the collection depend on the size of the work:

- `tabpfn_forward()` (`R/tabpfn-predict.R`) knows the size of each forward pass: `rows * width * members_in_pass`. When it is below a threshold, run the pass without collections; above it, keep them. Pass the decision down with a session-level flag in `tabpfn_env` (for example `tabpfn_env$release`), set and reset with `withr::defer()` around each pass, so the 11 call sites keep calling `tabpfn_release()` unchanged and the architecture files (`tabpfn-v3.R`, `tabpfn-v3_5.R`) don't change.
- `tabpfn_release()` reads the flag and returns immediately when it is off. If the measurements favor it, use `gc(full = FALSE)` when it is on.
- The threshold is an option, `brulee.tabpfn_release_cells`, with a default chosen from the measurements: the smallest size at which skipping the collections raises peak memory noticeably. Document the option next to the existing `brulee.tabpfn_mps_attention_cells` and `brulee.tabpfn_rows_per_pass`.

## 2. Remember verified checkpoints across sessions

- After `tabpfn_verify_cached()` hashes a checkpoint successfully, write a small record to `.verified/<file>.json` in the TabPFN cache directory (`tabpfn_cache_dir()`), with the file's path, size, modification time, and sha256. Write it for checkpoints in the Python cache too; the record lives in brulee's cache, never in Python's.
- On later calls, skip the hash when a record exists whose path, size, modification time, and sha256 all match the file and the registry. Any change to the file (re-download, truncation, a different release) changes the size or modification time and triggers a full check.
- The in-session `tabpfn_env$verified` stays as the first, cheapest check.
- `tab_pfn_clear_cache()` deletes the records of the files it removes (and the `.verified/` directory when it clears everything), and resets `tabpfn_env$verified` as now.
- Downloads already verify the sha256 as they finish (`brulee_download_file()`), so they write the record directly.

## Tests

In `tests/testthat/test-tabpfn-predict.R` and `test-tabpfn-weights.R`:

- **Release decision:** mock `gc` (or count calls through a mocked `tabpfn_release()`) and check that a small prediction makes no collections, and that a prediction above `brulee.tabpfn_release_cells` (set low in the test) makes them. Predictions with and without collections are identical.
- **Verification records:**
  - The first `tabpfn_verify_cached()` hashes and writes a record; a second call in a fresh state (`tabpfn_env$verified` reset) doesn't hash. Count hashes by mocking `cli::hash_file_sha256` with `local_mocked_bindings(.package = "cli")`.
  - Changing the file's size or modification time (`Sys.setFileTime()`) forces a new hash, and a bad file then still fails with the existing checksum error.
  - A record with a different sha256 than the registry is ignored.
  - `tab_pfn_clear_cache()` removes the records.
- The existing parity tests must pass unchanged, which confirms predictions don't change.

## Docs and NEWS

- `?predict.brulee_tab_pfn` details: one sentence on the memory option and when collections run.
- `?tab_pfn_download_weights`, "Cache location": checkpoints are verified once and the result is remembered until the file changes.
- NEWS bullet: faster `brulee_tab_pfn()` fits and predictions on small data (numbers from the final benchmark).

## Verification

- Rerun the profiling script on `mtcars` and record before and after timings in the PR description. Target: first session about 2 s for fit plus `augment()`, later sessions under 1 s.
- Rerun the `forested` example (CPU and MPS, `device = "mps"`) and confirm it still completes without running out of memory, with the same predictions.
- `air format .`, the full `devtools::test()`, and `R CMD check` (not as CRAN).
