# TabICL developer tooling (`dev/tabicl/`)

This folder holds the **offline tooling** used to bring the TabICL model into
brulee and to keep it honest. None of it is part of the installed R package and
none of it runs when a user fits or predicts: `dev/` is excluded from the package
build (it is listed in `.Rbuildignore`). It exists so a maintainer can

1. turn the released TabICL weights into a format R can read, and
2. generate the reference data that the package's tests compare against.

If you only want to *use* the model, you do not need anything here; you need a
converted checkpoint directory (see "What the package needs at runtime" below).

## 1. Background in plain terms

TabICL is a pretrained "tabular foundation model" written in Python/PyTorch by
its authors. brulee contains a from-scratch reimplementation of that model's math
in R using the `torch` package, so R users can run the model without Python. The
reimplementation must produce the *same numbers* as the original for the *same
weights*, or the predictions would be wrong.

Two practical problems follow from that goal, and this folder solves both:

- **The released weights are in a Python-only file format** (a "pickle",
  `.ckpt`). R cannot read it safely. So we convert it once, offline, into files R
  reads natively. That is what `convert_ckpt.py` does.
- **We need proof the R code matches the original.** So we run the real Python
  model, capture its inputs and outputs, and save them. The R tests then feed the
  same inputs to the R code and check the outputs match. Those saved
  inputs/outputs are the "fixtures" the test suite uses.

### A short glossary

- **Checkpoint**: a file containing a trained model's saved parameter values. The
  TabICL release ships two: one for classification, one for regression.
- **`.ckpt`**: TabICL's checkpoint format, a Python `torch.save` pickle holding a
  `config` (the model's shape settings) and a `state_dict` (the parameter
  values).
- **`state_dict`**: the dictionary mapping each parameter's name (for example
  `col_embedder.in_linear.weight`) to its numeric values.
- **safetensors**: a simple, language-neutral file format for storing named
  tensors. R can read it; it has no executable code, unlike a pickle.
- **`config.json`**: the model's shape settings (embedding size, number of
  layers, and so on), pulled out of the checkpoint and written as plain JSON.
- **Fixture**: a small saved test case (weights plus an input plus the expected
  output) that a test loads to check the R code.
- **Golden / reference values**: outputs taken from the original Python model,
  treated as the source of truth the R code must reproduce.
- **Parity**: agreement between the R outputs and the Python outputs, measured as
  the largest difference (we expect roughly 1e-6, the limit of 32-bit floats).

## 2. How the pieces fit together

```
                 (Python, offline, this folder)
 released .ckpt ──convert_ckpt.py──▶ <task>.config.json + <task>.model.safetensors
        │                                    │
        │                                    ▼
        │                         R loads these at predict time
        │                         (R/tabicl-load.R, the shipped package)
        ▼
 real Python model ──dump_*.py──▶ test fixtures (inputs + golden outputs)
                                          │
                                          ▼
                          R tests rebuild each piece, copy the
                          weights in, run it, and compare to the
                          golden values (tests/testthat/test-tabicl-*.R)
```

The package's own R code lives in `R/tabicl-*.R` (not here). This folder only
produces inputs for it (converted weights) and checks on it (fixtures).

## 3. What the package needs at runtime

`brulee_tab_icl()` reads its weights from a local cache (following the chronos2
pattern), not from this folder. The cache lives at
`~/.cache/TabICL/<version>/<date>/<Classification|Regression>/` and holds only
the two files brulee reads per task, named with the task prefix so a loose file
is self-identifying:

- classification: `classification.config.json` and
  `classification.model.safetensors`
- regression: `regression.config.json` and `regression.model.safetensors`

`brulee_tab_icl()` takes no path argument: it looks up the cached checkpoint for
the task and errors if none is present. The cache is populated by the downloader
(`tab_icl_download_weights()`), which fetches the files as individual assets from
a release of the `tidymodels/tabicl-weights` GitHub repo (tag `<version>-<date>`)
into the cache. The large safetensors are release assets rather than committed
files, so the source-archive tarball does not contain them; the downloader pulls
each asset directly. Everything else in this folder produces and validates those
two files. There is one such pair per task: a classifier pair and a regressor
pair.

### Populating the cache from local artifacts

`dev/tabicl/artifacts/` already has the same `<version>/<date>/<TaskLabel>/`
layout as the cache, so you can populate the cache by copying the two files brulee
reads per task into `~/.cache/TabICL/` (or wherever `brulee.tabicl_cache_dir`
points):

```r
art <- "dev/tabicl/artifacts"
cache <- getOption("brulee.tabicl_cache_dir",
                   file.path(Sys.getenv("HOME"), ".cache", "TabICL"))
for (rel in list.files(art, recursive = TRUE,
                       pattern = "\\.(config\\.json|model\\.safetensors)$")) {
  dest <- file.path(cache, rel)
  dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
  file.copy(file.path(art, rel), dest, overwrite = TRUE)
}
```

## 4. Two storage locations (and what is tracked in git)

The tooling writes to two different places, for two different purposes.

- **`dev/tabicl/artifacts/<version>/<date>/<Classification|Regression>/`**: the
  *real* converted weights and reference outputs, organized by model version and
  import date (for example `artifacts/v2/2026-02-12/Classification/`). The
  version and date are taken from the checkpoint filename. The large
  `*.model.safetensors` files are **git-ignored** (they are big and
  re-derivable). The small companion files (`*.config.json`,
  `state_dict_keys.txt`, `manifest.json`, and the `golden/` outputs) are tracked. These are used by the
  developer-only checks, which need the real weights present locally. The reader
  scripts locate the latest version/date automatically, so you still refer to a
  checkpoint by task (`classifier` / `regressor`).

- **`tests/testthat/fixtures/tabicl/`** (note: outside this folder): the
  **committed** test fixtures the automated test suite uses. These are small and
  use *random* weights, not the real trained weights, so they can be checked into
  the repository and run in continuous integration without the multi-hundred-MB
  checkpoints.

## 5. File-by-file guide

### Conversion and reference generation (Python)

- **`convert_ckpt.py`** turns a released `.ckpt` into the task-prefixed
  `<task>.config.json` and `<task>.model.safetensors` (plus an inventory and a
  provenance manifest). Run once per checkpoint. This is the only script whose
  output the package actually consumes.
- **`dump_golden.py`** runs the *real* TabICL model on a tiny fixed dataset and
  saves the output of each internal stage (column embedding, row interaction,
  in-context learning) plus, for the regressor, the final prediction statistics.
  These "real-weight golden" outputs let `check_stages.R` confirm the R code
  matches the genuine model.
- **`dump_primitives.py`** builds each R-ported building block and the full model
  with small *random* weights, runs them through the real Python code, and saves
  the weights, inputs, and outputs as the committed test fixtures.
- **`dump_preprocess.py`** runs scikit-learn's preprocessing steps (scaling,
  outlier clipping, the power transform) and saves their outputs as committed
  fixtures, so the R preprocessing can be checked against scikit-learn.
- **`dump_predict_golden.py`** runs the real `TabICLClassifier` /
  `TabICLRegressor` end to end (in their single, fully reproducible
  configuration) and saves the resulting probabilities / predictions. Used for
  developer-side, real-weight verification of the whole pipeline.

### Verification (R)

- **`harness.R`** confirms R can read everything the converter produced (config,
  every weight tensor with no gaps, the golden files) for a checkpoint.
- **`check_stages.R`** loads the *real* converted weights into the R model, runs
  it, and compares every stage and the full forward pass (and the regression
  statistics) to the golden values from `dump_golden.py`. This is the strongest
  developer check: it proves the shipped R code reproduces the real model.

### Records

- **`manifest.json`** (one per checkpoint, written by `convert_ckpt.py`): the
  upstream filename and checksums for that conversion.
- **`PROVENANCE.md`**: where the weights and code came from, the licensing
  situation, and the conversion details, with checksums and revisions.
- **`ticklish-whistling-walrus.md`**: the original feasibility study and roadmap.
- **`.venv/`** (git-ignored): the Python virtual environment used to run the
  scripts.

## 6. What is inside the test objects, and why

Almost every committed fixture is one self-contained test case made of three
parts:

1. **Weights**: the parameter values for one module (or the whole model). These
   are *random*, drawn with "fan-in" scaling (each weight's spread shrinks as the
   layer gets wider). Random is fine because the test checks that the R math
   equals the Python math for whatever weights are given; it is not checking the
   trained behavior. Fan-in scaling keeps the numbers a sensible size so a tight
   tolerance is meaningful.
2. **Inputs**: a fixed random input tensor (and, where relevant, training labels).
3. **Golden output**: what the real Python code produced from those weights and
   inputs.

A test loads the fixture, builds the matching R module, copies the weights in,
runs it on the input, and checks the output matches the golden values within
about 1e-6.

Some practical details that explain the fixtures' form:

- Each fixture is a pair: a `*.safetensors.gz` (the tensors, gzip-compressed) and
  a `*.json` (the shape settings and the random seed, so the test can rebuild the
  right-sized module). They are gzip-compressed because R's package checker
  otherwise mistakes the raw safetensors header for an executable file.
- The *real* trained weights are never committed. Where we want to validate
  against the genuine model (not just random weights), the reference values live
  in `artifacts/.../golden/` and the checks run locally via `check_stages.R`.

### The committed fixtures and the tests that use them

| Fixture(s) | What it exercises | Test file |
|---|---|---|
| `rope` | rotary position embedding | `test-tabicl-rope.R` |
| `ssmax`, `mha_rope`, `mha_ssmax_self`, `mha_ssmax_cross`, `mha_plain_cross` | scalable softmax and attention variants | `test-tabicl-attention.R` |
| `col_embedding`, `col_embedding_reg` | column-embedding stage | `test-tabicl-embedding.R` |
| `row_interaction`, `row_interaction_biasfree` | row-interaction stage | `test-tabicl-interaction.R` |
| `icl_learning`, `icl_learning_reg`, `full_model`, `full_model_reg` | in-context stage and full forward | `test-tabicl-learning.R` |
| `full_model`, `full_model_reg` | the weight loader (name mapping) | `test-tabicl-load.R` |
| `quantile_dist` | regression head (quantiles to statistics) | `test-tabicl-quantile.R` |
| `prep_standard_scaler`, `prep_outlier_remover`, `prep_pipeline_none`, `prep_pipeline_power` | preprocessing vs scikit-learn | `test-tabicl-preprocess.R` |
| `engine_clf`, `engine_reg` | end-to-end predict on a small model | `test-tabicl-ensemble.R`, `test-tabicl-fit.R` |
| `predict_clf`, `predict_reg` | real-weight end-to-end reference (developer check, not run by the test suite) | (manual / `check_stages.R`-style) |

The `_reg` / `_biasfree` variants exist because the regressor checkpoint differs
from the classifier in two ways the code must handle: it has no LayerNorm biases,
and it predicts numbers (quantiles) instead of class probabilities.

## 7. Setup (one time)

```sh
cd dev/tabicl
python -m venv .venv
.venv/bin/python -m pip install -e ~/github/tabicl safetensors scikit-learn numpy
```

`~/github/tabicl` is a local clone of the TabICL source repository (see
`PROVENANCE.md` for the exact revision used). The R scripts additionally need a
working `torch`, `safetensors`, and `jsonlite` in R.

## 8. Updating to new model weights

When the TabICL authors publish new weights, how much work it is depends entirely
on whether the model's *shape* changed. The deciding question is: **do the
parameter names and shapes match the current version?**

The answer comes from `state_dict_keys.txt`, an inventory the converter writes
listing every parameter name with its data type and shape. Convert the new
checkpoint, then compare its inventory to the old one. If the two inventories are
identical, you are in the simple case (Section 8a). If names or shapes differ,
you are in the harder case (Section 8b).

```sh
# Step shared by both cases: edit the filename/repo in convert_ckpt.py if the
# release name changed, then convert.
.venv/bin/python convert_ckpt.py classifier
.venv/bin/python convert_ckpt.py regressor

# The decision: compare the new parameter inventory to the committed one.
git diff -- 'dev/tabicl/artifacts/**/state_dict_keys.txt'
```

### 8a. Same architecture, new values only

This covers a retrained checkpoint, or a point release, where every parameter has
the same name and shape and only the numbers changed. The `git diff` above shows
no changes to `state_dict_keys.txt`. The R model is built from `config.json` and
walks its own module tree to load weights, so identical names and shapes mean
**no R code changes are needed**.

1. Convert the new checkpoints (done above).
2. Confirm the inventory is unchanged (the `git diff` is empty). Note that some
   "size" changes are still in this case: if only values inside `config.json`
   that drive existing shapes changed (for example a different embedding size or
   a different number of layers), the *names* are unchanged and the R code, which
   reads those from the config, still works. The test in Section 8b is "did any
   parameter name appear, disappear, or change role", not "did any number
   change".
3. Regenerate the real-weight reference outputs and confirm parity:

   ```sh
   .venv/bin/python dump_golden.py classifier
   .venv/bin/python dump_golden.py regressor
   Rscript check_stages.R   # expect every line "OK" at ~1e-6
   ```

4. The committed CI fixtures do **not** need regenerating: they use small random
   models of the same architecture and are independent of the real values. The
   existing `tests/testthat/test-tabicl-*.R` suite should still pass unchanged.
5. Update `PROVENANCE.md` with the new filename, checksums (from the new
   `manifest.json`), and the upstream revision. When the downloader is enabled,
   re-host the converted files and update the pinned revision in
   `R/tabicl-download.R`.

Net effect: convert, verify, refresh provenance. No package code or fixture
changes.

### 8b. New or changed parameters (different shapes, names, or modules)

This covers a genuine architecture change: a renamed parameter, a new submodule,
a removed one, a new option that adds parameters (for example turning on an
affine output layer), or a structural change. The `git diff` of
`state_dict_keys.txt` is non-empty and is your specification: it lists exactly
which names appeared, disappeared, or changed shape.

1. Convert the new checkpoints and read the inventory diff to understand what
   changed.
2. Update the R modules in `R/tabicl-*.R` so their structure matches the new
   `state_dict`. Add new parameters/submodules, rename, or remove as the diff
   dictates. If the change is only a different number/size that already flows
   from `config.json` (more layers, wider embeddings), this step may be empty
   because the modules and the loader are written to follow the config and the
   module tree.
3. Update the weight loader's name map in `R/tabicl-load.R` (the `tabicl_map_*`
   and `tabicl_state_map` functions) so every new parameter has a slot and every
   removed one is gone. The loader checks the mapping **in both directions** and
   errors with the offending names if the checkpoint has a parameter the model
   does not, or vice versa. Use those errors as a checklist:

   ```sh
   Rscript check_stages.R   # will first fail on unmatched/missing names; fix until it loads
   ```

4. Once it loads with no gaps, `check_stages.R` reports per-stage parity. If a
   stage's numbers are off, that pinpoints which module's `forward` logic needs
   correcting. Compare against the Python intermediates (regenerate with
   `dump_golden.py`, and if needed dump finer-grained intermediates as was done
   during the original port) to localize the discrepancy.
5. Regenerate the committed fixtures so the test suite exercises the new
   architecture. Edit the small-model settings in the dump scripts if dimensions
   or modules changed (for example `_small_tabicl_config` and the per-module
   sizes in `dump_primitives.py`), then:

   ```sh
   .venv/bin/python dump_primitives.py
   .venv/bin/python dump_preprocess.py        # only if preprocessing changed
   .venv/bin/python dump_predict_golden.py
   ```

6. Add or adjust R parity tests in `tests/testthat/test-tabicl-*.R` for any new
   module, and update the weight-copy helpers in
   `tests/testthat/helper-tabicl.R` if parameter names changed.
7. Format and run the full check:

   ```sh
   air format R/tabicl-*.R tests/testthat/test-tabicl-*.R
   NOT_CRAN=true R CMD check    # expect Status: OK
   ```

8. Update `PROVENANCE.md` and (when hosting) re-host plus re-pin the revision.

Net effect: code change driven by the inventory diff, then regenerate every
fixture and re-verify. The two automated safety nets are the loader's
both-direction name check (catches structural mismatches) and `check_stages.R`'s
per-stage parity (catches arithmetic mismatches).

## 9. Stage output shapes (fixed reference dataset)

For the tiny fixed dataset used by `dump_golden.py` (one table, 12 training rows,
5 test rows, 4 features), the per-stage output shapes are:

| Tensor | classifier | regressor |
|---|---|---|
| `X` (input) | `[1, 17, 4]` | `[1, 17, 4]` |
| `y_train` | `[1, 12]` | `[1, 12]` |
| `col_embed` | `[1, 17, 8, 128]` | `[1, 17, 8, 128]` |
| `row_interact` | `[1, 17, 512]` | `[1, 17, 512]` |
| `icl_out` | `[1, 5, 3]` | `[1, 5, 999]` |

These are a quick reference for sanity-checking a conversion: if the shapes here
change after an update, you are in the Section 8b case.

## 10. Current status

The R port is complete and validated end to end against both the released
checkpoints and the scikit-learn wrappers. The only outstanding piece is the live
weight downloader, which waits on the hosting decision; until then
`brulee_tab_icl()` reads from the local cache (`~/.cache/TabICL/`), which the
downloader populates once a URL is set; until then, populate it offline as shown
in Section 3.
