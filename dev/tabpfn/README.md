# dev/tabpfn

Python scripts that build the files `brulee_tab_pfn()` relies on: the model
registry and the test fixtures. They run the Python `tabpfn` package, which is the
reference that the R port is checked against. `dev/` is in
`.Rbuildignore`, so none of it goes into the built package.

## Licence

No Prior Labs weights or checkpoint contents may be committed. Prior Labs
releases the TabPFN-3 and TabPFN-3.5 weights under non-commercial licences.

- All test fixtures come from tiny checkpoints with **random** weights, built
  with the Python architecture code (Apache-2.0).
- The scripts write their own inference configs; none are copied from a
  released checkpoint.
- `checkpoints/` holds metadata dumped from the real checkpoints. That
  metadata is covered by the licence, so the directory is git-ignored and
  stays on your machine.

## Setup

The scripts need Python 3.12 with `tabpfn` pinned to the reference release
recorded in `inst/tabpfn-registry.json` (currently 9.1.0):

```sh
uv venv --python 3.12 dev/tabpfn/.venv
uv pip install --python dev/tabpfn/.venv/bin/python "tabpfn==9.1.0"
```

`.venv/` is git-ignored. Run every script with that Python from the package
root, e.g. `dev/tabpfn/.venv/bin/python dev/tabpfn/make_fixtures.py`.

`dump_checkpoints.py` and `make_registry.py` also need the real checkpoints
in the Python `tabpfn` cache, so you must first accept the licences and
download the weights. `make_registry.py` also reads the commit of the
release tag from a clone of the Python repository at `~/github/TabPFN-python`.

## Scripts

| Script | Writes | Used by |
| --- | --- | --- |
| `dump_checkpoints.py` | `checkpoints/*.json` (local only): each real checkpoint's sha256, size, config, inference config, and tensor names, shapes, and dtypes. No weights. | `make_registry.py`; checking the R config parsing against new checkpoints |
| `make_registry.py` | `inst/tabpfn-registry.json`: supported versions, Hugging Face repos, file names, sha256 sums, sizes, architectures, and data size limits | `R/tabpfn-weights.R` |
| `make_checkpoint_fixtures.py` | `tests/testthat/fixtures/tabpfn/checkpoints/tiny*`, `int64.ckpt.gz`: small `.ckpt` and `.safetensors` files with known values | the checkpoint-reader tests |
| `make_fixtures.py` | `tests/testthat/fixtures/tabpfn/v3-*`, `v3_5-*`: weights, inputs, and every submodule's inputs and outputs for a tiny random model | the architecture tests (`test-tabpfn-v3.R`, `test-tabpfn-v3_5.R`) |
| `make_pipeline_fixtures.py` | `tests/testthat/fixtures/tabpfn/pipeline-*` and `checkpoints/tabpfn-*-tiny.*`: tiny random checkpoints, plus each ensemble member's random choices and the model inputs, outputs, and predictions from `TabPFNClassifier`/`TabPFNRegressor` | the preprocessing, post-processing, and end-to-end tests |

Binary fixtures are gzipped so that `R CMD check` doesn't flag them.

## When to rerun

- **New Python `tabpfn` release or new checkpoints:**
  1. Update the pin in the venv.
  2. Run `dump_checkpoints.py`, then `make_registry.py`.
  3. Review the changes to `inst/tabpfn-registry.json` and the dumps. New config
     keys make the R parsers fail on purpose.
- **Python changes to the architecture or the preprocessing:** rerun
  `make_fixtures.py` or `make_pipeline_fixtures.py`, then
  `devtools::test()`.

## Adding a model version

The R code is split so that a new version doesn't change the code of the
existing ones:

- `R/tabpfn-v3.R` and `R/tabpfn-v3_5.R` each hold one architecture: its
  strict config parser `<architecture>_parse_config()` and its module
  `<architecture>_model()` (with `forward()` and `borders()`). The shared code
  finds them by the checkpoint's `architecture_name`.
- Everything else (`R/tabpfn-fit.R`, `-predict.R`, `-preprocess.R`,
  `-modules.R`, `-checkpoint.R`, `-weights.R`) reads what differs between
  versions from the registry and from each checkpoint's inference config.

To add a version:

1. Add it to `SOURCES` in `make_registry.py`, then rerun `dump_checkpoints.py`
   and `make_registry.py`.
2. If its checkpoints use an architecture that brulee already has, that's
   all.
3. Otherwise, add `R/tabpfn-<version>.R` with the two backend functions, a
   scenario in `make_fixtures.py`, and `tests/testthat/test-tabpfn-<version>.R`.
   Modules that several architectures share go in `R/tabpfn-modules.R`.
