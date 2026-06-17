# TabICL weights: provenance, licensing, and format conversion

This records where the TabICL model and weights came from, the license situation,
how the released checkpoints were converted into the format the R port consumes,
and the checksums/revisions that make the chain reproducible. Captured
2026-06-16.

## 1. Source materials

| Item | Source | Revision / identifier |
|---|---|---|
| Reference **code** (architecture, sklearn wrappers) | GitHub `soda-inria/tabicl` | commit `8f665edb1e3099907e15bf645bb319fbb2f99c24` (2026-06-09) |
| Released **weights** | Hugging Face Hub `jingang/TabICL` | repo commit `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` (lastModified 2026-02-18); not gated |
| Classifier checkpoint | `jingang/TabICL` | `tabicl-classifier-v2-20260212.ckpt` |
| Regressor checkpoint | `jingang/TabICL` | `tabicl-regressor-v2-20260212.ckpt` |

The `-v2-20260212` in the filenames is the TabICL v2 checkpoint version (dated
2026-02-12). The weights were downloaded with `huggingface_hub.hf_hub_download`,
which resolved to the repo commit SHA above (confirmed via the local HF cache
snapshot directory `models--jingang--TabICL/snapshots/4dcd344…`).

Conversion toolchain: Python `torch` 2.12.0 + `safetensors` (dev venv); the R
side loads with `torch`/libtorch 0.17.0 and `safetensors` 0.2.0.

### Checksums

Converted artifacts are stored under
`artifacts/<version>/<date>/<Classification|Regression>/` (for example
`artifacts/v2/2026-02-12/Classification/`), with the version and date taken from
the checkpoint filename.

Original `.ckpt` files (sha256), recorded in each `manifest.json` at conversion
time:

| File | sha256 |
|---|---|
| `tabicl-classifier-v2-20260212.ckpt` | `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` |
| `tabicl-regressor-v2-20260212.ckpt` | `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a` |

Converted artifacts (sha256):

| File | sha256 |
|---|---|
| `classification.model.safetensors` | `05a13ce66b439b03fd5f9f02ff0c41410ed2c03f36b1f4219874806b12b04a83` |
| `classification.config.json` | `74fcd36136518b50e4929cbd4fcafb0ba9f2bb799db741265902bd4d7e9bc3e1` |
| `regression.model.safetensors` | `90c5952b4e201c265b4444a3694f3d1fc07df3864157ed6361b6c31723413d30` |
| `regression.config.json` | `1448d3cbcc170aac2a3670aeda46257706da11860e08884128e42f14063a9960` |

(File contents, hence the sha256s, are independent of the filename, so these
match regardless of the task prefix.)

`convert_ckpt.py` writes a fresh `manifest.json` on every run capturing the
upstream filename, the `.ckpt` sha256, and the output sha256s, so any
re-conversion is self-documenting. Tensor counts: 391 (classifier), 347
(regressor).

## 2. Licensing

**Conclusion: redistribution of the converted weights is permitted under
BSD-3-Clause, provided the copyright notice, license text, and disclaimer travel
with them.**

- The Hugging Face model repo `jingang/TabICL` is tagged **`license: bsd-3-clause`**
  and is **not gated** (no access agreement, no token required).
- The GitHub repo's `LICENSE` is **BSD-3-Clause** (Copyright 2025, Soda team @
  Inria) for the code and weights. It additionally bundles an Apache-2.0 license
  block, but that block applies **only** to `src/tabicl/forecast/`, which is
  derived from TabPFN-TS. We do **not** use or port any code from
  `src/tabicl/forecast/`; everything ported comes from `src/tabicl/_model/` and
  `src/tabicl/_sklearn/`, which are BSD-3.
- Because the format conversion (below) is a **lossless repackaging** of the same
  tensors, the converted `model.safetensors` carries the same BSD-3 terms as the
  original `.ckpt`.

**Action items for hosting the converted weights:** include the upstream
`LICENSE` (BSD-3, Soda team @ Inria), attribution to the TabICL authors and the
TabICL paper, and this provenance record (or the per-checkpoint `manifest.json`)
alongside the files. BSD-3 only requires retaining the copyright notice +
license + disclaimer; no share-alike obligation.

**Citation:** Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. (2025).
TabICL: A Tabular Foundation Model for In-Context Learning on Large Data. arXiv
preprint arXiv:2502.05564.

## 3. Why and how the checkpoint format was converted

### Why convert

The released checkpoints are a `torch.save` **pickle** (`.ckpt`) of
`{"config": <dict>, "state_dict": <OrderedDict>}`. Pickle has no safe, pure-R
reader: loading it in R would require `reticulate` + a Python `torch`, which
defeats the goal of a pure-R `torch` implementation and adds a heavy runtime
dependency. So the checkpoints are converted **once, offline** into formats R
reads natively:

- `state_dict` → `<task>.model.safetensors` (read by
  `safetensors::safe_load_file`).
- `config` → `<task>.config.json` (read by `jsonlite::fromJSON`).

The two files brulee reads are prefixed with the task (`classification.` /
`regression.`) so a single file is self-identifying if moved on its own.

The conversion is a Python dev-time step (`dev/tabicl/convert_ckpt.py`), not part
of the R package or its runtime.

### What the conversion does (and does not) change

`convert_ckpt.py`:

1. Downloads the `.ckpt` from `jingang/TabICL`.
2. `torch.load(..., weights_only=True)` and asserts the dict has `config` and
   `state_dict`.
3. Writes `config` verbatim to `<task>.config.json`.
4. Writes the `state_dict` tensors to `<task>.model.safetensors` via
   `safetensors.torch.save_file` (each tensor made contiguous and cloned to
   avoid shared-storage issues; values unchanged).
5. Writes `state_dict_keys.txt` (name/dtype/shape inventory) and `manifest.json`
   (provenance + checksums).

It is **lossless**: the same parameter names, shapes, dtypes (float32), and
values are preserved. No quantization, pruning, or renaming. The weights are not
fine-tuned or otherwise modified.

### Verification that the conversion is faithful

The converted weights were validated end-to-end against the original model:

- Every checkpoint tensor maps 1:1 to an R parameter with no missing/unmatched
  keys and matching shapes (the loader enforces this in both directions).
- The full R forward pass reproduces the reference model's stage outputs to
  ~1e-6 relative error (classifier and regressor), and the prediction pipeline
  matches the Python sklearn wrappers (single-member, deterministic config) to
  <1e-5. See `dev/tabicl/check_stages.R` and the `tests/testthat/test-tabicl-*`
  suite.

### Distribution format note

The converted files are hosted/consumed as `<task>.config.json` +
`<task>.model.safetensors` (per checkpoint). The filenames are conventional, not
load-bearing: the readers parse by content, so the prefix is only a convenience
for identifying a loose file. The brulee code maps the task to these filenames in
one place (`tabicl_checkpoint_files()`), so the naming can change there if
needed.
