# TabICL port — development harness

Spike tooling for porting TabICL v2 to brulee as pure R `torch`. Not part of the
package build (`dev/` is in `.Rbuildignore`). See the roadmap in
`~/.claude/plans/ticklish-whistling-walrus.md`.

## Setup

```sh
python -m venv .venv
.venv/bin/python -m pip install -e ~/github/tabicl safetensors
```

## Workflow

1. **Convert released checkpoints** (`.ckpt` -> safetensors + JSON):

   ```sh
   .venv/bin/python convert_ckpt.py classifier
   .venv/bin/python convert_ckpt.py regressor
   ```

   Writes to `artifacts/<kind>/`: `config.json`, `model.safetensors`,
   `state_dict_keys.txt` (name/dtype/shape inventory), `manifest.json`
   (provenance + sha256). The large `model.safetensors` is git-ignored and
   re-derivable.

2. **Dump golden intermediate tensors** from the reference model:

   ```sh
   .venv/bin/python dump_golden.py classifier
   .venv/bin/python dump_golden.py regressor
   ```

   Writes `artifacts/<kind>/golden/`: `inputs.safetensors` (the fixed dataset),
   `stage_outputs.safetensors` (`col_embed`, `row_interact`, `icl_out`), and
   `meta.json` (seed + shapes). These are small and tracked; the R port
   validates against them tensor-for-tensor.

3. **R-side harness** confirms R consumes every artifact:

   ```sh
   Rscript harness.R classifier
   Rscript harness.R regressor
   ```

   Loads the config, all state_dict tensors (no skips), and the golden
   fixtures, checking shapes against `meta.json`.

4. **Per-primitive parity fixtures** (step 2): build each primitive with random
   (fan-in scaled) weights, run a forward pass, and save weights + inputs +
   golden output.

   ```sh
   .venv/bin/python dump_primitives.py
   ```

   Writes small gzipped safetensors + json fixtures to
   `tests/testthat/fixtures/tabicl/` (tracked): `rope`, `ssmax`, four MHA
   configurations, the stage-1 `col_embedding[_reg]`, stage-2
   `row_interaction[_biasfree]`, stage-3 `icl_learning[_reg]`, the end-to-end
   `full_model[_reg]`, and the regression head `quantile_dist`. The R modules
   (`R/tabicl-rope.R`, `tabicl-attention.R`, `tabicl-layers.R`,
   `tabicl-embedding.R`, `tabicl-interaction.R`, `tabicl-learning.R`,
   `tabicl-model.R`, `tabicl-quantile.R`) are validated against these by
   `tests/testthat/test-tabicl-*.R`. Weights use fan-in scaling so most
   activations stay O(1) (absolute tol 1e-5); the column embedder keeps -100 skip
   values through residuals, so that test uses a relative tolerance.

5. **Real-weight stage checks** (dev only, needs the large `model.safetensors`):

   ```sh
   Rscript dev/tabicl/check_stages.R
   ```

   Loads the actual converted checkpoint + step-1 golden stage outputs, builds
   each R stage (and the full model), copies the real weights in, and compares to
   the golden. Covers both `classifier` (biased LayerNorms) and `regressor`
   (`bias_free_ln`), all stages, the full forward, and the regression-head stats.

6. **Preprocessing + end-to-end prediction goldens** (sklearn references):

   ```sh
   .venv/bin/python dump_preprocess.py        # prep_* fixtures (scaler, outlier, pipelines)
   .venv/bin/python dump_predict_golden.py     # predict_{clf,reg}: real sklearn n_estimators=1
   ```

   `dump_predict_golden.py` runs the real `TabICLClassifier`/`TabICLRegressor`
   (single deterministic member) for authoritative end-to-end parity.

## Port status

The pure-R port is complete and validated end-to-end against the released
checkpoints and the sklearn wrappers: model forward (all stages), preprocessing,
regression head, the prediction engine, and the user-facing `brulee_tab_icl()`
fit/predict. The only remaining piece is the live weight **downloader**, which is
deferred pending the hosting decision; `brulee_tab_icl(path = ...)` currently
takes a local converted-checkpoint directory.

## Stage shapes (fixed dataset: B=1, n_train=12, n_test=5, H=4)

| Tensor       | classifier      | regressor       |
|--------------|-----------------|-----------------|
| `X`          | `[1, 17, 4]`    | `[1, 17, 4]`    |
| `y_train`    | `[1, 12]`       | `[1, 12]`       |
| `col_embed`  | `[1, 17, 8, 128]` | `[1, 17, 8, 128]` |
| `row_interact` | `[1, 17, 512]` | `[1, 17, 512]`  |
| `icl_out`    | `[1, 5, 3]`     | `[1, 5, 999]`   |
