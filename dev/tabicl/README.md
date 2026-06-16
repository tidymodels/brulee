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

## Stage shapes (fixed dataset: B=1, n_train=12, n_test=5, H=4)

| Tensor       | classifier      | regressor       |
|--------------|-----------------|-----------------|
| `X`          | `[1, 17, 4]`    | `[1, 17, 4]`    |
| `y_train`    | `[1, 12]`       | `[1, 12]`       |
| `col_embed`  | `[1, 17, 8, 128]` | `[1, 17, 8, 128]` |
| `row_interact` | `[1, 17, 512]` | `[1, 17, 512]`  |
| `icl_out`    | `[1, 5, 3]`     | `[1, 5, 999]`   |
