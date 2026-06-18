# Feasibility: Pure R torch port of TabICL v2 for brulee

## Context

TabICL is a tabular foundation model (in-context learning over tabular data). The
Python reference lives at `~/github/tabicl`. We want to know whether the **model
modules** can be reimplemented in pure R `torch` (libtorch) and shipped in **brulee**,
loading the **released pretrained checkpoints** (no retraining). Scope: **both
classification and regression** (v2). **GPU parity matters.** brulee owns data
handling / training-loop infrastructure; this assessment is about the network modules,
weight transfer, and inference parity.

**Verdict: feasible, low-to-moderate risk.** The architecture uses only standard ops
(Linear, LayerNorm, GELU, SDPA, RoPE, small MLPs). No custom autograd, no C++/CUDA/
Triton extensions, no `torch.compile`/JIT (verified by grep across `_model/`). Flash
Attention 3 is an *optional* CUDA speedup with a clean `F.scaled_dot_product_attention`
fallback. brulee's existing **chronos2** integration is a near-exact template for the
whole download → reimplement → load-weights → predict pipeline. The single substantial
non-standard piece is the **regression** output module (`quantile_dist.py`), and base
R's `isoreg()` already covers its hardest sub-part (PAVA isotonic regression).

## Architecture and module-by-module difficulty

TabICL is a three-stage pipeline (`src/tabicl/_model/`):

| Stage | File | What it is | R torch difficulty |
|---|---|---|---|
| 1. Column embedding | `embedding.py` | Per-column SetTransformer (induced self-attention) → cell embeddings; target-aware | Moderate (ISAB is new to brulee) |
| 2. Row interaction | `interaction.py` | Transformer over features per row, RoPE, learnable CLS tokens | Low (close to SAINT) |
| 3. In-context learning | `learning.py` | Dataset-level transformer; train rows as context, predicts test rows | Moderate |
| Attention core | `attention.py`, `layers.py` | MHA w/ RoPE + SSMax + optional KV cache; SDPA + FA3 fallback | Low |
| RoPE | `rope.py` | Rotary embeddings (interleaved / non-interleaved, base 100000) | Low |
| SSMax | `ssmax.py` | Small MLP query-scalers (`qassmax-mlp-elementwise` default) | Trivial |
| KV cache | `kv_cache.py` | Mutable K/V store for fast test-time inference | Low (optimization; can defer) |
| Regression head | `quantile_dist.py` | Quantiles → distribution: PAVA, GPD tails, CRPS (1543 lines) | **Highest — scope carefully** |

Building blocks brulee **already has** (so these map directly):
- Hand-rolled MHA with q/k/v/o projections — `R/saint-fit.R` (`saint_attention_module`).
- A **RoPE** module and RMS/T5-style LayerNorm — `R/chronos2-misc.R`.
- `nn_module` idiom, pre-norm transformer blocks, CLS/target tokens, mixed
  categorical+continuous embeddings — `R/saint-fit.R`.
- Device auto-detect / safe fallback — `guess_brulee_device()`, `get_safe_device()` in
  `R/validation.R`.

R `torch` 0.17.0 exports everything the modules need (verified in NAMESPACE):
`torch_scaled_dot_product_attention`, `torch_einsum`, `torch_searchsorted`,
`nnf_pad`, `nnf_gelu`, `nnf_multi_head_attention_forward`.

New modules with no brulee precedent (all standard ops, just not yet written):
- **Induced self-attention block** (Set Transformer): two MHA calls around a learned
  `nn_parameter` of inducing points — `layers.py` `InducedSelfAttentionBlock`.
- **SSMax** query-scalers — direct port of `ssmax.py`.
- **einops removal**: `rope.py` uses only `rearrange`/`repeat`/`einsum`; replace with
  `reshape`/`torch_stack`/`torch_cat`/`torch_einsum` (authors note einops "could be
  replaced"). chronos2's RoPE is a starting point but TabICL adds an interleaved/
  non-interleaved toggle and base 100000.

## GPU parity

Achievable. On CUDA, `torch_scaled_dot_product_attention` dispatches to libtorch's
fused flash / memory-efficient kernels — the same math as FA3, minus the variable-length
packing optimization (which produces identical results, just faster). Plan: always use
`torch_scaled_dot_product_attention`; do **not** attempt to bind the external
`flash_attn_interface` package. Mirror chronos2's device handling: build modules on CPU,
`$to(device)`, pass `device=` explicitly to every `torch_tensor()`, `$detach()$cpu()`
results. Default precision float32 (matches checkpoints); skip AMP/bf16 initially.

## Weight transfer (released checkpoints → R)

Released format differs from chronos2: TabICL ships **`.ckpt`** (a `torch.save` of
`{"config": ..., "state_dict": ...}`) on HF Hub repo `jingang/TabICL`
(`tabicl-classifier-v2-20260212.ckpt`, `tabicl-regressor-v2-20260212.ckpt`), loaded in
`src/tabicl/_sklearn/classifier.py:382` via `TabICL(**config); load_state_dict(...)`.

Pickle `.ckpt` is awkward to read in R. Recommended: a **one-time offline Python
conversion** that re-saves each checkpoint's `state_dict` as `model.safetensors` and its
`config` as `config.json`, then host/cache those. After conversion, **reuse chronos2's
loader verbatim**: `safetensors::safe_load_file(..., framework="torch")` + an explicit
Python→R parameter-name map + `param$copy_()` under `with_no_grad()` (see
`load_chronos2_weights()` in `R/chronos2-misc.R`). Names are dot-paths
(`col_embedder.*`, `row_interactor.*`, `icl_predictor.*`, regression: `quantile_dist.*`);
the R `nn_module` tree must mirror them, with R 1-indexing → Python 0-indexing on block
loops (chronos2 already does this). Reuse chronos2's HF download/SHA-pinning/cache code
(`chronos2_download`, `chronos2_resolve_revision`) — adapt to a shared, better-versioned
downloader as you noted.

## Inference details brulee must reproduce (for parity)

These live outside the NN in `src/tabicl/_sklearn/preprocessing.py` and `classifier.py`:
- Preprocessing: ordinal-encode categoricals, impute missing, drop constant features,
  outlier clipping (z-score, thresh 4), per-member normalization
  (none/power/quantile/quantile_rtdl/robust), standardize + clip to [-100, 100].
- **Ensembling** (default 8 members): feature-permutation shuffles + class-label shuffles
  per member; forward each; **average logits**; softmax with temperature (0.9); undo
  class shuffles. Determinism: dropout off in eval; results otherwise deterministic on a
  fixed device once FA3 is excluded.
- Forward contract: concat train+test rows into `X (B, T, H)`, pass `y_train (B, n_train)`
  (test labels withheld; split is positional) — `tabicl.py:347`.
- Regression: model emits quantiles; `quantile_dist.py` turns them into a distribution.
  **Action item:** read `quantile_dist.py` to separate the *inference* path (monotonize
  quantiles via `isoreg()`; interpolate; GPD tails only for extrapolation beyond the grid;
  point prediction) from the *training-only* CRPS loss, and port only the inference path.

## Phased roadmap

1. **Spike — numerical parity harness.** Build the Python-side `.ckpt`→safetensors+json
   converter. In R, stand up the chronos2-style download/cache + safetensors loader.
   Dump a few intermediate tensors (post-stage-1/2/3) from Python on a tiny fixed dataset
   to use as golden references. *Exit: weights load, names all matched, no skips.*
2. **Attention + RoPE + SSMax primitives.** Port MHA (SDPA path), RoPE (einops removed,
   interleaved+non-interleaved), SSMax variants. Unit-test each against Python on random
   inputs (atol ~1e-5 float32). *Exit: per-op parity.*
3. **Stage 2 (RowInteraction).** Closest to SAINT; validate stage-2 output vs golden.
4. **Stage 1 (ColEmbedding + Set Transformer / ISAB).** New induced-self-attention block,
   target-aware embedding. Validate stage-1 output.
5. **Stage 3 (ICLearning) — classification.** Wire full forward; reproduce ensembling +
   logit-averaging + temperature softmax. *Exit: `predict_proba` matches Python within
   tolerance on real datasets (CPU and CUDA).*
6. **Regression head.** Port the `quantile_dist.py` inference path (using `isoreg()` for
   PAVA). *Exit: regression predictions/intervals match Python.*
7. **brulee integration + optional KV cache.** User-facing `brulee_tab_icl()` fit/predict,
   device handling, serialization (`model_to_raw`/`revive_model` patterns). Add KV-cache
   only if test-time speed requires it.

## Verification

- **Golden-tensor tests** per stage from step 1 (Python references committed as fixtures).
- **End-to-end parity**: compare R vs Python `predict_proba` / regression output on
  several OpenML-style tables, classification and regression, on **both CPU and CUDA**
  (expect tight agreement; small FA-vs-SDPA float drift acceptable).
- testthat suite mirroring source files (e.g. `test-tabicl-attention.R`,
  `test-tabicl-rope.R`, `test-tabicl-embedding.R`), snapshot-based warnings/errors.
- Close with `air format` and a non-CRAN `R CMD check`.

## Open scoping items (resolve during step 1 / step 6)

- Exact fraction of `quantile_dist.py` needed at inference vs. training-only (CRPS).
- Whether numba paths in `quantile_dist.py` have pure-torch equivalents already in the
  file (numba is an optional accelerator, so a torch fallback should exist).
- KV cache: defer unless test-time latency is unacceptable without it.
