"""Dump golden intermediate tensors from the Python TabICL model.

Builds the real TabICL module from a converted checkpoint, loads its weights,
runs a tiny fixed dataset through the three stages, and saves the inputs and
every stage output as a safetensors fixture. The R port validates against these
tensor-for-tensor in later steps.

Run AFTER convert_ckpt.py. Usage:
    python dump_golden.py classifier
    python dump_golden.py regressor

Output (in <artifacts>/<kind>/golden/):
    inputs.safetensors    X, y_train (the fixed dataset)
    stage_outputs.safetensors   col_embed, row_interact, icl_out
    meta.json             seed, shapes, dtypes, config echo
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from tabicl._model.tabicl import TabICL

from convert_ckpt import artifact_dir, checkpoint_files

# A reproducible, non-special seed: low 31 bits of a digest of a fixed phrase.
# Emulates an RNG draw rather than hand-picking a round number.
SEED = int(hashlib.sha256(b"brulee-tabicl-golden-v1").hexdigest(), 16) % (2**31 - 1)


def build_model(kind: str) -> tuple[TabICL, dict, Path]:
    art = artifact_dir(kind)
    config_name, weights_name = checkpoint_files(kind)
    with open(art / config_name) as fh:
        config = json.load(fh)
    model = TabICL(**config)
    from safetensors.torch import load_file

    sd = load_file(str(art / weights_name))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"missing keys when loading: {missing}")
    if unexpected:
        raise RuntimeError(f"unexpected keys when loading: {unexpected}")
    model.eval()
    return model, config, art


def make_dataset(config: dict, gen: torch.Generator):
    # Tiny fixed table: one dataset (B=1), a handful of train + test rows, few
    # features. Values are standardized-looking floats, matching the model's
    # expected preprocessed input domain.
    B, n_train, n_test, H = 1, 12, 5, 4
    T = n_train + n_test
    X = torch.randn(B, T, H, generator=gen, dtype=torch.float32)

    if config["max_classes"] > 0:
        n_classes = 3
        y_train = torch.randint(
            0, n_classes, (B, n_train), generator=gen, dtype=torch.long
        )
    else:
        y_train = torch.randn(B, n_train, generator=gen, dtype=torch.float32)
    return X, y_train


def dump(kind: str) -> None:
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)

    model, config, art = build_model(kind)
    X, y_train = make_dataset(config, gen)

    # NOTE: row_interactor mutates its input in place (overwrites the first
    # num_cls slots with the CLS tokens), so snapshot each stage's output with a
    # clone before feeding it onward. Otherwise the saved col_embed is the
    # mutated tensor and is inconsistent with the saved row_interact.
    with torch.no_grad():
        col_embed = model.col_embedder(X, y_train=y_train, d=None, embed_with_test=False).clone()
        row_interact = model.row_interactor(col_embed.clone(), d=None).clone()
        icl_out = model.icl_predictor(row_interact.clone(), y_train=y_train).clone()

    out_dir = art / "golden"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_file(
        {"X": X.contiguous(), "y_train": y_train.contiguous().to(torch.float32)},
        str(out_dir / "inputs.safetensors"),
    )
    stage_tensors = {
        "col_embed": col_embed.contiguous(),
        "row_interact": row_interact.contiguous(),
        "icl_out": icl_out.contiguous(),
    }
    # Regression head stats from the raw quantiles, so the R port can validate
    # the full regression pipeline (X -> model -> quantiles -> stats).
    if config["max_classes"] == 0:
        with torch.no_grad():
            dist = model.quantile_dist(icl_out)
            stage_tensors["qd_mean"] = dist.quantiles.mean(dim=-1).contiguous()
            stage_tensors["qd_median"] = dist.icdf(torch.tensor(0.5)).contiguous()
            stage_tensors["qd_quantiles"] = dist.icdf(
                torch.tensor([0.1, 0.5, 0.9])
            ).contiguous()
    save_file(stage_tensors, str(out_dir / "stage_outputs.safetensors"))

    meta = {
        "kind": kind,
        "seed": SEED,
        "shapes": {
            "X": list(X.shape),
            "y_train": list(y_train.shape),
            "col_embed": list(col_embed.shape),
            "row_interact": list(row_interact.shape),
            "icl_out": list(icl_out.shape),
        },
        "dtypes": {
            "X": str(X.dtype),
            "y_train": str(y_train.dtype),
        },
        "config": config,
    }
    with open(out_dir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    print(f"seed = {SEED}")
    for k, v in meta["shapes"].items():
        print(f"  {k}: {v}")
    print(f"Golden fixtures in {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=["classifier", "regressor"])
    ap.parse_args()
    dump(ap.parse_args().kind)


if __name__ == "__main__":
    main()
