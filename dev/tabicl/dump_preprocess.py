"""Dump sklearn preprocessing golden fixtures for the brulee TabICL port.

The preprocessing pipeline (TransformToNumerical -> CustomStandardScaler ->
optional normalizer -> OutlierRemover) is reimplemented in R for fidelity. This
dumps the inputs, fitted parameters, and transformed outputs from the real
classes so the R port can be validated against them.

Covers the deterministic transforms exactly (standard scaler, outlier remover)
and the default normalization methods used by the ensemble: "none" and "power"
(Yeo-Johnson). The power transform's lambdas come from an optimizer, so the R
port matches it to optimizer tolerance rather than bit-for-bit.

Run after the venv is set up. Usage: python dump_preprocess.py
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save

from tabicl._sklearn.preprocessing import (
    CustomStandardScaler,
    OutlierRemover,
    PreprocessingPipeline,
)

OUT = Path(__file__).resolve().parents[2] / "tests" / "testthat" / "fixtures" / "tabicl"
SEED = int.from_bytes(b"brulee-tabicl-prep", "big") % (2**31 - 1)


def write(name: str, arrays: dict, meta: dict) -> None:
    tensors = {k: torch.from_numpy(np.ascontiguousarray(v)).float() for k, v in arrays.items()}
    blob = save(tensors)
    gz_path = OUT / f"{name}.safetensors.gz"
    with gzip.open(gz_path, "wb") as fh:
        fh.write(blob)
    with open(OUT / f"{name}.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    os.chmod(gz_path, 0o644)
    os.chmod(OUT / f"{name}.json", 0o644)
    print(f"{name}: { {k: list(v.shape) for k, v in tensors.items()} }")


def make_data(rng):
    # Fit on "train" rows; transform a separate set that includes an extreme
    # value to exercise the outlier clipping. Mixed scales across features.
    x_fit = rng.standard_normal((60, 4)) * np.array([1.0, 5.0, 0.2, 20.0]) + np.array([0.0, 3.0, -1.0, 10.0])
    x_apply = rng.standard_normal((12, 4)) * np.array([1.0, 5.0, 0.2, 20.0]) + np.array([0.0, 3.0, -1.0, 10.0])
    x_apply[0, 0] = 50.0  # outlier to trigger clipping
    return x_fit.astype(np.float64), x_apply.astype(np.float64)


def main():
    rng = np.random.default_rng(SEED)
    x_fit, x_apply = make_data(rng)

    # Standard scaler (exact).
    scaler = CustomStandardScaler()
    scaler.fit(x_fit)
    write(
        "prep_standard_scaler",
        {
            "x_fit": x_fit,
            "x_apply": x_apply,
            "mean": scaler.mean_,
            "scale": scaler.scale_,
            "out": scaler.transform(x_apply),
        },
        {"seed": SEED},
    )

    # Outlier remover (exact).
    remover = OutlierRemover(threshold=4.0)
    remover.fit(x_fit)
    write(
        "prep_outlier_remover",
        {
            "x_fit": x_fit,
            "x_apply": x_apply,
            "lower": remover.lower_bounds_,
            "upper": remover.upper_bounds_,
            "out": remover.transform(x_apply),
        },
        {"threshold": 4.0, "seed": SEED},
    )

    # Full pipelines for the default norm methods.
    for method in ("none", "power"):
        pipe = PreprocessingPipeline(normalization_method=method, outlier_threshold=4.0, random_state=SEED)
        pipe.fit(x_fit)
        arrays = {
            "x_fit": x_fit,
            "x_apply": x_apply,
            "out": pipe.transform(x_apply),
        }
        if method == "power":
            arrays["lambdas"] = pipe.normalizer_.lambdas_
            arrays["pt_mean"] = pipe.normalizer_._scaler.mean_
            arrays["pt_scale"] = pipe.normalizer_._scaler.scale_
        write(f"prep_pipeline_{method}", arrays, {"method": method, "seed": SEED})


if __name__ == "__main__":
    main()
