"""Authoritative end-to-end prediction goldens from the real sklearn wrappers.

Runs TabICLClassifier / TabICLRegressor with n_estimators=1 (a single "none"
normalization member with identity feature/class shuffles -- the only fully
deterministic, RNG-independent ensemble configuration) so the R port can be
validated end-to-end: preprocessing -> model -> logit/quantile -> probabilities
/ mean. Uses the locally cached released checkpoint.

Output: tests/testthat/fixtures/tabicl/predict_{clf,reg}.safetensors.gz + .json
holding X_train, y_train, X_test, and the reference probabilities / mean.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save

from tabicl import TabICLClassifier, TabICLRegressor

OUT = Path(__file__).resolve().parents[2] / "tests" / "testthat" / "fixtures" / "tabicl"
SEED = int.from_bytes(b"brulee-tabicl-predict", "big") % (2**31 - 1)


def write(name, arrays, meta):
    tensors = {k: torch.from_numpy(np.ascontiguousarray(v)).float() for k, v in arrays.items()}
    blob = save(tensors)
    gz = OUT / f"{name}.safetensors.gz"
    with gzip.open(gz, "wb") as fh:
        fh.write(blob)
    with open(OUT / f"{name}.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    os.chmod(gz, 0o644)
    os.chmod(OUT / f"{name}.json", 0o644)
    print(f"{name}: { {k: list(v.shape) for k, v in tensors.items()} }")


def main():
    rng = np.random.default_rng(SEED)
    n_train, n_test, n_feat = 48, 10, 4

    # Classification: 3 well-separated-ish classes.
    Xtr = rng.standard_normal((n_train, n_feat)) * np.array([1.0, 4.0, 0.5, 12.0])
    ytr = rng.integers(0, 3, n_train)
    Xte = rng.standard_normal((n_test, n_feat)) * np.array([1.0, 4.0, 0.5, 12.0])

    clf = TabICLClassifier(n_estimators=1, device="cpu", random_state=SEED)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    write(
        "predict_clf",
        {"X_train": Xtr, "y_train": ytr.astype(np.float64), "X_test": Xte, "proba": proba},
        {"n_classes": 3, "classes": sorted(set(int(v) for v in ytr)), "seed": SEED},
    )

    # Regression.
    Xtr_r = rng.standard_normal((n_train, n_feat)) * np.array([1.0, 4.0, 0.5, 12.0])
    ytr_r = rng.standard_normal(n_train) * 2.0 + 1.0
    Xte_r = rng.standard_normal((n_test, n_feat)) * np.array([1.0, 4.0, 0.5, 12.0])

    reg = TabICLRegressor(n_estimators=1, device="cpu", random_state=SEED)
    reg.fit(Xtr_r, ytr_r)
    mean = reg.predict(Xte_r, output_type="mean")
    write(
        "predict_reg",
        {"X_train": Xtr_r, "y_train": ytr_r, "X_test": Xte_r, "mean": np.asarray(mean)},
        {"seed": SEED},
    )


if __name__ == "__main__":
    main()
