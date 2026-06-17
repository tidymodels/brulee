"""Convert a released TabICL .ckpt into safetensors + config.json.

The released checkpoints on HF Hub repo ``jingang/TabICL`` are ``torch.save``
dicts of the form ``{"config": <dict>, "state_dict": <OrderedDict>}``. Pickle is
awkward to read from R, so we do a one-time offline conversion: the ``state_dict``
becomes ``model.safetensors`` (readable via ``safetensors::safe_load_file``) and
the ``config`` becomes ``config.json``.

Usage:
    python convert_ckpt.py classifier   # or: regressor
    python convert_ckpt.py classifier --outdir /path/to/out

The model version and date are parsed from the checkpoint filename
(`tabicl-<kind>-<version>-<YYYYMMDD>.ckpt`), and outputs are organized by them:

Output (in <outdir>/<version>/<date>/<Classification|Regression>/):
    <task>.config.json        model hyperparameters passed to TabICL(**config)
    <task>.model.safetensors  state_dict
    state_dict_keys.txt       parameter names/shapes (for the R name map)
    manifest.json             provenance (repo, filename, version, date, sha256s)

The two files brulee loads (`config.json`, `model.safetensors`) are prefixed
with the task (`classification.` / `regression.`) so a single file is
self-identifying when moved on its own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

REPO_ID = "jingang/TabICL"
CKPT = {
    "classifier": "tabicl-classifier-v2-20260212.ckpt",
    "regressor": "tabicl-regressor-v2-20260212.ckpt",
}
# Directory label per task (used for the <Classification|Regression> subfolder).
TASK_LABEL = {"classifier": "Classification", "regressor": "Regression"}
# Filename prefix per task for the two files brulee reads.
TASK_PREFIX = {"classifier": "classification", "regressor": "regression"}


def checkpoint_files(kind: str) -> tuple[str, str]:
    """The (config, weights) filenames brulee reads for a task."""
    prefix = TASK_PREFIX[kind]
    return f"{prefix}.config.json", f"{prefix}.model.safetensors"


def default_artifacts_root() -> Path:
    return Path(__file__).resolve().parent / "artifacts"


def artifact_dir(kind: str, artifacts_root: Path | None = None) -> Path:
    """Locate the converted artifacts for a task, choosing the latest import date.

    Globs ``<root>/<version>/<date>/<TaskLabel>`` and returns the one with the
    most recent date. Used by the reader scripts so they need only the task name.
    """
    root = artifacts_root or default_artifacts_root()
    label = TASK_LABEL[kind]
    # The immediate parent of the task folder is the date (YYYY-MM-DD), which
    # sorts chronologically as a string.
    candidates = sorted(root.glob(f"*/*/{label}"), key=lambda p: p.parent.name)
    if not candidates:
        raise FileNotFoundError(
            f"No converted {label} checkpoint under {root}. "
            f"Run: python convert_ckpt.py {kind}"
        )
    return candidates[-1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_version_date(filename: str) -> tuple[str, str]:
    """Pull the version and import date out of a checkpoint filename.

    `tabicl-classifier-v2-20260212.ckpt` -> ("v2", "2026-02-12").
    """
    m = re.search(r"-(v[\w.]+)-(\d{4})(\d{2})(\d{2})\.ckpt$", filename)
    if not m:
        raise ValueError(
            f"Cannot parse version/date from filename {filename!r}; "
            "expected '...-<version>-<YYYYMMDD>.ckpt'."
        )
    version, year, month, day = m.groups()
    return version, f"{year}-{month}-{day}"


def convert(kind: str, outdir: Path) -> None:
    filename = CKPT[kind]
    version, date = _parse_version_date(filename)
    print(f"Downloading {filename} from {REPO_ID} ...")
    ckpt_path = Path(hf_hub_download(repo_id=REPO_ID, filename=filename))

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert "config" in checkpoint, "checkpoint missing 'config'"
    assert "state_dict" in checkpoint, "checkpoint missing 'state_dict'"

    config = checkpoint["config"]
    state_dict = checkpoint["state_dict"]

    # Organize by model version and import date: <version>/<date>/<TaskLabel>/.
    dest = outdir / version / date / TASK_LABEL[kind]
    dest.mkdir(parents=True, exist_ok=True)
    config_name, weights_name = checkpoint_files(kind)

    # config.json (task-prefixed)
    config_path = dest / config_name
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2, default=str)

    # state_dict -> safetensors (task-prefixed). save_file needs contiguous
    # tensors and cannot store shared storage, so clone each tensor defensively.
    tensors = {k: v.contiguous().clone() for k, v in state_dict.items()}
    st_path = dest / weights_name
    save_file(tensors, str(st_path))

    # parameter names, sorted, for building the R name map
    keys_path = dest / "state_dict_keys.txt"
    with open(keys_path, "w") as fh:
        for k in state_dict.keys():
            shape = tuple(state_dict[k].shape)
            dtype = str(state_dict[k].dtype).replace("torch.", "")
            fh.write(f"{k}\t{dtype}\t{list(shape)}\n")

    manifest = {
        "repo_id": REPO_ID,
        "filename": filename,
        "kind": kind,
        "task_label": TASK_LABEL[kind],
        "version": version,
        "date": date,
        "ckpt_sha256": _sha256(ckpt_path),
        "config_sha256": _sha256(config_path),
        "safetensors_sha256": _sha256(st_path),
        "num_tensors": len(tensors),
        "config": config,
    }
    with open(dest / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print(f"Wrote {len(tensors)} tensors to {st_path}")
    print(f"Config keys: {sorted(config.keys())}")
    print(f"Artifacts in {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=sorted(CKPT.keys()))
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    args = ap.parse_args()
    convert(args.kind, args.outdir)


if __name__ == "__main__":
    main()
