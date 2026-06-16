"""Convert a released TabICL .ckpt into safetensors + config.json.

The released checkpoints on HF Hub repo ``jingang/TabICL`` are ``torch.save``
dicts of the form ``{"config": <dict>, "state_dict": <OrderedDict>}``. Pickle is
awkward to read from R, so we do a one-time offline conversion: the ``state_dict``
becomes ``model.safetensors`` (readable via ``safetensors::safe_load_file``) and
the ``config`` becomes ``config.json``.

Usage:
    python convert_ckpt.py classifier   # or: regressor
    python convert_ckpt.py classifier --outdir /path/to/out

Output (in <outdir>/<kind>/):
    config.json          model hyperparameters passed to TabICL(**config)
    model.safetensors     state_dict
    state_dict_keys.txt   newline-separated parameter names (for the R name map)
    manifest.json         provenance (repo, filename, sha256 of each artifact)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

REPO_ID = "jingang/TabICL"
CKPT = {
    "classifier": "tabicl-classifier-v2-20260212.ckpt",
    "regressor": "tabicl-regressor-v2-20260212.ckpt",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(kind: str, outdir: Path) -> None:
    filename = CKPT[kind]
    print(f"Downloading {filename} from {REPO_ID} ...")
    ckpt_path = Path(hf_hub_download(repo_id=REPO_ID, filename=filename))

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert "config" in checkpoint, "checkpoint missing 'config'"
    assert "state_dict" in checkpoint, "checkpoint missing 'state_dict'"

    config = checkpoint["config"]
    state_dict = checkpoint["state_dict"]

    dest = outdir / kind
    dest.mkdir(parents=True, exist_ok=True)

    # config.json
    config_path = dest / "config.json"
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2, default=str)

    # state_dict -> safetensors. save_file needs contiguous tensors and cannot
    # store shared storage, so clone each tensor defensively.
    tensors = {k: v.contiguous().clone() for k, v in state_dict.items()}
    st_path = dest / "model.safetensors"
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
