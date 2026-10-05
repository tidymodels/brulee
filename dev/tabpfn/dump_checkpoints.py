"""Dump the non-weight contents of every TabPFN v3 / v3.5 checkpoint.

Writes one JSON file per checkpoint to dev/tabpfn/checkpoints/ with the
checkpoint's metadata fields (config, inference_config, architecture_name, ...)
and the name, shape, and dtype of every state-dict tensor. No weights are
written, but the metadata is part of the licensed checkpoints, so the
directory is git-ignored and stays local: it is used to check the R port's
config parsing against new checkpoints and to build inst/tabpfn-registry.json.

Run with the dev venv (tabpfn pinned to the reference release):
    dev/tabpfn/.venv/bin/python dev/tabpfn/dump_checkpoints.py
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import tabpfn
from tabpfn.architectures import ARCHITECTURES
from tabpfn.checkpoint import Checkpoint
from tabpfn.inference_config import InferenceConfig
from tabpfn.model_loading import _rename_old_inference_config_keys, get_cache_dir

OUT = Path(__file__).parent / "checkpoints"
PATTERNS = ["tabpfn-v3-*.ckpt", "tabpfn-v3.5*.safetensors"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cache = Path(get_cache_dir())
    paths = sorted(p for pat in PATTERNS for p in cache.glob(pat))
    for path in paths:
        ckpt = Checkpoint(path).load()
        state = ckpt.pop("state_dict")
        # The checkpoint's inference config with InferenceConfig's class defaults
        # filled in for absent keys, as Python resolves it at load time.
        resolved = InferenceConfig(
            **_rename_old_inference_config_keys(ckpt["inference_config"])
        )
        parsed, unused = ARCHITECTURES[ckpt["architecture_name"]].parse_config(
            ckpt["config"]
        )
        record = {
            "config_parsed": json.loads(
                json.dumps(dataclasses.asdict(parsed), default=str)
            ),
            "config_unused_keys": sorted(unused),
            "inference_config_with_defaults": json.loads(
                json.dumps(dataclasses.asdict(resolved), default=str)
            ),
            "file": path.name,
            "sha256": sha256(path),
            "size": path.stat().st_size,
            "tabpfn_version": tabpfn.__version__,
            "metadata": json.loads(json.dumps(ckpt, default=str)),
            "tensors": {
                k: {"shape": list(v.shape), "dtype": str(v.dtype).removeprefix("torch.")}
                for k, v in state.items()
            },
        }
        (OUT / f"{path.name}.json").write_text(json.dumps(record, indent=1, sort_keys=True))
        print(path.name, len(state), "tensors")


if __name__ == "__main__":
    main()
