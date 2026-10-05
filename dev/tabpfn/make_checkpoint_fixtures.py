"""Small checkpoint files for the R checkpoint-reader tests.

Writes gzipped files to tests/testthat/fixtures/tabpfn/checkpoints/:

* `tiny.ckpt.gz`: a torch.save() checkpoint shaped like the TabPFN v3 ones
  (nested dict with state_dict, config, inference_config, ...). One tensor is
  a strided view into another tensor's storage, to test offsets and strides.
* `tiny.safetensors.gz`: the same checkpoint written by upstream's
  `save_as_safetensors()`.
* `int64.ckpt.gz`: a checkpoint with an int64 tensor, which the R reader
  rejects because no TabPFN checkpoint uses that storage type.
* `tiny-values.json`: the expected tensor values and metadata.

    dev/tabpfn/.venv/bin/python dev/tabpfn/make_checkpoint_fixtures.py
"""

import gzip
import io
import json
import tempfile
from collections import OrderedDict
from pathlib import Path

import torch
from tabpfn.checkpoint import save_as_safetensors

OUT = Path(__file__).parents[2] / "tests" / "testthat" / "fixtures" / "tabpfn" / "checkpoints"


def gz_write(path: Path, data: bytes) -> None:
    with gzip.open(path, "wb") as f:
        f.write(data)


def torch_bytes(obj) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(27183)
    base = torch.randn(4, 6, generator=gen)
    state = OrderedDict(
        [
            ("layer.weight", base),
            # Shares `base`'s storage, with an offset and a non-unit stride.
            ("layer.view", base[1:3, ::2]),
            ("layer.bias", torch.randn(3, generator=gen)),
            ("scalar", torch.tensor(2.5)),
        ]
    )
    checkpoint = {
        "state_dict": state,
        "config": {"embed_dim": 16, "rope_base": 100000.0, "name": "tiny", "kv": None},
        "inference_config": {
            "PREPROCESS_TRANSFORMS": [{"name": "none", "append_original": False}],
            "N_ESTIMATORS": "auto",
        },
        "architecture_name": "tabpfn_tiny",
        "optimizer_state": None,
        "trained_epochs_until_now": 0,
    }
    gz_write(OUT / "tiny.ckpt.gz", torch_bytes(checkpoint))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tiny.safetensors"
        save_as_safetensors(
            {**checkpoint, "state_dict": {k: v.clone() for k, v in state.items()}},
            path,
        )
        gz_write(OUT / "tiny.safetensors.gz", path.read_bytes())

    bad = {**checkpoint, "state_dict": {"index": torch.arange(3)}}
    gz_write(OUT / "int64.ckpt.gz", torch_bytes(bad))

    values = {
        "tensors": {
            k: {"shape": list(v.shape), "values": v.flatten().tolist()}
            for k, v in state.items()
        },
        "config": checkpoint["config"],
        "inference_config": checkpoint["inference_config"],
        "architecture_name": checkpoint["architecture_name"],
    }
    (OUT / "tiny-values.json").write_text(json.dumps(values, indent=1))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
