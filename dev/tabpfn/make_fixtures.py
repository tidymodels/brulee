"""Generate architecture parity fixtures for the R port.

For each scenario, build a tiny, randomly initialised TabPFN model with the
upstream architecture code, run one forward pass, and record the inputs and
outputs of the first call of every submodule with forward hooks. The R tests
load the weights into the R modules and compare each module, and the whole
forward pass, against these recordings.

Each scenario writes two files to tests/testthat/fixtures/tabpfn/:

* `<name>.json`: scenario settings, model config, non-tensor call arguments,
  and the upstream tabpfn version.
* `<name>.safetensors.gz`: weights (`param.<key>`), model inputs (`input.*`),
  per-module recordings (`call.<module>.args.<i>`, `.kwargs.<name>`,
  `.out[.<i>...]`), and the model output (`output`). Gzipped so that R CMD
  check doesn't flag the binary header.

Run with the dev venv:
    dev/tabpfn/.venv/bin/python dev/tabpfn/make_fixtures.py
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import tabpfn
import torch
from safetensors.torch import save as safetensors_save
from tabpfn.architectures import ARCHITECTURES

OUT = Path(__file__).parents[2] / "tests" / "testthat" / "fixtures" / "tabpfn"

TINY_V3_5 = {
    "embed_dim": 16,
    "dist_embed_num_blocks": 2,
    "dist_embed_num_heads": 2,
    "dist_embed_num_inducing_points": 4,
    "feature_group_size": 3,
    "feat_agg_num_blocks": 2,
    "feat_agg_num_heads": 2,
    "feat_agg_num_cls_tokens": 2,
    "nlayers": 2,
    "icl_num_heads": 4,
    "icl_num_kv_heads_test": 1,
    "decoder_head_dim": 8,
    "decoder_num_heads": 2,
    "softmax_scaling_mlp_hidden_dim": 8,
    "fourier_encoding_num_frequencies": 4,
    "cell_ecdf_num_frequencies": 2,
    "max_num_classes": 10,
    "num_buckets": 32,
}

TINY_V3 = {
    "embed_dim": 16,
    "dist_embed_num_blocks": 2,
    "dist_embed_num_heads": 2,
    "dist_embed_num_inducing_points": 4,
    "feature_group_size": 3,
    "feat_agg_num_blocks": 2,
    "feat_agg_num_heads": 2,
    "feat_agg_num_cls_tokens": 2,
    "nlayers": 2,
    "icl_num_heads": 4,
    "icl_num_kv_heads_test": 1,
    "decoder_head_dim": 8,
    "decoder_num_heads": 2,
    "softmax_scaling_mlp_hidden_dim": 8,
    "num_buckets": 32,
}

SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "v3_5-multiclass",
        "architecture": "tabpfn_v3_5",
        "record": "all",
        "config": TINY_V3_5,
        "task_type": "multiclass",
        "n_train": 14,
        "n_test": 4,
        "n_features": 5,
        "n_batch": 1,
        "n_classes": 3,
        "seed": 30517,
    },
    {
        "name": "v3_5-regression",
        "architecture": "tabpfn_v3_5",
        "config": TINY_V3_5,
        "task_type": "regression",
        "n_train": 14,
        "n_test": 4,
        "n_features": 5,
        "n_batch": 1,
        "seed": 74203,
    },
    {
        # More train rows than ECDF buckets, so ranks are interpolated, and two
        # ensemble members batched along B.
        "name": "v3_5-multiclass-ecdf-batch",
        "architecture": "tabpfn_v3_5",
        "config": {**TINY_V3_5, "cell_ecdf_num_buckets": 4},
        "task_type": "multiclass",
        "n_train": 16,
        "n_test": 3,
        "n_features": 4,
        "n_batch": 2,
        "n_classes": 4,
        "seed": 58861,
    },
    {
        "name": "v3-multiclass",
        "architecture": "tabpfn_v3",
        "record": "all",
        "config": {**TINY_V3, "max_num_classes": 10},
        "task_type": "multiclass",
        "n_train": 14,
        "n_test": 4,
        "n_features": 5,
        "n_batch": 1,
        "n_classes": 3,
        "seed": 41939,
    },
    {
        "name": "v3-regression",
        "architecture": "tabpfn_v3",
        "config": {**TINY_V3, "max_num_classes": 0},
        "task_type": "regression",
        "n_train": 14,
        "n_test": 4,
        "n_features": 5,
        "n_batch": 1,
        "seed": 86477,
    },
]


def randomize_parameters(model: torch.nn.Module) -> None:
    """Replace every parameter with random values.

    Upstream initialisation zeroes some projections, which would make many
    recorded activations trivially zero and the parity tests weak.
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1 and "norm" in name and name.endswith("weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            else:
                fan_in = p.shape[-1] if p.dim() > 1 else 1
                p.copy_(torch.randn_like(p) / math.sqrt(fan_in))


def is_stage(name: str) -> bool:
    """Top-level stages: direct children of the model, and each ICL block."""
    parts = name.split(".")
    return len(parts) == 1 or (len(parts) == 2 and parts[0] == "icl_blocks")


def flatten(prefix: str, value: Any, tensors: dict, meta: dict) -> None:
    """Store tensors under `prefix`; record everything else in `meta`."""
    if isinstance(value, torch.Tensor):
        tensors[prefix] = value.detach().clone().contiguous()
    elif isinstance(value, (list, tuple)):
        meta[prefix] = {"type": type(value).__name__, "length": len(value)}
        for i, v in enumerate(value):
            flatten(f"{prefix}.{i}", v, tensors, meta)
    elif isinstance(value, dict):
        meta[prefix] = {"type": "dict", "keys": sorted(value)}
        for k, v in value.items():
            flatten(f"{prefix}.{k}", v, tensors, meta)
    elif value is None or isinstance(value, (bool, int, float, str)):
        meta[prefix] = value
    else:
        meta[prefix] = {"type": type(value).__name__}


def make_inputs(sc: dict, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    n_rows = sc["n_train"] + sc["n_test"]
    x = torch.randn(n_rows, sc["n_batch"], sc["n_features"], generator=gen)
    # Exercise the NaN / inf indicator paths and a few ties.
    x[1, :, 0] = float("nan")
    x[sc["n_train"] + 1, :, 1] = float("nan")
    x[3, :, 2] = float("inf")
    x[sc["n_train"] + 2, :, 2] = float("-inf")
    x[:, :, -1] = torch.round(x[:, :, -1])
    if sc["task_type"] == "multiclass":
        y = torch.arange(sc["n_train"]) % sc["n_classes"]
        y = y[torch.randperm(sc["n_train"], generator=gen)].float()
    else:
        y = torch.randn(sc["n_train"], generator=gen)
    if sc["n_batch"] > 1:
        y = y.unsqueeze(1).expand(-1, sc["n_batch"]).contiguous()
    return x, y


def run(sc: dict) -> None:
    torch.manual_seed(sc["seed"])
    gen = torch.Generator().manual_seed(sc["seed"])
    arch = ARCHITECTURES[sc["architecture"]]
    config, unused = arch.parse_config(dict(sc["config"]))
    assert not unused, unused
    model = arch.get_architecture(config)
    randomize_parameters(model)
    model.eval()

    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, Any] = {}
    calls: list[str] = []

    def hook(name: str):
        def fn(module, args, kwargs, output):
            if name in calls:
                return
            calls.append(name)
            flatten(f"call.{name}.args", list(args), tensors, meta)
            flatten(f"call.{name}.kwargs", dict(kwargs), tensors, meta)
            flatten(f"call.{name}.out", output, tensors, meta)

        return fn

    # Only TabPFN's own module classes; torch builtins (Linear, GELU, ...) are
    # tested by torch itself and would bloat the fixtures.
    handles = [
        m.register_forward_hook(hook(n), with_kwargs=True)
        for n, m in model.named_modules()
        if n
        and not type(m).__module__.startswith("torch.")
        and (sc.get("record") == "all" or is_stage(n))
    ]

    x, y = make_inputs(sc, gen)
    perf = model.get_default_performance_options()
    with torch.no_grad():
        if sc["architecture"] == "tabpfn_v3_5":
            output = model(x, y, sc["task_type"], performance_options=perf)
        else:
            output = model(x, y, task_type=sc["task_type"], performance_options=perf)
    for h in handles:
        h.remove()

    for k, v in model.state_dict().items():
        tensors[f"param.{k}"] = v.detach().clone().contiguous()
    tensors["input.x"] = x
    tensors["input.y"] = y
    tensors["output"] = output.detach().clone().contiguous()

    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT / f"{sc['name']}.safetensors.gz", "wb") as f:
        f.write(safetensors_save(tensors))
    record = {
        **{k: v for k, v in sc.items() if k != "config"},
        "config": json.loads(json.dumps(__import__("dataclasses").asdict(config))),
        "tabpfn_version": tabpfn.__version__,
        "torch_version": torch.__version__,
        "call_order": calls,
        "call_meta": meta,
        "performance_options": json.loads(
            json.dumps(__import__("dataclasses").asdict(perf), default=str)
        ),
    }
    (OUT / f"{sc['name']}.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    print(sc["name"], "output", tuple(output.shape), len(calls), "modules")


if __name__ == "__main__":
    for scenario in SCENARIOS:
        run(scenario)
