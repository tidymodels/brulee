"""Generate preprocessing/ensemble parity fixtures for the R port.

No Prior Labs weights or checkpoint contents are used. The script builds tiny
checkpoints with randomly initialised weights, using the architecture code of
the Python package (Apache-2.0) and inference configs written here, and saves
them in the formats of the released checkpoints (safetensors for v3.5, a
torch zip `.ckpt` for v3) to tests/testthat/fixtures/tabpfn/checkpoints/.

It then fits TabPFNClassifier and TabPFNRegressor with those checkpoints on
small mixed-type datasets and records:

* the data (`<name>.json`);
* each ensemble member's random choices: class permutation, constant-column
  mask, column order, categorical code permutations, column shuffle, and
  (regression) the fitted target transform;
* the inputs and outputs of every model forward call;
* the final predictions.

R tests inject the members' choices to check preprocessing against the
recorded model inputs, feed the recorded model outputs to the R
post-processing, and run tab_pfn() end to end on the same tiny checkpoints.

    dev/tabpfn/.venv/bin/python dev/tabpfn/make_pipeline_fixtures.py
"""

from __future__ import annotations

import dataclasses
import gzip
import io
import json
import math
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tabpfn
import torch
from safetensors.torch import save as safetensors_save
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.preprocessing.torch.steps import TorchShuffleFeaturesStep
from tabpfn.utils import transform_borders_one
from tabpfn.preprocessing.steps.add_svd_features_step import get_svd_n_components
from tabpfn.architectures import ARCHITECTURES
from tabpfn.checkpoint import save_as_safetensors

warnings.filterwarnings("ignore")
OUT = Path(__file__).parents[2] / "tests" / "testthat" / "fixtures" / "tabpfn"
CHECKPOINTS = OUT / "checkpoints"

TINY = {
    "embed_dim": 16,
    "dist_embed_num_blocks": 2,
    "dist_embed_num_heads": 2,
    "dist_embed_num_inducing_points": 4,
    "feat_agg_num_blocks": 2,
    "feat_agg_num_heads": 2,
    "feat_agg_num_cls_tokens": 2,
    "nlayers": 2,
    "icl_num_heads": 4,
    "icl_num_kv_heads_test": 1,
    "decoder_head_dim": 8,
    "decoder_num_heads": 2,
    "softmax_scaling_mlp_hidden_dim": 8,
    "num_buckets": 64,
}

# Inference configs for the tiny checkpoints. The values are chosen here to
# exercise the preprocessing the R port implements; field names and transform
# names are those of the Python package's `InferenceConfig`.
COMMON_INFERENCE = {
    "ENABLE_GPU_PREPROCESSING": True,
    "FEATURE_SUBSAMPLING_METHOD": "balanced",
    "MAX_NUMBER_OF_CLASSES": 10,
    "MAX_NUMBER_OF_FEATURES": 500,
    "MAX_CPU_SAMPLES": 5000,
}
INFERENCE_V3_5 = {
    **COMMON_INFERENCE,
    "PREPROCESS_TRANSFORMS": [
        {
            "name": "none",
            "categorical_name": "ordinal_shuffled",
            "append_original": False,
            "max_features_per_estimator": 500,
        }
    ],
    "N_ESTIMATORS": 4,
    "OUTLIER_REMOVAL_STD": 12.0,
    "SOFTMAX_TEMPERATURE": 1.0,
    "MAX_UNIQUE_FOR_CATEGORICAL_FEATURES": 1000000,
}
INFERENCE_V3_CLASSIFIER = {
    **COMMON_INFERENCE,
    "PREPROCESS_TRANSFORMS": [
        {
            "name": "squashing_scaler_default",
            "categorical_name": "ordinal_very_common_categories_shuffled",
            "global_transformer_name": "svd_quarter_components",
            "append_original": False,
            "max_features_per_estimator": 200,
        },
        {
            "name": "quantile_uni",
            "categorical_name": "numeric",
            "append_original": False,
            "max_features_per_estimator": 200,
        },
    ],
    "N_ESTIMATORS": "auto",
    "OUTLIER_REMOVAL_STD": "auto",
    "SOFTMAX_TEMPERATURE": 0.9,
}
INFERENCE_V3_REGRESSOR = {
    **COMMON_INFERENCE,
    "PREPROCESS_TRANSFORMS": [
        {
            "name": "squashing_scaler_max10",
            "categorical_name": "ordinal_very_common_categories_shuffled",
            "global_transformer_name": "svd_quarter_components",
            "append_original": False,
            "max_features_per_estimator": 500,
        },
        {
            "name": "quantile_uni_extrapolate",
            "categorical_name": "numeric",
            "append_original": "auto",
            "max_features_per_estimator": 500,
        },
    ],
    "N_ESTIMATORS": "auto",
    "OUTLIER_REMOVAL_STD": "auto",
    "SOFTMAX_TEMPERATURE": 0.9,
}


def tiny_checkpoint(architecture: str, config: dict, inference: dict, seed: int) -> dict:
    """A checkpoint dict, in the released layout, with random weights."""
    torch.manual_seed(seed)
    arch = ARCHITECTURES[architecture]
    parsed, unused = arch.parse_config(dict(config))
    assert not unused, unused
    model = arch.get_architecture(parsed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1 and "norm" in name and name.endswith("weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            else:
                fan_in = p.shape[-1] if p.dim() > 1 else 1
                p.copy_(torch.randn_like(p) / math.sqrt(fan_in))
    return {
        "state_dict": {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()},
        "config": dataclasses.asdict(parsed),
        "inference_config": inference,
        "architecture_name": architecture,
    }


def write_checkpoints() -> dict[str, Path]:
    """Write the tiny checkpoints, plain for Python and gzipped for R."""
    tmp = Path(tempfile.mkdtemp())
    paths = {}
    v35 = tiny_checkpoint(
        "tabpfn_v3_5",
        {
            **TINY,
            "fourier_encoding_num_frequencies": 4,
            "cell_ecdf_num_frequencies": 2,
            "max_num_classes": 10,
        },
        INFERENCE_V3_5,
        seed=40961,
    )
    paths["v3.5"] = tmp / "tabpfn-v3.5-tiny.safetensors"
    save_as_safetensors(v35, paths["v3.5"])
    for task, max_classes, inference, seed in [
        ("classifier", 10, INFERENCE_V3_CLASSIFIER, 17389),
        ("regressor", 0, INFERENCE_V3_REGRESSOR, 62801),
    ]:
        ck = tiny_checkpoint(
            "tabpfn_v3", {**TINY, "max_num_classes": max_classes}, inference, seed
        )
        paths[f"v3-{task}"] = tmp / f"tabpfn-v3-{task}-tiny.ckpt"
        torch.save(ck, paths[f"v3-{task}"])
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    for key, path in paths.items():
        with gzip.open(CHECKPOINTS / f"{path.name}.gz", "wb") as f:
            f.write(path.read_bytes())
    return paths


def make_data(seed: int, n_train: int, n_test: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    df = pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n) * 5 + 2,
            "const": np.full(n, 3.0),
            "cat": rng.choice(["b", "a", "c"], size=n),
            "int": rng.integers(0, 6, size=n).astype(float),
        }
    )
    df.loc[[2, 7], "num2"] = np.nan
    df.loc[5, "num1"] = 1e4  # an outlier for the soft clip
    df.loc[[3, 11], "cat"] = None
    df.loc[n_train + 1, "cat"] = "unseen"
    df.loc[n_train + 2, "num1"] = np.nan
    df.loc[9] = df.loc[8]  # a duplicate train row for fingerprint collisions
    df["cat"] = df["cat"].astype("category")
    return df


def member_record(member, n_cols_after_cpu: int, transforms: list) -> dict:
    steps = {type(s).__name__: s for s, _ in member.cpu_preprocessor.steps}
    rm = steps["RemoveConstantFeaturesStep"]
    reshape = steps["ReshapeFeatureDistributionsStep"]
    enc = steps["EncodeCategoricalFeaturesStep"]
    gpu_steps = {type(s).__name__: s for s, _ in member.gpu_preprocessor.steps}
    shuffle: TorchShuffleFeaturesStep = gpu_steps["TorchShuffleFeaturesStep"]
    config = member.config
    n_train, n_cols = member.X_train.shape
    svd_k = 0
    if config.preprocess_config.global_transformer_name and n_cols >= 2:
        svd_k = min(
            get_svd_n_components(
                config.preprocess_config.global_transformer_name,
                n_samples=n_train,
                n_features=n_cols,
            ),
            n_train,
            n_cols,
        )
    perm = shuffle._fit(torch.zeros(1, 1, n_cols + svd_k + 1))["permutation"]
    rec = {
        "preprocessor": transforms.index(config.preprocess_config),
        "features": None
        if member.feature_indices is None
        else [int(v) for v in member.feature_indices],
        "keep": [bool(v) for v in rm.sel_],
        "column_order": [int(v) for v in reshape.column_order_]
        if reshape.column_order_ is not None
        else None,
        "category_maps": {
            str(k): [int(i) for i in v]
        for k, v in (getattr(enc, "random_mappings_", None) or {}).items()
        },
        "shuffle": [int(v) for v in perm],
        "categorical": [
            i
            for i, f in enumerate(member.feature_schema.features)
            if f.modality.value == "categorical"
        ],
    }
    if hasattr(config, "class_permutation") and config.class_permutation is not None:
        rec["class_permutation"] = [int(v) for v in config.class_permutation]
    tt = getattr(config, "target_transform", None)
    if tt is not None:
        power = tt.named_steps["input_transformer"]
        scaler = tt.named_steps["standard"].named_steps["standard"]
        rec["target_transform"] = {
            "name": "safepower",
            "lambda": float(power.lambdas_[0]),
            "mean": float(scaler.mean_[0]),
            "scale": float(scaler.scale_[0]),
        }
    return rec


def record_calls(model: torch.nn.Module):
    calls = []

    def hook(module, args, kwargs, output):
        calls.append(
            {
                "x": args[0].detach().clone(),
                "y": args[1].detach().clone(),
                "task_type": kwargs.get("task_type"),
                "out": output.detach().clone(),
            }
        )

    return calls, model.register_forward_hook(hook, with_kwargs=True)


def run(
    name: str, estimator, df: pd.DataFrame, y, n_train: int, predict, checkpoint: Path
) -> None:
    X_train, X_test = df.iloc[:n_train], df.iloc[n_train:]
    estimator.fit(X_train, y[:n_train])
    calls, handle = record_calls(estimator.models_[0])
    preds = predict(estimator, X_test)
    handle.remove()

    members = estimator.executor_.ensemble_members
    n_cols = members[0].X_train.shape[1]
    transforms = list(estimator.inference_config_.PREPROCESS_TRANSFORMS)
    tensors = {}
    for i, m in enumerate(members):
        tensors[f"member.{i}.X_train"] = torch.from_numpy(np.ascontiguousarray(m.X_train))
        tensors[f"member.{i}.y_train"] = torch.from_numpy(np.asarray(m.y_train, dtype=np.float64))
    for i, c in enumerate(calls):
        for k in ("x", "y", "out"):
            tensors[f"call.{i}.{k}"] = c[k].contiguous()
    if isinstance(estimator, TabPFNRegressor):
        std_borders = estimator.znorm_space_bardist_.borders.detach().cpu()
        tensors["borders"] = std_borders.contiguous()
        for i, m in enumerate(members):
            if m.config.target_transform is not None:
                _, _, bt = transform_borders_one(
                    std_borders.numpy(),
                    m.config.target_transform,
                    repair_nan_borders_after_transform=True,
                )
                tensors[f"member.{i}.borders"] = torch.from_numpy(np.ascontiguousarray(bt))
    for k, v in preds.items():
        tensors[f"pred.{k}"] = torch.as_tensor(np.asarray(v, dtype=np.float64))

    record = {
        "name": name,
        "tabpfn_version": tabpfn.__version__,
        "n_train": n_train,
        "data": {
            col: [None if pd.isna(v) else (v if isinstance(v, str) else float(v)) for v in df[col]]
            for col in df.columns
        },
        "categorical_columns": [c for c in df.columns if str(df[c].dtype) == "category"],
        "y": [v if isinstance(v, str) else float(v) for v in y],
        "n_estimators": len(members),
        "members": [member_record(m, n_cols, transforms) for m in members],
        "preprocess_transforms": [
            {
                "name": t.name,
                "categorical_name": t.categorical_name,
                "append_original": t.append_original,
                "global_transformer_name": t.global_transformer_name,
                "max_features_per_estimator": t.max_features_per_estimator,
            }
            for t in transforms
        ],
        "outlier_std": estimator.inference_config_.get_resolved_outlier_removal_std(
            "regressor" if isinstance(estimator, TabPFNRegressor) else "classifier"
        )
        if hasattr(estimator.inference_config_, "get_resolved_outlier_removal_std")
        else None,
        "softmax_temperature": float(estimator.softmax_temperature_),
        "version": "v3" if checkpoint.name.startswith("tabpfn-v3-") else "v3.5",
        "checkpoint": checkpoint.name,
        "calls": [{"task_type": c["task_type"], "batch": int(c["x"].shape[1])} for c in calls],
    }
    if hasattr(estimator, "classes_") and isinstance(estimator, TabPFNClassifier):
        record["classes"] = [str(c) for c in estimator.classes_]
    if isinstance(estimator, TabPFNRegressor):
        record["y_train_mean"] = float(estimator.y_train_mean_)
        record["y_train_std"] = float(estimator.y_train_std_)
        record["quantiles"] = [0.1, 0.5, 0.9]
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT / f"{name}.safetensors.gz", "wb") as f:
        f.write(safetensors_save(tensors))
    (OUT / f"{name}.json").write_text(json.dumps(record, indent=1))
    print(name, len(members), "members,", len(calls), "calls", {k: np.shape(v) for k, v in preds.items()})


def make_data_v3(seed: int, n_train: int, n_test: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    df = pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n) * 5 + 2,
            "const": np.full(n, 3.0),
            "common": rng.choice(["b", "a", "c"], size=n),
            "rare": rng.choice(list("pqrstuvw"), size=n),
            "int": rng.integers(0, 6, size=n).astype(float),
            "skew": rng.exponential(size=n) ** 3,
        }
    )
    df.loc[[2, 7], "num2"] = np.nan
    df.loc[5, "num1"] = 1e4
    df.loc[[3, 11], "rare"] = None
    df.loc[n_train + 1, "common"] = "unseen"
    df.loc[9] = df.loc[8]
    for col in ("common", "rare"):
        df[col] = df[col].astype("category")
    return df


def predict_class(est, X):
    return {"prob": est.predict_proba(X)}


def predict_reg(est, X):
    full = est.predict(X, output_type="full", quantiles=[0.1, 0.5, 0.9])
    return {
        "mean": full["mean"],
        "median": full["median"],
        "quantiles": np.stack(full["quantiles"], axis=1),
    }


def main() -> None:
    ckpts = write_checkpoints()

    n_train, n_test = 40, 6
    df = make_data(73019, n_train, n_test)
    rng = np.random.default_rng(73019)
    y_cls = rng.choice(["yes", "no", "maybe"], size=n_train + n_test)
    clf = TabPFNClassifier(device="cpu", random_state=0, model_path=str(ckpts["v3.5"]))
    run("pipeline-classification", clf, df, y_cls, n_train, predict_class, ckpts["v3.5"])
    y_reg = df["num2"].fillna(0).to_numpy() * 3 + rng.normal(size=len(df)) + 20
    reg = TabPFNRegressor(
        device="cpu", n_estimators=2, random_state=0, model_path=str(ckpts["v3.5"])
    )
    run("pipeline-regression", reg, df, y_reg, n_train, predict_reg, ckpts["v3.5"])

    # v3: squashing/quantile transforms, SVD features, very-common categorical
    # encoding, appended originals, and two preprocessing configs.
    n_train, n_test = 60, 6
    df = make_data_v3(52817, n_train, n_test)
    rng = np.random.default_rng(52817)
    y_cls = rng.choice(["yes", "no"], size=n_train + n_test)
    clf = TabPFNClassifier(
        device="cpu", n_estimators=4, random_state=0, model_path=str(ckpts["v3-classifier"])
    )
    run(
        "pipeline-v3-classification",
        clf,
        df,
        y_cls,
        n_train,
        predict_class,
        ckpts["v3-classifier"],
    )
    y_reg = df["num1"].clip(-3, 3).to_numpy() * 4 + rng.normal(size=len(df)) + 10
    reg = TabPFNRegressor(
        device="cpu", n_estimators=4, random_state=0, model_path=str(ckpts["v3-regressor"])
    )
    run(
        "pipeline-v3-regression",
        reg,
        df,
        y_reg,
        n_train,
        predict_reg,
        ckpts["v3-regressor"],
    )


if __name__ == "__main__":
    main()
