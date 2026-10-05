"""Build inst/tabpfn-registry.json: the TabPFN model versions brulee supports.

Combines Python tabpfn's `ModelSource` (repos and file names per version) with
the sha256, size, and architecture of each file, read from the local
checkpoint dumps (dev/tabpfn/checkpoints/, git-ignored; run dump_checkpoints.py
first), so nothing is copied by hand. Rerun after updating the pinned tabpfn release and
rerunning dump_checkpoints.py.

    dev/tabpfn/.venv/bin/python dev/tabpfn/make_registry.py
"""

import json
import subprocess
from pathlib import Path

import tabpfn
from tabpfn.constants import ModelVersion
from tabpfn.inference_config import cpu_sample_limit
from tabpfn.model_loading import ModelSource
from tabpfn.settings import settings

HERE = Path(__file__).parent
OUT = HERE.parents[1] / "inst" / "tabpfn-registry.json"
PYTHON_REPO = Path.home() / "github" / "TabPFN-python"

# Versions with an R architecture backend, and how Python finds their files.
SOURCES = {
    ModelVersion.V3: {
        "classification": ModelSource.get_classifier_v3,
        "regression": ModelSource.get_regressor_v3,
    },
    ModelVersion.V3_5: {
        "classification": ModelSource.get_v3_5,
        "regression": ModelSource.get_v3_5,
    },
    ModelVersion.V3_5_FAST: {
        "classification": ModelSource.get_v3_5_fast,
        "regression": ModelSource.get_v3_5_fast,
    },
}


def version_files(entry: dict) -> list[str]:
    return sorted({f for t in entry["tasks"].values() for f in t["files"]})


def version_limits(version: ModelVersion, configs: list[dict]) -> dict:
    """The data size limits of a version's checkpoints, and the CPU row limit
    Python applies in their place."""

    def common(key: str) -> int:
        values = {c[key] for c in configs}
        assert len(values) == 1, (version, key, values)
        return values.pop()

    return {
        "rows": common("MAX_NUMBER_OF_SAMPLES"),
        "rows_cpu": cpu_sample_limit(version),
        "predictors": common("MAX_NUMBER_OF_FEATURES"),
        # Regression checkpoints have 0 classes.
        "classes": max(c["MAX_NUMBER_OF_CLASSES"] for c in configs),
    }


def main() -> None:
    dumps = {
        p.name.removesuffix(".json"): json.loads(p.read_text())
        for p in (HERE / "checkpoints").glob("*.json")
    }
    tag = f"v{tabpfn.__version__}"
    commit = subprocess.run(
        ["git", "-C", str(PYTHON_REPO), "rev-parse", f"{tag}^{{commit}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    versions = {}
    files = {}
    for version, by_task in SOURCES.items():
        entry = {"tasks": {}}
        for task, get_source in by_task.items():
            source = get_source()
            entry["repo_id"] = source.repo_id
            entry["tasks"][task] = {
                "default": source.default_filename,
                "files": list(source.filenames),
            }
            for name in source.filenames:
                dump = dumps[name]
                files[name] = {
                    "sha256": dump["sha256"],
                    "size": dump["size"],
                    "format": Path(name).suffix.removeprefix("."),
                    "architecture": dump["metadata"]["architecture_name"],
                }
        archs = {
            files[f]["architecture"]
            for t in entry["tasks"].values()
            for f in t["files"]
        }
        assert len(archs) == 1, archs
        entry["architecture"] = archs.pop()
        entry["limits"] = version_limits(
            version,
            [dumps[f]["inference_config_with_defaults"] for f in version_files(entry)],
        )
        entry["license_repo"] = source.repo_id.split("/")[-1]
        versions[version.value] = entry

    registry = {
        "reference": {"tabpfn": tabpfn.__version__, "tag": tag, "commit": commit},
        "default_version": settings.tabpfn.model_version.value,
        "auth": {
            "gui_url": settings.tabpfn.auth_gui_url,
            "api_url": settings.tabpfn.auth_api_url,
        },
        "versions": versions,
        "files": dict(sorted(files.items())),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(registry, indent=2) + "\n")
    print(f"wrote {OUT} ({len(versions)} versions, {len(files)} files)")


if __name__ == "__main__":
    main()
