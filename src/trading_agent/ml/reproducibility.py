"""Stable dataset fingerprints and runtime identity for local experiments."""

import hashlib
import json
import platform
import subprocess
from collections.abc import Sequence
from importlib.metadata import version
from typing import Any

from trading_agent.ml.dataset import DatasetRow


def reproducibility(rows: Sequence[DatasetRow]) -> dict[str, Any]:
    payload = json.dumps([r.model_dump(mode="json") for r in rows], sort_keys=True, allow_nan=False)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", None
    return {
        "dataset_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "python_version": platform.python_version(),
        "git_commit": commit,
        "git_dirty": dirty,
        "dependency_versions": {
            name: version(name)
            for name in (
                "numpy",
                "scipy",
                "scikit-learn",
                "lightgbm",
                "joblib",
                "pydantic",
                "pyarrow",
            )
        },
    }
