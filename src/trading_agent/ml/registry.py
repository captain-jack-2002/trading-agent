"""Immutable local registry. Checksums detect corruption, not malicious publishers."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]

from trading_agent.ml.pipeline import ModelBundle


def save_model(bundle: ModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.parent / ("." + path.name + ".publish-lock")
    # Exclusive lock coordinates writers; staging is on the same filesystem.
    with lock.open("x"):
        staging: Path | None = None
        try:
            if path.exists():
                raise FileExistsError(path)
            staging = Path(tempfile.mkdtemp(prefix="." + path.name + "-", dir=path.parent))
            joblib.dump(bundle, staging / "model.joblib")
            digest = hashlib.sha256((staging / "model.joblib").read_bytes()).hexdigest()
            metadata = {**bundle.metadata, "sha256": digest}
            (staging / "metadata.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n"
            )
            if path.exists():
                raise FileExistsError(path)
            from trading_agent.ml.lifecycle import LifecycleRegistry

            registry = LifecycleRegistry(path.parent)
            published = False

            def cleanup_published() -> None:
                if published:
                    shutil.rmtree(path)

            # Enrollment cannot observe either a partial publication or failed cleanup.
            with registry.publication_gate(on_failure=cleanup_published) as connection:
                if path.exists():
                    raise FileExistsError(path)
                staging.rename(path)
                published = True
                registry._register(connection, path.name)
            bundle.registry_path = path.resolve()
        finally:
            if staging is not None and staging.exists():
                shutil.rmtree(staging)
            lock.unlink()


def load_model(path: Path, *, trusted: bool = False) -> ModelBundle:
    if not trusted:
        raise ValueError(
            "joblib executes code: explicit trusted=True required for your own local artifacts"
        )
    metadata = json.loads((path / "metadata.json").read_text())
    digest = hashlib.sha256((path / "model.joblib").read_bytes()).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError("model checksum mismatch")
    bundle = joblib.load(path / "model.joblib")
    if not isinstance(bundle, ModelBundle):
        raise ValueError("invalid model bundle")
    if bundle.metadata != {k: v for k, v in metadata.items() if k != "sha256"}:
        raise ValueError("metadata mismatch")
    bundle.registry_path = path.resolve()
    return bundle


def show_model(registry: Path, model_id: str) -> dict[str, Any]:
    if not model_id or Path(model_id).name != model_id or model_id in (".", ".."):
        raise ValueError("model_id must be a local registry identifier")
    path = registry / model_id
    if not path.resolve().is_relative_to(registry.resolve()):
        raise ValueError("model_id escapes registry")
    metadata = json.loads((path / "metadata.json").read_text())
    digest = hashlib.sha256((path / "model.joblib").read_bytes()).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError("model checksum mismatch")
    return {"metadata": metadata, "checksum_valid": True}


def list_models(registry: Path) -> dict[str, Any]:
    models = [json.loads(path.read_text()) for path in sorted(registry.glob("*/metadata.json"))]
    return {"models": models, "warning": "SYNTHETIC local research artifacts only"}
