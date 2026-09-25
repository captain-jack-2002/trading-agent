"""Reproducible local training runs with immutable, machine-readable evidence."""

from collections.abc import Sequence
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from trading_agent.data.storage import checksum
from trading_agent.ml.audit import SYNTHETIC_WARNING, verify_dataset_rebuild
from trading_agent.ml.dataset import DatasetRow
from trading_agent.ml.pipeline import (
    ModelBundle,
    eligible_barrier,
    predict_probabilities,
    predict_returns,
    train_model,
)
from trading_agent.ml.registry import save_model
from trading_agent.ml.split import chronological_split, walk_forward
from trading_agent.research_io import write_json


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    algorithm: Literal["logistic", "ridge", "random_forest", "lightgbm", "all"] = "logistic"
    seed: StrictInt = Field(default=42, ge=0, le=2**32 - 1)
    train_size: StrictInt = Field(gt=0)
    validation_size: StrictInt = Field(gt=0)
    test_size: StrictInt = Field(gt=0)
    mode: Literal["expanding", "rolling", "holdout"] = "expanding"
    step: StrictInt | None = Field(default=None, gt=0)


def prediction_records(bundle: ModelBundle, rows: Sequence[DatasetRow]) -> list[dict[str, Any]]:
    regression = bundle.metadata["task"] == "regression"
    values = predict_returns(bundle, rows) if regression else predict_probabilities(bundle, rows)
    field = bundle.metadata["target"]["field"]
    return [
        {
            "instrument_id": r.instrument_id,
            "timestamp": r.timestamp.isoformat(),
            "label_end": r.label_end.isoformat(),
            "synthetic": True,
            "model_id": bundle.metadata["model_id"],
            "prediction_kind": "future_return" if regression else "positive_probability",
            "prediction": value,
            "label": r.future_return
            if regression
            else int(r.barrier == "upper")
            if field == "barrier"
            else getattr(r, field),
            "metric_eligible": field != "barrier" or eligible_barrier(r),
        }
        for r, value in zip(rows, values, strict=True)
    ]


def aggregate_metrics(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Unweighted fold summaries; overlapping training sets are not independent trials."""
    keys = (
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier",
        "expected_calibration_error",
        "mae",
        "rmse",
        "r2",
        "directional_accuracy",
    )
    result = {}
    for key in keys:
        values = [float(r[key]) for r in reports if r.get(key) is not None]
        if values:
            result[key] = {
                "mean": mean(values),
                "std": pstdev(values),
                "min": min(values),
                "max": max(values),
                "folds": len(values),
            }
    return result


def run_training(
    rows: Sequence[DatasetRow],
    target_field: str,
    dataset: Path,
    registry: Path,
    config: TrainingConfig,
) -> dict[str, Any]:
    verify_dataset_rebuild(dataset, list(rows))
    if config.mode == "holdout":
        splits = [chronological_split(rows, config.train_size, config.validation_size)]
        if len({rows[i].timestamp for i in splits[0].test}) != config.test_size:
            raise ValueError("holdout test-size must equal all remaining timestamp groups")
    else:
        splits = walk_forward(
            rows,
            config.train_size,
            config.validation_size,
            config.test_size,
            config.step,
            expanding=config.mode == "expanding",
        )
    if not splits:
        raise ValueError("no nonempty purged folds: reduce windows or supply more observations")
    algorithms = (
        ["ridge" if target_field == "future_return" else "logistic", "random_forest", "lightgbm"]
        if config.algorithm == "all"
        else [config.algorithm]
    )
    run = registry / "runs" / f"SYNTHETIC-{uuid4().hex}"
    envelope = {"synthetic": True, "warning": SYNTHETIC_WARNING}
    manifest = {
        **envelope,
        **config.model_dump(mode="json"),
        "target_field": target_field,
        "dataset_artifact_sha256": checksum(dataset),
        "dataset": str(dataset),
        "probability_threshold": 0.5,
        "regression_threshold": 0,
        "selection_policy": "fixed parameters; no hyperparameter or return search",
    }
    write_json(run / "training_config.json", manifest)
    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    paths: list[str] = []
    for algorithm in algorithms:
        for number, split in enumerate(splits):
            bundle = train_model(
                rows, split, model_kind=algorithm, seed=config.seed, target_field=target_field
            )
            bundle.metadata["dataset_artifact_sha256"] = manifest["dataset_artifact_sha256"]
            bundle.metadata["evaluation_config"] = config.model_dump(mode="json")
            bundle.metadata["leakage_audit"]["source_rebuild_verified"] = True
            path = registry / str(bundle.metadata["model_id"])
            save_model(bundle, path)
            paths.append(str(path))
            records.append(
                {
                    "model_id": bundle.metadata["model_id"],
                    "algorithm": algorithm,
                    "fold": number,
                    "ranges": bundle.metadata["ranges"],
                    "metrics": bundle.metadata["metrics"],
                    "runtime_seconds": bundle.metadata["training_runtime_seconds"],
                    "audit": bundle.metadata["leakage_audit"],
                }
            )
            for partition, indices in (("validation", split.validation), ("test", split.test)):
                predictions.extend(
                    {**r, "partition": partition, "fold": number}
                    for r in prediction_records(bundle, [rows[i] for i in indices])
                )
    aggregates = {
        algorithm: aggregate_metrics(
            [r["metrics"]["test"] for r in records if r["algorithm"] == algorithm]
        )
        for algorithm in algorithms
    }
    write_json(run / "metrics.json", {**envelope, "models": records, "aggregate": aggregates})
    write_json(
        run / "walk_forward.json",
        {
            **envelope,
            "mode": config.mode,
            "folds": records,
            "aggregate": aggregates,
            "note": "Fold means, not independent market evidence",
        },
    )
    write_json(run / "predictions.json", {**envelope, "predictions": predictions})
    write_json(run / "leakage_audit.json", {**envelope, "audits": [r["audit"] for r in records]})
    write_json(
        run / "checksums.json",
        {**envelope, "sha256": {p.name: checksum(p) for p in sorted(run.glob("*.json"))}},
    )
    return {**envelope, "models": paths, "folds": len(splits), "artifacts": str(run)}
