"""Deterministic sklearn baselines. All transformations fit training observations only."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import numpy as np
import sklearn  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from trading_agent.features.research import FeatureRow
from trading_agent.ml.dataset import DatasetRow
from trading_agent.ml.split import Split


@dataclass
class ModelBundle:
    estimator: Any
    features: tuple[str, ...]
    metadata: dict[str, Any]


def _target(row: DatasetRow, field: str) -> int:
    return (
        int(row.barrier == "upper")
        if field == "barrier"
        else row.direction
        if field == "direction"
        else row.target
    )


def _matrix(rows: Sequence[FeatureRow], features: tuple[str, ...]) -> Any:
    return np.array(
        [
            [r.values.get(f) if r.values.get(f) is not None else np.nan for f in features]
            for r in rows
        ],
        dtype=float,
    )


def train_model(
    rows: Sequence[DatasetRow],
    split: Split,
    model_kind: str = "logistic",
    seed: int = 42,
    target_field: str = "target",
) -> ModelBundle:
    if model_kind not in ("logistic", "random_forest"):
        raise ValueError("unknown model_kind")
    if target_field not in ("target", "direction", "barrier"):
        raise ValueError(
            "target_field must be target, direction or barrier; regression unsupported"
        )
    if (
        len(
            {
                (r.horizon, r.threshold, r.label_version, r.upper_barrier, r.lower_barrier)
                for r in rows
            }
        )
        != 1
    ):
        raise ValueError("mixed target definitions")
    if len({(r.instrument_id, r.timestamp) for r in rows}) != len(rows):
        raise ValueError("duplicate instrument timestamps")
    if any(a.timestamp > b.timestamp for a, b in zip(rows, rows[1:], strict=False)):
        raise ValueError("rows must be chronological")
    if any(r.label_end <= r.timestamp for r in rows):
        raise ValueError("invalid label horizon")
    if len({(r.dataset_version, r.provenance, r.synthetic) for r in rows}) != 1:
        raise ValueError("mixed dataset lineage; explicitly assemble a versioned dataset first")
    partitions = [split.train, split.validation, split.test]
    if any(isinstance(i, bool) or not isinstance(i, int) for p in partitions for i in p):
        raise ValueError("split indices must be integers")
    if any(tuple(sorted(p)) != p for p in partitions):
        raise ValueError("split indices must be chronological")
    if any(not p for p in partitions):
        raise ValueError("all partitions required")
    if len(set(i for p in partitions for i in p)) != sum(map(len, partitions)):
        raise ValueError("overlapping split indices")
    if any(i < 0 or i >= len(rows) for p in partitions for i in p):
        raise ValueError("invalid split indices")
    for left, right in zip(partitions, partitions[1:], strict=False):
        if max(rows[i].label_end for i in left) >= min(rows[i].timestamp for i in right):
            raise ValueError("overlapping label horizons")
    if target_field == "barrier":
        partitions = [
            tuple(i for i in part if rows[i].barrier in ("upper", "lower")) for part in partitions
        ]
        if any(not p for p in partitions):
            raise ValueError("barrier filtering leaves empty partition")
    training = [rows[i] for i in partitions[0]]
    if len({_target(r, target_field) for r in training}) != 2:
        raise ValueError("training requires both binary classes")
    versions = {r.feature_version for r in rows}
    if len(versions) != 1:
        raise ValueError("mixed feature versions")
    features = tuple(sorted({f for r in training for f, v in r.values.items() if v is not None}))
    if not features:
        raise ValueError("no observed training features")
    classifier = (
        LogisticRegression(random_state=seed, max_iter=2000)
        if model_kind == "logistic"
        else RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=2, random_state=seed, n_jobs=1
        )
    )
    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("classifier", classifier),
        ]
    )
    estimator.fit(_matrix(training, features), [_target(r, target_field) for r in training])
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "model_id": str(uuid4()),
        "model_kind": model_kind,
        "seed": seed,
        "feature_version": next(iter(versions)),
        "features": list(features),
        "created_at": datetime.now(UTC).isoformat(),
        "provenance": sorted({r.provenance for r in rows}),
        "dataset_versions": sorted({r.dataset_version for r in rows}),
        "synthetic": any(r.synthetic for r in rows),
        "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__},
        "hyperparameters": classifier.get_params(),
        "target": {
            "field": target_field,
            "upper_barrier": training[0].upper_barrier,
            "lower_barrier": training[0].lower_barrier,
            "horizon": training[0].horizon,
            "threshold": training[0].threshold,
            "label_version": training[0].label_version,
        },
        "ranges": {
            name: {
                "start": min(rows[i].timestamp for i in part).isoformat(),
                "end": max(rows[i].timestamp for i in part).isoformat(),
                "label_end": max(rows[i].label_end for i in part).isoformat(),
                "count": len(part),
            }
            for name, part in zip(("train", "validation", "test"), partitions, strict=True)
        },
    }
    bundle = ModelBundle(estimator, features, metadata)
    metadata["metrics"] = {
        name: evaluate_model(bundle, [rows[i] for i in part])
        for name, part in zip(("validation", "test"), partitions[1:], strict=True)
    }
    return bundle


def evaluate_model(
    bundle: ModelBundle, rows: Sequence[DatasetRow], cost_bps: float = 0
) -> dict[str, Any]:
    if not rows or not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("nonempty rows and nonnegative finite costs required")
    if any(r.feature_version != bundle.metadata["feature_version"] for r in rows):
        raise ValueError("feature version mismatch")
    target_definition = bundle.metadata["target"]
    if any(
        any(
            getattr(r, key) != target_definition[key]
            for key in ("horizon", "threshold", "label_version", "upper_barrier", "lower_barrier")
        )
        for r in rows
    ):
        raise ValueError("target definition mismatch")
    target_field = target_definition["field"]
    excluded = 0
    if target_field == "barrier":
        retained = [r for r in rows if r.barrier in ("upper", "lower")]
        excluded = len(rows) - len(retained)
        rows = retained
        if not rows:
            raise ValueError("no unambiguous reached barriers")
    y = np.array([_target(r, target_field) for r in rows])
    probability = np.array(predict_probabilities(bundle, rows))
    prediction = (probability >= 0.5).astype(int)
    calibration = []
    for low in (0.0, 0.2, 0.4, 0.6, 0.8):
        mask = (probability >= low) & (probability < (low + 0.2) if low < 0.8 else probability <= 1)
        if mask.any():
            calibration.append(
                {
                    "lower": low,
                    "count": int(mask.sum()),
                    "predicted": float(probability[mask].mean()),
                    "observed": float(y[mask].mean()),
                }
            )
    return {
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)) if len(set(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, probability)) if len(set(y)) == 2 else None,
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
        "brier": float(brier_score_loss(y, probability)),
        "calibration": calibration,
        "count": len(rows),
        "excluded_barrier_rows": excluded,
        "synthetic": any(r.synthetic for r in rows),
        "consequence_diagnostic": {
            "warning": "NOT A BACKTEST: overlapping horizons, no portfolio, fills or risk gates",
            "cost_bps": cost_bps,
            "mean_signal_future_return_after_cost": float(
                np.mean(
                    [
                        p * (r.future_return - cost_bps / 10000)
                        for p, r in zip(prediction, rows, strict=True)
                    ]
                )
            ),
        },
    }


def predict_probabilities(bundle: ModelBundle, rows: Sequence[FeatureRow]) -> list[float]:
    """Probability of positive class; callers enforce independent deployment time policy."""
    if not rows:
        return []
    if any(r.feature_version != bundle.metadata["feature_version"] for r in rows):
        raise ValueError("feature version mismatch")
    if any(not set(bundle.features).issubset(r.values) for r in rows):
        raise ValueError("feature schema mismatch")
    if any(v is not None and not np.isfinite(v) for r in rows for v in r.values.values()):
        raise ValueError("nonfinite features")
    return [float(p) for p in bundle.estimator.predict_proba(_matrix(rows, bundle.features))[:, 1]]
