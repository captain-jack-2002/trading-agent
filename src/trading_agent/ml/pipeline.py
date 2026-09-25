"""Deterministic sklearn baselines. All transformations fit training observations only."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
import sklearn  # type: ignore[import-untyped]
from sklearn.ensemble import (  # type: ignore[import-untyped]
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression, Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from trading_agent.features.research import FeatureRow
from trading_agent.ml.audit import SYNTHETIC_WARNING, audit_features
from trading_agent.ml.dataset import DatasetRow
from trading_agent.ml.metrics import classification_metrics, regression_metrics
from trading_agent.ml.reproducibility import reproducibility
from trading_agent.ml.split import Split


@dataclass
class ModelBundle:
    estimator: Any
    features: tuple[str, ...]
    metadata: dict[str, Any]


def _target(row: DatasetRow, field: str) -> float:
    if field == "future_return":
        return row.future_return
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
    started = perf_counter()
    audit = audit_features(rows)
    if model_kind not in ("logistic", "ridge", "random_forest", "lightgbm"):
        raise ValueError("unknown model_kind")
    if target_field not in ("target", "direction", "barrier", "future_return"):
        raise ValueError("invalid target field")
    regression = target_field == "future_return"
    if (model_kind == "logistic" and regression) or (model_kind == "ridge" and not regression):
        raise ValueError("algorithm incompatible with target task")
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
        partitions[0] = tuple(i for i in partitions[0] if eligible_barrier(rows[i]))
        if any(not p for p in partitions):
            raise ValueError("barrier filtering leaves empty partition")
    training = [rows[i] for i in partitions[0]]
    if not regression and len({_target(r, target_field) for r in training}) != 2:
        raise ValueError("training requires both binary classes")
    versions = {r.feature_version for r in rows}
    if len(versions) != 1:
        raise ValueError("mixed feature versions")
    features = tuple(sorted({f for r in training for f, v in r.values.items() if v is not None}))
    if not features:
        raise ValueError("no observed training features")
    classifier: Any
    if model_kind == "logistic":
        classifier = LogisticRegression(random_state=seed, max_iter=2000)
    elif model_kind == "ridge":
        classifier = Ridge(alpha=1.0)
    elif model_kind == "random_forest":
        cls = RandomForestRegressor if regression else RandomForestClassifier
        classifier = cls(
            n_estimators=100, max_depth=6, min_samples_leaf=2, random_state=seed, n_jobs=1
        )
    else:
        from lightgbm import LGBMClassifier, LGBMRegressor

        cls = LGBMRegressor if regression else LGBMClassifier
        classifier = cls(
            n_estimators=80,
            max_depth=4,
            num_leaves=15,
            min_child_samples=10,
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
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
        "schema_version": 2,
        "model_version": "phase4-v1",
        "algorithm": type(classifier).__name__,
        "task": "regression" if regression else "classification",
        "warning": SYNTHETIC_WARNING,
        "leakage_audit": audit,
        **reproducibility(rows),
        "model_id": str(uuid4()),
        "model_kind": model_kind,
        "seed": seed,
        "feature_version": next(iter(versions)),
        "features": list(features),
        "created_at": datetime.now(UTC).isoformat(),
        "provenance": sorted({r.provenance for r in rows}),
        "dataset_versions": sorted({r.dataset_version for r in rows}),
        "synthetic": True,
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
    metadata["preprocessing"] = {
        "imputer": "training median; all-null training columns excluded",
        "scaler": "StandardScaler; training only",
        "fit_range": metadata["ranges"]["train"],
        "imputer_statistics": estimator.named_steps["imputer"].statistics_.tolist(),
        "scaler_mean": estimator.named_steps["scale"].mean_.tolist(),
        "scaler_variance": estimator.named_steps["scale"].var_.tolist(),
        "excluded_features": sorted(set(training[0].values) - set(features)),
    }
    audit["split_indices_disjoint"] = True
    audit["label_horizons_purged"] = True
    audit["transformers_fit_training_only"] = True
    bundle = ModelBundle(estimator, features, metadata)
    metadata["metrics"] = {
        name: evaluate_model(bundle, [rows[i] for i in part])
        for name, part in zip(("validation", "test"), partitions[1:], strict=True)
    }
    metadata["training_runtime_seconds"] = perf_counter() - started
    return bundle


def evaluate_model(
    bundle: ModelBundle, rows: Sequence[DatasetRow], cost_bps: float = 0
) -> dict[str, Any]:
    audit_features(rows)
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
        retained = [r for r in rows if eligible_barrier(r)]
        excluded = len(rows) - len(retained)
        rows = retained
        if not rows:
            raise ValueError("no unambiguous reached barriers")
    if target_field == "future_return":
        metrics = regression_metrics([r.future_return for r in rows], predict_returns(bundle, rows))
    else:
        metrics = classification_metrics(
            [int(_target(r, target_field)) for r in rows], predict_probabilities(bundle, rows)
        )
    return {
        **metrics,
        "count": len(rows),
        "excluded_barrier_rows": excluded,
        "synthetic": True,
        "warning": SYNTHETIC_WARNING,
        "cost_bps": cost_bps,
        "cost_note": "Supervised metrics only; trading costs belong to risk-gated backtests",
    }


def eligible_barrier(row: DatasetRow) -> bool:
    return (
        row.barrier in ("upper", "lower")
        if row.label_version == "targets-v1"
        else row.barrier != "ambiguous"
    )


def predict_probabilities(bundle: ModelBundle, rows: Sequence[FeatureRow]) -> list[float]:
    """Probability of positive class; callers enforce independent deployment time policy."""
    if bundle.metadata.get("task") == "regression":
        raise ValueError("probabilities require a classification model")
    if not rows:
        return []
    if any(r.feature_version != bundle.metadata["feature_version"] for r in rows):
        raise ValueError("feature version mismatch")
    if any(not set(bundle.features).issubset(r.values) for r in rows):
        raise ValueError("feature schema mismatch")
    if any(v is not None and not np.isfinite(v) for r in rows for v in r.values.values()):
        raise ValueError("nonfinite features")
    return [float(p) for p in bundle.estimator.predict_proba(_matrix(rows, bundle.features))[:, 1]]


def predict_returns(bundle: ModelBundle, rows: Sequence[FeatureRow]) -> list[float]:
    if bundle.metadata.get("task") != "regression":
        raise ValueError("return prediction requires a regression model")
    if not rows:
        return []
    if any(r.feature_version != bundle.metadata["feature_version"] for r in rows):
        raise ValueError("feature version mismatch")
    if any(not set(bundle.features).issubset(r.values) for r in rows):
        raise ValueError("feature schema mismatch")
    if any(v is not None and not np.isfinite(v) for r in rows for v in r.values.values()):
        raise ValueError("nonfinite features")
    return [float(p) for p in bundle.estimator.predict(_matrix(rows, bundle.features))]
