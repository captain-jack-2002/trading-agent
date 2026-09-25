"""Supervised metrics with explicit undefined cases and fixed calibration bins."""

from typing import Any

import numpy as np
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y: list[int], probability: list[float]) -> dict[str, Any]:
    actual, scores = np.asarray(y), np.asarray(probability)
    if not y or len(y) != len(probability) or not set(y) <= {0, 1}:
        raise ValueError("aligned nonempty binary targets required")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("finite probabilities in [0, 1] required")
    prediction = (scores >= 0.5).astype(int)
    edges = np.linspace(0, 1, 6)
    counts = np.histogram(scores, bins=edges)[0].tolist()
    calibration = []
    for index in range(5):
        mask = (scores >= edges[index]) & (scores < edges[index + 1] if index < 4 else scores <= 1)
        if mask.any():
            calibration.append(
                {
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "count": int(mask.sum()),
                    "predicted": float(scores[mask].mean()),
                    "observed": float(actual[mask].mean()),
                }
            )
    return {
        "precision": float(precision_score(actual, prediction, zero_division=0)),
        "recall": float(recall_score(actual, prediction, zero_division=0)),
        "f1": float(f1_score(actual, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, scores)) if len(set(y)) == 2 else None,
        "pr_auc": float(average_precision_score(actual, scores)) if len(set(y)) == 2 else None,
        "pr_auc_definition": "average precision (step integral)",
        "confusion_matrix": confusion_matrix(actual, prediction, labels=[0, 1]).tolist(),
        "brier": float(brier_score_loss(actual, scores)),
        "calibration": calibration,
        "expected_calibration_error": sum(
            c["count"] * abs(c["predicted"] - c["observed"]) for c in calibration
        )
        / len(y),
        "class_distribution": {str(k): int((actual == k).sum()) for k in (0, 1)},
        "probability_distribution": {
            "edges": edges.tolist(),
            "counts": counts,
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "quantiles": np.quantile(scores, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
        },
    }


def regression_metrics(y: list[float], prediction: list[float]) -> dict[str, Any]:
    if not y or len(y) != len(prediction) or not np.isfinite([y, prediction]).all():
        raise ValueError("aligned nonempty finite regression targets required")
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)) if len(y) > 1 and len(set(y)) > 1 else None,
        "directional_accuracy": float(np.mean((np.asarray(y) > 0) == (np.asarray(prediction) > 0))),
    }
