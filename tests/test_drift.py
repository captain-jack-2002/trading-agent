import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from pydantic import ValidationError
from test_features_research import bars

from trading_agent.features.research import FeatureRow
from trading_agent.ml.dataset import build_dataset
from trading_agent.ml.drift import (
    DriftLimits,
    DriftThresholds,
    MonitoringWindow,
    RealizedLabel,
    build_monitoring_baseline,
    evaluate_drift,
    ks_statistic,
    population_stability_index,
    wasserstein_distance,
)
from trading_agent.ml.pipeline import train_model
from trading_agent.ml.split import walk_forward


def rows(values):
    return [
        FeatureRow(
            instrument_id="x",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=i),
            provenance="synthetic",
            values={"x": v},
        )
        for i, v in enumerate(values)
    ]


def metadata(values, predictions=None, labels=None, task="classification"):
    return {
        "model_id": "test",
        "feature_version": "research-v2",
        "features": ["x"],
        "task": task,
        "monitoring_baseline": build_monitoring_baseline(
            rows(values), ("x",), predictions=predictions, labels=labels, task=task
        ).model_dump(mode="json"),
    }


def test_statistical_definitions_and_zero_bins():
    assert population_stability_index([10, 0], [10, 0]) == 0
    assert np.isfinite(population_stability_index([10, 0], [0, 10]))
    assert population_stability_index([10, 0], [0, 10]) > 1
    assert ks_statistic([0, 1, 2], [0, 1, 2]) == 0
    assert ks_statistic([0, 1], [2, 3]) == 1
    assert wasserstein_distance([0, 1, 2], [3, 4, 5]) == pytest.approx(3)
    assert wasserstein_distance([0, 2], [1]) == pytest.approx(1)
    for function in (ks_statistic, wasserstein_distance):
        with pytest.raises(ValueError):
            function([], [1])
        with pytest.raises(ValueError):
            function([float("nan")], [1])
    with pytest.raises(ValueError):
        population_stability_index([0, 0], [1, 1])


def test_identical_shifted_and_configured_health():
    values = list(range(100))
    meta = metadata(values)
    assert evaluate_drift(meta, rows(values)).status == "healthy"
    result = evaluate_drift(meta, rows([x + 100 for x in values]))
    assert result.status == "quarantined"
    assert not result.allows_signals
    assert any(d.metric == "ks" and d.observed == 1 for d in result.diagnostics)
    high = DriftLimits(watch=1000, quarantine=2000)
    config = DriftThresholds(version="custom", psi=high, ks=high, wasserstein=high)
    healthy = evaluate_drift(meta, rows([x + 100 for x in values]), thresholds=config)
    assert healthy.status == "healthy"
    assert healthy.threshold_version == "custom"
    with pytest.raises(ValidationError):
        healthy.status = "quarantined"
    with pytest.raises(ValidationError):
        DriftLimits(watch=2, quarantine=1)


def test_watch_transition_uses_configured_limits():
    values = list(range(100))
    high = DriftLimits(watch=1000, quarantine=2000)
    config = DriftThresholds(psi=high, wasserstein=high, ks=DriftLimits(watch=0.05, quarantine=0.5))
    assert (
        evaluate_drift(metadata(values), rows([v + 10 for v in values]), thresholds=config).status
        == "watch"
    )


@pytest.mark.parametrize("window", [[], [None] * 50, [float("inf")] * 50])
def test_bad_data_fails_closed(window):
    # model_copy deliberately bypasses validation to exercise the monitoring boundary.
    bad = [rows([0])[0].model_copy(update={"values": {"x": x}}) for x in window]
    assert evaluate_drift(metadata(list(range(50))), bad).status == "quarantined"


def test_schema_missing_version_and_baseline_corruption():
    meta = metadata(list(range(50)))
    missing = rows([0])[0].model_copy(update={"values": {}})
    assert evaluate_drift(meta, [missing]).status == "quarantined"
    changed = rows([0])[0].model_copy(update={"feature_version": "other"})
    assert evaluate_drift(meta, [changed]).status == "quarantined"
    for bad in ({}, {**meta, "monitoring_baseline": {}}, {**meta, "features": ["other"]}):
        assert evaluate_drift(bad, rows([0])).status == "quarantined"


def test_null_drift_and_constant_distributions():
    meta = metadata([1.0] * 50)
    assert evaluate_drift(meta, rows([1.0] * 50)).status == "healthy"
    assert evaluate_drift(meta, rows([2.0] * 50)).status == "quarantined"
    result = evaluate_drift(meta, rows([None] * 30 + [1.0] * 20))
    assert result.status == "quarantined"
    assert any(d.metric == "null_rate" and d.observed == 0.6 for d in result.diagnostics)


def labeled_window(values, predictions, labels):
    observations = rows(values)
    return MonitoringWindow(
        rows=tuple(observations),
        predictions=tuple(predictions),
        labels=tuple(
            RealizedLabel(
                instrument_id=r.instrument_id,
                timestamp=r.timestamp,
                label_end=r.timestamp + timedelta(hours=1),
                value=y,
            )
            for r, y in zip(observations, labels, strict=True)
        ),
        evaluated_at=datetime(2026, 2, 1, tzinfo=UTC),
    )


def test_prediction_calibration_and_performance_drift():
    values = list(range(100))
    labels = [0, 1] * 50
    predictions = [0.05, 0.95] * 50
    meta = metadata(values, predictions, labels)
    assert evaluate_drift(meta, labeled_window(values, predictions, labels)).status == "healthy"
    result = evaluate_drift(meta, labeled_window(values, [1 - p for p in predictions], labels))
    metrics = {d.metric: d for d in result.diagnostics}
    assert result.status == "quarantined"
    assert metrics["brier_increase"].observed > 0.8
    assert metrics["f1_degradation"].observed == 1
    shifted = evaluate_drift(meta, rows(values), predictions=[0.99] * 100)
    assert any(
        d.category == "prediction" and d.severity == "quarantined" for d in shifted.diagnostics
    )
    assert evaluate_drift(meta, rows(values), predictions=[2.0] * 100).status == "quarantined"
    assert evaluate_drift(meta, rows(values), predictions=[0.5]).status == "quarantined"
    assert evaluate_drift(meta, rows(values), labels=labels).status == "quarantined"


def test_regression_monitoring():
    values = list(range(50))
    meta = metadata(values, values, values, task="regression")
    assert evaluate_drift(meta, labeled_window(values, values, values)).status == "healthy"
    result = evaluate_drift(meta, labeled_window(values, values, [v + 10 for v in values]))
    assert result.status == "quarantined"
    assert any(d.metric == "mae_increase" and d.observed == 10 for d in result.diagnostics)
    json.dumps(meta, allow_nan=False)


@pytest.mark.parametrize("kind,target", [("logistic", "target"), ("ridge", "future_return")])
def test_pipeline_baselines_use_only_training(kind, target):
    data = build_dataset(bars(150), horizon=2)
    split = walk_forward(data, 70, 25, 25)[0]
    original = train_model(data, split, model_kind=kind, target_field=target)
    altered = [
        r.model_copy(update={"values": {k: 1e12 for k in r.values}}) if i not in split.train else r
        for i, r in enumerate(data)
    ]
    changed = train_model(altered, split, model_kind=kind, target_field=target)
    base = original.metadata["monitoring_baseline"]
    assert base == changed.metadata["monitoring_baseline"]
    assert base["training_count"] == len(split.train)
    assert base["prediction"] is not None
    json.dumps(base, allow_nan=False)


def test_small_window_has_explicit_insufficient_status():
    result = evaluate_drift(metadata(list(range(100))), rows([1, 2]))
    assert result.status in ("watch", "quarantined")
    assert any(
        d.metric == "sample_count" and "insufficient" in d.reason.lower()
        for d in result.diagnostics
    )


def test_labels_must_be_realized_and_aligned():
    values = list(range(50))
    meta = metadata(values, [0.5] * 50, [0, 1] * 25)
    assert (
        evaluate_drift(meta, rows(values), predictions=[0.5] * 50, labels=[0, 1] * 25).status
        == "quarantined"
    )
    valid = labeled_window(values, [0.5] * 50, [0, 1] * 25)
    with pytest.raises(ValidationError, match="realized"):
        MonitoringWindow(**{**valid.model_dump(), "evaluated_at": valid.rows[0].timestamp})
    bad_label = valid.labels[0].model_copy(update={"instrument_id": "other"})
    with pytest.raises(ValidationError, match="aligned"):
        MonitoringWindow(**{**valid.model_dump(), "labels": (bad_label, *valid.labels[1:])})
    forged = valid.model_copy(update={"evaluated_at": valid.rows[0].timestamp})
    assert evaluate_drift(meta, forged).status == "quarantined"


def test_baseline_is_immutable_and_detects_corrupted_summary():
    meta = metadata([1, 2, 3, 4] * 15)
    baseline = build_monitoring_baseline(rows([1, 2, 3]), ("x",))
    with pytest.raises(ValidationError):
        baseline.training_count = 99
    assert isinstance(baseline.features, tuple)
    assert isinstance(baseline.features[0].distribution.values, tuple)
    meta["monitoring_baseline"]["features"][0]["distribution"]["mean"] = 100
    assert evaluate_drift(meta, rows([1, 2, 3, 4] * 15)).status == "quarantined"


def test_chronology_and_duplicate_identity_fail_closed():
    meta = metadata(list(range(50)))
    assert evaluate_drift(meta, rows(list(range(50)))[::-1]).status == "quarantined"
    r = rows([0])[0]
    assert evaluate_drift(meta, [r, r]).status == "quarantined"


def test_missing_realized_labels_are_reported_separately():
    result = evaluate_drift(
        metadata(list(range(50)), [0.5] * 50, [0, 1] * 25),
        rows(list(range(50))),
        predictions=[0.5] * 50,
    )
    assert result.status == "healthy"
    pending = [d for d in result.diagnostics if d.metric == "realized_labels"]
    assert {d.category for d in pending} == {"calibration", "performance"}
    assert all(d.observed is None for d in pending)


def test_invalid_probability_and_unaligned_baseline_rejected():
    with pytest.raises(ValueError):
        build_monitoring_baseline(rows([1, 2]), ("x",), predictions=[0.5])
    with pytest.raises(ValueError):
        build_monitoring_baseline(rows([1, 2]), ("x",), predictions=[0.0, 2.0])
    with pytest.raises(ValueError):
        build_monitoring_baseline(rows([None] * 50), ("x",))


def test_precision_recall_ece_and_regression_directional_degradation():
    values = list(range(50))
    meta = metadata(values, [0.1, 0.9] * 25, [0, 1] * 25)
    result = evaluate_drift(meta, labeled_window(values, [0.9, 0.1] * 25, [0, 1] * 25))
    diagnostics = {d.metric: d for d in result.diagnostics}
    assert diagnostics["precision_degradation"].observed == 1
    assert diagnostics["recall_degradation"].observed == 1
    assert diagnostics["ece_increase"].observed == pytest.approx(0.8)
    targets = [-1.0, 1.0] * 25
    regression = metadata(values, targets, targets, task="regression")
    result = evaluate_drift(regression, labeled_window(values, targets, [-x for x in targets]))
    assert any(
        d.metric == "directional_accuracy_degradation" and d.observed == 1
        for d in result.diagnostics
    )


def test_missing_labels_are_explicit_and_small_windows_watch():
    values = list(range(50))
    result = evaluate_drift(metadata(values), rows(values))
    assert {
        d.category for d in result.diagnostics if d.metric in ("unavailable", "realized_labels")
    } == {"prediction", "calibration", "performance"}
    assert evaluate_drift(metadata([1.0] * 50), rows([1.0])).status == "watch"


@pytest.mark.parametrize(
    "field,value",
    [
        ("counts", [0]),
        ("bin_counts", [1]),
        ("null_rate", 0.3),
        ("mean", 999),
        ("minimum", -1),
        ("quantiles", [100] * 5),
    ],
)
def test_corrupt_baseline_statistics_fail_closed(field, value):
    meta = metadata([1.0] * 50)
    meta["monitoring_baseline"]["features"][0]["distribution"][field] = value
    assert evaluate_drift(meta, rows([1.0] * 50)).status == "quarantined"


def test_corrupt_reference_metrics_fail_closed():
    meta = metadata(list(range(50)), [0.5] * 50, [0, 1] * 25)
    meta["monitoring_baseline"]["metrics"][0]["value"] = 100
    assert (
        evaluate_drift(
            meta, rows(list(range(50))), predictions=[0.5] * 50, labels=[0, 1] * 25
        ).status
        == "quarantined"
    )


def test_numerical_overflow_fails_closed():
    meta = metadata([0.0] * 50)
    result = evaluate_drift(meta, rows([-1e308, 1e308] * 25))
    assert result.status == "quarantined"


def test_baseline_requires_training_schema_and_alignment():
    for kwargs in ({"predictions": [0.2]}, {"labels": [0.0] * 50}, {"predictions": [1.5] * 50}):
        with pytest.raises(ValueError):
            build_monitoring_baseline(rows([0.0] * 50), ("x",), **kwargs)
    with pytest.raises(ValueError):
        build_monitoring_baseline(rows([0.0] * 50), ("missing",))
    with pytest.raises(ValueError):
        build_monitoring_baseline([], ("x",))
    with pytest.raises(ValueError):
        build_monitoring_baseline(rows([None] * 50), ("x",))


def test_psi_is_sample_size_invariant_and_validates_inputs():
    assert population_stability_index([1, 2, 3], [10, 20, 30]) == pytest.approx(0)
    for left, right in (([-1, 2], [1, 2]), ([1], [1, 2]), ([1, float("inf")], [1, 2])):
        with pytest.raises(ValueError):
            population_stability_index(left, right)
    with pytest.raises(ValueError):
        population_stability_index([1], [1], epsilon=0)


def test_unexpected_feature_columns_fail_closed():
    observations = rows(list(range(50)))
    observations[-1] = observations[-1].model_copy(update={"values": {"x": 49, "future_target": 1}})
    result = evaluate_drift(metadata(list(range(50))), observations)
    assert result.status == "quarantined"
    assert any(d.metric == "schema_unexpected" for d in result.diagnostics)


@pytest.mark.parametrize(
    "preprocessing", [None, {"excluded_features": [{}]}, {"excluded_features": "bad"}]
)
def test_corrupt_training_schema_metadata_returns_quarantined(preprocessing):
    meta = metadata(list(range(50)))
    meta["preprocessing"] = preprocessing
    assert evaluate_drift(meta, rows(list(range(50)))).status == "quarantined"
