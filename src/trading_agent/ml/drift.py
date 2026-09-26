"""Deterministic training-reference drift diagnostics, never deployment authority."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator

from trading_agent.features.research import FeatureRow
from trading_agent.ml.dataset import DatasetRow
from trading_agent.ml.metrics import classification_metrics, regression_metrics

HealthStatus = Literal["healthy", "watch", "quarantined"]
Task = Literal["classification", "regression"]


class FrozenContract(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, revalidate_instances="always"
    )


class RealizedLabel(FrozenContract):
    """A supplied outcome with the full observation horizon; never a forecast label."""

    instrument_id: str = Field(min_length=1)
    timestamp: AwareDatetime
    label_end: AwareDatetime
    value: float

    @model_validator(mode="after")
    def valid_horizon(self) -> Self:
        if self.label_end <= self.timestamp:
            raise ValueError("label_end must follow prediction timestamp")
        return self


class MonitoringWindow(FrozenContract):
    rows: tuple[DatasetRow | FeatureRow, ...]
    predictions: tuple[float, ...] | None = None
    labels: tuple[RealizedLabel, ...] | None = None
    evaluated_at: AwareDatetime

    @model_validator(mode="after")
    def valid_window(self) -> Self:
        # Revalidate external model instances: model_copy and mutable values can bypass validators.
        for row in self.rows:
            type(row).model_validate(row.model_dump())
        identities = [(r.instrument_id, r.timestamp) for r in self.rows]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate monitoring row identities")
        if any(a.timestamp > b.timestamp for a, b in zip(self.rows, self.rows[1:], strict=False)):
            raise ValueError("monitoring rows must be chronological")
        if any(r.timestamp > self.evaluated_at for r in self.rows):
            raise ValueError("window contains observations not yet realized at evaluated_at")
        if self.predictions is not None and len(self.predictions) != len(self.rows):
            raise ValueError("predictions must be aligned with monitoring rows")
        if self.labels is not None:
            if self.predictions is None or len(self.labels) != len(self.rows):
                raise ValueError("labels require aligned predictions and rows")
            for row, label in zip(self.rows, self.labels, strict=True):
                if (row.instrument_id, row.timestamp) != (label.instrument_id, label.timestamp):
                    raise ValueError("realized labels must be aligned by identity and timestamp")
                if label.label_end > self.evaluated_at:
                    raise ValueError("labels are not realized at evaluated_at")
        return self


class DriftLimits(FrozenContract):
    watch: float = Field(ge=0)
    quarantine: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.watch >= self.quarantine:
            raise ValueError("watch limit must be below quarantine limit")
        return self


class DriftThresholds(FrozenContract):
    """Illustrative policy defaults, not universal statistical or finance standards."""

    version: str = Field(default="drift-thresholds-v1", min_length=1)
    psi: DriftLimits = DriftLimits(watch=0.1, quarantine=0.25)
    ks: DriftLimits = DriftLimits(watch=0.15, quarantine=0.3)
    wasserstein: DriftLimits = DriftLimits(watch=0.5, quarantine=1.0)
    null_rate: DriftLimits = DriftLimits(watch=0.1, quarantine=0.3)
    brier: DriftLimits = DriftLimits(watch=0.05, quarantine=0.15)
    calibration: DriftLimits = DriftLimits(watch=0.05, quarantine=0.15)
    performance: DriftLimits = DriftLimits(watch=0.1, quarantine=0.25)
    regression_error: DriftLimits = DriftLimits(watch=0.1, quarantine=0.5)
    minimum_samples: int = Field(default=20, ge=1, strict=True)


class DriftDiagnostic(FrozenContract):
    category: Literal["feature", "prediction", "calibration", "performance", "data_quality"]
    metric: str
    feature: str | None = None
    observed: float | None = None
    watch_limit: float | None = None
    quarantine_limit: float | None = None
    severity: HealthStatus
    reason: str


class ModelHealth(FrozenContract):
    status: HealthStatus
    diagnostics: tuple[DriftDiagnostic, ...]
    threshold_version: str
    baseline_version: str | None = None
    model_id: str | None = None
    feature_version: str | None = None
    window_count: int = Field(ge=0)

    @property
    def allows_signals(self) -> bool:
        return self.status != "quarantined"


class DistributionBaseline(FrozenContract):
    count: int = Field(ge=1, strict=True)
    null_count: int = Field(ge=0, strict=True)
    null_rate: float = Field(ge=0, le=1)
    mean: float
    std: float = Field(ge=0)
    minimum: float
    maximum: float
    quantiles: tuple[float, ...]
    # Finite cut points define bins (-inf, first), [first, second), ..., [last, inf).
    bin_edges: tuple[float, ...]
    bin_counts: tuple[int, ...]
    # Lossless empirical distribution, compressed as unique value / count pairs.
    values: tuple[float, ...]
    counts: tuple[int, ...]

    @property
    def reference_values(self) -> tuple[float, ...]:
        """Unique sorted values of the lossless empirical reference (counts are weights)."""
        return self.values

    @model_validator(mode="after")
    def consistent(self) -> Self:
        finite_count = self.count - self.null_count
        if finite_count <= 0 or abs(self.null_rate - self.null_count / self.count) > 1e-12:
            raise ValueError("invalid null counts")
        if (
            not self.values
            or len(self.values) != len(self.counts)
            or any(c <= 0 for c in self.counts)
            or sum(self.counts) != finite_count
            or tuple(sorted(set(self.values))) != self.values
        ):
            raise ValueError("invalid empirical distribution")
        if (
            not self.bin_edges
            or tuple(sorted(set(self.bin_edges))) != self.bin_edges
            or len(self.bin_counts) != len(self.bin_edges) + 1
            or any(c < 0 for c in self.bin_counts)
            or sum(self.bin_counts) != finite_count
        ):
            raise ValueError("invalid fixed histogram")
        if self.minimum != self.values[0] or self.maximum != self.values[-1]:
            raise ValueError("invalid extrema")
        weights = np.asarray(self.counts) / finite_count
        mean = float(np.dot(self.values, weights))
        std = float(np.sqrt(np.dot((np.asarray(self.values) - mean) ** 2, weights)))
        expected = np.bincount(
            np.searchsorted(self.bin_edges, self.values, side="right"),
            weights=self.counts,
            minlength=len(self.bin_edges) + 1,
        )
        if (
            not np.allclose(expected, self.bin_counts, rtol=0, atol=0)
            or not np.isclose(mean, self.mean, rtol=1e-12, atol=1e-12)
            or not np.isclose(std, self.std, rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("inconsistent distribution statistics")
        # Check the stored quantiles against exact empirical order statistics without expanding
        # repeated samples. All baseline summaries are derived from the same training reference.
        expected_quantiles = _weighted_quantiles(
            self.values, self.counts, (0.05, 0.25, 0.5, 0.75, 0.95)
        )
        if len(self.quantiles) != 5 or not np.allclose(expected_quantiles, self.quantiles):
            raise ValueError("invalid quantiles")
        expected_edges = tuple(
            np.unique(_weighted_quantiles(self.values, self.counts, tuple(np.linspace(0, 1, 11))))
        )
        if self.bin_edges != expected_edges:
            raise ValueError("histogram edges must derive from training reference")
        return self


class FeatureBaseline(FrozenContract):
    name: str = Field(min_length=1)
    distribution: DistributionBaseline


class ReferenceMetric(FrozenContract):
    name: str
    value: float


class MonitoringBaseline(FrozenContract):
    version: Literal["monitoring-baseline-v1"] = "monitoring-baseline-v1"
    source: Literal["training_only"] = "training_only"
    task: Task
    feature_version: str = Field(min_length=1)
    training_count: int = Field(ge=1, strict=True)
    features: tuple[FeatureBaseline, ...]
    prediction: DistributionBaseline | None = None
    metrics: tuple[ReferenceMetric, ...] = ()
    training_start: AwareDatetime
    training_end: AwareDatetime
    training_label_end: AwareDatetime | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.training_start > self.training_end:
            raise ValueError("invalid training chronology")
        if self.training_label_end is not None and self.training_label_end <= self.training_end:
            raise ValueError("invalid training label horizon")
        names = [f.name for f in self.features]
        if not names or len(set(names)) != len(names):
            raise ValueError("nonempty unique features required")
        if any(f.distribution.count != self.training_count for f in self.features):
            raise ValueError("baseline feature counts differ")
        if self.prediction is not None:
            if self.prediction.count != self.training_count or self.prediction.null_count:
                raise ValueError("baseline prediction counts differ")
            if self.task == "classification" and (
                self.prediction.minimum < 0 or self.prediction.maximum > 1
            ):
                raise ValueError("baseline probabilities outside [0, 1]")
        if self.metrics and self.prediction is None:
            raise ValueError("metrics require prediction baseline")
        if len({m.name for m in self.metrics}) != len(self.metrics):
            raise ValueError("duplicate reference metrics")
        expected_metrics = (
            {"precision", "recall", "f1", "brier", "expected_calibration_error"}
            if self.task == "classification"
            else {"mae", "rmse", "directional_accuracy"}
        )
        if self.metrics and {m.name for m in self.metrics} != expected_metrics:
            raise ValueError("incomplete or unsupported reference metrics")
        for metric in self.metrics:
            if metric.value < 0 or (metric.name not in ("mae", "rmse") and metric.value > 1):
                raise ValueError("invalid reference metric range")
        return self


def _weighted_quantiles(
    values: tuple[float, ...], counts: tuple[int, ...], quantiles: tuple[float, ...]
) -> tuple[float, ...]:
    """Linear empirical quantiles, retaining the compressed exact empirical distribution."""
    positions = np.asarray(quantiles) * (sum(counts) - 1)
    cumulative = np.cumsum(counts)
    lower = np.asarray(values)[np.searchsorted(cumulative, np.floor(positions), side="right")]
    upper = np.asarray(values)[np.searchsorted(cumulative, np.ceil(positions), side="right")]
    return tuple(float(v) for v in lower + (upper - lower) * (positions - np.floor(positions)))


def _finite(values: Sequence[float]) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError("nonempty finite one-dimensional values required")
    return result


def population_stability_index(
    reference_counts: Sequence[float], observed_counts: Sequence[float], epsilon: float = 1e-6
) -> float:
    """PSI on aligned fixed-bin counts, with additive pseudocount smoothing."""
    left, right = _finite(reference_counts), _finite(observed_counts)
    if (
        len(left) != len(right)
        or np.any(left < 0)
        or np.any(right < 0)
        or left.sum() <= 0
        or right.sum() <= 0
        or not np.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("aligned nonnegative nonempty histograms and positive epsilon required")
    # Smooth proportions so scores are independent of total window sample count.
    p, q = left / left.sum() + epsilon, right / right.sum() + epsilon
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def _cdf(
    values: NDArray[np.float64], weights: NDArray[np.float64], points: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.concatenate(([0.0], np.cumsum(weights) / weights.sum()))[
        np.searchsorted(values, points, side="right")
    ]


def _distances(
    left: NDArray[np.float64], counts: NDArray[np.float64], right: NDArray[np.float64]
) -> tuple[float, float]:
    points = np.union1d(left, right)
    delta = np.abs(_cdf(left, counts, points) - _cdf(np.sort(right), np.ones(len(right)), points))
    with np.errstate(over="ignore", invalid="ignore"):
        distance = float(np.dot(delta[:-1], np.diff(points)))
    return float(np.max(delta)), distance


def ks_statistic(reference: Sequence[float], observed: Sequence[float]) -> float:
    """Two-sample empirical KS statistic; no p-value or independence claim."""
    left, right = np.sort(_finite(reference)), _finite(observed)
    return _distances(left, np.ones(len(left)), right)[0]


def wasserstein_distance(reference: Sequence[float], observed: Sequence[float]) -> float:
    """Exact one-dimensional empirical first Wasserstein distance in original units."""
    left, right = np.sort(_finite(reference)), _finite(observed)
    return _distances(left, np.ones(len(left)), right)[1]


def _summarize(values: Sequence[float | None]) -> DistributionBaseline:
    finite = _finite([x for x in values if x is not None])
    unique, counts = np.unique(finite, return_counts=True)
    edges = np.unique(
        _weighted_quantiles(
            tuple(unique), tuple(int(c) for c in counts), tuple(np.linspace(0, 1, 11))
        )
    )
    histogram = np.bincount(np.searchsorted(edges, finite, side="right"), minlength=len(edges) + 1)
    return DistributionBaseline(
        count=len(values),
        null_count=len(values) - len(finite),
        null_rate=(len(values) - len(finite)) / len(values),
        mean=float(finite.mean()),
        std=float(finite.std()),
        minimum=float(finite.min()),
        maximum=float(finite.max()),
        quantiles=_weighted_quantiles(
            tuple(unique), tuple(int(c) for c in counts), (0.05, 0.25, 0.5, 0.75, 0.95)
        ),
        bin_edges=tuple(edges),
        bin_counts=tuple(int(x) for x in histogram),
        values=tuple(unique),
        counts=tuple(int(x) for x in counts),
    )


def _metrics(task: Task, labels: Sequence[float], predictions: Sequence[float]) -> dict[str, float]:
    keys: tuple[str, ...]
    _finite(labels)
    _finite(predictions)
    if task == "classification":
        if any(x not in (0, 1) for x in labels):
            raise ValueError("binary labels required")
        result = classification_metrics([int(x) for x in labels], list(predictions))
        keys = ("precision", "recall", "f1", "brier", "expected_calibration_error")
    else:
        result = regression_metrics(list(labels), list(predictions))
        keys = ("mae", "rmse", "directional_accuracy")
    return {key: float(result[key]) for key in keys}


def build_monitoring_baseline(
    training_rows: Sequence[FeatureRow],
    features: tuple[str, ...],
    *,
    predictions: Sequence[float] | None = None,
    labels: Sequence[float] | None = None,
    task: Task = "classification",
) -> MonitoringBaseline:
    """Caller must supply training partition only, after any target-specific filtering."""
    if not training_rows or len({r.feature_version for r in training_rows}) != 1:
        raise ValueError("nonempty training rows with one feature version required")
    MonitoringWindow(
        rows=tuple(training_rows), evaluated_at=max(r.timestamp for r in training_rows)
    )
    if any(not set(features).issubset(row.values) for row in training_rows):
        raise ValueError("training schema mismatch")
    if predictions is not None and len(predictions) != len(training_rows):
        raise ValueError("predictions must align with training rows")
    if labels is not None and (predictions is None or len(labels) != len(training_rows)):
        raise ValueError("labels require aligned predictions")
    return MonitoringBaseline(
        task=task,
        feature_version=training_rows[0].feature_version,
        training_count=len(training_rows),
        training_start=min(r.timestamp for r in training_rows),
        training_end=max(r.timestamp for r in training_rows),
        training_label_end=max(r.label_end for r in training_rows if isinstance(r, DatasetRow))
        if all(isinstance(r, DatasetRow) for r in training_rows)
        else None,
        features=tuple(
            FeatureBaseline(name=f, distribution=_summarize([r.values[f] for r in training_rows]))
            for f in features
        ),
        prediction=_summarize(predictions) if predictions is not None else None,
        metrics=tuple(
            ReferenceMetric(name=k, value=v) for k, v in _metrics(task, labels, predictions).items()
        )
        if predictions is not None and labels is not None
        else (),
    )


def evaluate_drift(
    metadata: Mapping[str, Any],
    rows: Sequence[FeatureRow] | MonitoringWindow,
    *,
    predictions: Sequence[float] | None = None,
    labels: Sequence[float] | None = None,
    thresholds: DriftThresholds | None = None,
    evaluated_at: datetime | None = None,
) -> ModelHealth:
    """Evaluate an aligned window; invalid monitoring evidence fails closed.

    Predictions/labels are optional. Their absence is explicitly reported, never imputed.
    Healthy describes evaluated diagnostics, not future returns or deployment approval.
    """
    policy = DriftThresholds.model_validate(thresholds or DriftThresholds())
    supplied_window = rows if isinstance(rows, MonitoringWindow) else None
    observations: Sequence[FeatureRow] = (
        supplied_window.rows
        if supplied_window is not None
        else rows
        if not isinstance(rows, MonitoringWindow)
        else rows.rows
    )
    # Keep identity/count reporting available even if window validation later fails.
    rows = observations
    diagnostics: list[DriftDiagnostic] = []
    baseline: MonitoringBaseline | None = None

    def fail(metric: str, reason: str, feature: str | None = None) -> None:
        diagnostics.append(
            DriftDiagnostic(
                category="data_quality",
                metric=metric,
                severity="quarantined",
                reason=reason,
                feature=feature,
            )
        )

    def finish() -> ModelHealth:
        status: HealthStatus = (
            "quarantined"
            if any(d.severity == "quarantined" for d in diagnostics)
            else "watch"
            if any(d.severity == "watch" for d in diagnostics)
            else "healthy"
        )
        return ModelHealth(
            status=status,
            diagnostics=tuple(diagnostics),
            threshold_version=policy.version,
            baseline_version=baseline.version if baseline is not None else None,
            model_id=str(metadata["model_id"]) if metadata.get("model_id") else None,
            feature_version=baseline.feature_version if baseline else None,
            window_count=len(observations),
        )

    def measure(
        category: Any, metric: str, observed: float, limit: DriftLimits, feature: str | None = None
    ) -> None:
        if not np.isfinite(observed):
            fail(metric, "Nonfinite diagnostic; numerical integrity failure", feature)
            return
        severity: HealthStatus = (
            "quarantined"
            if observed >= limit.quarantine
            else "watch"
            if observed >= limit.watch and observed > 0
            else "healthy"
        )
        diagnostics.append(
            DriftDiagnostic(
                category=category,
                metric=metric,
                feature=feature,
                observed=observed,
                watch_limit=limit.watch,
                quarantine_limit=limit.quarantine,
                severity=severity,
                reason=f"{metric}={observed:.6g}; watch >= {limit.watch:g}, "
                f"quarantine >= {limit.quarantine:g}",
            )
        )

    def distributions(
        category: Any,
        reference: DistributionBaseline,
        observed: Sequence[float],
        feature: str | None = None,
    ) -> None:
        actual = _finite(observed)
        counts = np.bincount(
            np.searchsorted(reference.bin_edges, actual, side="right"),
            minlength=len(reference.bin_edges) + 1,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            ks, distance = _distances(
                np.asarray(reference.values), np.asarray(reference.counts, dtype=float), actual
            )
        if not np.isfinite(distance):
            fail(
                "numerical_integrity", "Distribution distance exceeds finite numeric range", feature
            )
            return
        measure(
            category,
            "psi",
            population_stability_index(reference.bin_counts, counts.tolist()),
            policy.psi,
            feature,
        )
        measure(category, "ks", ks, policy.ks, feature)
        # Normalize by training standard deviation; constant references use unit scale.
        scale = reference.std if reference.std > 1e-12 else 1.0
        measure(category, "wasserstein_normalized", distance / scale, policy.wasserstein, feature)
        diagnostics.append(
            DriftDiagnostic(
                category=category,
                metric="wasserstein",
                feature=feature,
                observed=distance,
                severity="healthy",
                reason="Raw distance in feature/prediction units; normalized score gates",
            )
        )

    expected_schema: set[str] = set()
    try:
        baseline = MonitoringBaseline.model_validate(metadata.get("monitoring_baseline"))
        preprocessing = metadata.get("preprocessing", {})
        if not isinstance(preprocessing, Mapping):
            raise ValueError("invalid preprocessing schema")
        excluded = preprocessing.get("excluded_features", ())
        if not isinstance(excluded, (tuple, list)) or any(
            not isinstance(name, str) or not name for name in excluded
        ):
            raise ValueError("invalid excluded feature schema")
        expected_schema = {f.name for f in baseline.features} | set(excluded)
        if (
            metadata.get("feature_version") != baseline.feature_version
            or metadata.get("task") != baseline.task
            or tuple(metadata.get("features", ())) != tuple(f.name for f in baseline.features)
        ):
            raise ValueError("metadata and baseline schema mismatch")
    except (ValidationError, ValueError, TypeError):
        fail("baseline", "Missing, corrupt, unsupported, or incompatible training baseline")
        return finish()
    try:
        if supplied_window is not None:
            if predictions is not None or labels is not None or evaluated_at is not None:
                raise ValueError("do not override an explicit monitoring window")
            window = MonitoringWindow.model_validate(supplied_window)
        else:
            realized: tuple[RealizedLabel, ...] | None = None
            if labels is not None:
                if len(labels) != len(rows) or not all(isinstance(r, DatasetRow) for r in rows):
                    raise ValueError("bare labels require aligned DatasetRow observation horizons")
                target = metadata.get("target", {}).get("field")
                for row, label in zip(rows, labels, strict=True):
                    if not isinstance(row, DatasetRow):
                        raise ValueError("realized DatasetRow required")
                    expected = (
                        row.future_return
                        if target == "future_return"
                        else row.direction
                        if target == "direction"
                        else int(row.barrier == "upper")
                        if target == "barrier"
                        else row.target
                    )
                    if target not in ("future_return", "direction", "barrier", "target"):
                        raise ValueError("baseline target definition required for raw labels")
                    if expected != label:
                        raise ValueError("supplied label differs from realized DatasetRow outcome")
                realized = tuple(
                    RealizedLabel(
                        instrument_id=r.instrument_id,
                        timestamp=r.timestamp,
                        label_end=r.label_end,
                        value=y,
                    )
                    for r, y in zip(rows, labels, strict=True)
                    if isinstance(r, DatasetRow)
                )
            window = MonitoringWindow(
                rows=tuple(rows),
                predictions=tuple(predictions) if predictions is not None else None,
                labels=realized,
                evaluated_at=evaluated_at or datetime.now(UTC),
            )
        rows = window.rows
        predictions = window.predictions
        labels = (
            tuple(label.value for label in window.labels) if window.labels is not None else None
        )
    except (ValidationError, ValueError, TypeError, AttributeError):
        fail("window_integrity", "Invalid chronology, nonfinite data, or unproven realized labels")
        return finish()
    if not rows:
        fail("empty_window", "A nonempty observation window is required")
        return finish()
    if len({(r.instrument_id, r.timestamp) for r in rows}) != len(rows) or any(
        a.timestamp > b.timestamp for a, b in zip(rows, rows[1:], strict=False)
    ):
        fail("chronology", "Duplicate observations or nonchronological window")
    if len(rows) < policy.minimum_samples:
        diagnostics.append(
            DriftDiagnostic(
                category="data_quality",
                metric="sample_count",
                observed=len(rows),
                watch_limit=float(policy.minimum_samples),
                severity="watch",
                reason="insufficient observations: window below configured minimum sample count",
            )
        )
    if any(row.feature_version != baseline.feature_version for row in rows):
        fail("feature_version", "Window feature version differs from the training baseline")
    if any(set(row.values) - expected_schema for row in rows):
        fail("schema_unexpected", "Window has columns outside the training feature schema")
    for feature in baseline.features:
        if any(feature.name not in row.values for row in rows):
            fail("schema", "Required feature is missing", feature.name)
            continue
        values = [r.values[feature.name] for r in rows]
        actual = [x for x in values if x is not None]
        if any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x)
            for x in actual
        ):
            fail(
                "nonfinite", "Features require finite numeric values or explicit null", feature.name
            )
            continue
        null_rate = (len(values) - len(actual)) / len(values)
        measure(
            "data_quality",
            "null_rate",
            abs(null_rate - feature.distribution.null_rate),
            policy.null_rate,
            feature.name,
        )
        if not actual:
            fail("all_null", "No observed values for required feature", feature.name)
            continue
        distributions("feature", feature.distribution, actual, feature.name)
    if predictions is None:
        if labels is not None:
            fail("labels", "Realized labels require aligned predictions")
        diagnostics.append(
            DriftDiagnostic(
                category="prediction",
                metric="unavailable",
                severity="healthy",
                reason="Prediction drift not evaluated: no predictions",
            )
        )
    else:
        try:
            actual_predictions = _finite(predictions)
            if len(actual_predictions) != len(rows):
                raise ValueError("prediction length differs from window")
            if baseline.task == "classification" and np.any(
                (actual_predictions < 0) | (actual_predictions > 1)
            ):
                raise ValueError("probabilities outside [0, 1]")
            if baseline.prediction is None:
                fail("prediction_baseline", "Training-reference predictions unavailable")
            else:
                distributions("prediction", baseline.prediction, predictions)
            if labels is not None:
                if len(labels) != len(rows):
                    raise ValueError("label length differs from window")
                current = _metrics(baseline.task, labels, predictions)
                reference = {m.name: m.value for m in baseline.metrics}
                if not set(current).issubset(reference):
                    fail("metric_baseline", "Training-reference realized metrics unavailable")
                else:
                    for key, value in current.items():
                        higher_is_bad = key in (
                            "brier",
                            "expected_calibration_error",
                            "mae",
                            "rmse",
                        )
                        change = max(
                            0.0, value - reference[key] if higher_is_bad else reference[key] - value
                        )
                        category = (
                            "calibration"
                            if key in ("brier", "expected_calibration_error")
                            else "performance"
                        )
                        limit = (
                            policy.brier
                            if key == "brier"
                            else policy.calibration
                            if key == "expected_calibration_error"
                            else policy.regression_error
                            if key in ("mae", "rmse")
                            else policy.performance
                        )
                        measure(
                            category,
                            ("ece" if key == "expected_calibration_error" else key)
                            + ("_increase" if higher_is_bad else "_degradation"),
                            change,
                            limit,
                        )
        except (ValueError, TypeError, OverflowError):
            fail(
                "prediction_labels",
                "Invalid, nonfinite, out-of-range, or misaligned predictions/labels",
            )
    if labels is None:
        for category in ("calibration", "performance"):
            diagnostics.append(
                DriftDiagnostic(
                    category=category,
                    metric="realized_labels",
                    severity="healthy",
                    reason="Not evaluated: realized labels unavailable",
                )
            )
    return finish()
