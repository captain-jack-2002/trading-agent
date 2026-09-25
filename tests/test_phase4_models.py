"""Model contracts: train-only preprocessing, calibrated metrics and persistence."""

import hashlib
import json

import numpy as np
import pytest
from test_features_research import bars

from trading_agent.ml.dataset import build_dataset
from trading_agent.ml.pipeline import evaluate_model, predict_probabilities, train_model
from trading_agent.ml.registry import load_model, save_model
from trading_agent.ml.split import walk_forward


def sample():
    rows = build_dataset(bars(150), horizon=2)
    return rows, walk_forward(rows, 70, 25, 25)[0]


@pytest.mark.parametrize("kind", ["logistic", "random_forest", "lightgbm"])
def test_classifier_reproducible_metadata_and_roundtrip(tmp_path, kind):
    rows, split = sample()
    one = train_model(rows, split, model_kind=kind, seed=42)
    two = train_model(rows, split, model_kind=kind, seed=42)
    selected = [rows[i] for i in split.test]
    assert predict_probabilities(one, selected) == pytest.approx(
        predict_probabilities(two, selected)
    )
    assert one.metadata["model_version"]
    assert one.metadata["git_commit"]
    assert one.metadata["python_version"]
    assert one.metadata["dataset_sha256"] == two.metadata["dataset_sha256"]
    assert one.metadata["preprocessing"]["fit_range"] == one.metadata["ranges"]["train"]
    assert "SYNTHETIC" in one.metadata["warning"]
    target = tmp_path / kind
    save_model(one, target)
    metadata = json.loads((target / "metadata.json").read_text())
    assert metadata["sha256"] == hashlib.sha256((target / "model.joblib").read_bytes()).hexdigest()
    assert predict_probabilities(load_model(target, trusted=True), selected) == pytest.approx(
        predict_probabilities(one, selected)
    )


@pytest.mark.parametrize("kind", ["ridge", "random_forest", "lightgbm"])
def test_regression(kind):
    from trading_agent.ml.pipeline import predict_returns

    rows, split = sample()
    bundle = train_model(rows, split, model_kind=kind, target_field="future_return")
    selected = [rows[i] for i in split.test]
    metrics = evaluate_model(bundle, selected)
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= metrics["mae"]
    assert np.isfinite(metrics["r2"])
    assert 0 <= metrics["directional_accuracy"] <= 1
    assert all(np.isfinite(predict_returns(bundle, selected)))
    with pytest.raises(ValueError, match="classification"):
        predict_probabilities(bundle, selected)


def test_imputer_and_scaler_ignore_all_future_values():
    rows, split = sample()
    original = train_model(rows, split)
    modified = [
        r.model_copy(update={"values": {k: 1e12 for k in r.values}})
        if i in (*split.validation, *split.test)
        else r
        for i, r in enumerate(rows)
    ]
    changed = train_model(modified, split)
    for name, attr in (("imputer", "statistics_"), ("scale", "mean_"), ("scale", "var_")):
        assert np.array_equal(
            getattr(original.estimator.named_steps[name], attr),
            getattr(changed.estimator.named_steps[name], attr),
        )
    assert original.estimator.named_steps["scale"].n_samples_seen_ == len(split.train)


def test_class_distribution_probability_distribution_and_calibration():
    rows, split = sample()
    bundle = train_model(rows, split)
    metrics = evaluate_model(bundle, [rows[i] for i in split.test])
    assert sum(metrics["class_distribution"].values()) == len(split.test)
    assert sum(metrics["probability_distribution"]["counts"]) == len(split.test)
    assert sum(c["count"] for c in metrics["calibration"]) == len(split.test)
    assert 0 <= metrics["expected_calibration_error"] <= 1


def test_missing_values_imputed_and_empty_features_rejected():
    rows, split = sample()
    bundle = train_model(rows, split)
    null = rows[-1].model_copy(update={"values": {k: None for k in rows[-1].values}})
    assert 0 <= predict_probabilities(bundle, [null])[0] <= 1
    with pytest.raises(ValueError, match="observed training features"):
        train_model(
            [r.model_copy(update={"values": {k: None for k in r.values}}) for r in rows], split
        )


def test_v2_barrier_neither_is_negative_and_ambiguous_is_excluded():
    rows, split = sample()
    bundle = train_model(rows, split, target_field="barrier")
    selected = [rows[i] for i in split.test]
    selected[0] = selected[0].model_copy(update={"barrier": "neither"})
    selected[1] = selected[1].model_copy(update={"barrier": "ambiguous"})
    metrics = evaluate_model(bundle, selected)
    assert metrics["count"] == sum(r.barrier != "ambiguous" for r in selected)
