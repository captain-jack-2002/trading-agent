import json

import pytest
from test_features_research import bars

from trading_agent.ml.dataset import build_dataset
from trading_agent.ml.pipeline import evaluate_model, train_model
from trading_agent.ml.registry import load_model, save_model
from trading_agent.ml.split import walk_forward


def test_labels_and_purged_splits():
    data = bars(120)
    rows = build_dataset(data, horizon=3)
    assert len(rows) == 117
    assert rows[0].label_end == data[3].timestamp
    assert rows[0].future_return == pytest.approx(data[3].close / data[0].close - 1)
    for split in walk_forward(rows, train_size=50, validation_size=20, test_size=20):
        assert max(rows[i].label_end for i in split.train) < min(
            rows[i].timestamp for i in split.validation
        )
        assert max(rows[i].label_end for i in split.validation) < min(
            rows[i].timestamp for i in split.test
        )


def test_ambiguous_barrier():
    data = bars(5)
    data[1] = data[1].model_copy(update={"high": 110.0, "low": 90.0})
    assert build_dataset(data, horizon=2)[0].barrier == "ambiguous"


@pytest.mark.parametrize("kind", ["logistic", "random_forest"])
def test_train_registry_reproducible_and_trust(tmp_path, kind):
    rows = build_dataset(bars(150), horizon=2)
    split = walk_forward(rows, train_size=70, validation_size=25, test_size=25)[0]
    bundle = train_model(rows, split, model_kind=kind)
    test = [rows[i] for i in split.test]
    metrics = evaluate_model(bundle, test)
    assert 0 <= metrics["brier"] <= 1
    assert metrics == evaluate_model(train_model(rows, split, model_kind=kind), test)
    path = tmp_path / "model"
    save_model(bundle, path)
    with pytest.raises(ValueError, match="trusted"):
        load_model(path)
    assert evaluate_model(load_model(path, trusted=True), test) == metrics
    meta = json.loads((path / "metadata.json").read_text())
    assert meta["feature_version"] and meta["ranges"]["train"]
    with (path / "model.joblib").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_model(path, trusted=True)


def test_invalid_rows_and_mixed_targets_rejected():
    rows = build_dataset(bars(120), horizon=2)
    with pytest.raises(ValueError):
        type(rows[0]).model_validate({**rows[0].model_dump(), "label_end": rows[0].timestamp})
    with pytest.raises(ValueError):
        walk_forward([rows[0], rows[0], *rows[1:]], 30, 20, 20)
    split = walk_forward(rows, 50, 20, 20)[0]
    rows[1] = rows[1].model_copy(update={"threshold": 0.5})
    with pytest.raises(ValueError, match="target"):
        train_model(rows, split)


def test_training_only_imputation_and_target_selection():
    rows = build_dataset(bars(150), horizon=2)
    split = walk_forward(rows, 70, 25, 25)[0]
    bundle = train_model(rows, split, target_field="direction")
    assert bundle.metadata["target"]["field"] == "direction"
    altered = [
        r.model_copy(update={"values": {k: 1e12 for k in r.values}}) if i in split.test else r
        for i, r in enumerate(rows)
    ]
    other = train_model(altered, split, target_field="direction")
    assert (
        bundle.estimator.named_steps["imputer"].statistics_
        == other.estimator.named_steps["imputer"].statistics_
    ).all()


def test_prediction_schema_checks():
    from trading_agent.ml.pipeline import predict_probabilities

    rows = build_dataset(bars(150), horizon=2)
    split = walk_forward(rows, 70, 25, 25)[0]
    bundle = train_model(rows, split)
    assert len(predict_probabilities(bundle, rows[-3:])) == 3
    row = rows[-1].model_copy(update={"values": {}})
    with pytest.raises(ValueError, match="schema"):
        predict_probabilities(bundle, [row])


def test_evaluation_rejects_changed_target():
    rows = build_dataset(bars(150), horizon=2)
    split = walk_forward(rows, 70, 25, 25)[0]
    bundle = train_model(rows, split)
    with pytest.raises(ValueError, match="target"):
        evaluate_model(bundle, [rows[-1].model_copy(update={"horizon": 3})])


def test_barrier_classifier_excludes_unresolved():
    rows = build_dataset(bars(150), horizon=2)
    split = walk_forward(rows, 70, 25, 25)[0]
    bundle = train_model(rows, split, target_field="barrier")
    selected = [rows[i] for i in split.test]
    report = evaluate_model(bundle, selected)
    assert report["count"] + report["excluded_barrier_rows"] == len(selected)


@pytest.mark.parametrize(
    "field,value",
    [("horizon", True), ("horizon", 2.0), ("direction", True), ("target", False), ("synthetic", 1)],
)
def test_strict_label_types(field, value):
    row = build_dataset(bars(5), horizon=2)[0]
    with pytest.raises(ValueError):
        type(row).model_validate({**row.model_dump(), field: value})


def test_registry_failed_serialization_is_not_published(tmp_path):
    rows = build_dataset(bars(150), horizon=2)
    bundle = train_model(rows, walk_forward(rows, 70, 25, 25)[0])
    assert bundle.metadata["model_id"]
    bundle.metadata["invalid_json"] = float("nan")
    path = tmp_path / "failed"
    with pytest.raises(ValueError):
        save_model(bundle, path)
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_forged_training_inputs_rejected():
    from trading_agent.ml.split import Split

    rows = build_dataset(bars(150), horizon=2)
    split = walk_forward(rows, 70, 25, 25)[0]
    with pytest.raises(ValueError, match="chronolog"):
        train_model(rows[::-1], split)
    changed = rows.copy()
    changed[0] = changed[0].model_copy(update={"dataset_version": "different"})
    with pytest.raises(ValueError, match="lineage"):
        train_model(changed, split)
    with pytest.raises(ValueError, match="indices"):
        train_model(rows, Split((True, *split.train[1:]), split.validation, split.test))
