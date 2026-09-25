"""Independent-review regressions: exact partitions and explicit data controls."""

import json

import pytest
from test_features_research import bars
from test_phase4_workflow import call, dataset, train

from trading_agent.data.providers import CSVMapping
from trading_agent.data.storage import import_csv
from trading_agent.features.research import FeatureRow, ResearchBar, build_features
from trading_agent.ml.audit import require_synthetic
from trading_agent.ml.dataset import build_dataset


def test_saved_test_membership_preserves_irregular_instrument_purges(tmp_path):
    source = sorted(
        [*bars(160, "A"), *bars(160, "B")[::4]], key=lambda b: (b.timestamp, b.instrument_id)
    )
    rows = build_dataset(source, horizon=2)
    path = tmp_path / "irregular.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "research-dataset-v1",
                "synthetic": False,
                "dataset_version": "unspecified",
                "target_field": "direction",
                "source_bars": [b.model_dump(mode="json") for b in source],
                "rows": [r.model_dump(mode="json") for r in rows],
            }
        )
    )
    trained = call(
        "model",
        "train",
        path,
        "--registry",
        tmp_path / "models",
        "--train-size",
        70,
        "--validation-size",
        25,
        "--test-size",
        25,
    )
    model = trained["models"][0]
    evaluated = call(
        "model",
        "evaluate",
        model,
        path,
        "--output",
        tmp_path / "evaluation.json",
        "--trust-local-artifact",
    )
    shown = call("model", "show", model.split("/")[-1], "--registry", tmp_path / "models")
    stored = shown["metadata"]["metrics"]["test"]
    assert evaluated["count"] == stored["count"]
    assert evaluated["brier"] == stored["brier"]


@pytest.mark.parametrize("tamper", ["feature", "label", "control"])
def test_out_of_sample_evaluation_rebuilds_and_honors_controls(tmp_path, tamper):
    _, path = dataset(tmp_path)
    trained = train(path, tmp_path)
    content = json.loads(path.read_text())
    if tamper == "feature":
        content["rows"][200]["values"]["ema_12"] = 1e9
    elif tamper == "label":
        content["rows"][200]["direction"] = 1 - content["rows"][200]["direction"]
    else:
        content["training_eligible"] = False
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="rebuild|informational"):
        call(
            "model",
            "evaluate",
            trained["models"][0],
            changed,
            "--partition",
            "out-of-sample",
            "--output",
            tmp_path / "evaluation.json",
            "--trust-local-artifact",
        )


def test_non_synthetic_substring_is_not_an_authorization():
    with pytest.raises(ValueError, match="SYNTHETIC"):
        require_synthetic("non-synthetic-market-feed", False)
    require_synthetic("SYNTHETIC", False)  # Explicit pre-Phase-2 unit-test lineage.


def test_import_does_not_mislabel_non_synthetic_provider(tmp_path):
    from pathlib import Path

    path = tmp_path / "observations.csv"
    path.write_bytes(Path("examples/research/SYNTHETIC.csv").read_bytes())
    mapping = CSVMapping.model_validate_json(Path("examples/research/mapping.json").read_text())
    mapping = mapping.model_copy(update={"provider": "non-synthetic-market-feed"})
    manifest = import_csv(path, mapping, tmp_path / "imports")
    assert manifest.synthetic is False


@pytest.mark.parametrize(
    "control,value", [("training_eligible", False), ("informational_only", True)]
)
def test_source_controls_cannot_be_silently_discarded(control, value):
    source = bars(10)
    with pytest.raises(ValueError):
        ResearchBar.model_validate({**source[0].model_dump(), control: value})
    with pytest.raises(ValueError):
        FeatureRow.model_validate({**build_features(source)[0].model_dump(), control: value})


def test_source_bar_deny_control_rejected_before_training(tmp_path):
    _, path = dataset(tmp_path)
    content = json.loads(path.read_text())
    content["source_bars"][0]["training_eligible"] = False
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        train(path, tmp_path)


def test_backtest_cannot_reclassify_mcp_bars_as_executable_prices(tmp_path):
    from pathlib import Path

    mapping = CSVMapping.model_validate_json(Path("examples/research/mapping.json").read_text())
    mapping = mapping.model_copy(update={"provider": "SYNTHETIC NSE MCP"})
    manifest = import_csv(
        Path("examples/research/SYNTHETIC.csv"), mapping, tmp_path / "imports", synthetic=True
    )
    with pytest.raises(ValueError, match="MCP"):
        call(
            "backtest",
            "run",
            tmp_path / "imports" / manifest.import_id / "manifest.json",
            "--calendar",
            "examples/research/calendar.json",
            "--config",
            "examples/research/backtest.json",
            "--output",
            tmp_path / "backtest",
        )
