"""Artifact, CLI and risk integration tests on the existing synthetic fixture."""

import json
from pathlib import Path

import pytest
from test_cli import cli

from trading_agent.cli import dispatch, parser


def call(*args):
    return dispatch(parser().parse_args([str(a) for a in args]))


def dataset(tmp_path):
    imported = call(
        "data",
        "import",
        "examples/research/SYNTHETIC.csv",
        "--mapping",
        "examples/research/mapping.json",
        "--root",
        tmp_path / "imports",
        "--synthetic",
    )
    path = tmp_path / "dataset.json"
    call("features", "build", imported["manifest_path"], "--output", path, "--horizon", 5)
    return imported["manifest_path"], path


def train(path, tmp_path, *extra):
    return call(
        "model",
        "train",
        path,
        "--registry",
        tmp_path / "models",
        "--train-size",
        120,
        "--validation-size",
        60,
        "--test-size",
        60,
        *extra,
    )


def test_training_writes_machine_readable_evidence_and_registry_commands(tmp_path):
    _, path = dataset(tmp_path)
    trained = train(path, tmp_path)
    assert "SYNTHETIC" in trained["warning"]
    run = Path(trained["artifacts"])
    for name in (
        "training_config.json",
        "metrics.json",
        "walk_forward.json",
        "predictions.json",
        "leakage_audit.json",
        "checksums.json",
    ):
        assert (run / name).is_file()
    predictions = json.loads((run / "predictions.json").read_text())
    assert predictions["predictions"]
    model_id = Path(trained["models"][0]).name
    listing = call("model", "list", "--registry", tmp_path / "models")
    assert model_id in [m["model_id"] for m in listing["models"]]
    shown = call("model", "show", model_id, "--registry", tmp_path / "models")
    assert shown["metadata"]["model_id"] == model_id
    assert shown["checksum_valid"] is True
    with pytest.raises(ValueError, match="model_id"):
        call("model", "show", "../escape", "--registry", tmp_path / "models")


def test_training_rebuilds_features_and_labels_to_detect_tampering(tmp_path):
    _, path = dataset(tmp_path)
    content = json.loads(path.read_text())
    content["rows"][30]["values"]["ema_12"] = 1e9
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="causal|rebuild"):
        train(path, tmp_path)


def test_cash_and_naive_backtest_baselines(tmp_path):
    manifest, _ = dataset(tmp_path)
    for name in ("cash", "naive"):
        output = tmp_path / name
        call(
            "backtest",
            "run",
            manifest,
            "--calendar",
            "examples/research/calendar.json",
            "--config",
            "examples/research/backtest.json",
            "--output",
            output,
            "--baseline",
            name,
        )
        report = json.loads((output / "report.json").read_text())
        assert report["risk_settings"]["execution_mode"] == "paper"
        if name == "cash":
            assert report["metrics"]["fill_count"] == 0
            assert report["metrics"]["total_return"] == 0
        else:
            assert report["metrics"]["fill_count"] > 0


def test_model_predictions_cannot_bypass_risk_engine(tmp_path):
    manifest, path = dataset(tmp_path)
    trained = train(path, tmp_path)
    config = json.loads(Path("examples/research/backtest.json").read_text())
    config["risk"]["max_capital_per_trade"] = "1"
    restricted = tmp_path / "restricted.json"
    restricted.write_text(json.dumps(config))
    out = tmp_path / "backtest"
    call(
        "backtest",
        "run",
        manifest,
        "--calendar",
        "examples/research/calendar.json",
        "--config",
        restricted,
        "--output",
        out,
        "--model",
        trained["models"][0],
        "--trust-local-artifact",
    )
    report = json.loads((out / "report.json").read_text())
    assert not report["trades"]
    assert any("capital_per_trade" in e["reasons"] for e in report["audit"])


def test_model_cli_documents_new_workflow_and_handles_bad_input(tmp_path):
    for action in ("train", "evaluate", "list", "show"):
        result = cli("model", action, "--help")
        assert result.returncode == 0
    result = cli("model", "show", "missing", "--registry", str(tmp_path))
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
