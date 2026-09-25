"""CLI contract and end-to-end SYNTHETIC engineering workflow."""

import json
import subprocess
import sys
from pathlib import Path


def cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "trading_agent.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_lists_research_groups() -> None:
    result = cli("--help")
    assert result.returncode == 0
    assert all(group in result.stdout for group in ("data", "features", "backtest", "model"))


def test_invalid_input_is_a_nonzero_actionable_error(tmp_path: Path) -> None:
    result = cli("data", "inspect", str(tmp_path / "absent.csv"))
    assert result.returncode != 0
    assert "error" in result.stderr.lower()
    assert "Traceback" not in result.stderr


def test_json_artifact_never_overwrites(tmp_path: Path) -> None:
    from trading_agent.research_io import write_json

    target = tmp_path / "artifact.json"
    write_json(target, {"synthetic": True})
    try:
        write_json(target, {"synthetic": False})
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing artifact overwritten")
    assert json.loads(target.read_text()) == {"synthetic": True}


def test_import_validates_and_preserves_synthetic_provenance(tmp_path: Path) -> None:
    result = cli(
        "data",
        "import",
        "examples/research/SYNTHETIC.csv",
        "--mapping",
        "examples/research/mapping.json",
        "--synthetic",
        "--root",
        str(tmp_path / "imports"),
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["row_count"] == 300
    assert output["synthetic"] is True
    summary = cli("data", "summarize", output["manifest_path"])
    assert summary.returncode == 0, summary.stderr
    assert json.loads(summary.stdout)["raw_sha256"] == output["raw_sha256"]
    repeated = cli(
        "data",
        "import",
        "examples/research/SYNTHETIC.csv",
        "--mapping",
        "examples/research/mapping.json",
        "--synthetic",
        "--root",
        str(tmp_path / "imports"),
    )
    assert repeated.returncode != 0
    assert "exists" in repeated.stderr


def test_synthetic_feature_and_training_cli(tmp_path: Path) -> None:
    imported = cli(
        "data",
        "import",
        "examples/research/SYNTHETIC.csv",
        "--mapping",
        "examples/research/mapping.json",
        "--synthetic",
        "--root",
        str(tmp_path / "imports"),
    )
    manifest = json.loads(imported.stdout)["manifest_path"]
    dataset = tmp_path / "dataset.json"
    built = cli("features", "build", manifest, "--output", str(dataset), "--horizon", "5")
    assert built.returncode == 0, built.stderr
    content = json.loads(dataset.read_text())
    assert content["synthetic"] is True
    assert len(content["features"]) == 300
    assert len(content["rows"]) == 295
    trained = cli(
        "model",
        "train",
        str(dataset),
        "--registry",
        str(tmp_path / "models"),
        "--train-size",
        "120",
        "--validation-size",
        "60",
        "--test-size",
        "60",
    )
    assert trained.returncode == 0, trained.stderr
    models = json.loads(trained.stdout)["models"]
    assert models
    evaluated = cli(
        "model",
        "evaluate",
        models[0],
        str(dataset),
        "--output",
        str(tmp_path / "evaluation.json"),
        "--trust-local-artifact",
    )
    assert evaluated.returncode == 0, evaluated.stderr
    metrics = json.loads((tmp_path / "evaluation.json").read_text())
    assert metrics["synthetic"] is True
    assert metrics["count"] == 60
    assert len(metrics["confusion_matrix"]) == 2

    metadata = json.loads((Path(models[0]) / "metadata.json").read_text())
    test_start = metadata["ranges"]["test"]["start"]
    test_end = metadata["ranges"]["test"]["end"]
    baseline = cli(
        "backtest",
        "run",
        manifest,
        "--calendar",
        "examples/research/calendar.json",
        "--config",
        "examples/research/backtest.json",
        "--output",
        str(tmp_path / "same-window-sma"),
        "--start",
        test_start,
        "--end",
        test_end,
    )
    assert baseline.returncode == 0, baseline.stderr

    consequences = cli(
        "backtest",
        "run",
        manifest,
        "--calendar",
        "examples/research/calendar.json",
        "--config",
        "examples/research/backtest.json",
        "--output",
        str(tmp_path / "model-backtest"),
        "--model",
        models[0],
        "--trust-local-artifact",
    )
    assert consequences.returncode == 0, consequences.stderr
    consequence_report = json.loads((tmp_path / "model-backtest/report.json").read_text())
    assert consequence_report["config"]["strategy_name"] == "ml_probability_baseline"
    assert consequence_report["config"]["synthetic"] is True
    assert consequence_report["risk_settings"]["execution_mode"] == "paper"
    baseline_report = json.loads((tmp_path / "same-window-sma/report.json").read_text())
    assert baseline_report["start"] == consequence_report["start"]
    assert baseline_report["end"] == consequence_report["end"]
    assert consequence_report["config"]["strategy_config"]["model_id"] == metadata["model_id"]
    assert consequence_report["config"]["strategy_config"]["model_lineage"]

    # A different artifact cannot masquerade as this model's saved test partition.
    content["rows"][0]["values"]["ema_12"] = 999
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(content))
    mismatched = cli(
        "model",
        "evaluate",
        models[0],
        str(changed),
        "--output",
        str(tmp_path / "bad-evaluation.json"),
        "--trust-local-artifact",
    )
    assert mismatched.returncode != 0
    assert "original immutable dataset" in mismatched.stderr


def test_backtest_cli_uses_calendar_and_labels_report(tmp_path: Path) -> None:
    imported = cli(
        "data",
        "import",
        "examples/research/SYNTHETIC.csv",
        "--mapping",
        "examples/research/mapping.json",
        "--synthetic",
        "--root",
        str(tmp_path / "imports"),
    )
    manifest = json.loads(imported.stdout)["manifest_path"]
    output = tmp_path / "backtest"
    result = cli(
        "backtest",
        "run",
        manifest,
        "--calendar",
        "examples/research/calendar.json",
        "--config",
        "examples/research/backtest.json",
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["config"]["synthetic"] is True
    assert report["trades"]
    assert "SYNTHETIC" in (output / "report.md").read_text()
    assert report["risk_settings"]["execution_mode"] == "paper"
    assert "database_url" not in report["risk_settings"]


def test_cli_keeps_phase1_risk_gate_authoritative(tmp_path: Path) -> None:
    imported = cli(
        "data",
        "import",
        "examples/research/SYNTHETIC.csv",
        "--mapping",
        "examples/research/mapping.json",
        "--synthetic",
        "--root",
        str(tmp_path / "imports"),
    )
    manifest = json.loads(imported.stdout)["manifest_path"]
    config = json.loads(Path("examples/research/backtest.json").read_text())
    config["risk"]["max_capital_per_trade"] = "1"
    path = tmp_path / "restrictive.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "denied"
    result = cli(
        "backtest",
        "run",
        manifest,
        "--calendar",
        "examples/research/calendar.json",
        "--config",
        str(path),
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["trades"] == []
    assert any("capital_per_trade" in event["reasons"] for event in report["audit"])
    assert report["final_portfolio"]["cash"] == report["initial_capital"]
