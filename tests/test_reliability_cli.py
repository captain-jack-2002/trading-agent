import json
from datetime import UTC, datetime

import pytest
from test_model_lifecycle import artifact

from trading_agent.cli import main


def test_grounding_cli_roundtrip_and_missing(tmp_path, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(
        "# Limitations\nSynthetic tests are engineering validation only."
    )
    index = tmp_path / "index"
    assert main(["grounding", "index", "--source", str(docs), "--output", str(index)]) == 0
    capsys.readouterr()
    assert main(["grounding", "query", "engineering validation", "--index", str(index)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "supported"
    evidence_id = report["evidence"][0]["evidence_id"]
    assert main(["grounding", "inspect", evidence_id, "--index", str(index)]) == 0
    assert json.loads(capsys.readouterr().out)
    main(["grounding", "query", "price", "--index", str(tmp_path / "absent")])
    assert json.loads(capsys.readouterr().out)["status"] == "abstained"


def test_model_lifecycle_cli(tmp_path, capsys):
    artifact(tmp_path)
    main(["model", "health", "one", "--registry", str(tmp_path)])
    assert json.loads(capsys.readouterr().out)["lifecycle"]["status"] == "challenger"
    main(["model", "quarantine", "one", "--registry", str(tmp_path), "--reason", "severe drift"])
    assert json.loads(capsys.readouterr().out)["status"] == "quarantined"
    main(["model", "audit", "--registry", str(tmp_path)])
    assert len(json.loads(capsys.readouterr().out)["events"]) == 2


def test_cli_drift_schema_failure_quarantines_without_loading_pickle(tmp_path, capsys):
    artifact(tmp_path)
    window = tmp_path / "window.json"
    window.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "instrument_id": "DEMO",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "provenance": "synthetic",
                        "values": {},
                    }
                ],
            }
        )
    )
    main(["model", "drift", "one", "--registry", str(tmp_path), "--window", str(window)])
    report = json.loads(capsys.readouterr().out)
    assert report["health"]["status"] == "quarantined"
    assert report["lifecycle"]["status"] == "quarantined"
    assert report["health_reference"]


def test_promote_cli_requires_review_evidence(tmp_path):
    artifact(tmp_path)
    with pytest.raises(SystemExit):
        main(["model", "promote", "one", "--registry", str(tmp_path)])


def test_monitoring_report_tampering_is_rejected(tmp_path, capsys):
    from test_reliability_execution import window as model_window
    from test_risk import NOW

    bundle = artifact(tmp_path)
    path = tmp_path / "window.json"
    path.write_text(
        json.dumps(
            {
                "rows": [r.model_dump(mode="json") for r in model_window(bundle)],
                "evaluated_at": NOW.isoformat(),
            }
        )
    )
    argv = [
        "model",
        "health",
        "one",
        "--registry",
        str(tmp_path),
        "--window",
        str(path),
        "--trust-local-artifact",
    ]
    main(argv)
    report = json.loads(capsys.readouterr().out)["report"]
    from pathlib import Path

    Path(report).write_text("{}")
    with pytest.raises(SystemExit):
        main(argv)


def test_invalid_grounding_input_is_not_echoed_into_cli_errors(tmp_path, capsys):
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(
            {"trusted": True, "validated": True, "content": {"marker": "PRIVATE_INPUT_MARKER"}}
        )
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    with pytest.raises(SystemExit):
        main(
            [
                "grounding",
                "index",
                "--source",
                str(docs),
                "--trusted-artifact",
                str(path),
                "--output",
                str(tmp_path / "index"),
            ]
        )
    captured = capsys.readouterr()
    assert "PRIVATE_INPUT_MARKER" not in captured.err


def test_monitoring_report_hashes_the_exact_evaluated_snapshot(tmp_path, capsys, monkeypatch):
    import hashlib

    from test_reliability_execution import window as model_window
    from test_risk import NOW

    from trading_agent import cli_reliability

    bundle = artifact(tmp_path)
    path = tmp_path / "window.json"
    payload = {
        "rows": [r.model_dump(mode="json") for r in model_window(bundle)],
        "evaluated_at": NOW.isoformat(),
    }
    path.write_text(json.dumps(payload))
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    original = cli_reliability.load_model

    def replace_window_then_load(*args, **kwargs):
        changed = {
            **payload,
            "rows": [{**r, "values": {k: 1e6 for k in r["values"]}} for r in payload["rows"]],
        }
        path.write_text(json.dumps(changed))
        return original(*args, **kwargs)

    monkeypatch.setattr(cli_reliability, "load_model", replace_window_then_load)
    main(
        [
            "model",
            "health",
            "one",
            "--registry",
            str(tmp_path),
            "--window",
            str(path),
            "--trust-local-artifact",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    from pathlib import Path

    report = json.loads(Path(result["report"]).read_text())
    assert result["health"]["status"] == "healthy"
    assert report["window_sha256"] == expected_hash
    assert report["window_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
