"""Local reliability commands and explicit artifact trust; no network required."""

import json

import pytest
from test_model_lifecycle import artifact
from test_phase4_workflow import call
from test_reliability_execution import window
from test_risk import NOW


def test_grounding_cli_indexes_queries_inspects_and_abstains(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "limits.md").write_text("Synthetic model limitations: no market performance claims.")
    index = tmp_path / "index"
    indexed = call("grounding", "index", "--source", docs, "--output", index)
    assert indexed["chunks"] == 1
    result = call(
        "grounding", "query", "model limitations", "--index", index, "--as-of", NOW.isoformat()
    )
    assert result["status"] == "supported"
    item = result["evidence"][0]
    inspected = call("grounding", "inspect", item["evidence_id"], "--index", index)
    assert inspected["chunk"]["content_hash"]
    missing = call("grounding", "query", "model", "--index", tmp_path / "absent")
    assert missing["status"] == "abstained"


def test_health_and_drift_cli_infers_predictions_only_with_local_trust(tmp_path):
    bundle = artifact(tmp_path / "models")
    path = tmp_path / "window.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "monitoring-window-v1",
                "evaluated_at": NOW.isoformat(),
                "rows": [r.model_dump(mode="json") for r in window(bundle)],
            }
        )
    )
    with pytest.raises(ValueError, match="trust"):
        call("model", "health", "one", "--window", path, "--registry", tmp_path / "models")
    for action in ("health", "drift"):
        result = call(
            "model",
            action,
            "one",
            "--window",
            path,
            "--registry",
            tmp_path / "models",
            "--trust-local-artifact",
        )
        assert result["health"]["status"] == "healthy"
        assert result["lifecycle"]["status"] == "challenger"
        assert result["health"]["diagnostics"]


def test_lifecycle_cli_requires_explicit_verified_validation(tmp_path):
    from trading_agent.data.storage import checksum
    from trading_agent.ml.lifecycle import LifecycleRegistry
    from trading_agent.ml.registry import show_model

    bundle = artifact(tmp_path / "models")
    meta = show_model(tmp_path / "models", "one")["metadata"]
    validation = {
        "model_id": bundle.metadata["model_id"],
        "model_sha256": meta["sha256"],
        "passed": True,
    }
    walk = tmp_path / "walk.json"
    paper = tmp_path / "paper.json"
    walk.write_text(json.dumps({**validation, "kind": "purged_walk_forward", "purged": True}))
    paper.write_text(
        json.dumps({**validation, "kind": "paper_shadow_validation", "paper_only": True})
    )
    result = call(
        "model",
        "promote",
        "one",
        "--registry",
        tmp_path / "models",
        "--walk-forward",
        walk,
        "--paper-validation",
        paper,
        "--approved-by",
        "engineering-reviewer",
        "--reason",
        "reviewed engineering evidence",
    )
    assert result["status"] == "champion"
    assert result["promotion"]["walk_forward_sha256"] == checksum(walk)
    result = call(
        "model",
        "quarantine",
        "one",
        "--registry",
        tmp_path / "models",
        "--reason",
        "severe schema drift",
    )
    assert result["status"] == "quarantined"
    events = LifecycleRegistry(tmp_path / "models").events()
    assert len(events) == 3
