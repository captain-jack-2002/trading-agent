"""Fail-closed Phase 4 data policy and auditable causal feature boundaries."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from trading_agent.features.research import FEATURE_NAMES, FeatureRow, ResearchBar
from trading_agent.ml.dataset import DatasetRow, build_dataset

SYNTHETIC_WARNING = "SYNTHETIC engineering validation only; no expected market performance"


def require_synthetic(provenance: str, synthetic: bool) -> None:
    if "mcp" in provenance.lower():
        raise ValueError("MCP data is informational only and never training eligible")
    # Legacy test fixtures predate the explicit synthetic flag.
    if not synthetic and "synthetic" not in provenance.lower():
        raise ValueError("Phase 4 permits SYNTHETIC fixtures only")


def audit_features(rows: Sequence[FeatureRow]) -> dict[str, Any]:
    if not rows:
        raise ValueError("nonempty dataset required")
    for row in rows:
        require_synthetic(row.provenance, row.synthetic)
        if set(row.values) - FEATURE_NAMES:
            raise ValueError("unknown feature column: possible target/future leakage")
    if len({(r.instrument_id, r.timestamp) for r in rows}) != len(rows):
        raise ValueError("duplicate samples/instrument timestamps")
    if any(set(r.values) != set(rows[0].values) for r in rows):
        raise ValueError("inconsistent feature schema")
    return {
        "warning": SYNTHETIC_WARNING,
        "synthetic": True,
        "duplicate_samples": 0,
        "unknown_feature_columns": 0,
        "instrument_policy": "instrument-local features; global timestamp partitions",
        "limitation": "schema alone cannot prove causality; CLI rebuilds supplied values",
    }


def verify_dataset_rebuild(path: Path, rows: list[DatasetRow]) -> None:
    """Recompute supplied features/labels from fixture bars before fitting anything."""
    document = json.loads(path.read_text())
    if not rows or not document.get("source_bars"):
        raise ValueError("causal source rebuild requires source_bars; rerun features build")
    if document.get("training_eligible") is False or document.get("informational_only") is True:
        raise ValueError("informational/MCP data cannot be used for training")
    source = [ResearchBar.model_validate(bar) for bar in document["source_bars"]]
    for bar in source:
        require_synthetic(bar.provenance, bar.synthetic)
    first = rows[0]
    rebuilt = build_dataset(
        source,
        horizon=first.horizon,
        threshold=first.threshold,
        upper_barrier=first.upper_barrier,
        lower_barrier=first.lower_barrier,
    )
    if rebuilt != rows:
        raise ValueError("causal feature/target rebuild mismatch")
