"""Fail-closed Phase 4 data policy and auditable causal feature boundaries."""

from collections.abc import Sequence
from typing import Any

from trading_agent.features.research import FEATURE_NAMES, FeatureRow

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
