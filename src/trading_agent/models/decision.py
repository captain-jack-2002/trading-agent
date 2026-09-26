"""Compact immutable decision provenance; no raw retrieval text or credentials."""

from typing import Literal

from pydantic import AwareDatetime, Field

from trading_agent.grounding.contracts import GroundingDecision
from trading_agent.ml.drift import ModelHealth
from trading_agent.models.domain import DomainModel, OrderRequest, OrderResult, RiskDecision


class DecisionRecord(DomainModel):
    schema_version: Literal["decision-v1"] = "decision-v1"
    decision_id: str
    instrument: str
    event_timestamp: AwareDatetime
    dataset_versions: tuple[str, ...]
    feature_version: str
    features_sha256: str
    feature_timestamp: AwareDatetime | None = None
    model_id: str | None = None
    model_version: str | None = None
    model_artifact_sha256: str | None = None
    model_metadata_sha256: str | None = None
    prediction: float | None = Field(default=None, allow_inf_nan=False)
    prediction_kind: Literal["positive_probability", "future_return"] | None = None
    model_health: Literal["healthy", "watch", "quarantined"] | None = None
    drift_reference: str | None = None
    health: ModelHealth | None = None
    lifecycle_reference: str | None = None
    evidence_ids: tuple[str, ...] = ()
    source_hashes: tuple[str, ...] = ()
    fact_policy_sha256: str | None = None
    fact_decisions: tuple[GroundingDecision, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    proposal: OrderRequest
    risk_decision: RiskDecision | None = None
    execution: OrderResult | None = None
    market_data_sha256: str | None = None
    outcome: Literal["proposed", "abstained", "rejected", "filled"] = "proposed"
    reasons: tuple[str, ...] = ()
