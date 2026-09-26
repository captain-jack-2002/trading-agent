"""Immutable evidence contracts shared by retrieval and the fact policy."""

from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Literal

from pydantic import AwareDatetime, ConfigDict, Field, PrivateAttr, model_validator

from trading_agent.models.domain import DomainModel


class SourceType(StrEnum):
    REPOSITORY_DOC = "repository_doc"
    PHASE_REPORT = "phase_report"
    MODEL_CARD = "model_card"
    REGISTRY_METADATA = "registry_metadata"
    VALIDATED_RESEARCH = "validated_research"
    EXECUTABLE_QUOTE = "executable_quote"
    LEDGER = "ledger"
    ORDER_ADAPTER = "order_adapter"
    RISK_CONFIG = "risk_config"
    MODEL_INFERENCE = "model_inference"
    MARKET_DATA = "market_data"
    NSE_MCP = "nse_mcp"


DOCUMENT_SOURCES = frozenset(
    {
        SourceType.REPOSITORY_DOC,
        SourceType.PHASE_REPORT,
        SourceType.MODEL_CARD,
        SourceType.REGISTRY_METADATA,
        SourceType.VALIDATED_RESEARCH,
    }
)
TIME_SENSITIVE_SOURCES = frozenset(set(SourceType) - DOCUMENT_SOURCES)


class FactClass(StrEnum):
    EXECUTABLE_PRICE = "executable_price"
    ACCOUNT_BALANCE = "account_balance"
    ACCOUNT_EQUITY = "account_equity"
    OPEN_POSITION = "open_position"
    ORDER_STATE = "order_state"
    RISK_LIMIT = "risk_limit"
    MODEL_PROBABILITY = "model_probability"
    MODEL_SCORE = "model_score"
    MODEL_METRIC = "model_metric"
    MARKET_VOLUME = "market_volume"
    OPEN_INTEREST = "open_interest"
    DERIVATIVE_FIELD = "derivative_field"


def content_hash(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


class GroundingContract(DomainModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class GroundingDocument(GroundingContract):
    source_id: Annotated[str, Field(min_length=1)]
    content: Annotated[str, Field(min_length=1)]
    source_type: SourceType
    source_timestamp: AwareDatetime | None = None
    explicitly_trusted: bool = False

    @model_validator(mode="after")
    def trusted_document_class(self) -> "GroundingDocument":
        if self.source_type not in DOCUMENT_SOURCES:
            raise ValueError("tool/provider sources cannot be ingested as documents")
        if self.source_type == SourceType.VALIDATED_RESEARCH and not self.explicitly_trusted:
            raise ValueError("validated research requires explicit trust")
        return self


class SourceProvenance(GroundingContract):
    source_id: str
    source_type: SourceType
    source_hash: str
    content_hash: str
    retrieved_at: AwareDatetime
    source_timestamp: AwareDatetime | None = None
    retrieval_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    trust_class: Literal["trusted_document", "typed_adapter", "informational"]
    informational_only: bool = True
    training_eligible: Literal[False] = False
    executable_price: bool = False

    @model_validator(mode="after")
    def enforce_flags(self) -> "SourceProvenance":
        if self.executable_price and self.source_type != SourceType.EXECUTABLE_QUOTE:
            raise ValueError("executable flag must match typed executable quote source")
        if self.source_type == SourceType.NSE_MCP and not self.informational_only:
            raise ValueError("NSE MCP must remain informational only")
        return self


class GroundingChunk(GroundingContract):
    evidence_id: str
    source_id: str
    source_type: SourceType
    source_hash: str
    content_hash: str
    content: str
    ordinal: Annotated[int, Field(ge=0)]
    source_timestamp: AwareDatetime | None = None


class RetrievedEvidence(GroundingContract):
    evidence_id: str
    content: str
    provenance: SourceProvenance


class FactRequirement(GroundingContract):
    fact_class: FactClass
    key: Annotated[str, Field(min_length=1)]
    min_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0
    max_age_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 60


class FactObservation(GroundingContract):
    fact_class: FactClass
    key: str
    value: str


class ToolEvidence(GroundingContract):
    """Request-time evidence. Only typed adapter factories attach authoritative facts.

    JSON/deserialized/free-form instances deliberately have no authoritative facts.
    This is an application trust boundary, not a sandbox for hostile Python code.
    """

    source_id: str
    source_type: SourceType
    source_timestamp: AwareDatetime | None = None
    content: str
    _facts: tuple[FactObservation, ...] = PrivateAttr(default=())
    _binding: str | None = PrivateAttr(default=None)

    @property
    def authoritative_facts(self) -> tuple[FactObservation, ...]:
        # Copies/deserialization or relabeling never retain authority over changed content.
        return self._facts if self._binding == content_hash(self.model_dump_json()) else ()

    def __setattr__(self, name: str, value: object) -> None:
        if name in ("_facts", "_binding"):
            raise TypeError("authoritative observations are immutable adapter output")
        super().__setattr__(name, value)


class GroundingRequest(GroundingContract):
    query: Annotated[str, Field(min_length=1)]
    as_of: AwareDatetime
    top_k: Annotated[int, Field(gt=0, le=1000)] = 5
    min_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.1
    source_types: tuple[SourceType, ...] = ()
    source_ids: tuple[str, ...] = ()
    max_age_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    fact_requirements: tuple[FactRequirement, ...] = ()
    tool_evidence: tuple[ToolEvidence, ...] = ()


class GroundingDecision(GroundingContract):
    requirement: FactRequirement
    status: Literal["supported", "unsupported", "stale", "conflicting", "below_threshold"]
    value: str | None = None
    evidence_ids: tuple[str, ...] = ()
    reason: str


class GroundingResult(GroundingContract):
    status: Literal["supported", "abstained"]
    evidence: tuple[RetrievedEvidence, ...] = ()
    decisions: tuple[GroundingDecision, ...] = ()
    reasons: tuple[str, ...] = ()


def is_fresh(timestamp: datetime | None, as_of: datetime, max_age: float | None) -> bool:
    if timestamp is None:
        return max_age is None
    age = (as_of - timestamp).total_seconds()
    return age >= 0 and (max_age is None or age <= max_age)
