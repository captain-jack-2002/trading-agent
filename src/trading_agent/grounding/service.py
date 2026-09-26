"""Evidence composition and fail-closed authoritative fact policy."""

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from .contracts import (
    TIME_SENSITIVE_SOURCES,
    FactClass,
    FactRequirement,
    GroundingDecision,
    GroundingRequest,
    GroundingResult,
    RetrievedEvidence,
    SourceProvenance,
    SourceType,
    ToolEvidence,
    content_hash,
    is_fresh,
)
from .retrieval import GroundingRetriever, LocalGroundingIndex

ALLOWED_FACT_SOURCES: Mapping[FactClass, frozenset[SourceType]] = MappingProxyType(
    {
        FactClass.EXECUTABLE_PRICE: frozenset({SourceType.EXECUTABLE_QUOTE}),
        FactClass.ACCOUNT_BALANCE: frozenset({SourceType.LEDGER}),
        FactClass.ACCOUNT_EQUITY: frozenset({SourceType.LEDGER}),
        FactClass.OPEN_POSITION: frozenset({SourceType.LEDGER}),
        FactClass.ORDER_STATE: frozenset({SourceType.ORDER_ADAPTER}),
        FactClass.RISK_LIMIT: frozenset({SourceType.RISK_CONFIG}),
        FactClass.MODEL_PROBABILITY: frozenset({SourceType.MODEL_INFERENCE}),
        FactClass.MODEL_SCORE: frozenset({SourceType.MODEL_INFERENCE}),
        FactClass.MODEL_METRIC: frozenset({SourceType.REGISTRY_METADATA, SourceType.PHASE_REPORT}),
        FactClass.MARKET_VOLUME: frozenset({SourceType.MARKET_DATA}),
        FactClass.OPEN_INTEREST: frozenset({SourceType.MARKET_DATA}),
        FactClass.DERIVATIVE_FIELD: frozenset({SourceType.MARKET_DATA}),
    }
)


def _tool_packet(tool: ToolEvidence, request: GroundingRequest) -> RetrievedEvidence:
    digest = content_hash(tool.content)
    identity = json.dumps(
        [
            tool.source_type,
            tool.source_id,
            digest,
            tool.source_timestamp.isoformat() if tool.source_timestamp else None,
        ]
    )
    executable = tool.source_type == SourceType.EXECUTABLE_QUOTE and bool(tool.authoritative_facts)
    return RetrievedEvidence(
        evidence_id=content_hash(identity),
        content=tool.content,
        provenance=SourceProvenance(
            source_id=tool.source_id,
            source_type=tool.source_type,
            source_hash=digest,
            content_hash=digest,
            retrieved_at=request.as_of,
            source_timestamp=tool.source_timestamp,
            retrieval_score=1,
            trust_class="typed_adapter" if tool.authoritative_facts else "informational",
            informational_only=not bool(tool.authoritative_facts),
            executable_price=executable,
        ),
    )


def _decide(
    requirement: FactRequirement,
    request: GroundingRequest,
    tools: tuple[tuple[ToolEvidence, RetrievedEvidence], ...],
) -> GroundingDecision:
    candidates = [
        (tool, packet, fact)
        for tool, packet in tools
        if tool.source_type in ALLOWED_FACT_SOURCES[requirement.fact_class]
        for fact in tool.authoritative_facts
        if fact.fact_class == requirement.fact_class and fact.key == requirement.key
    ]
    if not candidates:
        return GroundingDecision(
            requirement=requirement,
            status="unsupported",
            reason="Required authoritative typed evidence is absent",
        )
    fresh = [
        (tool, packet, fact)
        for tool, packet, fact in candidates
        if is_fresh(tool.source_timestamp, request.as_of, requirement.max_age_seconds)
    ]
    if not fresh:
        return GroundingDecision(
            requirement=requirement,
            status="stale",
            evidence_ids=tuple(sorted({p.evidence_id for _, p, _ in candidates})),
            reason="Authoritative evidence is stale, future-dated, or has no source timestamp",
        )
    ids = tuple(sorted({packet.evidence_id for _, packet, _ in fresh}))
    if any(packet.provenance.retrieval_score < requirement.min_score for _, packet, _ in fresh):
        return GroundingDecision(
            requirement=requirement,
            status="below_threshold",
            evidence_ids=ids,
            reason="Authoritative evidence below required score",
        )
    values = {fact.value for _, _, fact in fresh}
    if len(values) != 1:
        return GroundingDecision(
            requirement=requirement,
            status="conflicting",
            evidence_ids=ids,
            reason="Authoritative sources disagree; no value selected",
        )
    return GroundingDecision(
        requirement=requirement,
        status="supported",
        evidence_ids=ids,
        value=next(iter(values)),
        reason="Supported by fresh authoritative typed evidence",
    )


class GroundingService:
    def __init__(self, index: GroundingRetriever | None = None):
        self.index = index

    def query(self, request: GroundingRequest) -> GroundingResult:
        documents = self.index.retrieve(request) if self.index else ()
        tools = tuple(
            (tool, _tool_packet(tool, request))
            for tool in request.tool_evidence
            if (not request.source_types or tool.source_type in request.source_types)
            and (not request.source_ids or tool.source_id in request.source_ids)
        )
        decisions = tuple(
            _decide(requirement, request, tools) for requirement in request.fact_requirements
        )
        required_ids = {eid for decision in decisions for eid in decision.evidence_ids}
        # All matching typed observations are assessed before retrieval top-k, so a
        # low top-k cannot conceal contradictory authoritative values.
        packets = list(documents)
        seen_ids = {p.evidence_id for p in packets}
        for tool, packet in tools:
            fresh = tool.source_type not in TIME_SENSITIVE_SOURCES or is_fresh(
                tool.source_timestamp, request.as_of, request.max_age_seconds or 300
            )
            if packet.evidence_id not in seen_ids and (fresh or packet.evidence_id in required_ids):
                packets.append(packet)
                seen_ids.add(packet.evidence_id)
        reasons = tuple(d.reason for d in decisions if d.status != "supported")
        supported = not reasons and bool(packets)
        if not packets:
            reasons += ("Insufficient evidence: no relevant trusted evidence is available",)
        return GroundingResult(
            status="supported" if supported else "abstained",
            evidence=tuple(packets),
            decisions=decisions,
            reasons=reasons,
        )


def query_index(path: Path, request: GroundingRequest) -> GroundingResult:
    try:
        index = LocalGroundingIndex.load(path)
    except (OSError, ValueError):
        index = None
    return GroundingService(index).query(request)
