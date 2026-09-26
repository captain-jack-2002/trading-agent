from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_agent.config.settings import Settings
from trading_agent.grounding import (
    FactClass,
    FactRequirement,
    GroundingDocument,
    GroundingRequest,
    GroundingService,
    LocalGroundingIndex,
    ModelInference,
    ModelMetric,
    SourceType,
    evidence_from_inference,
    evidence_from_metric,
    evidence_from_portfolio,
    evidence_from_risk_settings,
    load_documents,
)
from trading_agent.models.domain import PortfolioSnapshot

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_authoritative_portfolio_settings_inference_and_metric_facts() -> None:
    portfolio = PortfolioSnapshot(
        cash=Decimal("1000"),
        equity=Decimal("1100"),
        day_start_equity=Decimal("1000"),
        as_of=NOW,
    )
    inference = ModelInference(
        model_id="m1",
        symbol="TEST",
        probability=0.8,
        score=0.2,
        timestamp=NOW,
    )
    metric = ModelMetric(
        model_id="m1",
        metric="f1",
        value=0.7,
        source_id="data/models/m1/metadata.json",
        timestamp=NOW,
    )
    request = GroundingRequest(
        query="account model risk",
        as_of=NOW,
        fact_requirements=(
            FactRequirement(fact_class=FactClass.ACCOUNT_EQUITY, key="equity"),
            FactRequirement(fact_class=FactClass.ACCOUNT_BALANCE, key="cash"),
            FactRequirement(fact_class=FactClass.RISK_LIMIT, key="max_open_positions"),
            FactRequirement(fact_class=FactClass.MODEL_PROBABILITY, key="m1:TEST"),
            FactRequirement(fact_class=FactClass.MODEL_METRIC, key="m1:f1"),
        ),
        tool_evidence=(
            evidence_from_portfolio(portfolio),
            evidence_from_risk_settings(Settings(), as_of=NOW),
            evidence_from_inference(inference),
            evidence_from_metric(metric),
        ),
    )
    result = GroundingService(None).query(request)
    assert result.status == "supported"
    assert [decision.value for decision in result.decisions] == [
        "1100",
        "1000",
        "5",
        "0.8",
        "0.7",
    ]
    assert all(decision.evidence_ids for decision in result.decisions)
    assert isinstance(request.tool_evidence, tuple)
    assert isinstance(result.decisions, tuple)
    assert isinstance(result.evidence, tuple)


def test_generic_retrieval_cannot_be_relabelled_as_tool_evidence() -> None:
    index = LocalGroundingIndex.from_documents(
        [
            GroundingDocument(
                source_id="docs/a.md",
                content="TEST price 100",
                source_type=SourceType.REPOSITORY_DOC,
            ),
        ]
    )
    retrieved = (
        GroundingService(index).query(GroundingRequest(query="price", as_of=NOW)).evidence[0]
    )
    forged = retrieved.model_dump()
    forged["provenance"]["source_type"] = SourceType.EXECUTABLE_QUOTE
    forged["provenance"]["executable_price"] = True
    with pytest.raises(ValidationError):
        GroundingRequest.model_validate({"query": "price", "as_of": NOW, "tool_evidence": [forged]})


def test_freshness_source_filters_unknown_timestamp_and_future() -> None:
    documents = [
        GroundingDocument(
            source_id="fresh",
            content="alpha model",
            source_type=SourceType.MODEL_CARD,
            source_timestamp=NOW,
        ),
        GroundingDocument(
            source_id="stale",
            content="alpha model stale",
            source_type=SourceType.MODEL_CARD,
            source_timestamp=NOW - timedelta(seconds=61),
        ),
        GroundingDocument(
            source_id="future",
            content="alpha model future",
            source_type=SourceType.MODEL_CARD,
            source_timestamp=NOW + timedelta(seconds=1),
        ),
        GroundingDocument(
            source_id="unknown", content="alpha model unknown", source_type=SourceType.MODEL_CARD
        ),
    ]
    result = GroundingService(LocalGroundingIndex.from_documents(documents)).query(
        GroundingRequest(
            query="alpha",
            as_of=NOW,
            max_age_seconds=60,
            source_ids=("fresh", "stale", "future", "unknown"),
        ),
    )
    assert [e.provenance.source_id for e in result.evidence] == ["fresh"]


def test_empty_and_punctuation_documents_and_single_letter_tokens() -> None:
    index = LocalGroundingIndex.from_documents(
        [
            GroundingDocument(
                source_id="blank", content="... ", source_type=SourceType.REPOSITORY_DOC
            ),
            GroundingDocument(
                source_id="single", content="x y", source_type=SourceType.REPOSITORY_DOC
            ),
        ]
    )
    assert (
        GroundingService(index).query(GroundingRequest(query="x", as_of=NOW)).status == "supported"
    )
    assert (
        GroundingService(LocalGroundingIndex.from_documents([]))
        .query(GroundingRequest(query="x", as_of=NOW))
        .status
        == "abstained"
    )


def test_loader_requires_explicit_trust_for_artifacts_and_skips_symlink_escape(
    tmp_path: Path,
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("approved alpha")
    secret = tmp_path / "outside.md"
    secret.write_text("outside secret")
    (root / "link.md").symlink_to(secret)
    research = root / "research.json"
    research.write_text('{"trusted":true,"validated":true,"content":"alpha beta"}')
    assert [doc.source_id for doc in load_documents(root)] == [str(root / "a.md")]
    documents = load_documents(root, trusted_artifacts=(research,))
    assert len(documents) == 2
    assert documents[1].source_type == SourceType.VALIDATED_RESEARCH
    research.write_text('{"trusted":true,"content":"alpha"}')
    with pytest.raises(ValueError, match="validated"):
        load_documents(root, trusted_artifacts=(research,))


def test_model_metric_text_does_not_authorize_numeric_metric() -> None:
    index = LocalGroundingIndex.from_documents(
        [
            GroundingDocument(
                source_id="model-card.md",
                content="model f1 0.99",
                source_type=SourceType.MODEL_CARD,
            ),
        ]
    )
    result = GroundingService(index).query(
        GroundingRequest(
            query="model f1",
            as_of=NOW,
            fact_requirements=(FactRequirement(fact_class=FactClass.MODEL_METRIC, key="m1:f1"),),
        )
    )
    assert result.status == "abstained"
    assert result.decisions[0].value is None
