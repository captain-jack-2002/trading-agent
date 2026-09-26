from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_agent.grounding import (
    FactClass,
    FactRequirement,
    GroundingDocument,
    GroundingRequest,
    GroundingService,
    LocalGroundingIndex,
    SourceType,
    build_index,
    chunk_document,
    evidence_from_quote,
)
from trading_agent.models.domain import ExecutableMarketQuote, Instrument, Quote

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def document(
    path: str = "docs/a.md", content: str = "Model limitations include synthetic data."
) -> GroundingDocument:
    return GroundingDocument(source_id=path, content=content, source_type=SourceType.REPOSITORY_DOC)


def request(**kwargs: object) -> GroundingRequest:
    return GroundingRequest.model_validate({"query": "model limitations", "as_of": NOW, **kwargs})


def test_chunks_and_ranking_are_deterministic_and_keep_hashes() -> None:
    doc = document(content="alpha beta gamma delta epsilon zeta")
    assert chunk_document(doc, chunk_words=3) == chunk_document(doc, chunk_words=3)
    assert len(chunk_document(doc, chunk_words=3)) == 2
    index = LocalGroundingIndex.from_documents([document("b.md"), document("a.md")])
    result = GroundingService(index).query(request())
    assert result == GroundingService(index).query(request())
    assert result.status == "supported"
    assert len(result.evidence) == 1  # duplicate content suppression
    assert result.evidence[0].provenance.source_id == "a.md"
    assert len(result.evidence[0].provenance.content_hash) == 64
    assert result.evidence[0].provenance.retrieved_at == NOW
    assert not result.evidence[0].provenance.executable_price
    with pytest.raises(ValidationError):
        doc.content = "changed"  # type: ignore[misc]


def test_top_k_threshold_filters_and_missing_index_abstain() -> None:
    index = LocalGroundingIndex.from_documents(
        [document(), document("b.md", "model limitations risk")]
    )
    assert len(GroundingService(index).query(request(top_k=1)).evidence) == 1
    assert GroundingService(index).query(request(min_score=1)).status == "abstained"
    assert (
        GroundingService(index).query(request(source_types=[SourceType.MODEL_CARD])).status
        == "abstained"
    )
    assert GroundingService(None).query(request()).status == "abstained"
    assert GroundingService(index).query(request(query="unfindable words")).status == "abstained"


def test_json_roundtrip_inspection_and_corruption_detection(tmp_path: Path) -> None:
    index = LocalGroundingIndex.from_documents([document()])
    index.save(tmp_path)
    loaded = LocalGroundingIndex.load(tmp_path)
    evidence = GroundingService(loaded).query(request()).evidence[0]
    assert loaded.inspect(evidence.evidence_id).content == evidence.content
    assert loaded.inspect("absent") is None
    target = tmp_path / "index.json"
    target.write_text(target.read_text().replace("synthetic", "fabricated"))
    with pytest.raises(ValueError, match="hash|checksum"):
        LocalGroundingIndex.load(tmp_path)


def test_build_index_only_reads_explicit_trusted_sources(tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "notes.md").write_text("model limitations synthetic validation")
    (source / "untrusted.json").write_text('{"text": "never ingest by default"}')
    index = build_index(source, tmp_path / "index")
    assert len(index.chunks) == 1


def quote(price: str = "100", timestamp: datetime = NOW) -> ExecutableMarketQuote:
    return ExecutableMarketQuote(
        quote=Quote(instrument=Instrument(symbol="TEST"), price=Decimal(price), timestamp=timestamp)
    )


def price_request(**kwargs: object) -> GroundingRequest:
    return request(
        fact_requirements=[
            FactRequirement(fact_class=FactClass.EXECUTABLE_PRICE, key="TEST", max_age_seconds=60)
        ],
        **kwargs,
    )


def test_rag_text_cannot_supply_an_executable_price() -> None:
    index = LocalGroundingIndex.from_documents(
        [document(content="model limitations TEST executable price 100")]
    )
    result = GroundingService(index).query(price_request())
    assert result.status == "abstained"
    assert result.decisions[0].status == "unsupported"
    assert result.decisions[0].value is None
    with pytest.raises(ValueError):
        GroundingDocument(source_id="liar", content="100", source_type=SourceType.EXECUTABLE_QUOTE)


def test_typed_quote_can_support_fact_without_rag_index() -> None:
    result = GroundingService(None).query(
        price_request(tool_evidence=[evidence_from_quote(quote())])
    )
    assert result.status == "supported"
    assert result.decisions[0].value == "100"
    assert result.evidence[0].provenance.executable_price
    assert result.decisions[0].evidence_ids == (result.evidence[0].evidence_id,)


@pytest.mark.parametrize("timestamp", [NOW - timedelta(seconds=61), NOW + timedelta(seconds=1)])
def test_stale_or_future_quote_abstains(timestamp: datetime) -> None:
    result = GroundingService(None).query(
        price_request(tool_evidence=[evidence_from_quote(quote(timestamp=timestamp))])
    )
    assert result.status == "abstained"
    assert result.decisions[0].status == "stale"


def test_conflicting_authoritative_facts_abstain_even_with_top_k_one() -> None:
    result = GroundingService(None).query(
        price_request(
            top_k=1, tool_evidence=[evidence_from_quote(quote()), evidence_from_quote(quote("101"))]
        )
    )
    assert result.status == "abstained"
    assert result.decisions[0].status == "conflicting"
    assert result.decisions[0].value is None
