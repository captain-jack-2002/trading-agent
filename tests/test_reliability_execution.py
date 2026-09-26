"""Engineering validation only: model confidence never authorizes a fill."""

from datetime import timedelta

import pytest
from sqlalchemy import select
from test_execution import broker  # noqa: F401
from test_model_lifecycle import artifact
from test_risk import NOW, order

from trading_agent.features.research import FeatureRow
from trading_agent.ml.lifecycle import LifecycleRegistry
from trading_agent.ml.registry import load_model
from trading_agent.persistence.tables import FillRecord, OrderRecord


def window(bundle):
    # Use the saved training-reference feature distribution, with new event times.
    from test_features_research import bars

    from trading_agent.ml.dataset import build_dataset
    from trading_agent.ml.split import walk_forward

    rows = build_dataset(bars(150), horizon=2)
    split = walk_forward(rows, 70, 25, 25)[0]
    return [
        FeatureRow(
            **{
                **r.model_dump(
                    exclude={
                        "label_end",
                        "horizon",
                        "threshold",
                        "upper_barrier",
                        "lower_barrier",
                        "label_version",
                        "future_return",
                        "direction",
                        "target",
                        "barrier",
                    }
                ),
                "timestamp": NOW - timedelta(seconds=len(split.train) - i),
                "instrument_id": "DEMO",
            }
        )
        for i, r in enumerate(rows[j] for j in split.train)
    ]


def test_healthy_model_valid_quote_still_faces_risk(tmp_path, broker):  # noqa: F811
    from trading_agent.agent.reliability import ModelPaperExecutor

    bundle = artifact(tmp_path / "models")
    executor = ModelPaperExecutor(broker, tmp_path / "models" / "one", trusted=True)
    # Low threshold selects the classifier's positive proposal, independent of risk.
    result = executor.submit(window(bundle), order(quantity=1000), probability_threshold=0.000001)
    assert result.model_health == "healthy"
    assert result.prediction is not None
    assert result.risk_decision is not None and not result.risk_decision.approved
    assert result.outcome == "rejected"
    assert "capital_per_trade" in result.risk_decision.reasons
    assert any(a.get("event") == "model_decision" for a in broker.store.audit())


def test_quarantine_blocks_already_loaded_high_confidence_model(tmp_path, broker):  # noqa: F811
    from trading_agent.agent.reliability import ModelPaperExecutor

    bundle = artifact(tmp_path / "models")
    executor = ModelPaperExecutor(broker, tmp_path / "models" / "one", trusted=True)
    LifecycleRegistry(tmp_path / "models").quarantine("one", reason="severe drift")
    record = executor.submit(window(bundle), order(), probability_threshold=0.000001)
    assert record.outcome == "abstained"
    assert record.model_health == "quarantined"
    with broker.store.sessions() as session:
        assert list(session.scalars(select(OrderRecord))) == []
        assert list(session.scalars(select(FillRecord))) == []


def test_missing_quote_has_no_text_fallback(tmp_path, broker):  # noqa: F811
    from trading_agent.agent.reliability import ModelPaperExecutor

    bundle = artifact(tmp_path / "models")

    class MissingQuotes:
        def quote(self, symbol):
            raise LookupError("unavailable")

    broker.provider = MissingQuotes()
    result = ModelPaperExecutor(broker, tmp_path / "models" / "one", trusted=True).submit(
        window(bundle),
        order(),
        probability_threshold=0.000001,
    )
    assert result.outcome == "rejected"
    assert "quote_unavailable_or_invalid" in result.risk_decision.reasons
    assert result.execution.fill_price is None


def test_paper_success_records_complete_provenance(tmp_path, broker):  # noqa: F811
    from trading_agent.agent.reliability import ModelPaperExecutor

    bundle = artifact(tmp_path / "models")
    result = ModelPaperExecutor(broker, tmp_path / "models" / "one", trusted=True).submit(
        window(bundle),
        order(),
        probability_threshold=0.000001,
    )
    assert result.outcome == "filled"
    assert result.risk_decision.approved
    assert result.dataset_versions and result.features_sha256 and result.drift_reference
    assert result.model_id == bundle.metadata["model_id"]
    assert result.market_data_sha256
    assert result.decision_id


def test_corrupt_registry_and_invalid_schema_abstain(tmp_path, broker):  # noqa: F811
    from trading_agent.agent.reliability import ModelPaperExecutor

    bundle = artifact(tmp_path / "models")
    executor = ModelPaperExecutor(broker, tmp_path / "models" / "one", trusted=True)
    rows = window(bundle)
    rows[-1] = rows[-1].model_copy(update={"values": {}})
    result = executor.submit(rows, order())
    assert result.outcome == "abstained"
    assert result.model_health == "quarantined"
    assert LifecycleRegistry(tmp_path / "models").state("one").status == "quarantined"


def test_loaded_research_model_cannot_ignore_new_quarantine(tmp_path):
    from trading_agent.ml.lifecycle import require_bundle_usable

    artifact(tmp_path)
    bundle = load_model(tmp_path / "one", trusted=True)
    require_bundle_usable(bundle)
    LifecycleRegistry(tmp_path).quarantine("one", reason="new drift")
    with pytest.raises(ValueError, match="quarantined"):
        require_bundle_usable(bundle)
