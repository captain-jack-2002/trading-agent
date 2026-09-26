import logging
from collections.abc import Callable
from datetime import datetime
from uuid import uuid4

from trading_agent.config.settings import Settings
from trading_agent.execution.base import BrokerExecutionAdapter
from trading_agent.grounding import (
    FactClass,
    FactRequirement,
    GroundingRequest,
    GroundingService,
    ModelInference,
    evidence_from_inference,
    evidence_from_portfolio,
    evidence_from_quote,
    evidence_from_risk_settings,
)
from trading_agent.market_data.interfaces import QuoteProvider
from trading_agent.ml.lifecycle import digest
from trading_agent.models.decision import DecisionRecord
from trading_agent.models.domain import (
    ExecutableMarketQuote,
    OrderRequest,
    OrderResult,
    PortfolioSnapshot,
    Quote,
    RiskDecision,
)
from trading_agent.nse.mock import utc_now
from trading_agent.persistence.store import Store
from trading_agent.persistence.tables import (
    AuditRecord,
    FillRecord,
    ObservationRecord,
    OrderRecord,
    RiskRecord,
)
from trading_agent.portfolio.accounting import apply_fill, mark_portfolio
from trading_agent.risk.engine import RiskEngine

logger = logging.getLogger(__name__)


class PaperBrokerAdapter(BrokerExecutionAdapter):
    """Only execution entry point: load/mark -> risk -> fill -> durable commit.

    No caller-supplied approval, price, portfolio, or alternate unguarded fill API.
    """

    def __init__(
        self,
        settings: Settings,
        store: Store,
        provider: QuoteProvider,
        clock: Callable[[], datetime] = utc_now,
    ):
        self.settings = settings
        self.store = store
        self.provider = provider
        self.clock = clock
        self._risk = RiskEngine(settings)

    def _quote(self, symbol: str) -> Quote:
        quote = self.provider.quote(symbol)
        if not isinstance(quote, Quote):
            raise ValueError("configured provider did not return a typed executable quote")
        quote = Quote.model_validate(quote.model_dump())
        if "mcp" in quote.source.lower():
            raise ValueError("MCP quote lineage is informational only")
        return quote

    def _validate_held_quotes(self, quotes: dict[str, Quote], now: datetime) -> None:
        for symbol, q in quotes.items():
            age = (now - q.timestamp).total_seconds()
            if (
                q.instrument.symbol != symbol
                or q.instrument.instrument_type != "equity"
                or not 0 <= age <= self.settings.quote_max_age_seconds
            ):
                raise ValueError("invalid_portfolio_quote")

    def portfolio(self) -> PortfolioSnapshot:
        with self.store.transaction() as session:
            p = self.store.load_locked(session)
            quotes = {x.symbol: self._quote(x.symbol) for x in p.positions}
            now = self.clock()
            self._validate_held_quotes(quotes, now)
            p = mark_portfolio(p, quotes, now)
            self.store.save(session, p)
        return p

    def submit(self, order: OrderRequest) -> OrderResult:
        return self._submit(order)[0]

    def _submit(
        self,
        order: OrderRequest,
        provenance: DecisionRecord | None = None,
    ) -> tuple[OrderResult, DecisionRecord | None]:
        # Revalidate at boundary, including model_construct/model_copy inputs.
        order = OrderRequest.model_validate(order.model_dump())
        with self.store.transaction() as session:
            portfolio = self.store.load_locked(session)
            now = self.clock()
            is_model = provenance is not None
            if provenance is None:
                provenance = DecisionRecord(
                    decision_id=str(uuid4()),
                    instrument=order.symbol,
                    event_timestamp=now,
                    dataset_versions=(),
                    feature_version="not_applicable",
                    features_sha256=digest([]),
                    proposal=order,
                )
            seen = (
                {order.client_order_id}
                if session.get(OrderRecord, order.client_order_id)
                else set()
            )
            quote: Quote | None = None
            try:
                held = {x.symbol: self._quote(x.symbol) for x in portfolio.positions}
                quote = held.get(order.symbol)
                if quote is None:
                    quote = self._quote(order.symbol)
                now = self.clock()
                self._validate_held_quotes(held, now)
                portfolio = mark_portfolio(portfolio, held, now)
                decision = self._risk.evaluate(order, quote, portfolio, seen, now)
            except (LookupError, ValueError, TypeError, OSError, TimeoutError):
                decision = RiskDecision(
                    approved=False,
                    reasons=("invalid_portfolio_quote", "quote_unavailable_or_invalid"),
                    evaluated_at=now,
                    client_order_id=order.client_order_id,
                )
            tools = [
                evidence_from_portfolio(portfolio),
                evidence_from_risk_settings(self.settings, as_of=now),
            ]
            requirements = [
                FactRequirement(
                    fact_class=FactClass.EXECUTABLE_PRICE,
                    key=order.symbol,
                    max_age_seconds=self.settings.quote_max_age_seconds,
                ),
                FactRequirement(fact_class=FactClass.ACCOUNT_BALANCE, key="cash"),
                FactRequirement(fact_class=FactClass.RISK_LIMIT, key="max_capital_per_trade"),
            ]
            if quote is not None:
                tools.append(evidence_from_quote(ExecutableMarketQuote(quote=quote)))
            if provenance.prediction is not None and provenance.model_id is not None:
                tools.append(
                    evidence_from_inference(
                        ModelInference(
                            model_id=provenance.model_id,
                            symbol=order.symbol,
                            timestamp=provenance.event_timestamp,
                            probability=provenance.prediction
                            if provenance.prediction_kind == "positive_probability"
                            else None,
                            score=provenance.prediction
                            if provenance.prediction_kind == "future_return"
                            else None,
                        )
                    )
                )
                requirements.append(
                    FactRequirement(
                        fact_class=FactClass.MODEL_PROBABILITY
                        if provenance.prediction_kind == "positive_probability"
                        else FactClass.MODEL_SCORE,
                        key=f"{provenance.model_id}:{order.symbol}",
                    )
                )
            packet = GroundingService().query(
                GroundingRequest(
                    query="paper execution facts",
                    as_of=now,
                    fact_requirements=tuple(requirements),
                    tool_evidence=tuple(tools),
                )
            )
            facts = provenance.fact_decisions + packet.decisions
            provenance = provenance.model_copy(
                update={
                    "evidence_ids": tuple(
                        dict.fromkeys(
                            (*provenance.evidence_ids, *(e.evidence_id for e in packet.evidence))
                        )
                    ),
                    "source_hashes": tuple(
                        dict.fromkeys(
                            (
                                *provenance.source_hashes,
                                *(e.provenance.source_hash for e in packet.evidence),
                            )
                        )
                    ),
                    "fact_decisions": facts,
                    "fact_policy_sha256": digest([d.model_dump(mode="json") for d in facts]),
                    "unsupported_claims": provenance.unsupported_claims
                    + tuple(d.requirement.key for d in packet.decisions if d.status != "supported"),
                }
            )
            if packet.status == "abstained" and decision.approved:
                decision = RiskDecision(
                    approved=False,
                    reasons=("authoritative_fact_evidence_invalid",),
                    evaluated_at=now,
                    client_order_id=order.client_order_id,
                )
            # Phase 1 ledger supports only equities, even if policy is widened.
            if order.instrument_type != "equity":
                decision = RiskDecision(
                    approved=False,
                    reasons=(*decision.reasons, "unsupported_paper_instrument"),
                    evaluated_at=now,
                    client_order_id=order.client_order_id,
                )
            result = OrderResult(
                client_order_id=order.client_order_id, status="rejected", risk=decision
            )
            if decision.approved and quote is not None:
                portfolio = apply_fill(portfolio, order, quote, now)
                result = OrderResult(
                    client_order_id=order.client_order_id,
                    status="filled",
                    filled_quantity=order.quantity,
                    fill_price=quote.price,
                    risk=decision,
                )
            if quote is not None:
                session.add(
                    ObservationRecord(created_at=now, payload=quote.model_dump(mode="json"))
                )
            session.add(RiskRecord(created_at=now, payload=decision.model_dump(mode="json")))
            if not seen:
                session.add(
                    OrderRecord(
                        client_order_id=order.client_order_id,
                        created_at=now,
                        payload={
                            "request": order.model_dump(mode="json"),
                            "result": result.model_dump(mode="json"),
                        },
                    )
                )
                session.flush()  # Order FK must exist before its fill.
            if result.status == "filled":
                session.add(
                    FillRecord(
                        created_at=now,
                        client_order_id=order.client_order_id,
                        payload=result.model_dump(mode="json"),
                    )
                )
            provenance = provenance.model_copy(
                update={
                    "risk_decision": decision,
                    "execution": result,
                    "outcome": result.status,
                    "market_data_sha256": digest(quote.model_dump(mode="json")) if quote else None,
                }
            )
            session.add(
                AuditRecord(
                    created_at=now,
                    payload={
                        "event": "paper_order",
                        "timestamp": now.isoformat(),
                        "request": order.model_dump(mode="json"),
                        "result": result.model_dump(mode="json"),
                        "decision": provenance.model_dump(mode="json"),
                    },
                )
            )
            if is_model:
                session.add(
                    AuditRecord(
                        created_at=now,
                        payload={
                            "event": "model_decision",
                            "decision": provenance.model_dump(mode="json"),
                        },
                    )
                )
            self.store.save(session, portfolio)
        logger.info(
            "paper_order", extra={"order_id": order.client_order_id, "status": result.status}
        )
        return result, provenance
