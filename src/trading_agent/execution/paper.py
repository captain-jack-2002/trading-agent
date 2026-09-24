import logging
from collections.abc import Callable
from datetime import datetime

from trading_agent.config.settings import Settings
from trading_agent.execution.base import BrokerExecutionAdapter
from trading_agent.market_data.interfaces import QuoteProvider
from trading_agent.models.domain import (
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
            quotes = {x.symbol: self.provider.quote(x.symbol) for x in p.positions}
            now = self.clock()
            self._validate_held_quotes(quotes, now)
            p = mark_portfolio(p, quotes, now)
            self.store.save(session, p)
        return p

    def submit(self, order: OrderRequest) -> OrderResult:
        # Revalidate at boundary, including model_construct/model_copy inputs.
        order = OrderRequest.model_validate(order.model_dump())
        with self.store.transaction() as session:
            portfolio = self.store.load_locked(session)
            now = self.clock()
            seen = (
                {order.client_order_id}
                if session.get(OrderRecord, order.client_order_id)
                else set()
            )
            quote: Quote | None = None
            try:
                held = {x.symbol: self.provider.quote(x.symbol) for x in portfolio.positions}
                quote = held.get(order.symbol)
                if quote is None:
                    quote = self.provider.quote(order.symbol)
                now = self.clock()
                self._validate_held_quotes(held, now)
                portfolio = mark_portfolio(portfolio, held, now)
                decision = self._risk.evaluate(order, quote, portfolio, seen, now)
            except (LookupError, ValueError):
                decision = RiskDecision(
                    approved=False,
                    reasons=("invalid_portfolio_quote", "quote_unavailable_or_invalid"),
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
            session.add(
                AuditRecord(
                    created_at=now,
                    payload={
                        "event": "paper_order",
                        "timestamp": now.isoformat(),
                        "request": order.model_dump(mode="json"),
                        "result": result.model_dump(mode="json"),
                    },
                )
            )
            self.store.save(session, portfolio)
        logger.info(
            "paper_order", extra={"order_id": order.client_order_id, "status": result.status}
        )
        return result
