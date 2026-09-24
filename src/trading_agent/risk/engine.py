from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from trading_agent.config.settings import Settings
from trading_agent.models.domain import OrderRequest, PortfolioSnapshot, Quote, RiskDecision

IST = ZoneInfo("Asia/Kolkata")


class RiskEngine:
    """Pure policy evaluation. A decision is diagnostic, never an execution capability."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def evaluate(
        self,
        order: OrderRequest,
        quote: Quote,
        portfolio: PortfolioSnapshot,
        seen: set[str],
        now: datetime,
    ) -> RiskDecision:
        s = self.settings
        reasons: list[str] = []
        if now.tzinfo is None:
            raise ValueError("aware clock required")
        local = now.astimezone(IST)
        if (
            local.weekday() >= 5
            or local.date().isoformat() in s.holidays
            or not s.market_open <= local.time() < s.market_close
        ):
            reasons.append("market_closed")
        if order.client_order_id in seen:
            reasons.append("duplicate_order")
        if order.symbol not in s.allowed_symbols:
            reasons.append("symbol_not_allowed")
        if order.instrument_type not in s.allowed_instrument_types:
            reasons.append("instrument_type_not_allowed")
        if (
            quote.instrument.symbol != order.symbol
            or quote.instrument.instrument_type != order.instrument_type
        ):
            reasons.append("quote_mismatch")
        age = (now - quote.timestamp).total_seconds()
        if age < 0:
            reasons.append("future_quote")
        if age > s.quote_max_age_seconds:
            reasons.append("stale_quote")
        amount = quote.price * order.quantity
        if amount > s.max_capital_per_trade:
            reasons.append("capital_per_trade")
        if portfolio.daily_pnl <= -s.max_daily_loss:
            reasons.append("daily_loss")
        position = next((p for p in portfolio.positions if p.symbol == order.symbol), None)
        if order.side == "buy":
            if amount > portfolio.cash:
                reasons.append("insufficient_cash")
            if portfolio.exposure + amount > s.max_portfolio_exposure:
                reasons.append("portfolio_exposure")
            if position is None and len(portfolio.positions) >= s.max_open_positions:
                reasons.append("open_positions")
        elif position is None or order.quantity > position.quantity:
            reasons.append("insufficient_position")
        stop = order.stop_loss
        if (s.mandatory_stop_loss and stop is None) or (
            stop is not None
            and not (
                Decimal("0") < stop < quote.price if order.side == "buy" else stop > quote.price
            )
        ):
            reasons.append("invalid_stop_loss")
        return RiskDecision(
            approved=not reasons,
            reasons=tuple(reasons),
            evaluated_at=now,
            client_order_id=order.client_order_id,
        )
