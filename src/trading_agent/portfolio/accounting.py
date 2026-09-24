from datetime import datetime
from decimal import Decimal

from trading_agent.models.domain import OrderRequest, PortfolioSnapshot, Position, Quote
from trading_agent.risk.engine import IST


def mark_portfolio(
    p: PortfolioSnapshot, quotes: dict[str, Quote], now: datetime
) -> PortfolioSnapshot:
    day = now.astimezone(IST).date()
    # Last stored equity is the prior day's closing proxy. Preserve overnight gaps.
    baseline = p.equity if p.trading_day != day else p.day_start_equity
    positions = tuple(
        Position(
            symbol=x.symbol,
            quantity=x.quantity,
            average_price=x.average_price,
            mark_price=quotes[x.symbol].price,
        )
        for x in p.positions
    )
    exposure = sum((x.mark_price * x.quantity for x in positions), Decimal("0"))
    unrealized = sum((x.unrealized_pnl for x in positions), Decimal("0"))
    return PortfolioSnapshot(
        cash=p.cash,
        equity=p.cash + exposure,
        exposure=exposure,
        realized_pnl=p.realized_pnl,
        unrealized_pnl=unrealized,
        day_start_equity=baseline,
        trading_day=day,
        positions=positions,
        as_of=now,
    )


def apply_fill(
    p: PortfolioSnapshot, order: OrderRequest, quote: Quote, now: datetime
) -> PortfolioSnapshot:
    """Internal accounting transformation. Only risk-gated adapter persists this result."""
    positions = {x.symbol: x for x in p.positions}
    current = positions.get(order.symbol)
    amount = quote.price * order.quantity
    realized = p.realized_pnl
    if order.side == "buy":
        if amount > p.cash:
            raise ValueError("insufficient cash")
        old_qty = current.quantity if current else 0
        old_cost = current.average_price * old_qty if current else Decimal("0")
        quantity = old_qty + order.quantity
        positions[order.symbol] = Position(
            symbol=order.symbol,
            quantity=quantity,
            average_price=(old_cost + amount) / quantity,
            mark_price=quote.price,
        )
        cash = p.cash - amount
    else:
        if current is None or order.quantity > current.quantity:
            raise ValueError("insufficient position")
        realized += (quote.price - current.average_price) * order.quantity
        remaining = current.quantity - order.quantity
        if remaining:
            positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=remaining,
                average_price=current.average_price,
                mark_price=quote.price,
            )
        else:
            del positions[order.symbol]
        cash = p.cash + amount
    exposure = sum((x.mark_price * x.quantity for x in positions.values()), Decimal("0"))
    unrealized = sum((x.unrealized_pnl for x in positions.values()), Decimal("0"))
    return PortfolioSnapshot(
        cash=cash,
        equity=cash + exposure,
        exposure=exposure,
        realized_pnl=realized,
        unrealized_pnl=unrealized,
        day_start_equity=p.day_start_equity,
        trading_day=p.trading_day,
        positions=tuple(positions.values()),
        as_of=now,
    )
