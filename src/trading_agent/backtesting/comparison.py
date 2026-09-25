"""Compare caller-supplied recorded fills without account or database access."""

from collections.abc import Sequence
from decimal import Decimal

from trading_agent.backtesting.costs import Frozen, NonNegative
from trading_agent.models.domain import PositiveMoney, Quantity


class RecordedTrade(Frozen):
    client_order_id: str
    quantity: Quantity
    price: PositiveMoney
    fees: NonNegative = Decimal(0)


class TradeDifference(Frozen):
    client_order_id: str
    quantity_difference: int
    price_difference: Decimal
    fee_difference: Decimal


class TradeComparison(Frozen):
    matched: tuple[TradeDifference, ...]
    only_simulated: tuple[str, ...]
    only_paper: tuple[str, ...]


def compare_trades(
    simulated: Sequence[RecordedTrade], paper: Sequence[RecordedTrade]
) -> TradeComparison:
    """Match unique client order IDs; differences are simulation minus paper."""
    left = {t.client_order_id: t for t in simulated}
    right = {t.client_order_id: t for t in paper}
    if len(left) != len(simulated) or len(right) != len(paper):
        raise ValueError(
            "duplicate trade IDs; aggregate partial fills explicitly before comparison"
        )
    return TradeComparison(
        matched=tuple(
            TradeDifference(
                client_order_id=key,
                quantity_difference=left[key].quantity - right[key].quantity,
                price_difference=left[key].price - right[key].price,
                fee_difference=left[key].fees - right[key].fees,
            )
            for key in sorted(left.keys() & right.keys())
        ),
        only_simulated=tuple(sorted(left.keys() - right.keys())),
        only_paper=tuple(sorted(right.keys() - left.keys())),
    )
