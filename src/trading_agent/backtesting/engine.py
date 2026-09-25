"""Offline, fully funded, long-only event simulation behind Phase 1 risk policy."""

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from itertools import groupby
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from trading_agent.backtesting.costs import (
    CostBreakdown,
    CostSchedule,
    Frozen,
    NonNegative,
    Product,
    Slippage,
)
from trading_agent.backtesting.metrics import compute_metrics
from trading_agent.config.settings import Settings
from trading_agent.data.calendar import MarketCalendar
from trading_agent.data.schemas.canonical import CanonicalBar, Contract
from trading_agent.models.domain import (
    Instrument,
    OrderRequest,
    PortfolioSnapshot,
    Position,
    PositiveMoney,
    Quote,
)
from trading_agent.risk.engine import IST, RiskEngine

ZERO = Decimal(0)


class SimulationOrder(Frozen):
    order: OrderRequest
    order_type: Literal["market", "limit"] = "market"
    limit_price: PositiveMoney | None = None
    product: Product = "equity_delivery"

    @model_validator(mode="after")
    def limit_required(self) -> "SimulationOrder":
        if (self.order_type == "limit") != (self.limit_price is not None):
            raise ValueError("limit price required exactly for limit orders")
        return self


def zero_costs() -> dict[Product, CostSchedule]:
    return {
        "equity_delivery": CostSchedule(),
        "equity_intraday": CostSchedule(),
        "future": CostSchedule(),
        "option": CostSchedule(),
    }


class BacktestConfig(Frozen):
    bar_seconds: Annotated[int, Field(gt=0)] = 60
    equity_tick_size: PositiveMoney
    costs: dict[Product, CostSchedule] = Field(default_factory=zero_costs)
    slippage: Slippage = Slippage()
    periods_per_year: Annotated[int, Field(gt=0)] = 252 * 375
    risk_free_rate: Annotated[float, Field(gt=-1, allow_inf_nan=False)] = 0
    fill_mode: Literal["full", "partial", "reject"] = "full"
    fill_fraction: Annotated[Decimal, Field(gt=0, le=1, allow_inf_nan=False)] = Decimal(1)
    strategy_name: str = "unspecified research strategy"
    strategy_config: dict[str, str] = Field(default_factory=dict)
    dataset_version: str = "unspecified local dataset"
    synthetic: bool = True
    provenance: str = "SYNTHETIC engineering fixture; no investment evidence"


class AuditEvent(Frozen):
    timestamp: datetime
    client_order_id: str
    status: Literal["filled", "partially_filled", "rejected", "unfilled"]
    reasons: tuple[str, ...] = ()


class SimulatedTrade(Frozen):
    timestamp: datetime
    order: OrderRequest
    fill_price: PositiveMoney
    costs: CostBreakdown
    slippage: NonNegative
    realized_pnl: Decimal = ZERO


class BacktestResult(Frozen):
    config: BacktestConfig
    risk_settings: dict[str, object]
    start: datetime
    end: datetime
    instruments: tuple[str, ...]
    initial_capital: Decimal
    final_portfolio: PortfolioSnapshot
    equity: tuple[Decimal, ...]
    equity_timestamps: tuple[datetime, ...]
    trades: tuple[SimulatedTrade, ...]
    audit: tuple[AuditEvent, ...]
    metrics: dict[str, float | int | None]
    assumptions: tuple[str, ...] = (
        "Bar timestamps are closes; open time is close minus configured fixed duration.",
        "Signals execute at next matching bar open; limits require marketable open.",
        "Full/partial/reject fills use only prior closed bar volume; unfilled remainder cancelled.",
        "Long-only fully funded derivatives; no margin, exercise or settlement emulation.",
        "Stop-loss intent is risk eligibility only; automatic stops are not simulated.",
        "Costs are supplied research assumptions, not authoritative exchange rates.",
    )

    def write_reports(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if any((directory / name).exists() for name in ("report.json", "report.md")):
            raise FileExistsError("report destination already exists")
        with (directory / "report.json").open("x") as handle:
            handle.write(self.model_dump_json(indent=2))
        label = "SYNTHETIC ENGINEERING RESULTS" if self.config.synthetic else "RESEARCH RESULTS"
        lines = [
            f"# {label}",
            "",
            self.config.provenance,
            "",
            f"Period: {self.start.isoformat()} to {self.end.isoformat()}",
            f"Instruments: {', '.join(self.instruments)}",
            f"Initial capital: {self.initial_capital}",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
        lines += [
            f"| {k} | {v if v is not None else 'undefined'} |" for k, v in self.metrics.items()
        ]
        lines += ["", "## Assumptions", ""] + [f"- {a}" for a in self.assumptions]
        lines += ["", "Full configuration, risk policy, equity and audit: report.json."]
        with (directory / "report.md").open("x") as handle:
            handle.write("\n".join(lines) + "\n")


class Ledger:
    def __init__(self, cash: Decimal):
        self.cash = cash
        self.positions: dict[str, Position] = {}
        self.entry_costs: dict[str, Decimal] = {}
        self.realized = ZERO
        self.day_start = cash
        self.day: date | None = None

    def snapshot(self, now: datetime) -> PortfolioSnapshot:
        exposure = sum((p.mark_price * p.quantity for p in self.positions.values()), ZERO)
        unrealized = sum(
            (p.unrealized_pnl - self.entry_costs[p.symbol] for p in self.positions.values()), ZERO
        )
        return PortfolioSnapshot(
            cash=self.cash,
            equity=self.cash + exposure,
            exposure=exposure,
            realized_pnl=self.realized,
            unrealized_pnl=unrealized,
            day_start_equity=self.day_start,
            trading_day=self.day,
            positions=tuple(self.positions.values()),
            as_of=now,
        )

    def mark(self, symbol: str, price: Decimal) -> None:
        if symbol in self.positions:
            self.positions[symbol] = self.positions[symbol].model_copy(update={"mark_price": price})

    def fill(self, order: OrderRequest, price: Decimal, cost: Decimal) -> Decimal:
        old = self.positions.get(order.symbol)
        value = price * order.quantity
        pnl = ZERO
        if order.side == "buy":
            quantity = order.quantity + (old.quantity if old else 0)
            basis = value + (old.average_price * old.quantity if old else ZERO)
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=quantity,
                average_price=basis / quantity,
                mark_price=price,
            )
            self.entry_costs[order.symbol] = self.entry_costs.get(order.symbol, ZERO) + cost
            self.cash -= value + cost
        else:
            if old is None or old.quantity < order.quantity:
                raise ValueError("ledger disallows short positions")
            entry_cost = self.entry_costs[order.symbol] * order.quantity / old.quantity
            pnl = (price - old.average_price) * order.quantity - entry_cost - cost
            self.realized += pnl
            self.cash += value - cost
            if old.quantity == order.quantity:
                del self.positions[order.symbol]
                del self.entry_costs[order.symbol]
            else:
                self.positions[order.symbol] = old.model_copy(
                    update={"quantity": old.quantity - order.quantity, "mark_price": price}
                )
                self.entry_costs[order.symbol] -= entry_cost
        return pnl


Strategy = Callable[[tuple[CanonicalBar, ...]], Sequence[SimulationOrder]]


class BacktestEngine:
    def __init__(self, settings: Settings, config: BacktestConfig, calendar: MarketCalendar):
        self.settings, self.config, self.calendar = settings, config, calendar
        self.risk = RiskEngine(settings)

    def run(
        self,
        bars: Sequence[CanonicalBar],
        strategy: Strategy,
        contracts: Mapping[str, Contract] | None = None,
        *,
        warmup_bars: Sequence[CanonicalBar] = (),
    ) -> BacktestResult:
        if not bars:
            raise ValueError("nonempty bars required")
        duration = timedelta(seconds=self.config.bar_seconds)
        first_open = bars[0].timestamp - duration
        if any(bar.timestamp > first_open for bar in warmup_bars):
            raise ValueError("warmup bars must be available by the first execution open")
        validation_bars = [*warmup_bars, *bars]
        keys: set[tuple[datetime, tuple[str, ...]]] = set()
        identities: dict[str, tuple[str, ...]] = {}
        previous: datetime | None = None
        for bar in validation_bars:
            if previous is not None and bar.timestamp < previous:
                raise ValueError("bars must be chronological")
            if (
                previous is not None
                and bar.timestamp != previous
                and bar.timestamp < previous + duration
            ):
                raise ValueError("overlapping bar chronology")
            key = (bar.timestamp, bar.instrument_key)
            if key in keys:
                raise ValueError("duplicate instrument bar")
            keys.add(key)
            if bar.symbol in identities and identities[bar.symbol] != bar.instrument_key:
                raise ValueError("symbol must identify one unambiguous contract")
            identities[bar.symbol] = bar.instrument_key
            previous = bar.timestamp
        ledger = Ledger(self.settings.initial_cash)
        history: list[CanonicalBar] = list(warmup_bars)
        pending: list[SimulationOrder] = []
        seen: set[str] = set()
        audit: list[AuditEvent] = []
        trades: list[SimulatedTrade] = []
        equity = [ledger.cash]
        timestamps = [bars[0].timestamp - duration]
        exposures = [ZERO]
        for closed_at, grouped in groupby(bars, key=lambda b: b.timestamp):
            timestamp = closed_at - duration
            group = list(grouped)
            if ledger.day != timestamp.astimezone(IST).date():
                ledger.day_start = ledger.snapshot(timestamp).equity
                ledger.day = timestamp.astimezone(IST).date()
            for bar in group:
                ledger.mark(bar.symbol, bar.open)
            available = {bar.symbol: bar for bar in group}
            remaining: list[SimulationOrder] = []
            consumed_volume: dict[str, int] = {}
            for signal in pending:
                if signal.order.symbol not in available:
                    remaining.append(signal)
                    continue
                bar = available[signal.order.symbol]
                trade, event = self._execute(
                    signal,
                    bar.model_copy(update={"timestamp": timestamp}),
                    ledger,
                    seen,
                    history,
                    contracts,
                    consumed_volume.get(bar.symbol, 0),
                )
                audit.append(event)
                if trade is not None:
                    trades.append(trade)
                    consumed_volume[bar.symbol] = (
                        consumed_volume.get(bar.symbol, 0) + trade.order.quantity
                    )
            pending = remaining
            for bar in group:
                ledger.mark(bar.symbol, bar.close)
            history.extend(group)
            snapshot = ledger.snapshot(closed_at)
            equity.append(snapshot.equity)
            timestamps.append(closed_at)
            exposures.append(snapshot.exposure)
            session = self.calendar.session_for(timestamp)
            if session is None or closed_at > session.closes_at:
                audit.append(
                    AuditEvent(
                        timestamp=closed_at,
                        client_order_id="calendar",
                        status="rejected",
                        reasons=("calendar_closed",),
                    )
                )
                continue
            for signal in strategy(tuple(history)):
                try:
                    validated = SimulationOrder.model_validate(
                        signal.model_dump() if isinstance(signal, SimulationOrder) else signal
                    )
                except (ValueError, TypeError):
                    audit.append(
                        AuditEvent(
                            timestamp=closed_at,
                            client_order_id="invalid",
                            status="rejected",
                            reasons=("invalid_strategy_signal",),
                        )
                    )
                    continue
                pending.append(validated)
        for signal in pending:
            audit.append(
                AuditEvent(
                    timestamp=timestamps[-1],
                    client_order_id=signal.order.client_order_id,
                    status="unfilled",
                    reasons=("no_next_bar",),
                )
            )
        total_costs = sum((t.costs.total for t in trades), ZERO)
        total_slip = sum((t.slippage for t in trades), ZERO)
        metrics = compute_metrics(
            equity,
            [t.realized_pnl for t in trades if t.order.side == "sell"],
            periods_per_year=self.config.periods_per_year,
            risk_free_rate=self.config.risk_free_rate,
            turnover=sum((t.fill_price * t.order.quantity for t in trades), ZERO),
            exposures=exposures,
            costs=total_costs,
            slippage=total_slip,
        )
        metrics["fill_count"] = len(trades)
        return BacktestResult(
            config=self.config,
            risk_settings=self.settings.model_dump(
                mode="json", exclude={"database_url", "valkey_url"}
            ),
            start=timestamps[0],
            end=timestamps[-1],
            instruments=tuple(sorted(identities)),
            initial_capital=self.settings.initial_cash,
            final_portfolio=ledger.snapshot(timestamps[-1]),
            equity=tuple(equity),
            equity_timestamps=tuple(timestamps),
            trades=tuple(trades),
            audit=tuple(audit),
            metrics=metrics,
        )

    def _execute(
        self,
        signal: SimulationOrder,
        bar: CanonicalBar,
        ledger: Ledger,
        seen: set[str],
        history: list[CanonicalBar],
        contracts: Mapping[str, Contract] | None,
        consumed_volume: int = 0,
    ) -> tuple[SimulatedTrade | None, AuditEvent]:
        order = signal.order
        reasons: list[str] = []
        session = self.calendar.session_for(bar.timestamp)
        if (
            session is None
            or bar.timestamp + timedelta(seconds=self.config.bar_seconds) > session.closes_at
        ):
            reasons.append("calendar_closed")
        contract = bar.contract or (contracts or {}).get(bar.symbol)
        if bar.exchange != "NSE" or order.instrument_type != bar.asset_class:
            reasons.append("instrument_mismatch")
        expected = "future" if bar.asset_class == "future" else "option"
        if bar.asset_class != "equity":
            if contract is None:
                reasons.append("missing_contract")
            elif (
                contract.symbol != bar.symbol
                or contract.expiry != bar.expiry
                or contract.underlying != bar.underlying
                or contract.strike != bar.strike
                or contract.option_type != bar.option_type
                or contract.exchange != bar.exchange
                or ("option" if contract.instrument_type.startswith("OPT") else "future")
                != expected
            ):
                reasons.append("contract_mismatch")
            if signal.product != bar.asset_class:
                reasons.append("cost_product_mismatch")
        elif signal.product not in ("equity_delivery", "equity_intraday"):
            reasons.append("cost_product_mismatch")
        if contract is not None and order.quantity % contract.lot_size:
            reasons.append("invalid_lot_quantity")
        prior = [b for b in history if b.symbol == bar.symbol]
        capacity = max(0, (prior[-1].volume if prior else 0) - consumed_volume)
        if self.config.fill_mode == "reject":
            reasons.append("fill_model_reject")
        elif self.config.fill_mode == "full" and order.quantity > capacity:
            reasons.append("partial_fill_unsupported")
        tick = contract.tick_size if contract else self.config.equity_tick_size
        quote = Quote(
            instrument=Instrument(symbol=bar.symbol, instrument_type=bar.asset_class),
            price=bar.open,
            timestamp=bar.timestamp,
            source="offline-backtest",
        )
        portfolio = ledger.snapshot(bar.timestamp)
        reasons.extend(self.risk.evaluate(order, quote, portfolio, seen, bar.timestamp).reasons)
        volatility = abs(prior[-1].close / prior[-2].close - 1) if len(prior) > 1 else ZERO
        try:
            slipped = self.config.slippage.price(bar.open, order.side, volatility)
            price = (slipped / tick).to_integral_value(
                rounding=ROUND_CEILING if order.side == "buy" else ROUND_FLOOR
            ) * tick
            if price <= 0:
                raise ValueError("nonpositive rounded fill")
        except ValueError:
            price = bar.open
            reasons.append("invalid_slippage_price")
        if signal.limit_price is not None:
            limit = signal.limit_price
            if limit % tick:
                reasons.append("invalid_limit_tick")
            if (order.side == "buy" and price > limit) or (order.side == "sell" and price < limit):
                reasons.append("limit_not_marketable_at_open")
        reasons.extend(
            self.risk.evaluate(
                order, quote.model_copy(update={"price": price}), portfolio, seen, bar.timestamp
            ).reasons
        )
        proposed_quantity = order.quantity
        if self.config.fill_mode == "partial":
            lot = contract.lot_size if contract else 1
            quantity = min(int(order.quantity * self.config.fill_fraction), capacity)
            quantity = quantity // lot * lot
            if quantity <= 0:
                reasons.append("no_fill_quantity")
            else:
                order = order.model_copy(update={"quantity": quantity})
                reasons.extend(
                    self.risk.evaluate(
                        order,
                        quote.model_copy(update={"price": price}),
                        portfolio,
                        seen,
                        bar.timestamp,
                    ).reasons
                )
        schedule = self.config.costs.get(signal.product)
        if schedule is None:
            reasons.append("missing_cost_schedule")
        costs = (
            schedule.calculate(order.side, price, order.quantity) if schedule else CostBreakdown()
        )
        cash_required = (
            price * order.quantity + costs.total
            if order.side == "buy"
            else costs.total - price * order.quantity
        )
        if cash_required > ledger.cash:
            reasons.append("insufficient_cash_with_costs")
        seen.add(order.client_order_id)
        event = AuditEvent(
            timestamp=bar.timestamp,
            client_order_id=order.client_order_id,
            status="rejected"
            if reasons
            else ("partially_filled" if order.quantity < proposed_quantity else "filled"),
            reasons=tuple(dict.fromkeys(reasons)),
        )
        if reasons:
            return None, event
        pnl = ledger.fill(order, price, costs.total)
        # Execution friction affects basis/cash, not the observable market mark.
        ledger.mark(order.symbol, bar.open)
        return SimulatedTrade(
            timestamp=bar.timestamp,
            order=order,
            fill_price=price,
            costs=costs,
            slippage=abs(price - bar.open) * order.quantity,
            realized_pnl=pnl,
        ), event
