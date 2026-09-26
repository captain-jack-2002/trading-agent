"""Engineering benchmarks: only propose orders; the backtester owns risk and fills."""

import math
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Literal

from trading_agent.backtesting.engine import SimulationOrder
from trading_agent.cli_research import research_bars
from trading_agent.data.schemas.canonical import CanonicalBar
from trading_agent.data.storage import ImportManifest
from trading_agent.features.research import build_features
from trading_agent.ml.drift import DriftThresholds, evaluate_drift
from trading_agent.ml.lifecycle import require_bundle_usable
from trading_agent.ml.pipeline import ModelBundle, predict_probabilities, predict_returns
from trading_agent.models.domain import Instrument, OHLCVBar, OrderRequest
from trading_agent.strategy.crossover import MovingAverageCrossover


class ResearchStrategy:
    def __init__(
        self,
        manifest: ImportManifest,
        *,
        fast: int = 5,
        slow: int = 20,
        quantity: int = 1,
        model: ModelBundle | None = None,
        threshold: float = 0.5,
        baseline_kind: str = "sma",
        return_threshold: float = 0.0,
        drift_thresholds: DriftThresholds | None = None,
    ):
        if quantity < 1 or not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("positive quantity and probability threshold in (0,1) required")
        if baseline_kind not in ("sma", "cash", "naive") or not math.isfinite(return_threshold):
            raise ValueError("invalid baseline or return threshold")
        self.drift_thresholds = drift_thresholds or DriftThresholds()
        self.baseline_kind = baseline_kind
        self.return_threshold = return_threshold
        self.baseline = MovingAverageCrossover(fast, slow)
        self.manifest, self.quantity, self.model, self.threshold = (
            manifest,
            quantity,
            model,
            threshold,
        )
        self.desired: dict[str, bool] = {}
        self.last: dict[str, datetime] = {}

    def __call__(self, history: tuple[CanonicalBar, ...]) -> Sequence[SimulationOrder]:
        if self.model is None and self.baseline_kind == "cash":
            return []
        if self.model is not None:
            require_bundle_usable(self.model)
        latest = history[-1].timestamp
        orders: list[SimulationOrder] = []
        for bar in (b for b in history if b.timestamp == latest):
            if self.last.get(bar.symbol) == latest:
                continue
            self.last[bar.symbol] = latest
            group = [b for b in history if b.instrument_key == bar.instrument_key]
            side: Literal["buy", "sell", "hold"] = "hold"
            health = None
            if self.model is not None:
                if len(group) < 26:
                    continue
                features = build_features(research_bars(group, self.manifest))
                window = features[-max(20, self.drift_thresholds.minimum_samples) :]
                health = evaluate_drift(
                    self.model.metadata,
                    window,
                    thresholds=self.drift_thresholds,
                    evaluated_at=bar.timestamp,
                )
                if health.allows_signals:
                    predictions = (
                        predict_returns(self.model, window)
                        if self.model.metadata.get("task") == "regression"
                        else predict_probabilities(self.model, window)
                    )
                    health = evaluate_drift(
                        self.model.metadata,
                        window,
                        predictions=predictions,
                        thresholds=self.drift_thresholds,
                        evaluated_at=bar.timestamp,
                    )
                feature = features[-1]
                positive = (
                    predict_returns(self.model, [feature])[0] > self.return_threshold
                    if self.model.metadata.get("task") == "regression"
                    else predict_probabilities(self.model, [feature])[0] >= self.threshold
                )
                if positive != self.desired.get(bar.symbol, False):
                    side = "buy" if positive else "sell"
                self.desired[bar.symbol] = positive
            elif self.baseline_kind == "naive":
                positive = len(group) >= 2 and group[-1].close > group[-2].close
                if positive != self.desired.get(bar.symbol, False):
                    side = "buy" if positive else "sell"
                self.desired[bar.symbol] = positive
            else:
                bars = [
                    OHLCVBar(
                        instrument=Instrument(symbol=b.symbol, instrument_type=b.asset_class),
                        timestamp=b.timestamp,
                        open=b.open,
                        high=b.high,
                        low=b.low,
                        close=b.close,
                        volume=b.volume,
                    )
                    for b in group
                ]
                side = self.baseline.generate(bars).side
            if side == "hold":
                continue
            stop = bar.close * (Decimal("0.95") if side == "buy" else Decimal("1.05"))
            orders.append(
                SimulationOrder(
                    order=OrderRequest(
                        client_order_id=f"research-{len(history)}-{len(orders)}",
                        symbol=bar.symbol,
                        instrument_type=bar.asset_class,
                        side=side,
                        quantity=self.quantity,
                        stop_loss=stop,
                    ),
                    model_health=health,
                    product="equity_delivery" if bar.asset_class == "equity" else bar.asset_class,
                )
            )
        return orders
