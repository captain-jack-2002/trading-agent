"""Causal, instrument-local research features. Missing values are explicit nulls."""

import math
from collections.abc import Sequence
from datetime import datetime
from statistics import mean, stdev

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from trading_agent.features import indicators as ind

FEATURE_VERSION = "research-v2"


class ResearchBar(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False, extra="forbid")
    instrument_id: str = Field(min_length=1)
    timestamp: datetime
    close: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    volume: float = Field(ge=0)
    open_interest: float | None = Field(default=None, ge=0)
    underlying_price: float | None = Field(default=None, gt=0)
    strike: float | None = Field(default=None, gt=0)
    expiry: datetime | None = None
    provenance: str = Field(min_length=1)
    dataset_version: str = "unspecified"
    synthetic: StrictBool = False

    @model_validator(mode="after")
    def valid(self) -> "ResearchBar":
        if self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone aware")
        if not self.low <= self.close <= self.high:
            raise ValueError("invalid OHLC")
        if self.expiry is not None and self.expiry.utcoffset() is None:
            raise ValueError("expiry must be timezone aware")
        return self


class FeatureRow(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False, extra="forbid")
    instrument_id: str
    timestamp: datetime
    feature_version: str = FEATURE_VERSION
    provenance: str
    dataset_version: str = "unspecified"
    synthetic: StrictBool = False
    values: dict[str, float | None]

    @model_validator(mode="after")
    def valid_row(self) -> "FeatureRow":
        if self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be aware")
        if (
            not self.instrument_id
            or not self.feature_version
            or not self.dataset_version
            or not self.provenance
        ):
            raise ValueError("identity, versions and provenance required")
        return self


def build_features(bars: Sequence[ResearchBar]) -> list[FeatureRow]:
    groups: dict[str, list[ResearchBar]] = {}
    for bar in bars:
        group = groups.setdefault(bar.instrument_id, [])
        if group and bar.timestamp <= group[-1].timestamp:
            raise ValueError("each instrument must have strictly increasing timestamps")
        if group and (bar.provenance, bar.dataset_version, bar.synthetic) != (
            group[0].provenance,
            group[0].dataset_version,
            group[0].synthetic,
        ):
            raise ValueError("mixed lineage within instrument history")
        group.append(bar)
    result: dict[tuple[str, datetime], FeatureRow] = {}
    for instrument, group in groups.items():
        close = [b.close for b in group]
        ema12, ema26 = ind.ema(close, 12), ind.ema(close, 26)
        macd = [a - b for a, b in zip(ema12, ema26, strict=True)]
        signal = ind.ema(macd, 9)
        rsi = ind.rsi(close)
        atr = ind.atr([b.high for b in group], [b.low for b in group], close)
        volume = ind.volume_ratio([b.volume for b in group], 20)
        moving_averages = {period: ind.sma(close, period) for period in (5, 10, 20, 50)}
        for i, bar in enumerate(group):
            values: dict[str, float | None] = {}
            for n in (1, 5, 10, 20):
                values[f"return_{n}"] = close[i] / close[i - n] - 1 if i >= n else None
                values[f"log_return_{n}"] = math.log(close[i] / close[i - n]) if i >= n else None
            window = close[i - 19 : i + 1] if i >= 19 else []
            avg = mean(window) if window else None
            std = stdev(window) if window else None
            log_changes = (
                [math.log(close[j] / close[j - 1]) for j in range(i - 19, i + 1)] if i >= 20 else []
            )
            oi_prev = group[i - 1].open_interest if i else None
            oi_change = (
                bar.open_interest - oi_prev
                if bar.open_interest is not None and oi_prev is not None
                else None
            )
            volumes = [b.volume for b in group[i - 19 : i + 1]] if window else []
            volume_std = stdev(volumes) if volumes else None
            values["volume_zscore_20"] = (
                (bar.volume - mean(volumes)) / volume_std if volume_std else None
            )
            values["volume_acceleration"] = (
                bar.volume - 2 * group[i - 1].volume + group[i - 2].volume if i >= 2 else None
            )
            values["bollinger_position"] = (
                (bar.close - (avg - 2 * std)) / (4 * std) if avg is not None and std else None
            )
            for period in (5, 10, 20, 50):
                average = moving_averages[period][i]
                values[f"sma_distance_{period}"] = bar.close / average - 1 if average else None
            values["ema_distance_12"] = bar.close / ema12[i] - 1
            values["ema_distance_26"] = bar.close / ema26[i] - 1
            values["rolling_range_pct_20"] = (
                (
                    max(b.high for b in group[i - 19 : i + 1])
                    - min(b.low for b in group[i - 19 : i + 1])
                )
                / bar.close
                if window
                else None
            )
            values["open_interest"] = bar.open_interest
            values.update(
                {
                    "sma_5": moving_averages[5][i],
                    "sma_10": moving_averages[10][i],
                    "sma_20": avg,
                    "sma_50": moving_averages[50][i],
                    "ema_12": ema12[i],
                    "ema_26": ema26[i],
                    "rsi_14": rsi[i],
                    "macd": macd[i],
                    "macd_signal": signal[i],
                    "macd_histogram": macd[i] - signal[i],
                    "atr_14": atr[i],
                    "atr_pct": float(atr[i] or 0) / bar.close if atr[i] is not None else None,
                    "bollinger_upper": avg + 2 * std
                    if avg is not None and std is not None
                    else None,
                    "bollinger_lower": avg - 2 * std
                    if avg is not None and std is not None
                    else None,
                    "realized_volatility_20": stdev(log_changes) if log_changes else None,
                    "high_low_volatility_20": math.sqrt(
                        mean([math.log(b.high / b.low) ** 2 for b in group[i - 19 : i + 1]])
                        / (4 * math.log(2))
                    )
                    if window
                    else None,
                    "volume_ratio_20": volume[i],
                    "zscore_20": (bar.close - avg) / std if avg is not None and std else None,
                    "return_acceleration": close[i] / close[i - 1] - close[i - 1] / close[i - 2]
                    if i >= 2
                    else None,
                    "ma_distance_20": bar.close / avg - 1 if avg else None,
                    "momentum_20": close[i] - close[i - 20] if i >= 20 else None,
                    "rolling_high_20": max(b.high for b in group[i - 19 : i + 1])
                    if window
                    else None,
                    "rolling_low_20": min(b.low for b in group[i - 19 : i + 1]) if window else None,
                    "oi_change": oi_change,
                    "price_oi_relationship": (bar.close - close[i - 1]) * oi_change
                    if i and oi_change is not None
                    else None,
                    "basis": bar.close / bar.underlying_price - 1 if bar.underlying_price else None,
                    "time_to_expiry_days": (bar.expiry - bar.timestamp).total_seconds() / 86400
                    if bar.expiry
                    else None,
                    "moneyness": bar.underlying_price / bar.strike
                    if bar.underlying_price and bar.strike
                    else None,
                }
            )
            result[instrument, bar.timestamp] = FeatureRow(
                instrument_id=instrument,
                timestamp=bar.timestamp,
                provenance=bar.provenance,
                dataset_version=bar.dataset_version,
                synthetic=bar.synthetic,
                values=values,
            )
    return [result[b.instrument_id, b.timestamp] for b in bars]


# Exact schema owned by the causal builder, never inferred from supplied training columns.
FEATURE_NAMES = frozenset(
    {
        *(f"{kind}_{n}" for kind in ("return", "log_return") for n in (1, 5, 10, 20)),
        *(f"{kind}_{n}" for kind in ("sma", "sma_distance") for n in (5, 10, 20, 50)),
        "ema_12",
        "ema_26",
        "ema_distance_12",
        "ema_distance_26",
        "rolling_range_pct_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "atr_14",
        "atr_pct",
        "bollinger_upper",
        "bollinger_lower",
        "bollinger_position",
        "realized_volatility_20",
        "high_low_volatility_20",
        "volume_ratio_20",
        "volume_zscore_20",
        "volume_acceleration",
        "zscore_20",
        "return_acceleration",
        "ma_distance_20",
        "momentum_20",
        "rolling_high_20",
        "rolling_low_20",
        "open_interest",
        "oi_change",
        "price_oi_relationship",
        "basis",
        "time_to_expiry_days",
        "moneyness",
    }
)
