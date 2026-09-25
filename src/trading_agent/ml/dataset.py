"""Future targets retain their full information horizon for purging."""

from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from pydantic import Field, StrictInt, model_validator

from trading_agent.features.research import FeatureRow, ResearchBar, build_features


class DatasetRow(FeatureRow):
    label_end: datetime
    horizon: StrictInt = Field(gt=0)
    future_return: float
    direction: StrictInt
    target: StrictInt
    threshold: float
    upper_barrier: float
    lower_barrier: float
    barrier: Literal["upper", "lower", "neither", "ambiguous"]
    label_version: str = "targets-v1"

    @model_validator(mode="after")
    def valid_label(self) -> "DatasetRow":
        if self.label_end.utcoffset() is None or self.label_end <= self.timestamp:
            raise ValueError("label_end must be aware and follow timestamp")
        if self.horizon < 1 or self.direction not in (0, 1) or self.target not in (0, 1):
            raise ValueError("invalid label")
        if self.upper_barrier <= 0 or not 0 < self.lower_barrier < 1 or not self.label_version:
            raise ValueError("invalid barrier or label version")
        return self


def build_dataset(
    bars: Sequence[ResearchBar],
    horizon: int = 5,
    threshold: float = 0,
    upper_barrier: float = 0.02,
    lower_barrier: float = 0.01,
) -> list[DatasetRow]:
    import math

    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be positive")
    if not all(math.isfinite(x) for x in (threshold, upper_barrier, lower_barrier)):
        raise ValueError("thresholds must be finite")
    if upper_barrier <= 0 or not 0 < lower_barrier < 1:
        raise ValueError("invalid barriers")
    features = build_features(bars)
    groups: dict[str, list[ResearchBar]] = {}
    lookup = {(f.instrument_id, f.timestamp): f for f in features}
    for bar in bars:
        groups.setdefault(bar.instrument_id, []).append(bar)
    rows: list[DatasetRow] = []
    for group in groups.values():
        for i in range(len(group) - horizon):
            bar = group[i]
            future = group[i + 1 : i + horizon + 1]
            change = future[-1].close / bar.close - 1
            barrier: Literal["upper", "lower", "neither", "ambiguous"] = "neither"
            for next_bar in future:
                up = next_bar.high >= bar.close * (1 + upper_barrier)
                down = next_bar.low <= bar.close * (1 - lower_barrier)
                if up or down:
                    barrier = "ambiguous" if up and down else "upper" if up else "lower"
                    break
            rows.append(
                DatasetRow(
                    **lookup[bar.instrument_id, bar.timestamp].model_dump(),
                    label_end=future[-1].timestamp,
                    horizon=horizon,
                    future_return=change,
                    direction=int(change > 0),
                    target=int(change > threshold),
                    threshold=threshold,
                    upper_barrier=upper_barrier,
                    lower_barrier=lower_barrier,
                    barrier=barrier,
                )
            )
    return sorted(rows, key=lambda row: (row.timestamp, row.instrument_id))
