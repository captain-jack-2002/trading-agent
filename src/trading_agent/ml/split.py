"""Time-grouped expanding/rolling splits; boundary-touching labels are purged."""

from collections.abc import Sequence
from dataclasses import dataclass

from trading_agent.ml.dataset import DatasetRow


@dataclass(frozen=True)
class Split:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


def walk_forward(
    rows: Sequence[DatasetRow],
    train_size: int,
    validation_size: int,
    test_size: int,
    step: int | None = None,
    expanding: bool = True,
) -> list[Split]:
    sizes = (train_size, validation_size, test_size, step if step is not None else test_size)
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in sizes):
        raise ValueError("split sizes must be positive")
    if sizes[3] < test_size:
        raise ValueError("overlapping test windows are not permitted")
    if any(a.timestamp > b.timestamp for a, b in zip(rows, rows[1:], strict=False)):
        raise ValueError("rows must be chronological")
    if any(r.label_end <= r.timestamp for r in rows):
        raise ValueError("label_end must follow timestamp")
    if len({(r.instrument_id, r.timestamp) for r in rows}) != len(rows):
        raise ValueError("duplicate instrument timestamps")
    times = sorted({r.timestamp for r in rows})
    splits = []
    ends = list(range(train_size, len(times) - validation_size - test_size + 1, sizes[3]))
    for end in ends:
        start = 0 if expanding else end - train_size
        val_start = times[end]
        test_start = times[end + validation_size]
        test_times = set(times[end + validation_size : end + validation_size + test_size])
        train = tuple(
            i
            for i, r in enumerate(rows)
            if times[start] <= r.timestamp < val_start and r.label_end < val_start
        )
        validation = tuple(
            i
            for i, r in enumerate(rows)
            if val_start <= r.timestamp < test_start and r.label_end < test_start
        )
        # Cross-fold test labels must not consume the next fold's test observations.
        next_test = times[end + validation_size + sizes[3]] if end != ends[-1] else None
        test = tuple(
            i
            for i, r in enumerate(rows)
            if r.timestamp in test_times and (next_test is None or r.label_end < next_test)
        )
        if train and validation and test:
            splits.append(Split(train, validation, test))
    return splits


def chronological_split(rows: Sequence[DatasetRow], train_size: int, validation_size: int) -> Split:
    """Time-group counts before purging; every remaining timestamp is final holdout."""
    test_size = len({r.timestamp for r in rows}) - train_size - validation_size
    if test_size < 1:
        raise ValueError("no test observations remain")
    splits = walk_forward(rows, train_size, validation_size, test_size)
    if len(splits) != 1:
        raise ValueError("empty partition after label purging")
    return splits[0]


def development_rows(rows: Sequence[DatasetRow], holdout: Split) -> list[DatasetRow]:
    """Exclude any observation whose label uses final-holdout information."""
    start = min(rows[i].timestamp for i in holdout.test)
    return [row for row in rows if row.label_end < start]
