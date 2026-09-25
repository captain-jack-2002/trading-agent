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
    if any(isinstance(n, bool) or n < 1 for n in sizes):
        raise ValueError("split sizes must be positive")
    if any(a.timestamp > b.timestamp for a, b in zip(rows, rows[1:], strict=False)):
        raise ValueError("rows must be chronological")
    if any(r.label_end <= r.timestamp for r in rows):
        raise ValueError("label_end must follow timestamp")
    if len({(r.instrument_id, r.timestamp) for r in rows}) != len(rows):
        raise ValueError("duplicate instrument timestamps")
    times = sorted({r.timestamp for r in rows})
    splits = []
    for end in range(train_size, len(times) - validation_size - test_size + 1, sizes[3]):
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
        test = tuple(i for i, r in enumerate(rows) if r.timestamp in test_times)
        if train and validation and test:
            splits.append(Split(train, validation, test))
    return splits
