"""Regression guards for Phase 4 labels, temporal isolation and data eligibility."""

import pytest
from test_features_research import bars

from trading_agent.features.research import build_features
from trading_agent.ml.dataset import build_dataset
from trading_agent.ml.pipeline import train_model
from trading_agent.ml.split import walk_forward


def test_threshold_is_inclusive_and_versioned():
    data = bars(5)
    data[1] = data[1].model_copy(update={"close": data[0].close})
    row = build_dataset(data, horizon=1, threshold=0)[0]
    assert row.direction == 0
    assert row.target == 1
    assert row.label_version == "targets-v2"


@pytest.mark.parametrize("first,expected", [("upper", "upper"), ("lower", "lower")])
def test_barrier_first_touch_and_full_horizon(first, expected):
    data = bars(5)
    data[1] = data[1].model_copy(
        update={
            "high": 105 if first == "upper" else 102,
            "low": 99 if first == "upper" else 95,
        }
    )
    data[2] = data[2].model_copy(update={"high": 106, "low": 94})
    row = build_dataset(data, horizon=3, upper_barrier=0.03, lower_barrier=0.03)[0]
    assert row.barrier == expected
    assert row.label_end == data[3].timestamp


def test_causal_distances_and_range():
    data = bars(80)
    full = build_features(data)
    assert full[:40] == build_features(data[:40])
    assert full[39].values["ema_distance_12"] == pytest.approx(
        data[39].close / full[39].values["ema_12"] - 1
    )
    assert full[39].values["rolling_range_pct_20"] == pytest.approx(
        (max(b.high for b in data[20:40]) - min(b.low for b in data[20:40])) / data[39].close
    )


@pytest.mark.parametrize(
    "provenance,synthetic",
    [
        ("nse-mcp", True),
        ("SYNTHETIC mcp.nseindia.in", True),
        ("unlicensed-market", False),
    ],
)
def test_training_rejects_mcp_and_nonfixture_data(provenance, synthetic):
    rows = build_dataset(bars(150), horizon=2)
    rows = [r.model_copy(update={"provenance": provenance, "synthetic": synthetic}) for r in rows]
    with pytest.raises(ValueError, match="SYNTHETIC|MCP"):
        train_model(rows, walk_forward(rows, 70, 25, 25)[0])


def test_target_column_cannot_become_feature():
    rows = build_dataset(bars(150), horizon=2)
    rows = [
        r.model_copy(update={"values": {**r.values, "future_return": r.future_return}})
        for r in rows
    ]
    with pytest.raises(ValueError, match="feature"):
        train_model(rows, walk_forward(rows, 70, 25, 25)[0])


def test_walk_forward_rejects_overlapping_test_windows():
    rows = build_dataset(bars(150), horizon=3)
    with pytest.raises(ValueError, match="overlap"):
        walk_forward(rows, 50, 20, 20, step=10)


def test_holdout_and_walk_forward_have_disjoint_information():
    from trading_agent.ml.split import chronological_split, development_rows

    rows = build_dataset(bars(300), horizon=5)
    split = chronological_split(rows, 150, 60)
    cutoff = rows[split.test[0]].timestamp
    dev = development_rows(rows, split)
    assert max(r.label_end for r in dev) < cutoff
    assert len(split.test) == 85
    for expanding in (True, False):
        folds = walk_forward(dev, 70, 30, 30, expanding=expanding)
        assert len(folds) >= 2
        for left, right in zip(folds, folds[1:], strict=False):
            assert max(dev[i].label_end for i in left.test) < min(
                dev[i].timestamp for i in right.test
            )
        assert (folds[0].train[0] == folds[-1].train[0]) is expanding


def test_multi_instrument_split_uses_global_time_boundaries():
    from trading_agent.ml.split import chronological_split

    rows = build_dataset(
        [x for pair in zip(bars(100, "A"), bars(100, "B"), strict=True) for x in pair], horizon=3
    )
    split = chronological_split(rows, 50, 20)
    for left, right in ((split.train, split.validation), (split.validation, split.test)):
        assert max(rows[i].label_end for i in left) < min(rows[i].timestamp for i in right)
        assert {rows[i].instrument_id for i in left} == {"A", "B"}


def test_future_price_perturbation_cannot_change_prior_features():
    data = bars(80)
    altered = [
        b.model_copy(
            update={
                "close": b.close * 10,
                "high": b.high * 10,
                "low": b.low * 10,
                "volume": b.volume * 100,
            }
        )
        if i >= 40
        else b
        for i, b in enumerate(data)
    ]
    assert build_features(data)[:40] == build_features(altered)[:40]
    original = build_dataset(data, horizon=5)
    changed = build_dataset(altered, horizon=5)
    assert original[39].values == changed[39].values
    assert original[39].future_return != changed[39].future_return


def test_fo_features_use_current_instrument_observations():
    from datetime import timedelta

    data = [
        b.model_copy(
            update={
                "open_interest": 1000.0 + i * 10,
                "underlying_price": 100.0,
                "strike": 110.0,
                "expiry": bars(1)[0].timestamp + timedelta(days=90),
            }
        )
        for i, b in enumerate(bars(40))
    ]
    row = build_features(data)[-1]
    assert row.values["oi_change"] == 10
    assert row.values["price_oi_relationship"] == pytest.approx(
        (data[-1].close - data[-2].close) * 10
    )
    assert row.values["basis"] == pytest.approx(data[-1].close / 100 - 1)
    assert row.values["moneyness"] == pytest.approx(100 / 110)
    assert row.values["time_to_expiry_days"] == 51
    assert build_features(data[:30]) == build_features(data)[:30]
