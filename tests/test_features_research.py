from datetime import UTC, datetime, timedelta

import pytest

from trading_agent.features.research import ResearchBar, build_features


def bars(n=90, instrument="SYNTHETIC"):
    return [
        ResearchBar(
            instrument_id=instrument,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=i),
            close=100 + i % 7 + i / 10,
            high=102 + i % 7 + i / 10,
            low=98 + i % 7 + i / 10,
            volume=100 + i,
            provenance="SYNTHETIC",
        )
        for i in range(n)
    ]


def test_prefix_causal_and_missing():
    data = bars()
    a = build_features(data[:40])
    b = build_features(data)
    assert a == b[:40]
    assert a[0].values["return_20"] is None
    assert a[-1].values["oi_change"] is None
    assert a[20].values["return_20"] == pytest.approx(data[20].close / data[0].close - 1)
    assert a[-1].values["rsi_14"] is not None
    assert a[4].values["sma_5"] == pytest.approx(sum(x.close for x in data[:5]) / 5)
    assert a[4].values["sma_50"] is None
    assert b[49].values["sma_50"] is not None


def test_instruments_do_not_mix_and_order_rejected():
    a = bars(30, "A")
    b = bars(30, "B")
    rows = build_features([x for pair in zip(a, b, strict=True) for x in pair])
    assert [r.values for r in rows if r.instrument_id == "A"] == [
        r.values for r in build_features(a)
    ]
    with pytest.raises(ValueError):
        build_features(a[::-1])


def test_volume_features_and_metadata():
    data = bars(30)
    row = build_features(data)[-1]
    assert row.values["volume_zscore_20"] is not None
    assert row.values["volume_acceleration"] is not None
    assert row.values["bollinger_position"] is not None
    assert "open_interest" in row.values


@pytest.mark.parametrize(
    "field,value", [("provenance", "OTHER"), ("dataset_version", "other"), ("synthetic", True)]
)
def test_mixed_history_lineage_rejected(field, value):
    data = bars(30)
    data[-1] = data[-1].model_copy(update={field: value})
    with pytest.raises(ValueError, match="lineage"):
        build_features(data)
