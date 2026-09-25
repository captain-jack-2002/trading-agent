from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_agent.data.calendar import MarketCalendar, TradingSession
from trading_agent.data.providers import CSVMapping, CSVProvider
from trading_agent.data.schemas.canonical import CanonicalBar, Contract
from trading_agent.data.storage import import_csv, load_bars


def bar(**changes: object) -> CanonicalBar:
    values: dict[str, object] = dict(
        asset_class="equity",
        symbol="TEST",
        exchange="TEST",
        timestamp="2025-01-01T10:00:00+05:30",
        open="10",
        high="12",
        low="9",
        close="11",
        volume=2,
    )
    values.update(changes)
    return CanonicalBar.model_validate(values)


@pytest.mark.parametrize(
    "changes",
    [
        dict(timestamp="2025-01-01T10:00:00"),
        dict(high="NaN"),
        dict(low="12"),
        dict(volume=-1),
        dict(volume=1.5),
        dict(asset_class="future"),
        dict(asset_class="equity", strike="10"),
    ],
)
def test_invalid_bars(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        bar(**changes)


def test_contract_and_immutable_bar() -> None:
    contract = Contract(
        contract_id="test-future",
        instrument_type="FUTIDX",
        symbol="TESTF",
        exchange="TEST",
        underlying="TEST",
        expiry="2025-01-30",
        lot_size=25,
        tick_size="0.05",
    )
    future = bar(
        asset_class="future",
        symbol="TESTF",
        underlying="TEST",
        expiry="2025-01-30",
        contract=contract,
    )
    assert future.close == Decimal("11")
    with pytest.raises(ValidationError):
        future.volume = 4
    with pytest.raises(ValidationError):
        bar(asset_class="future", underlying="WRONG", expiry="2025-01-30", contract=contract)


def test_explicit_calendar() -> None:
    session = TradingSession(
        session_date="2025-01-04",
        opens_at="2025-01-04T09:00:00+05:30",
        closes_at="2025-01-04T12:00:00+05:30",
        special=True,
    )
    calendar = MarketCalendar(sessions=(session,), holidays=("2025-01-06",))
    assert calendar.is_open(datetime.fromisoformat("2025-01-04T10:00:00+05:30"))
    assert not calendar.is_open(datetime.fromisoformat("2025-01-06T10:00:00+05:30"))
    assert not calendar.is_open(datetime.fromisoformat("2025-01-07T10:00:00+05:30"))
    with pytest.raises(ValueError):
        calendar.is_open(datetime(2025, 1, 4))


def mapping() -> CSVMapping:
    return CSVMapping(
        provider="synthetic-test",
        columns={x: x for x in ("symbol", "timestamp", "open", "high", "low", "close", "volume")},
        defaults={"asset_class": "equity", "exchange": "TEST"},
    )


def csv_file(tmp_path: Path, duplicate: bool = False) -> Path:
    path = tmp_path / "SYNTHETIC.csv"
    rows = [
        "symbol,timestamp,open,high,low,close,volume",
        "TEST,2025-01-01T10:00:00+05:30,10,12,9,11,2",
        f"TEST,2025-01-0{1 if duplicate else 2}T10:00:00+05:30,11,13,10,12,3",
    ]
    path.write_text("\n".join(rows) + "\n")
    return path


def test_csv_rejects_duplicate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="row 3.*chronology"):
        list(CSVProvider(mapping()).iter_bars(csv_file(tmp_path, True)))


def test_import_roundtrip_identity_and_partitions(tmp_path: Path) -> None:
    source = csv_file(tmp_path)
    manifest = import_csv(source, mapping(), tmp_path / "store", synthetic=True, batch_size=1)
    assert manifest.row_count == 2
    assert manifest.synthetic
    directory = tmp_path / "store" / manifest.import_id
    assert len(list(load_bars(directory / "normalized.jsonl"))) == 2
    assert len(list(directory.glob("parquet/asset_class=equity/year=2025/month=01/*.parquet"))) == 2
    assert (directory / "raw.csv").read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        import_csv(source, mapping(), tmp_path / "store", synthetic=True)
    changed = mapping().model_copy(update={"provider": "other"})
    assert import_csv(source, changed, tmp_path / "store").import_id != manifest.import_id


def test_bad_import_never_publishes(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        import_csv(csv_file(tmp_path, True), mapping(), tmp_path / "store")
    assert not list((tmp_path / "store").glob("*/manifest.json"))


def test_timezone_and_decimal_identity() -> None:
    a = bar(timestamp="2025-01-01T10:00:00+05:30")
    b = bar(timestamp="2025-01-01T04:30:00Z")
    assert a.model_dump_json() == b.model_dump_json()
    a = bar(
        asset_class="option",
        underlying="TEST",
        expiry="2025-01-30",
        strike="10.0",
        option_type="CE",
    )
    b = bar(
        asset_class="option",
        underlying="TEST",
        expiry="2025-01-30",
        strike="10.00",
        option_type="CE",
    )
    assert a.instrument_key == b.instrument_key


def test_verify_and_parquet_roundtrip(tmp_path: Path) -> None:
    from trading_agent.data.storage import load_parquet, verify_import

    manifest = import_csv(csv_file(tmp_path), mapping(), tmp_path / "store")
    assert manifest.synthetic  # Prominent source label is preserved without caller flag.
    directory = tmp_path / "store" / manifest.import_id
    assert verify_import(directory) == manifest
    assert list(load_parquet(directory / "parquet")) == list(
        load_bars(directory / "normalized.jsonl")
    )
    manifest_path = directory / "manifest.json"
    import json

    value = json.loads(manifest_path.read_text())
    value["checksums"] = {}
    manifest_path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        verify_import(directory)


def test_malformed_option_and_calendar_overlap() -> None:
    with pytest.raises(ValidationError):
        Contract(
            contract_id="x",
            instrument_type="OPTSTK",
            symbol="X",
            exchange="TEST",
            underlying="X",
            expiry="2025-01-30",
            lot_size=1,
            tick_size="0.05",
        )
    with pytest.raises(ValidationError):
        MarketCalendar(
            sessions=(
                TradingSession(
                    session_date="2025-01-04",
                    opens_at="2025-01-04T09:00:00+05:30",
                    closes_at="2025-01-04T12:00:00+05:30",
                ),
            ),
            holidays=("2025-01-04",),
        )


@pytest.mark.parametrize(
    "csv_text",
    [
        "symbol,symbol,timestamp,open,high,low,close,volume\n",
        "symbol,timestamp,open,high,low,close,volume\nX,2025-01-01T10:00:00Z,1,2,1,1,5,extra\n",
        "symbol,timestamp,open,high,low,close,volume\nX,2025-01-01T10:00:00Z,1,2,1,1\n",
        "symbol,timestamp,open,high,low,close,volume\nX,2025-01-01T10:00:00Z,1,2,1,1,1.5\n",
    ],
)
def test_csv_structural_validation(tmp_path: Path, csv_text: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(csv_text)
    with pytest.raises(ValueError):
        list(CSVProvider(mapping()).iter_bars(path))


def test_expiry_uses_exchange_date() -> None:
    with pytest.raises(ValidationError, match="after contract expiry"):
        bar(
            asset_class="future",
            underlying="TEST",
            expiry="2025-01-30",
            timestamp="2025-01-30T20:00:00Z",
        )


def test_checksum_tampering(tmp_path: Path) -> None:
    from trading_agent.data.storage import verify_import

    manifest = import_csv(csv_file(tmp_path), mapping(), tmp_path / "store")
    directory = tmp_path / "store" / manifest.import_id
    with (directory / "raw.csv").open("a") as stream:
        stream.write("corrupted\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_import(directory)


def test_interleaved_instrument_chronology(tmp_path: Path) -> None:
    path = tmp_path / "interleaved.csv"
    path.write_text(
        "symbol,timestamp,open,high,low,close,volume\n"
        "A,2025-01-02T10:00:00+05:30,10,12,9,11,2\n"
        "B,2025-01-01T10:00:00+05:30,10,12,9,11,2\n"
        "A,2025-01-02T04:30:00Z,10,12,9,11,2\n"
    )
    with pytest.raises(ValueError, match="row 4.*chronology"):
        list(CSVProvider(mapping()).iter_bars(path))
