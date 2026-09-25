"""Content-addressed imports published only after successful streaming validation."""

import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import AwareDatetime

from trading_agent.data.providers import CSVMapping, CSVProvider
from trading_agent.data.schemas.canonical import CanonicalBar
from trading_agent.models.domain import DomainModel

SCHEMA_VERSION = "canonical-v1"


class ImportManifest(DomainModel):
    import_id: str
    provider: str
    source_filename: str
    imported_at: AwareDatetime
    schema_version: str = SCHEMA_VERSION
    raw_sha256: str
    mapping_sha256: str
    row_count: int
    first_timestamp: AwareDatetime
    last_timestamp: AwareDatetime
    instruments: tuple[tuple[str, ...], ...]
    validation_status: str = "passed"
    synthetic: bool
    checksums: dict[str, str]


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_batch(bars: list[CanonicalBar], directory: Path, batch_number: int) -> None:
    partitions: dict[tuple[str, int, int], list[dict[str, str | int | None]]] = {}
    for bar in bars:
        # Decimal strings preserve arbitrary provider precision without float coercion.
        row: dict[str, str | int | None] = {}
        for key, value in bar.model_dump(mode="json").items():
            row[key] = json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
        partition_key = (bar.asset_class, bar.timestamp.year, bar.timestamp.month)
        partitions.setdefault(partition_key, []).append(row)
    fields = [
        pa.field(name, pa.int64() if name in {"volume", "open_interest"} else pa.string())
        for name in CanonicalBar.model_fields
    ]
    schema = pa.schema(
        fields,
        metadata={
            b"schema_version": SCHEMA_VERSION.encode(),
            b"decimal_encoding": b"exact-base10-string",
        },
    )
    for (asset, year, month), rows in partitions.items():
        target = directory / f"asset_class={asset}" / f"year={year}" / f"month={month:02d}"
        target.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema),
            target / f"part-{batch_number:06d}.parquet",
            compression="zstd",
        )


def import_csv(
    path: Path,
    mapping: CSVMapping,
    storage_dir: Path,
    *,
    synthetic: bool = False,
    batch_size: int = 10_000,
) -> ImportManifest:
    if batch_size < 1 or batch_size > 1_000_000:
        raise ValueError("batch_size must be between 1 and 1000000")
    synthetic = (
        synthetic or "synthetic" in mapping.provider.lower() or "synthetic" in path.name.lower()
    )
    storage_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".import-", dir=storage_dir))
    try:
        raw = staging / "raw.csv"
        shutil.copyfile(path, raw)
        raw_hash = checksum(raw)
        mapping_json = json.dumps(
            mapping.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        mapping_hash = hashlib.sha256(mapping_json.encode()).hexdigest()
        identity = f"{SCHEMA_VERSION}:{raw_hash}:{mapping_hash}:{synthetic}"
        import_id = hashlib.sha256(identity.encode()).hexdigest()
        destination = storage_dir / import_id
        if destination.exists():
            raise FileExistsError(f"import already exists: {destination}")
        (staging / "mapping.json").write_text(mapping_json + "\n", encoding="utf-8")
        count = 0
        first: datetime | None = None
        last: datetime | None = None
        instruments: set[tuple[str, ...]] = set()
        batch: list[CanonicalBar] = []
        batch_number = 0
        with (staging / "normalized.jsonl").open("x", encoding="utf-8") as output:
            for bar in CSVProvider(mapping).iter_bars(raw):
                output.write(bar.model_dump_json() + "\n")
                count += 1
                first = min(first, bar.timestamp) if first else bar.timestamp
                last = max(last, bar.timestamp) if last else bar.timestamp
                instruments.add(bar.instrument_key)
                batch.append(bar)
                if len(batch) >= batch_size:
                    _write_batch(batch, staging / "parquet", batch_number)
                    batch.clear()
                    batch_number += 1
            if batch:
                _write_batch(batch, staging / "parquet", batch_number)
        if first is None or last is None:
            raise ValueError("CSV contains no data rows")
        hashes = {
            str(p.relative_to(staging)): checksum(p)
            for p in sorted(staging.rglob("*"))
            if p.is_file()
        }
        manifest = ImportManifest(
            import_id=import_id,
            provider=mapping.provider,
            source_filename=path.name,
            imported_at=datetime.now(UTC),
            raw_sha256=raw_hash,
            mapping_sha256=mapping_hash,
            row_count=count,
            first_timestamp=first,
            last_timestamp=last,
            instruments=tuple(sorted(instruments)),
            synthetic=synthetic,
            checksums=hashes,
        )
        (staging / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        # An exclusive lock coordinates writers; rename publishes the complete bundle.
        lock = storage_dir / f".{import_id}.lock"
        with lock.open("x"):
            try:
                if destination.exists():
                    raise FileExistsError(f"import already exists: {destination}")
                staging.rename(destination)
            finally:
                lock.unlink()
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def load_bars(path: Path) -> Iterator[CanonicalBar]:
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            try:
                yield CanonicalBar.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"normalized row {number}: {exc}") from exc


def load_parquet(directory: Path) -> Iterator[CanonicalBar]:
    """Read bounded record batches; partition order is not global chronology."""
    for path in sorted(directory.rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=10_000):
            for row in batch.to_pylist():
                if row.get("contract") is not None:
                    row["contract"] = json.loads(row["contract"])
                yield CanonicalBar.model_validate(row)


def verify_import(directory: Path) -> ImportManifest:
    manifest = ImportManifest.model_validate_json((directory / "manifest.json").read_text())
    required = {"raw.csv", "mapping.json", "normalized.jsonl"}
    if not required <= manifest.checksums.keys():
        raise ValueError("manifest omits required checksums")
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError("unsupported schema version")
    actual_files = {
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    if actual_files != manifest.checksums.keys():
        raise ValueError("manifest files do not match import files")
    if not any(name.startswith("parquet/") for name in actual_files):
        raise ValueError("manifest omits Parquet data")
    for relative, expected in manifest.checksums.items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("manifest path escapes import directory")
        if not path.is_file() or checksum(path) != expected:
            raise ValueError(f"checksum mismatch: {relative}")
    mapping = CSVMapping.model_validate_json((directory / "mapping.json").read_text())
    encoded_mapping = json.dumps(
        mapping.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    mapping_hash = hashlib.sha256(encoded_mapping.encode()).hexdigest()
    raw_hash = checksum(directory / "raw.csv")
    identity = f"{SCHEMA_VERSION}:{raw_hash}:{mapping_hash}:{manifest.synthetic}"
    if (
        manifest.raw_sha256 != raw_hash
        or manifest.mapping_sha256 != mapping_hash
        or manifest.provider != mapping.provider
        or manifest.import_id != hashlib.sha256(identity.encode()).hexdigest()
        or directory.name != manifest.import_id
    ):
        raise ValueError("import identity or provenance mismatch")
    count = 0
    first: datetime | None = None
    last: datetime | None = None
    instruments: set[tuple[str, ...]] = set()
    for bar in load_bars(directory / "normalized.jsonl"):
        count += 1
        first = min(first, bar.timestamp) if first else bar.timestamp
        last = max(last, bar.timestamp) if last else bar.timestamp
        instruments.add(bar.instrument_key)
    if (
        count != manifest.row_count
        or count == 0
        or first != manifest.first_timestamp
        or last != manifest.last_timestamp
        or tuple(sorted(instruments)) != manifest.instruments
        or manifest.validation_status != "passed"
    ):
        raise ValueError("manifest row statistics mismatch")
    return manifest
