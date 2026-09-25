"""CSV is an input format, never an assumption about a particular exchange feed."""

import csv
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Protocol

from pydantic import Field, model_validator

from trading_agent.data.schemas.canonical import CanonicalBar
from trading_agent.models.domain import DomainModel


class CSVMapping(DomainModel):
    provider: str = Field(min_length=1)
    columns: dict[str, str]
    defaults: dict[str, object] = Field(default_factory=dict)
    delimiter: str = Field(default=",", min_length=1, max_length=1)
    encoding: str = "utf-8-sig"
    timestamp_format: str | None = None

    @model_validator(mode="after")
    def validate_mapping(self) -> "CSVMapping":
        unknown = (self.columns.keys() | self.defaults.keys()) - CanonicalBar.model_fields.keys()
        if unknown:
            raise ValueError(f"unknown canonical fields: {sorted(unknown)}")
        if self.columns.keys() & self.defaults.keys():
            raise ValueError("mapped columns and defaults cannot overlap")
        required = {k for k, v in CanonicalBar.model_fields.items() if v.is_required()}
        if required - (self.columns.keys() | self.defaults.keys()):
            raise ValueError("mapping does not supply every required canonical field")
        return self


class BarProvider(Protocol):
    def iter_bars(self, path: Path) -> Iterator[CanonicalBar]: ...


class CSVProvider:
    def __init__(self, mapping: CSVMapping) -> None:
        self.mapping = mapping

    def iter_bars(self, path: Path) -> Iterator[CanonicalBar]:
        previous: dict[tuple[str, ...], datetime] = {}
        with path.open(encoding=self.mapping.encoding, newline="") as stream:
            reader = csv.DictReader(stream, delimiter=self.mapping.delimiter, strict=True)
            headers = reader.fieldnames
            if not headers or len(headers) != len(set(headers)):
                raise ValueError("missing or duplicate CSV headers")
            if set(self.mapping.columns.values()) - set(headers):
                raise ValueError("CSV missing mapped columns")
            try:
                for number, row in enumerate(reader, start=2):
                    try:
                        if None in row or any(v is None for v in row.values()):
                            raise ValueError("malformed CSV column count")
                        values = dict(self.mapping.defaults)
                        for field, column in self.mapping.columns.items():
                            value = row[column]
                            if value != "":
                                values[field] = value
                        if self.mapping.timestamp_format is not None:
                            values["timestamp"] = datetime.strptime(
                                str(values["timestamp"]), self.mapping.timestamp_format
                            )
                        bar = CanonicalBar.model_validate(values)
                        last = previous.get(bar.instrument_key)
                        if last is not None and bar.timestamp <= last:
                            raise ValueError(
                                "strict chronology violated (duplicate or out of order)"
                            )
                        previous[bar.instrument_key] = bar.timestamp
                    except (ValueError, KeyError) as exc:
                        raise ValueError(f"CSV row {number}: {exc}") from exc
                    yield bar
            except csv.Error as exc:
                raise ValueError(f"CSV row {reader.line_num}: malformed CSV: {exc}") from exc
