from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from threading import RLock
from typing import Any

from sqlalchemy import create_engine, delete, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from trading_agent.models.domain import PortfolioSnapshot
from trading_agent.persistence.tables import (
    AuditRecord,
    Base,
    DailyPnLRecord,
    LedgerRecord,
    PositionRecord,
)
from trading_agent.risk.engine import IST


class Store:
    """PostgreSQL row lock serializes all paper-account changes across processes.

    SQLite is a test/local option: BEGIN IMMEDIATE serializes database writers.
    In-memory SQLite is rejected to avoid unsafe shared-connection transaction lifetimes.
    """

    def __init__(self, url: str):
        options: dict[str, Any] = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            options["connect_args"] = {"check_same_thread": False, "timeout": 10}
            if make_url(url).database in (None, "", ":memory:"):
                raise ValueError("SQLite requires a file-backed database")
        else:
            options["connect_args"] = {"connect_timeout": 3}
        self.engine = create_engine(url, **options)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        self._lock = RLock()

    def initialize(self, initial_cash: Decimal, now: datetime) -> None:
        Base.metadata.create_all(self.engine)
        with self.transaction() as session:
            # Protect first account creation too; normal orders lock the existing row.
            if self.engine.dialect.name == "postgresql":
                session.execute(text("SELECT pg_advisory_xact_lock(710001)"))
            if session.get(LedgerRecord, 1) is None:
                p = PortfolioSnapshot(
                    cash=initial_cash,
                    equity=initial_cash,
                    day_start_equity=initial_cash,
                    as_of=now,
                    trading_day=now.astimezone(IST).date(),
                )
                session.add(LedgerRecord(id=1, payload=p.model_dump(mode="json")))

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        with self._lock, self.sessions() as session:
            with session.begin():
                if self.engine.dialect.name == "sqlite":
                    session.execute(text("BEGIN IMMEDIATE"))
                yield session

    def load_locked(self, session: Session) -> PortfolioSnapshot:
        row = session.scalar(select(LedgerRecord).where(LedgerRecord.id == 1).with_for_update())
        if row is None:
            raise RuntimeError("paper ledger is not initialized")
        return PortfolioSnapshot.model_validate(row.payload)

    def save(self, session: Session, portfolio: PortfolioSnapshot) -> None:
        row = session.get(LedgerRecord, 1)
        if row is None:
            raise RuntimeError("paper ledger is not initialized")
        row.payload = portfolio.model_dump(mode="json")
        session.execute(delete(PositionRecord))
        session.add_all(
            [
                PositionRecord(symbol=p.symbol, payload=p.model_dump(mode="json"))
                for p in portfolio.positions
            ]
        )
        if portfolio.trading_day is not None:
            session.merge(
                DailyPnLRecord(
                    trading_day=portfolio.trading_day.isoformat(),
                    payload={
                        "equity": str(portfolio.equity),
                        "day_start_equity": str(portfolio.day_start_equity),
                        "daily_pnl": str(portfolio.daily_pnl),
                        "as_of": portfolio.as_of.isoformat(),
                    },
                )
            )

    def ready(self) -> bool:
        try:
            with self.sessions() as session:
                return session.get(LedgerRecord, 1) is not None
        except Exception:
            return False

    def audit(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.sessions() as session:
            return [
                row.payload
                for row in session.scalars(
                    select(AuditRecord).order_by(AuditRecord.id.desc()).limit(limit)
                )
            ]
