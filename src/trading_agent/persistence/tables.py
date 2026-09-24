"""Portable SQLAlchemy schema; JSON uses JSONB on PostgreSQL."""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

JSON_TYPE = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class LedgerRecord(Base):
    __tablename__ = "paper_ledger"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)


class EventMixin:
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)


class ObservationRecord(EventMixin, Base):
    __tablename__ = "market_observations"


class SignalRecord(EventMixin, Base):
    __tablename__ = "generated_signals"


class RiskRecord(EventMixin, Base):
    __tablename__ = "risk_decisions"


class OrderRecord(Base):
    __tablename__ = "orders"
    client_order_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)


class FillRecord(EventMixin, Base):
    __tablename__ = "fills"
    client_order_id: Mapped[str] = mapped_column(ForeignKey("orders.client_order_id"), unique=True)


class PositionRecord(Base):
    __tablename__ = "positions"
    symbol: Mapped[str] = mapped_column(String(30), primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)


class DailyPnLRecord(Base):
    __tablename__ = "daily_pnl"
    trading_day: Mapped[str] = mapped_column(String(10), primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)


class AuditRecord(EventMixin, Base):
    __tablename__ = "audit_events"
