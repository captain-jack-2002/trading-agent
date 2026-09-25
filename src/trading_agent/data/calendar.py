"""An explicitly supplied calendar: absent dates are never presumed tradable."""

from datetime import date, datetime

from pydantic import AwareDatetime, model_validator

from trading_agent.models.domain import DomainModel


class TradingSession(DomainModel):
    session_date: date
    opens_at: AwareDatetime
    closes_at: AwareDatetime
    special: bool = False

    @model_validator(mode="after")
    def validate_bounds(self) -> "TradingSession":
        if self.opens_at >= self.closes_at:
            raise ValueError("session must open before it closes")
        if self.opens_at.date() != self.session_date or self.closes_at.date() != self.session_date:
            raise ValueError("session endpoints must match supplied session_date")
        return self


class MarketCalendar(DomainModel):
    sessions: tuple[TradingSession, ...] = ()
    holidays: tuple[date, ...] = ()
    expiries: tuple[date, ...] = ()
    source: str = "explicit-user-input"

    @model_validator(mode="after")
    def validate_sessions(self) -> "MarketCalendar":
        dates = [s.session_date for s in self.sessions]
        if len(dates) != len(set(dates)):
            raise ValueError("duplicate session dates")
        if set(dates) & set(self.holidays):
            raise ValueError("holiday cannot also have a session")
        ordered = sorted(self.sessions, key=lambda s: s.opens_at)
        if any(a.closes_at >= b.opens_at for a, b in zip(ordered, ordered[1:], strict=False)):
            raise ValueError("overlapping sessions")
        return self

    def session_for(self, timestamp: datetime) -> TradingSession | None:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("calendar requires timezone-aware timestamp")
        return next((s for s in self.sessions if s.opens_at <= timestamp <= s.closes_at), None)

    def is_open(self, timestamp: datetime) -> bool:
        return self.session_for(timestamp) is not None
