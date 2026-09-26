"""Append-only local lifecycle journal, serialized with model-backed paper submission.

Artifacts remain immutable. SQLite transactions serialize cooperating processes;
state is replayed from validated, hash-chained events, never a mutable model flag.
"""

import fcntl
import hashlib
import json
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import AwareDatetime, Field, model_validator

from trading_agent.models.domain import DomainModel

if TYPE_CHECKING:
    from trading_agent.ml.pipeline import ModelBundle

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]
Status = Literal["candidate", "challenger", "champion", "quarantined", "retired"]


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class PromotionEvidence(DomainModel):
    walk_forward_sha256: Digest
    paper_validation_sha256: Digest
    walk_forward_path: Path | None = None
    paper_validation_path: Path | None = None
    approved_by: Annotated[str, Field(min_length=1, max_length=100, pattern=r"^\S+(?: \S+)*$")]
    reason: Annotated[str, Field(min_length=1, max_length=500)]

    @model_validator(mode="after")
    def paired_paths(self) -> "PromotionEvidence":
        if (self.walk_forward_path is None) != (self.paper_validation_path is None):
            raise ValueError("both local validation paths are required together")
        if not self.reason.strip():
            raise ValueError("nonempty promotion reason required")
        return self


class LifecycleEvent(DomainModel):
    sequence: int = Field(gt=0)
    model_id: Identifier
    artifact_model_id: str
    metadata_sha256: Digest
    status: Status
    timestamp: AwareDatetime
    actor: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=500)
    promotion: PromotionEvidence | None = None
    previous_hash: Digest
    event_hash: Digest


class ModelState(DomainModel):
    model_id: Identifier
    artifact_model_id: str
    status: Status
    metadata_sha256: Digest
    event_hash: Digest


class LifecycleRegistry:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.path = self.root / "lifecycle.db"

    @contextmanager
    def _transaction(
        self,
        *,
        create: bool = False,
        on_failure: Callable[[], None] | None = None,
    ) -> Iterator[sqlite3.Connection]:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.root / ".lifecycle.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
            )
        except OSError as exc:
            raise ValueError("unsafe lifecycle lock path") from exc
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            with self._locked_transaction(create=create) as connection:
                yield connection
        except BaseException:
            # Rollback cleanup must complete before waiting enrollment can acquire this lock.
            if on_failure is not None:
                on_failure()
            raise
        finally:
            os.close(descriptor)

    @contextmanager
    def _locked_transaction(self, *, create: bool = False) -> Iterator[sqlite3.Connection]:
        if self.path.is_symlink() or (self.path.exists() and not self.path.is_file()):
            raise ValueError("unsafe lifecycle registry path")
        if not create and not self.path.is_file():
            raise ValueError("missing lifecycle registry; model use rejected")
        self.root.mkdir(parents=True, exist_ok=True)
        created = False
        if create:
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
                created = True
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            if created:
                connection.execute(
                    "CREATE TABLE events (sequence INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE journal_head (singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
                    "sequence INTEGER NOT NULL, event_hash TEXT NOT NULL)"
                )
                connection.execute("INSERT INTO journal_head VALUES(1,0,?)", ("0" * 64,))
                for action in ("UPDATE", "DELETE"):
                    connection.execute(
                        f"CREATE TRIGGER no_{action.lower()} BEFORE {action} "
                        "ON events BEGIN SELECT RAISE(ABORT, 'append-only lifecycle'); END"
                    )
                connection.execute("PRAGMA user_version=1")
                connection.commit()
                connection.execute("BEGIN IMMEDIATE")
            if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise ValueError("invalid lifecycle registry schema")
            triggers = dict(
                connection.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger'")
            )
            for action in ("UPDATE", "DELETE"):
                expected = (
                    f"CREATE TRIGGER no_{action.lower()} BEFORE {action} "
                    "ON events BEGIN SELECT RAISE(ABORT, 'append-only lifecycle'); END"
                )
                if triggers.get(f"no_{action.lower()}") != expected:
                    raise ValueError("invalid append-only lifecycle controls")
            yield connection
            connection.commit()
        except sqlite3.DatabaseError as exc:
            connection.rollback()
            raise ValueError("invalid lifecycle registry") from exc
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def publication_gate(
        self,
        *,
        on_failure: Callable[[], None] | None = None,
    ) -> Iterator[sqlite3.Connection]:
        """Coordinate publication and enrollment with lifecycle mutations."""
        with self._transaction(create=True, on_failure=on_failure) as connection:
            self._replay(connection)
            yield connection

    def _artifact(self, model_id: str) -> tuple[str, str]:
        from trading_agent.ml.registry import show_model

        path = self.root / model_id
        if path.is_symlink() or any(
            (path / name).is_symlink() for name in ("metadata.json", "model.joblib")
        ):
            raise ValueError("symlinked model artifact")
        try:
            metadata = show_model(self.root, model_id)["metadata"]
            return str(metadata["model_id"]), digest(metadata)
        except (KeyError, TypeError, OSError) as exc:
            raise ValueError("invalid lifecycle artifact") from exc

    @staticmethod
    def _replay(
        connection: sqlite3.Connection,
    ) -> tuple[list[LifecycleEvent], dict[str, ModelState]]:
        events: list[LifecycleEvent] = []
        states: dict[str, ModelState] = {}
        previous = "0" * 64
        identities: set[str] = set()
        for sequence, payload in connection.execute(
            "SELECT sequence,payload FROM events ORDER BY sequence"
        ):
            event = LifecycleEvent.model_validate_json(payload)
            if (
                sequence != len(events) + 1
                or event.sequence != sequence
                or event.previous_hash != previous
                or event.event_hash != digest(event.model_dump(mode="json", exclude={"event_hash"}))
            ):
                raise ValueError("corrupt lifecycle hash chain")
            old = states.get(event.model_id)
            if old is None:
                if event.artifact_model_id in identities:
                    raise ValueError("duplicate lifecycle artifact identity")
                identities.add(event.artifact_model_id)
                if event.status not in ("candidate", "challenger"):
                    raise ValueError("invalid initial lifecycle state")
            else:
                if (
                    old.metadata_sha256 != event.metadata_sha256
                    or old.artifact_model_id != event.artifact_model_id
                ):
                    raise ValueError("lifecycle artifact identity changed")
                allowed = {
                    "candidate": {"challenger", "quarantined", "retired"},
                    "challenger": {"champion", "quarantined", "retired"},
                    "champion": {"quarantined", "retired"},
                    "quarantined": {"quarantined", "retired"},
                    "retired": {"quarantined"},
                }
                if event.status not in allowed[old.status]:
                    raise ValueError("invalid lifecycle transition")
            if event.status == "champion":
                if event.promotion is None:
                    raise ValueError("promotion evidence required")
                if (
                    event.actor != event.promotion.approved_by
                    or event.reason != event.promotion.reason
                ):
                    raise ValueError("promotion approval mismatch")
            elif event.promotion is not None:
                raise ValueError("promotion evidence on non-promotion event")
            states[event.model_id] = ModelState(
                model_id=event.model_id,
                artifact_model_id=event.artifact_model_id,
                status=event.status,
                metadata_sha256=event.metadata_sha256,
                event_hash=event.event_hash,
            )
            if sum(s.status == "champion" for s in states.values()) > 1:
                raise ValueError("multiple champions in registry")
            events.append(event)
            previous = event.event_hash
        head = connection.execute(
            "SELECT sequence,event_hash FROM journal_head WHERE singleton=1"
        ).fetchone()
        if head != (len(events), previous):
            raise ValueError("corrupt lifecycle journal head")
        return events, states

    def _append(
        self,
        connection: sqlite3.Connection,
        model_id: str,
        status: Status,
        actor: str,
        reason: str,
        promotion: PromotionEvidence | None = None,
    ) -> LifecycleEvent:
        events, _ = self._replay(connection)
        artifact_id, metadata_hash = self._artifact(model_id)
        values: dict[str, Any] = dict(
            sequence=len(events) + 1,
            model_id=model_id,
            artifact_model_id=artifact_id,
            metadata_sha256=metadata_hash,
            status=status,
            timestamp=datetime.now(UTC),
            actor=actor,
            reason=reason,
            promotion=promotion,
            previous_hash=events[-1].event_hash if events else "0" * 64,
            event_hash="0" * 64,
        )
        event = LifecycleEvent(**values)
        event = event.model_copy(
            update={"event_hash": digest(event.model_dump(mode="json", exclude={"event_hash"}))}
        )
        connection.execute(
            "INSERT INTO events VALUES (?,?)", (event.sequence, event.model_dump_json())
        )
        connection.execute(
            "UPDATE journal_head SET sequence=?,event_hash=? WHERE singleton=1",
            (event.sequence, event.event_hash),
        )
        self._replay(connection)  # Validate transition before commit.
        return event

    def _register(
        self,
        connection: sqlite3.Connection,
        model_id: str,
        *,
        status: Literal["candidate", "challenger"] = "challenger",
        actor: str = "local-training",
        reason: str = "artifact registered; explicit promotion required",
    ) -> LifecycleEvent:
        _, states = self._replay(connection)
        if model_id in states:
            raise ValueError("model already registered")
        return self._append(connection, model_id, status, actor, reason)

    def register(
        self, model_id: str, *, status: Literal["candidate", "challenger"] = "challenger"
    ) -> None:
        with self.publication_gate() as connection:
            self._register(connection, model_id, status=status)

    def enroll(
        self,
        model_id: str,
        *,
        actor: str,
        reason: str,
        status: Literal["candidate", "challenger"] = "candidate",
    ) -> LifecycleEvent:
        """Explicitly enroll a verified legacy artifact; never assume healthy state."""
        with self.publication_gate() as connection:
            return self._register(connection, model_id, status=status, actor=actor, reason=reason)

    def challenge(self, model_id: str, *, actor: str, reason: str) -> LifecycleEvent:
        with self._transaction() as connection:
            if self._state(connection, model_id).status != "candidate":
                raise ValueError("only a candidate can advance to challenger")
            return self._append(connection, model_id, "challenger", actor, reason)

    def events(self) -> tuple[LifecycleEvent, ...]:
        with self._transaction() as connection:
            return tuple(self._replay(connection)[0])

    def _state(self, connection: sqlite3.Connection, model_id: str) -> ModelState:
        _, states = self._replay(connection)
        if model_id not in states:
            raise ValueError("unregistered model")
        state = states[model_id]
        artifact_id, metadata_hash = self._artifact(model_id)
        if artifact_id != state.artifact_model_id or metadata_hash != state.metadata_sha256:
            raise ValueError("lifecycle metadata checksum mismatch")
        return state

    def state(self, model_id: str) -> ModelState:
        with self._transaction() as connection:
            return self._state(connection, model_id)

    @contextmanager
    def signal_gate(self, model_id: str) -> Iterator[ModelState]:
        """Hold registry lock until paper submission finishes; no quarantine race."""
        with self._transaction() as connection:
            state = self._state(connection, model_id)
            if state.status not in ("challenger", "champion"):
                raise ValueError(f"model {state.status}: signals rejected")
            yield state

    def require_usable(self, model_id: str) -> ModelState:
        with self.signal_gate(model_id) as state:
            return state

    def quarantine(
        self, model_id: str, *, reason: str, actor: str = "local-operator"
    ) -> LifecycleEvent:
        with self._transaction() as connection:
            self._state(connection, model_id)
            return self._append(connection, model_id, "quarantined", actor, reason)

    def retire(
        self, model_id: str, *, reason: str, actor: str = "local-operator"
    ) -> LifecycleEvent:
        with self._transaction() as connection:
            self._state(connection, model_id)
            return self._append(connection, model_id, "retired", actor, reason)

    def promote(self, model_id: str, evidence: PromotionEvidence) -> LifecycleEvent:
        evidence = PromotionEvidence.model_validate(evidence.model_dump())
        with self._transaction() as connection:
            state = self._state(connection, model_id)
            if state.status != "challenger":
                raise ValueError("only a non-quarantined challenger can be promoted")
            self._validate_promotion(model_id, evidence)
            _, states = self._replay(connection)
            for old in states.values():
                if old.status == "champion":
                    self._state(connection, old.model_id)
                    self._append(
                        connection,
                        old.model_id,
                        "retired",
                        evidence.approved_by,
                        "superseded by explicit promotion",
                    )
            return self._append(
                connection, model_id, "champion", evidence.approved_by, evidence.reason, evidence
            )

    def _validate_promotion(self, model_id: str, evidence: PromotionEvidence) -> None:
        from trading_agent.ml.registry import show_model

        metadata = show_model(self.root, model_id)["metadata"]
        for path, expected, kind, flag in (
            (
                evidence.walk_forward_path,
                evidence.walk_forward_sha256,
                "purged_walk_forward",
                "purged",
            ),
            (
                evidence.paper_validation_path,
                evidence.paper_validation_sha256,
                "paper_shadow_validation",
                "paper_only",
            ),
        ):
            if path is None:  # Reviewed attestation, not an automatic certification.
                continue
            try:
                if path.is_symlink() or not path.is_file():
                    raise ValueError("unsafe validation artifact")
                content = path.read_bytes()
                manifest = json.loads(content)
                if (
                    hashlib.sha256(content).hexdigest() != expected
                    or manifest["kind"] != kind
                    or manifest["model_id"] != metadata["model_id"]
                    or manifest["model_sha256"] != metadata["sha256"]
                    or manifest["passed"] is not True
                    or manifest[flag] is not True
                ):
                    raise ValueError("validation artifact does not certify this model")
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError("invalid validation artifact") from exc


def require_bundle_usable(bundle: "ModelBundle") -> None:
    """Offline unsaved experiments remain supported; loaded artifacts use live state."""
    if bundle.registry_path is not None:
        state = LifecycleRegistry(bundle.registry_path.parent).require_usable(
            bundle.registry_path.name
        )
        if state.artifact_model_id != bundle.metadata["model_id"]:
            raise ValueError("bundle identity mismatch")
