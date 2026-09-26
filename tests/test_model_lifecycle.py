"""Lifecycle is explicit, audited and fail-closed, never a mutable model flag."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_features_research import bars

from trading_agent.ml.dataset import build_dataset
from trading_agent.ml.pipeline import train_model
from trading_agent.ml.registry import save_model
from trading_agent.ml.split import walk_forward


def artifact(tmp_path, name="one"):
    rows = build_dataset(bars(150), horizon=2)
    bundle = train_model(rows, walk_forward(rows, 70, 25, 25)[0])
    save_model(bundle, tmp_path / name)
    return bundle


def test_publication_creates_challenger_and_retraining_preserves_champion(tmp_path):
    from trading_agent.ml.lifecycle import LifecycleRegistry, PromotionEvidence

    artifact(tmp_path)
    registry = LifecycleRegistry(tmp_path)
    assert registry.state("one").status == "challenger"
    evidence = PromotionEvidence(
        walk_forward_sha256="a" * 64,
        paper_validation_sha256="b" * 64,
        approved_by="local-reviewer",
        reason="engineering validation reviewed",
    )
    registry.promote("one", evidence)
    before = registry.events()
    artifact(tmp_path, "two")
    assert registry.state("one").status == "champion"
    assert registry.state("two").status == "challenger"
    assert registry.events()[: len(before)] == before
    registry.promote("two", evidence)
    assert registry.state("one").status == "retired"
    assert registry.state("two").status == "champion"


def test_quarantine_is_latched_and_audited(tmp_path):
    from trading_agent.ml.lifecycle import LifecycleRegistry, PromotionEvidence

    artifact(tmp_path)
    registry = LifecycleRegistry(tmp_path)
    previous = registry.events()
    registry.quarantine("one", reason="severe drift", actor="monitor")
    assert registry.state("one").status == "quarantined"
    assert registry.events()[: len(previous)] == previous
    with pytest.raises(ValueError, match="quarantined"):
        registry.require_usable("one")
    with pytest.raises(ValueError):
        registry.promote(
            "one",
            PromotionEvidence(
                walk_forward_sha256="a" * 64,
                paper_validation_sha256="b" * 64,
                approved_by="reviewer",
                reason="cannot bypass quarantine",
            ),
        )


def test_registry_corruption_and_unknown_fail_closed(tmp_path):
    from trading_agent.ml.lifecycle import LifecycleRegistry

    artifact(tmp_path)
    registry = LifecycleRegistry(tmp_path)
    with pytest.raises(ValueError):
        registry.require_usable("absent")
    with pytest.raises(ValueError):
        registry.state("../one")
    (tmp_path / "one" / "metadata.json").write_text(json.dumps({"model_id": "changed"}))
    with pytest.raises(ValueError):
        registry.require_usable("one")


def test_concurrent_quarantine_preserves_append_only_sequence(tmp_path):
    from trading_agent.ml.lifecycle import LifecycleRegistry

    artifact(tmp_path)
    registry = LifecycleRegistry(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: registry.quarantine("one", reason=f"drift-{i}"), range(8)))
    assert len(registry.events()) == 9
    assert registry.state("one").status == "quarantined"


def test_publication_failure_does_not_leave_usable_artifact(tmp_path, monkeypatch):
    from trading_agent.ml.lifecycle import LifecycleRegistry

    rows = build_dataset(bars(150), horizon=2)
    bundle = train_model(rows, walk_forward(rows, 70, 25, 25)[0])

    def fail(*args, **kwargs):
        raise ValueError("audit storage unavailable")

    monkeypatch.setattr(LifecycleRegistry, "_append", fail)
    with pytest.raises(ValueError, match="audit storage unavailable"):
        save_model(bundle, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()


def test_registered_identity_cannot_be_reused_in_another_directory(tmp_path):
    from trading_agent.ml.lifecycle import LifecycleRegistry

    bundle = artifact(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        save_model(bundle, tmp_path / "duplicate")
    assert not (tmp_path / "duplicate").exists()
    assert len(LifecycleRegistry(tmp_path).events()) == 1


def test_model_and_journal_symlinks_fail_closed(tmp_path):
    from trading_agent.ml.lifecycle import LifecycleRegistry

    artifact(tmp_path / "real")
    registry = LifecycleRegistry(tmp_path / "real")
    (tmp_path / "real" / "alias").symlink_to(tmp_path / "real" / "one", target_is_directory=True)
    with pytest.raises(ValueError):
        registry.register("alias")
    database = tmp_path / "real" / "lifecycle.db"
    database.rename(tmp_path / "other.db")
    database.symlink_to(tmp_path / "other.db")
    with pytest.raises(ValueError):
        registry.require_usable("one")


def test_candidate_requires_audited_advancement_and_legacy_requires_enrollment(tmp_path):
    import shutil

    from trading_agent.ml.lifecycle import LifecycleRegistry

    artifact(tmp_path / "published")
    shutil.copytree(tmp_path / "published" / "one", tmp_path / "legacy" / "one")
    registry = LifecycleRegistry(tmp_path / "legacy")
    with pytest.raises(ValueError):
        registry.require_usable("one")
    registry.enroll(
        "one", actor="operator", reason="verified own local artifact", status="candidate"
    )
    with pytest.raises(ValueError, match="candidate"):
        registry.require_usable("one")
    registry.challenge("one", actor="operator", reason="training review completed")
    assert registry.require_usable("one").status == "challenger"
    registry.retire("one", reason="withdrawn")
    with pytest.raises(ValueError, match="retired"):
        registry.require_usable("one")


def test_journal_rejects_tampering_and_truncation(tmp_path):
    import sqlite3

    from trading_agent.ml.lifecycle import LifecycleRegistry

    artifact(tmp_path)
    registry = LifecycleRegistry(tmp_path)
    registry.quarantine("one", reason="severe drift")
    with sqlite3.connect(registry.path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM events WHERE sequence=2")
        connection.execute("DROP TRIGGER no_delete")
        connection.execute("DELETE FROM events WHERE sequence=2")
    with pytest.raises(ValueError):
        registry.require_usable("one")


def test_hash_chain_rejects_changed_event_payload(tmp_path):
    import sqlite3

    from trading_agent.ml.lifecycle import LifecycleRegistry

    artifact(tmp_path)
    registry = LifecycleRegistry(tmp_path)
    with sqlite3.connect(registry.path) as connection:
        payload = json.loads(connection.execute("SELECT payload FROM events").fetchone()[0])
        payload["reason"] = "edited after publication"
        connection.execute("DROP TRIGGER no_update")
        connection.execute("UPDATE events SET payload=?", (json.dumps(payload),))
    with pytest.raises(ValueError):
        registry.events()


def test_signal_gate_serializes_quarantine_until_paper_submission_finishes(tmp_path):
    from threading import Event

    from trading_agent.ml.lifecycle import LifecycleRegistry

    artifact(tmp_path)
    entered, finished = Event(), Event()

    def quarantine():
        entered.set()
        LifecycleRegistry(tmp_path).quarantine("one", reason="severe drift")
        finished.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with LifecycleRegistry(tmp_path).signal_gate("one") as state:
            future = pool.submit(quarantine)
            assert entered.wait(2)
            assert not finished.wait(0.05)
            assert state.status == "challenger"
        future.result(timeout=5)
    with pytest.raises(ValueError, match="quarantined"):
        LifecycleRegistry(tmp_path).require_usable("one")


def test_corrupt_existing_registry_is_not_reinitialized_by_publication(tmp_path):
    (tmp_path / "lifecycle.db").write_bytes(b"")
    with pytest.raises(ValueError):
        artifact(tmp_path)
    assert not (tmp_path / "one").exists()


def test_local_promotion_evidence_must_bind_the_immutable_model(tmp_path):
    import hashlib

    from trading_agent.ml.lifecycle import LifecycleRegistry, PromotionEvidence

    bundle = artifact(tmp_path)
    metadata = json.loads((tmp_path / "one" / "metadata.json").read_text())
    paths, hashes = [], []
    for kind in ("purged_walk_forward", "paper_shadow_validation"):
        path = tmp_path / f"{kind}.json"
        path.write_text(
            json.dumps(
                {
                    "kind": kind,
                    "model_id": bundle.metadata["model_id"],
                    "model_sha256": metadata["sha256"],
                    "passed": True,
                    "purged": True,
                    "paper_only": True,
                }
            )
        )
        paths.append(path)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    evidence = PromotionEvidence(
        walk_forward_sha256=hashes[0],
        paper_validation_sha256=hashes[1],
        walk_forward_path=paths[0],
        paper_validation_path=paths[1],
        approved_by="reviewer",
        reason="reviewed artifacts",
    )
    paths[0].write_text("{}")
    registry = LifecycleRegistry(tmp_path)
    before = registry.events()
    with pytest.raises(ValueError, match="validation"):
        registry.promote("one", evidence)
    assert registry.events() == before
    paths[0].write_text(
        json.dumps(
            {
                "kind": "purged_walk_forward",
                "model_id": bundle.metadata["model_id"],
                "model_sha256": metadata["sha256"],
                "passed": True,
                "purged": True,
                "paper_only": True,
            }
        )
    )
    assert registry.promote("one", evidence).status == "champion"


def test_enrollment_cannot_observe_half_published_model(tmp_path, monkeypatch):
    from pathlib import Path
    from threading import Event

    from trading_agent.ml.lifecycle import LifecycleRegistry

    published, attempted, finished, release = Event(), Event(), Event(), Event()
    original = Path.rename
    target = tmp_path / "one"

    def paused_rename(path, destination):
        result = original(path, destination)
        if destination == target:
            published.set()
            assert release.wait(5)
        return result

    monkeypatch.setattr(Path, "rename", paused_rename)

    def enroll():
        attempted.set()
        try:
            LifecycleRegistry(tmp_path).enroll("one", actor="operator", reason="reviewed")
        except ValueError:
            return "already_registered"
        finally:
            finished.set()
        return "enrolled"

    with ThreadPoolExecutor(max_workers=2) as pool:
        publisher = pool.submit(artifact, tmp_path)
        assert published.wait(5)
        enroller = pool.submit(enroll)
        assert attempted.wait(2)
        try:
            assert not finished.wait(0.1)
        finally:
            release.set()
        publisher.result(timeout=5)
        assert enroller.result(timeout=5) == "already_registered"
    assert target.is_dir()
    assert LifecycleRegistry(tmp_path).state("one").status == "challenger"


def test_original_published_bundle_cannot_ignore_quarantine(tmp_path):
    from trading_agent.ml.lifecycle import LifecycleRegistry, require_bundle_usable

    bundle = artifact(tmp_path)
    require_bundle_usable(bundle)
    LifecycleRegistry(tmp_path).quarantine("one", reason="severe drift")
    with pytest.raises(ValueError, match="quarantined"):
        require_bundle_usable(bundle)


def test_failed_publication_cleanup_blocks_concurrent_enrollment(tmp_path, monkeypatch):
    import shutil
    from threading import Event

    from trading_agent.ml.lifecycle import LifecycleRegistry

    cleanup, attempted, finished, release = Event(), Event(), Event(), Event()
    real_remove = shutil.rmtree

    def paused_cleanup(path, *args, **kwargs):
        if path == tmp_path / "one":
            cleanup.set()
            assert release.wait(5)
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", paused_cleanup)
    original = LifecycleRegistry._register

    def fail_publication(self, connection, model_id, **kwargs):
        if kwargs.get("actor", "local-training") == "local-training":
            raise ValueError("publication registration failed")
        return original(self, connection, model_id, **kwargs)

    monkeypatch.setattr(LifecycleRegistry, "_register", fail_publication)

    def enroll():
        attempted.set()
        try:
            LifecycleRegistry(tmp_path).enroll("one", actor="operator", reason="reviewed")
        except ValueError:
            return "unavailable"
        finally:
            finished.set()
        return "enrolled"

    with ThreadPoolExecutor(max_workers=2) as pool:
        publisher = pool.submit(artifact, tmp_path)
        assert cleanup.wait(5)
        enroller = pool.submit(enroll)
        assert attempted.wait(2)
        try:
            assert not finished.wait(0.1)
        finally:
            release.set()
        with pytest.raises(ValueError, match="registration failed"):
            publisher.result(timeout=5)
        assert enroller.result(timeout=5) == "unavailable"
    assert not (tmp_path / "one").exists()
    assert LifecycleRegistry(tmp_path).events() == ()
