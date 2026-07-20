from __future__ import annotations

import contextlib

from plugins.memory.memory_v2.artifacts import register_local_artifact
from plugins.memory.memory_v2.extraction import OfflineSessionExtractor
from plugins.memory.memory_v2.index import MemoryV2Index
from plugins.memory.memory_v2.operations import MemoryOperationService
from plugins.memory.memory_v2.schemas import CandidateMemory, MemoryType
from plugins.memory.memory_v2.store import MemoryV2Store


def _store_index(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    return store, index


def test_resolve_open_loop_holds_lock_and_journals_transition(tmp_path, monkeypatch):
    store, index = _store_index(tmp_path)
    store.write_open_loops([{"id": "loop_1", "status": "open", "text": "finish review", "source_refs": []}])
    held = False
    real_lock = store.profile_lock

    @contextlib.contextmanager
    def tracked_lock(*args, **kwargs):
        nonlocal held
        with real_lock(*args, **kwargs):
            held = True
            try:
                yield
            finally:
                held = False

    real_list = store.list_open_loops

    def checked_list(*args, **kwargs):
        assert held
        return real_list(*args, **kwargs)

    monkeypatch.setattr(store, "profile_lock", tracked_lock)
    monkeypatch.setattr(store, "list_open_loops", checked_list)
    result = MemoryOperationService(store, index).resolve_open_loop("loop_1", "resolved", resolution="done")
    assert result.success is True
    statuses = [row["status"] for row in store.list_operation_records() if row["operation_id"] == result.operation_id]
    assert statuses == ["prepared", "committed"]


def test_resolve_open_loop_fails_closed_on_interrupted_operation(tmp_path):
    store, index = _store_index(tmp_path)
    store.write_open_loops([{"id": "loop_1", "status": "open", "text": "finish review", "source_refs": []}])
    store.append_operation_record({"operation_id": "op_interrupted", "type": "promote_candidate", "status": "prepared"})
    result = MemoryOperationService(store, index).resolve_open_loop("loop_1", "resolved")
    assert result.success is False
    assert store.list_open_loops()[0]["status"] == "open"


def test_extractor_candidate_upsert_holds_profile_lock(tmp_path, monkeypatch):
    store, index = _store_index(tmp_path)
    held = False
    real_lock = store.profile_lock

    @contextlib.contextmanager
    def tracked_lock(*args, **kwargs):
        nonlocal held
        with real_lock(*args, **kwargs):
            held = True
            try:
                yield
            finally:
                held = False

    real_list = store.list_candidates

    def checked_list(*args, **kwargs):
        assert held
        return real_list(*args, **kwargs)

    monkeypatch.setattr(store, "profile_lock", tracked_lock)
    monkeypatch.setattr(store, "list_candidates", checked_list)
    candidate = CandidateMemory(
        id="cand_locked",
        type=MemoryType.PREFERENCE,
        claim="User prefers concise answers",
        proposed_destination="semantic/items",
        confidence=0.8,
        importance=0.7,
        promotion_reason="test",
        source_refs=[],
    )
    assert OfflineSessionExtractor()._upsert_candidate(store, index, candidate) == "created"


def test_append_repairs_missing_or_corrupt_manifest_without_undercount(tmp_path):
    store, _index = _store_index(tmp_path)
    store.append_raw_event({"id": "evt_1", "type": "turn", "user_content": "one"})
    store.raw_archive_manifest_path.unlink()
    store.append_raw_event({"id": "evt_2", "type": "turn", "user_content": "two"})
    assert store.read_raw_archive_manifest()["event_count"] == 2
    store.raw_archive_manifest_path.write_text("event_count: [broken", encoding="utf-8")
    store.append_raw_event({"id": "evt_3", "type": "turn", "user_content": "three"})
    manifest = store.read_raw_archive_manifest()
    assert manifest["event_count"] == 3
    assert manifest["verified_event_count"] == 3


def test_append_updates_raw_index_incrementally_without_full_rebuild(tmp_path, monkeypatch):
    store, index = _store_index(tmp_path)

    def forbidden_rebuild(*_args, **_kwargs):
        raise AssertionError("hot append must not rebuild the full index")

    monkeypatch.setattr(MemoryV2Index, "rebuild_from_store", forbidden_rebuild)
    event = store.append_raw_event({"id": "evt_incremental", "type": "turn", "user_content": "incremental needle"})
    metadata = index.raw_event_metadata(event["id"])
    assert metadata is not None
    assert metadata["byte_offset"] == 0
    assert metadata["byte_length"] > 0
    assert store.read_raw_archive_manifest()["derived_index_status"] == "ok"


def test_incremental_append_leaves_manifest_stale_when_fts_or_offsets_are_incomplete(tmp_path):
    store, index = _store_index(tmp_path)
    first = store.append_raw_event({"id": "evt_first", "type": "turn", "user_content": "first"})
    with index._connect() as conn:
        conn.execute("DELETE FROM raw_events_fts WHERE id = ?", (first["id"],))
        conn.execute(
            """
            INSERT INTO raw_events_fts
              (id, user_content, assistant_content, content, tool, session_id, event_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("orphan_evt", "orphan", "", "", "", "", "turn"),
        )
    store.append_raw_event({"id": "evt_second", "type": "turn", "user_content": "second"})
    assert store.read_raw_archive_manifest()["derived_index_status"] == "stale"

    index.rebuild_from_store(store)
    with index._connect() as conn:
        conn.execute("UPDATE raw_events SET byte_length = NULL WHERE id = ?", (first["id"],))
    store.append_raw_event({"id": "evt_third", "type": "turn", "user_content": "third"})
    assert store.read_raw_archive_manifest()["derived_index_status"] == "stale"


def test_duplicate_artifact_registration_cannot_downgrade_privacy_or_provenance(tmp_path):
    store, _index = _store_index(tmp_path)
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("same bytes", encoding="utf-8")
    second.write_text("same bytes", encoding="utf-8")
    confidential_record = register_local_artifact(store, first, privacy_level="secret")
    duplicate = register_local_artifact(store, second, privacy_level="public")
    assert duplicate.id == confidential_record.id
    assert getattr(duplicate.privacy_level, "value", duplicate.privacy_level) == "secret"
    assert first.resolve().as_uri() in duplicate.metadata["source_uris"]
    assert second.resolve().as_uri() in duplicate.metadata["source_uris"]


def test_duplicate_artifact_registration_preserves_security_and_processing_state(tmp_path):
    store, _index = _store_index(tmp_path)
    first = tmp_path / "risky-first.txt"
    second = tmp_path / "risky-second.txt"
    first.write_text("same risky bytes", encoding="utf-8")
    second.write_text("same risky bytes", encoding="utf-8")
    record = register_local_artifact(store, first)
    record.injection_risk = "confirmed"
    record.processing_status["text_extract"] = "done"
    record.source_refs = ["source_existing"]
    record.derived_refs = ["segment_existing"]
    record.metadata.update(
        {
            "retrieval_disabled": True,
            "injection_risk_reason": "instruction-shaped evidence",
            "injection_risk_flagged_at": "2026-07-20T00:00:00+00:00",
        }
    )
    store.write_artifact_record(record)

    duplicate = register_local_artifact(store, second, privacy_level="public")

    assert duplicate.injection_risk == "confirmed"
    assert duplicate.processing_status["text_extract"] == "done"
    assert duplicate.source_refs == ["source_existing"]
    assert duplicate.derived_refs == ["segment_existing"]
    assert duplicate.metadata["retrieval_disabled"] is True
    assert duplicate.metadata["injection_risk_reason"] == "instruction-shaped evidence"
    assert duplicate.metadata["injection_risk_flagged_at"] == "2026-07-20T00:00:00+00:00"


def test_negated_or_failed_user_completion_does_not_create_completed_action():
    extractor = OfflineSessionExtractor()
    examples = [
        "I did not complete the migration.",
        "We completed neither migration nor rollout.",
        "I completed the migration but it failed validation.",
    ]
    for ordinal, text in enumerate(examples):
        rows = extractor._extract_turn({"id": f"evt_{ordinal}", "type": "turn", "user_content": text})
        assert not any(row.claim_kind == "completed_action" for row in rows)
    positive = extractor._extract_turn({"id": "evt_ok", "type": "turn", "user_content": "I completed the migration."})
    assert any(row.claim_kind == "completed_action" for row in positive)
    mixed_examples = [
        "I did not complete the migration. I completed the docs.",
        "We completed neither migration nor rollout. We completed the docs.",
        "I completed the migration but it failed validation. I completed the docs.",
    ]
    for ordinal, text in enumerate(mixed_examples):
        mixed = extractor._extract_turn(
            {"id": f"evt_mixed_{ordinal}", "type": "turn", "user_content": text}
        )
        completed = [row for row in mixed if row.claim_kind == "completed_action"]
        assert len(completed) == 1
        assert "docs" in completed[0].claim


def test_structured_completion_validation_uses_same_negation_and_tool_outcome_rules():
    extractor = OfflineSessionExtractor()

    def span(text, *, role="user"):
        return {
            "source_id": "evt",
            "field": "user_content" if role == "user" else "result",
            "role": role,
            "text": text,
            "start": 0,
            "end": len(text),
        }

    events = {"evt": {"id": "evt", "type": "turn"}}
    assert not extractor._model_evidence_supports(
        "completed_action",
        [span("I completed the migration but it failed validation.")],
        [],
        events=events,
    )
    assert not extractor._model_evidence_supports(
        "completed_action",
        [span("Completed with 2 failed tests.", role="tool")],
        [],
        events=events,
    )
    assert extractor._model_evidence_supports(
        "completed_action",
        [span("12 passed, 0 failed.", role="tool")],
        [],
        events=events,
    )
