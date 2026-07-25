"""File-store tests for Memory v2 canonical records."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

from plugins.memory.memory_v2.schemas import (
    ArtifactRecord,
    ArtifactSegment,
    CandidateMemory,
    CoreMemoryRecord,
    GateDecision,
    MemoryItem,
    MemoryType,
    ProjectCard,
    SourceRef,
    ValidationError,
)
from plugins.memory.memory_v2.store import MemoryV2Store


EXPECTED_STORE_DIRS = [
    "working",
    "core",
    "sources",
    "inbox",
    "semantic/projects",
    "semantic/environment",
    "episodic/daily",
    "episodic/sessions",
    "graph",
    "indexes/vector",
    "evals",
    "reports/daily_consolidation",
    "reports/weekly_reflection",
    "audit",
    "artifacts",
    "artifacts/raw",
    "artifacts/derived",
    "artifacts/manifests",
    "artifacts/indexes",
]


def test_store_initialize_creates_profile_scoped_layout(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")

    store.initialize()

    for rel in EXPECTED_STORE_DIRS:
        assert (tmp_path / "memory_v2" / rel).is_dir(), rel
    assert (tmp_path / "memory_v2" / "README.md").is_file()
    assert (tmp_path / "memory_v2" / "config.yaml").is_file()
    assert (tmp_path / "memory_v2" / "inbox" / "raw_events.jsonl").is_file()
    assert (tmp_path / "memory_v2" / "inbox" / "candidates.jsonl").is_file()
    assert (tmp_path / "memory_v2" / "inbox" / "rejected.jsonl").is_file()
    assert (tmp_path / "memory_v2" / "audit").is_dir()
    assert not (tmp_path / "memory_v2" / "audit" / "operations.jsonl").exists()


def test_operation_log_is_materialized_only_when_operation_is_appended(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()

    assert not store.operations_path.exists()

    store.append_operation_record({"operation": "test_operation", "status": "ok"})

    assert store.operations_path.is_file()
    records = store.list_operation_records()
    assert len(records) == 1
    assert records[0]["operation"] == "test_operation"


def test_atomic_yaml_write_failure_preserves_existing_file_and_removes_temp(tmp_path, monkeypatch):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    path = store.projects_dir / "atomic.yaml"
    original = {"version": 1, "name": "original"}
    store._atomic_write_yaml(path, original)

    def crash_after_tmp_write(src, dst):
        raise RuntimeError("simulated crash before replace")

    monkeypatch.setattr(os, "replace", crash_after_tmp_write)

    with pytest.raises(RuntimeError, match="simulated crash"):
        store._atomic_write_yaml(path, {"version": 1, "name": "replacement"})

    assert yaml.safe_load(path.read_text(encoding="utf-8")) == original
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_concurrent_atomic_jsonl_rewrites_leave_parseable_file_without_temp_files(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    candidate_sets = [
        [CandidateMemory(id=f"cand_concurrent_{worker}_{idx}", type="fact", claim=f"claim {worker}-{idx}") for idx in range(3)]
        for worker in range(8)
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(store.rewrite_candidates, candidate_sets))

    lines = store.candidates_path.read_text(encoding="utf-8").splitlines()
    assert lines
    parsed = [json.loads(line) for line in lines]
    assert all(isinstance(item, dict) and item["id"].startswith("cand_concurrent_") for item in parsed)
    assert store.list_candidates()
    assert list(store.candidates_path.parent.glob(f".{store.candidates_path.name}.*.tmp")) == []


def test_write_read_and_list_artifact_records_as_profile_scoped_manifests(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    record = ArtifactRecord(
        id="artifact_design_doc",
        content_hash="sha256:abc123",
        modality="pdf",
        source_type="upload",
        source_uri="file:///tmp/design.pdf",
        project_id="project:hermes-memory-v2",
        processing_status={"ocr": "done"},
    )

    path = store.write_artifact_record(record)
    loaded = store.read_artifact_record("artifact_design_doc")
    listed = store.list_artifact_records()

    assert (
        path
        == tmp_path
        / "memory_v2"
        / "artifacts"
        / "manifests"
        / "artifact_design_doc.yaml"
    )
    assert path.is_file()
    assert loaded == record
    assert listed == [record]


def test_write_and_list_artifact_segments_under_safe_artifact_directory(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    first = ArtifactSegment(
        id="segment_1",
        artifact_id="artifact/design:doc",
        segment_type="page",
        text="Memory v2 should stay source-grounded.",
    )
    second = ArtifactSegment(
        id="segment/2",
        artifact_id="artifact/design:doc",
        segment_type="page",
        summary="Second page.",
    )

    first_path = store.write_artifact_segment(first)
    second_path = store.write_artifact_segment(second)
    listed = store.list_artifact_segments("artifact/design:doc")

    assert first_path.is_file()
    assert second_path.is_file()
    assert first_path != second_path
    assert first_path.parent == second_path.parent
    assert first_path.parent.name == "segments"
    assert listed == [second, first]


def test_artifact_paths_sanitize_unsafe_ids_without_escaping_base_dir(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    record = ArtifactRecord(
        id="../../outside/artifact",
        content_hash="sha256:escape",
        modality="text",
        source_type="manual",
        source_uri="note://escape",
    )
    segment = ArtifactSegment(
        id="../../outside/segment",
        artifact_id="../../outside/artifact",
        segment_type="chunk",
    )

    record_path = store.write_artifact_record(record)
    segment_path = store.write_artifact_segment(segment)

    record_path.resolve().relative_to(store.base_dir)
    segment_path.resolve().relative_to(store.base_dir)
    assert ".." not in record_path.relative_to(store.base_dir).parts
    assert ".." not in segment_path.relative_to(store.base_dir).parts
    assert not (tmp_path / "outside").exists()


def test_artifact_write_rejects_manifest_path_outside_base_dir_even_if_helper_regresses(
    tmp_path, monkeypatch
):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    record = ArtifactRecord(
        id="artifact_design_doc",
        content_hash="sha256:abc123",
        modality="pdf",
        source_type="upload",
        source_uri="file:///tmp/design.pdf",
    )
    outside_path = tmp_path / "outside" / "artifact_design_doc.yaml"
    monkeypatch.setattr(
        store, "_artifact_record_path", lambda _artifact_id: outside_path
    )

    with pytest.raises(ValidationError, match="base_dir"):
        store.write_artifact_record(record)

    assert not outside_path.exists()


def test_artifact_write_rejects_segment_path_outside_base_dir_even_if_helper_regresses(
    tmp_path, monkeypatch
):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    segment = ArtifactSegment(
        id="segment_1",
        artifact_id="artifact_design_doc",
        segment_type="chunk",
    )
    outside_path = tmp_path / "outside" / "segment_1.yaml"
    monkeypatch.setattr(
        store, "_artifact_segment_path", lambda _artifact_id, _segment_id: outside_path
    )

    with pytest.raises(ValidationError, match="base_dir"):
        store.write_artifact_segment(segment)

    assert not outside_path.exists()


def test_append_and_read_raw_events_jsonl(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()

    first = store.append_raw_event({
        "type": "turn",
        "session_id": "session-1",
        "content": "hello",
    })
    second = store.append_raw_event({
        "type": "tool",
        "session_id": "session-1",
        "tool": "memory_v2_status",
    })

    events = store.read_raw_events()

    assert first["id"].startswith("event_")
    assert first["created_at"]
    assert first["schema_version"] == 1
    assert first["trust_level"] == "untrusted"
    assert first["can_instruct"] is False
    assert first["chain_index"] == 0
    assert first["previous_record_sha256"] == ""
    assert first["content_sha256"].startswith("sha256:")
    assert first["record_sha256"].startswith("sha256:")
    assert second["previous_record_sha256"] == first["record_sha256"]
    assert second["chain_index"] == 1
    assert second["id"] != first["id"]
    assert [event["type"] for event in events] == ["turn", "tool"]
    assert events[0]["session_id"] == "session-1"


def test_raw_event_blank_timestamps_are_filled_and_invalid_timestamps_rejected(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()

    event = store.append_raw_event(
        {
            "type": "turn",
            "content": "timestamp evidence",
            "created_at": "",
            "observed_at": "",
        }
    )

    assert event["created_at"]
    assert event["observed_at"] == event["created_at"]
    assert store.read_source_ref(event["id"]).observed_at == event["created_at"]

    with pytest.raises(ValidationError, match="created_at.*ISO-8601"):
        store.append_raw_event(
            {
                "type": "turn",
                "content": "invalid timestamp must not become evidence",
                "created_at": "not-a-timestamp",
            }
        )

    assert store.count_raw_events() == 1


def test_bounded_raw_hydration_recomputes_hashes_and_rejects_degraded_manifest(tmp_path):
    from plugins.memory.memory_v2.index import MemoryV2Index

    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    index = MemoryV2Index(store.default_index_path)
    index.initialize()
    event = store.append_raw_event(
        {"type": "turn", "content": "canonical alpha", "created_at": "2026-07-20T00:00:00Z"}
    )
    original = store.raw_events_path.read_text(encoding="utf-8")
    store.raw_events_path.write_text(
        original.replace("canonical alpha", "canonical omega"),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="content hash mismatch"):
        store.get_raw_event_by_id(event["id"], index=index)

    manifest = store.rebuild_raw_archive_manifest()
    assert manifest["status"] == "degraded"
    with pytest.raises(ValidationError, match="integrity is degraded"):
        store.get_raw_event_by_id(event["id"], index=index)

    with pytest.raises(ValidationError, match="integrity is degraded"):
        store.append_raw_event({"type": "turn", "content": "must not extend degraded evidence"})


def test_store_rejects_profile_root_symlink_escape(tmp_path):
    profile_root = tmp_path / "profile"
    outside = tmp_path / "outside"
    profile_root.mkdir()
    outside.mkdir()
    memory_link = profile_root / "memory_v2"
    try:
        memory_link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable on this platform: {exc}")

    with pytest.raises(ValidationError, match="symlink or junction escape"):
        MemoryV2Store(memory_link)

    assert list(outside.iterdir()) == []


def test_raw_archive_manifest_and_integrity_report_are_privacy_safe(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()

    store.append_raw_event({
        "type": "turn",
        "session_id": "session-1",
        "user_content": "token sk-1234567890abcdef should be redacted",
        "assistant_content": "Stored.",
    })

    manifest = store.read_raw_archive_manifest()
    report = store.verify_raw_archive()
    raw_text = store.raw_events_path.read_text(encoding="utf-8")
    serialized = json.dumps({"manifest": manifest, "report": report})

    assert manifest["path"] == "inbox/raw_events.jsonl"
    assert manifest["event_count"] == 1
    assert manifest["verified_event_count"] == 1
    assert manifest["status"] == "ok"
    assert report["status"] == "ok"
    assert report["issue_count"] == 0
    assert "sk-1234567890abcdef" not in raw_text
    assert "sk-1234567890abcdef" not in serialized
    assert str(tmp_path) not in serialized


def test_verify_raw_archive_detects_payload_tampering_without_raw_text_leak(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    event = store.append_raw_event({
        "type": "turn",
        "session_id": "session-1",
        "user_content": "private archive text",
    })
    lines = store.raw_events_path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["user_content"] = "tampered private archive text"
    store.raw_events_path.write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8")

    report = store.verify_raw_archive()
    serialized = json.dumps(report)

    assert report["status"] == "degraded"
    assert any(issue["code"] == "raw_event_content_hash_mismatch" for issue in report["issues"])
    assert any(issue["code"] == "raw_event_record_hash_mismatch" for issue in report["issues"])
    assert event["id"] in serialized
    assert "tampered private archive text" not in serialized


def test_verify_raw_archive_detects_deleted_middle_event(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    store.append_raw_event({"type": "turn", "content": "first"})
    store.append_raw_event({"type": "turn", "content": "second"})
    store.append_raw_event({"type": "turn", "content": "third"})
    lines = store.raw_events_path.read_text(encoding="utf-8").splitlines()
    store.raw_events_path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")

    report = store.verify_raw_archive()

    assert report["status"] == "degraded"
    assert any(issue["code"] == "raw_event_previous_hash_mismatch" for issue in report["issues"])


def test_concurrent_raw_event_appends_preserve_parseable_hash_chain(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()

    def append_event(index):
        return store.append_raw_event({"type": "turn", "content": f"event {index}"})

    with ThreadPoolExecutor(max_workers=8) as executor:
        events = list(executor.map(append_event, range(20)))

    assert len({event["id"] for event in events}) == 20
    assert store.count_raw_events() == 20
    report = store.verify_raw_archive()
    assert report["status"] == "ok"
    assert report["verified_event_count"] == 20


def test_append_raw_event_materializes_canonical_source_ref(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()

    event = store.append_raw_event({
        "type": "turn",
        "session_id": "session-1",
        "user_content": "Remember that source refs should be canonical.",
        "assistant_content": "Queued.",
    })

    source = store.read_source_ref(event["id"])
    assert source is not None
    assert source.id == event["id"]
    assert source.type.value == "message"
    assert source.uri == f"raw_event:{event['id']}"
    assert source.title == "Raw turn evidence from session session-1"
    assert source.observed_at == event["created_at"]
    assert source.quote == "Remember that source refs should be canonical."
    assert (tmp_path / "memory_v2" / "sources" / f"{event['id']}.yaml").is_file()


def test_append_raw_event_rejects_non_object_payload(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()

    with pytest.raises(ValidationError, match="raw event"):
        store.append_raw_event(["not", "a", "dict"])  # type: ignore[arg-type]


def test_append_and_list_candidates(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    candidate = CandidateMemory(
        id="cand_memory_v2_goal",
        type=MemoryType.PROJECT_STATE,
        claim="Alex wants robust low-compute Memory v2.",
        proposed_destination="semantic/projects/hermes-memory-v2.yaml",
        confidence=0.9,
        importance=0.8,
        source_refs=["source_memory_v2_thread"],
    )

    store.append_candidate(candidate)

    candidates = store.list_candidates()
    assert candidates == [candidate]
    assert store.count_pending_candidates() == 1


def test_append_rejected_candidate_records_decision(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    candidate = CandidateMemory(
        id="cand_rejected",
        type="fact",
        claim="Temporary detail",
        gate_decision=GateDecision.REJECTED,
        decision_reason="Too ephemeral for durable memory.",
    )

    store.append_rejected_candidate(candidate)

    assert store.list_rejected_candidates() == [candidate]
    assert store.count_pending_candidates() == 0


def test_write_read_and_list_core_memory_records_by_category(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    user_record = CoreMemoryRecord(
        id="core_user_style",
        category="user",
        statement="Alex prefers direct, grounded answers.",
        priority=0.95,
        source_refs=["source_user_profile"],
    )
    identity_record = CoreMemoryRecord(
        id="core_assistant_identity",
        category="assistant_identity",
        statement="Hermes should be intellectually honest and tool-grounded.",
        priority=0.9,
        source_refs=["source_soul"],
    )

    user_path = store.write_core_memory_record(user_record)
    identity_path = store.write_core_memory_record(identity_record)

    assert user_path == tmp_path / "memory_v2" / "core" / "user.yaml"
    assert identity_path == tmp_path / "memory_v2" / "core" / "assistant_identity.yaml"
    assert store.read_core_memory_record("core_user_style") == user_record
    assert store.list_core_memory_records(category="user") == [user_record]
    assert store.list_core_memory_records() == [user_record, identity_record]


def test_write_read_and_list_project_cards(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    card = ProjectCard(
        id="Hermes Memory v2",
        name="Hermes Memory v2",
        goal="Build robust source-grounded memory for Hermes.",
        current_state="File store layer under implementation.",
        decisions=["Use human-readable canonical files."],
        source_refs=["source_memory_v2_thread"],
    )

    path = store.write_project_card(card)
    loaded_by_id = store.read_project_card("project:hermes-memory-v2")
    loaded_by_name = store.read_project_card("Hermes Memory v2")
    listed = store.list_project_cards()

    assert (
        path
        == tmp_path / "memory_v2" / "semantic" / "projects" / "hermes-memory-v2.yaml"
    )
    assert path.is_file()
    assert loaded_by_id == card
    assert loaded_by_name == card
    assert listed == [card]


def test_write_read_and_list_memory_items_as_canonical_yaml(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    item = MemoryItem(
        id="pref_response_style",
        type="preference",
        subject="Alex",
        predicate="prefers_response_style",
        value="direct no-BS tool-grounded help",
        summary="Alex prefers direct, no-BS, tool-grounded help.",
        confidence=0.98,
        importance=0.95,
        source_refs=["source_user_profile"],
        tags=["user_preference", "style"],
    )

    path = store.write_memory_item(item)
    loaded = store.read_memory_item("pref_response_style")
    listed = store.list_memory_items()

    assert (
        path
        == tmp_path / "memory_v2" / "semantic" / "items" / "pref_response_style.yaml"
    )
    assert path.is_file()
    assert loaded == item
    assert listed == [item]


def test_memory_item_paths_do_not_collide_for_distinct_unsafe_ids(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    slash_id = MemoryItem(id="a/b", type="fact", subject="first", value="first")
    colon_id = MemoryItem(id="a:b", type="fact", subject="second", value="second")

    slash_path = store.write_memory_item(slash_id)
    colon_path = store.write_memory_item(colon_id)

    assert slash_path != colon_path
    assert store.read_memory_item("a/b") == slash_id
    assert store.read_memory_item("a:b") == colon_id
    assert sorted(item.id for item in store.list_memory_items()) == ["a/b", "a:b"]


def test_source_ref_paths_do_not_collide_for_distinct_unsafe_ids(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    slash_source = SourceRef(id="source/a", type="manual", uri="memory://source/a")
    colon_source = SourceRef(id="source:a", type="manual", uri="memory://source:a")

    slash_path = store.write_source_ref(slash_source)
    colon_path = store.write_source_ref(colon_source)

    assert slash_path != colon_path
    assert store.read_source_ref("source/a") == slash_source
    assert store.read_source_ref("source:a") == colon_source


def test_list_memory_items_can_filter_by_type_and_status(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    active_pref = MemoryItem(
        id="pref_active", type="preference", subject="Alex", status="active"
    )
    superseded_pref = MemoryItem(
        id="pref_old",
        type="preference",
        subject="Alex",
        status="superseded",
        superseded_by="pref_active",
    )
    env = MemoryItem(id="env_host", type="environment", subject="Hermes runtime")
    for item in (active_pref, superseded_pref, env):
        store.write_memory_item(item)

    assert store.list_memory_items(memory_type="preference", status="active") == [
        active_pref
    ]
    assert store.list_memory_items(memory_type="preference") == [
        active_pref,
        superseded_pref,
    ]
    assert store.list_memory_items(status="active") == [env, active_pref]


def test_write_read_and_list_source_refs_as_canonical_yaml(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    source = SourceRef(
        id="source_memory_v2_thread",
        type="session",
        uri="discord://thread/1508915896054452264",
        title="Memory v2 design thread",
        observed_at="2026-05-26T00:00:00Z",
        quote="Alex asked for robust low-compute memory.",
    )

    path = store.write_source_ref(source)
    loaded = store.read_source_ref("source_memory_v2_thread")
    listed = store.list_source_refs()

    assert path == tmp_path / "memory_v2" / "sources" / "source_memory_v2_thread.yaml"
    assert path.is_file()
    assert loaded == source
    assert listed == [source]


def test_read_missing_project_card_returns_none(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()

    assert store.read_project_card("project:missing") is None


def test_store_counts_raw_event_lines_without_blank_lines(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    raw_path = tmp_path / "memory_v2" / "inbox" / "raw_events.jsonl"
    raw_path.write_text('\n{"id": "event_1"}\n\n{"id": "event_2"}\n', encoding="utf-8")

    assert store.count_raw_events() == 2


def test_read_raw_events_limit_zero_returns_empty_list(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    for i in range(3):
        store.append_raw_event({"id": f"event_{i}", "content": str(i)})

    assert store.read_raw_events(limit=0) == []


def test_read_raw_events_rejects_negative_limit(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()

    try:
        store.read_raw_events(limit=-1)
    except ValidationError as exc:
        assert "limit" in str(exc)
    else:
        raise AssertionError("negative limit should raise ValidationError")
