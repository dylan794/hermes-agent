"""Rebuildable raw archive index tests for Memory v2."""

from __future__ import annotations

import json
import sqlite3

from plugins.memory.memory_v2.index import MemoryV2Index
from plugins.memory.memory_v2.store import MemoryV2Store


def _store(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    return store


def _index(store):
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    return index


def test_initialize_creates_rebuildable_raw_archive_tables(tmp_path):
    store = _store(tmp_path)
    index = _index(store)

    tables = index.table_names()

    assert "raw_events" in tables
    assert "raw_events_fts" in tables
    with sqlite3.connect(str(index.db_path)) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(raw_events)")}
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(raw_events)")}

    assert {
        "id",
        "event_type",
        "source_system",
        "session_id",
        "provider_session_id",
        "message_id",
        "created_at",
        "observed_at",
        "archive_status",
        "trust_level",
        "privacy_level",
        "can_instruct",
        "chain_index",
        "record_sha256",
        "previous_record_sha256",
        "content_sha256",
        "byte_offset",
        "byte_length",
        "line_no",
        "source_ref_id",
        "import_key_sha256",
        "indexed_at",
    }.issubset(columns)
    assert {
        "raw_events_session_created_idx",
        "raw_events_type_created_idx",
        "raw_events_created_idx",
        "raw_events_import_key_idx",
        "raw_events_message_idx",
        "raw_events_record_hash_idx",
        "raw_events_chain_index_idx",
    }.issubset(indexes)


def test_index_raw_archive_event_persists_metadata_and_fts_without_trusting_text(tmp_path):
    store = _store(tmp_path)
    index = _index(store)
    event = store.append_raw_event(
        {
            "type": "turn",
            "source_system": "discord",
            "session_id": "session-raw",
            "provider_session_id": "provider-1",
            "message_id": "msg-1",
            "user_content": "Archive evidence says rebuildable indexes matter.",
            "assistant_content": "Noted.",
            "can_instruct": True,
            "privacy_level": "standard",
        }
    )

    assert index.index_raw_archive_event(event, byte_offset=12, byte_length=345, line_no=7)

    metadata = index.raw_event_metadata(event["id"])
    assert metadata is not None
    assert metadata["event_type"] == "turn"
    assert metadata["source_system"] == "discord"
    assert metadata["session_id"] == "session-raw"
    assert metadata["provider_session_id"] == "provider-1"
    assert metadata["message_id"] == "msg-1"
    assert metadata["trust_level"] == "untrusted"
    # Store normalization preserves the raw archive safety gate even if input attempted otherwise.
    assert metadata["can_instruct"] == 0
    assert metadata["byte_offset"] == 12
    assert metadata["byte_length"] == 345
    assert metadata["line_no"] == 7
    assert metadata["source_ref_id"] == event["id"]
    assert metadata["import_key_sha256"].startswith("sha256:")
    assert index.search_raw_archive("rebuildable indexes", limit=5)[0]["id"] == event["id"]
    assert index.raw_import_key_exists(metadata["import_key_sha256"])


def test_index_raw_archive_event_persists_event_provided_import_key_sha256(tmp_path):
    store = _store(tmp_path)
    index = _index(store)
    provided_import_key = "sha256:" + "1" * 64
    event = {
        "id": "raw-import-key-preserved",
        "type": "turn",
        "source_system": "hermes_sessiondb",
        "session_id": "session-import-key",
        "provider_session_id": "session-import-key",
        "message_id": "42",
        "import_key_sha256": provided_import_key,
        "content": "session backfill import keys should be dedupe-stable",
    }
    fallback_import_key = index._raw_import_key_sha256(event)
    assert fallback_import_key != provided_import_key

    assert index.index_raw_archive_event(event)

    metadata = index.raw_event_metadata(event["id"])
    assert metadata is not None
    assert metadata["import_key_sha256"] == provided_import_key
    assert index.raw_import_key_exists(provided_import_key)
    assert not index.raw_import_key_exists(fallback_import_key)


def test_direct_raw_archive_index_forces_untrusted_non_instructing_metadata(tmp_path):
    store = _store(tmp_path)
    index = _index(store)
    event = {
        "id": "raw-direct-trust-attempt",
        "type": "turn",
        "session_id": "direct-session",
        "user_content": "direct archive event should remain safe evidence",
        "trust_level": "trusted",
        "can_instruct": True,
    }

    assert index.index_raw_archive_event(event)

    metadata = index.raw_event_metadata(event["id"])
    assert metadata is not None
    assert metadata["trust_level"] == "untrusted"
    assert metadata["can_instruct"] == 0


def test_rebuild_raw_archive_index_streams_jsonl_offsets_and_updates_manifest(tmp_path):
    store = _store(tmp_path)
    first = store.append_raw_event({"type": "turn", "session_id": "s1", "user_content": "first raw archive line"})
    second = store.append_raw_event({"type": "tool", "session_id": "s1", "tool": "memory_v2_status"})
    raw_lines = store.raw_events_path.read_bytes().splitlines(keepends=True)
    second_offset = len(raw_lines[0])
    index = _index(store)

    status = index.rebuild_raw_archive_index(store)
    again = index.rebuild_raw_archive_index(store)

    assert status == again
    assert status["derived_index_status"] == "ok"
    assert status["indexed_event_count"] == 2
    assert status["last_indexed_record_sha256"] == second["record_sha256"]
    assert status["raw_index_schema_version"] == 1
    assert index.raw_event_count() == 2
    assert index.raw_event_metadata(first["id"])["byte_offset"] == 0
    second_meta = index.raw_event_metadata(second["id"])
    assert second_meta["byte_offset"] == second_offset
    assert second_meta["byte_length"] == len(raw_lines[1])
    assert second_meta["line_no"] == 2
    manifest = store.read_raw_archive_manifest()
    assert manifest["derived_index_status"] == "ok"
    assert manifest["indexed_event_count"] == 2
    assert manifest["last_indexed_record_sha256"] == second["record_sha256"]
    assert manifest["raw_index_schema_version"] == 1


def test_raw_fts_schema_mismatch_marks_derived_content_needing_rebuild(tmp_path):
    store = _store(tmp_path)
    db_path = store.base_dir / "indexes" / "memory.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_events (
              id TEXT PRIMARY KEY,
              event_type TEXT,
              source_system TEXT,
              session_id TEXT,
              provider_session_id TEXT,
              message_id TEXT,
              created_at TEXT,
              observed_at TEXT,
              archive_status TEXT,
              trust_level TEXT,
              privacy_level TEXT,
              can_instruct INTEGER NOT NULL DEFAULT 0,
              chain_index INTEGER,
              record_sha256 TEXT,
              previous_record_sha256 TEXT,
              content_sha256 TEXT,
              byte_offset INTEGER,
              byte_length INTEGER,
              line_no INTEGER,
              source_ref_id TEXT,
              import_key_sha256 TEXT,
              indexed_at TEXT NOT NULL
            );
            INSERT INTO raw_events (id, session_id, event_type, indexed_at) VALUES ('legacy-raw', 'legacy-session', 'turn', '2026-01-01T00:00:00Z');
            CREATE VIRTUAL TABLE raw_events_fts USING fts5(id UNINDEXED, content);
            INSERT INTO raw_events_fts (id, content) VALUES ('legacy-raw', 'legacy searchable content');
            """
        )

    index = MemoryV2Index(db_path)
    index.initialize()

    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT value FROM raw_index_metadata WHERE key = 'raw_events_fts_status'"
        ).fetchone()
    assert row == ("needs_rebuild",)


def test_append_raw_event_updates_existing_derived_raw_index_after_canonical_append(tmp_path):
    store = _store(tmp_path)
    index = _index(store)

    event = store.append_raw_event(
        {
            "type": "turn",
            "session_id": "append-session",
            "user_content": "canonical append updates derived raw index",
        }
    )

    assert json.loads(store.raw_events_path.read_text(encoding="utf-8").splitlines()[0])["id"] == event["id"]
    metadata = index.raw_event_metadata(event["id"])
    assert metadata is not None
    assert metadata["byte_offset"] == 0
    assert metadata["byte_length"] == store.raw_events_path.stat().st_size
    assert metadata["line_no"] == 1
    assert index.search_raw_archive("derived raw index", limit=5)[0]["id"] == event["id"]
    manifest = store.read_raw_archive_manifest()
    assert manifest["derived_index_status"] == "ok"
    assert manifest["indexed_event_count"] == 1
    assert manifest["last_indexed_record_sha256"] == event["record_sha256"]


def test_store_raw_archive_hot_lookups_hydrate_from_indexed_byte_slices(tmp_path):
    store = _store(tmp_path)
    index = _index(store)
    first = store.append_raw_event({"type": "turn", "session_id": "s1", "user_content": "first bounded lookup"})
    second = store.append_raw_event({"type": "turn", "session_id": "s1", "user_content": "second bounded lookup"})
    third = store.append_raw_event({"type": "tool", "session_id": "s2", "tool": "archive_show"})

    second_meta = index.raw_event_metadata(second["id"])

    assert store.read_raw_event_at_offset(second_meta["byte_offset"], second_meta["byte_length"])["id"] == second["id"]
    assert store.get_raw_event_by_id(second["id"])["id"] == second["id"]
    assert [event["id"] for event in store.get_raw_events_by_ids([first["id"], third["id"]])] == [first["id"], third["id"]]
    assert store.raw_event_exists(first["id"])
    assert not store.raw_event_exists("missing-raw-event")
    assert store.raw_import_key_exists(second_meta["import_key_sha256"])
    assert [event["id"] for event in store.search_raw_events("bounded lookup", session_id="s1", limit=5)] == [second["id"], first["id"]]
    neighbors = store.get_raw_event_neighbors(second["id"])
    assert neighbors["previous"]["id"] == first["id"]
    assert neighbors["next"]["id"] == third["id"]


def test_append_marks_manifest_stale_when_existing_derived_raw_index_is_missing(tmp_path):
    store = _store(tmp_path)
    index = _index(store)
    first = store.append_raw_event(
        {
            "type": "turn",
            "session_id": "append-session",
            "user_content": "first append updates available index",
        }
    )
    assert store.read_raw_archive_manifest()["derived_index_status"] == "ok"

    for suffix in ("", "-wal", "-shm"):
        path = type(index.db_path)(str(index.db_path) + suffix)
        if path.exists():
            path.unlink()

    second = store.append_raw_event(
        {
            "type": "turn",
            "session_id": "append-session",
            "user_content": "second append cannot update missing derived index",
        }
    )

    manifest = store.read_raw_archive_manifest()
    assert manifest["event_count"] == 2
    assert manifest["last_record_sha256"] == second["record_sha256"]
    assert manifest["derived_index_status"] == "stale"
    assert manifest["indexed_event_count"] == 1
    assert manifest["last_indexed_record_sha256"] == first["record_sha256"]
