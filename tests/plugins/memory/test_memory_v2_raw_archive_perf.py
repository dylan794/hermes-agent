"""Structural/perf guardrails for bounded raw archive APIs."""

from __future__ import annotations

import pytest

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


def _seed(store, count=25):
    _index(store)
    events = []
    for idx in range(count):
        events.append(
            store.append_raw_event(
                {
                    "type": "turn" if idx % 2 == 0 else "tool",
                    "session_id": "hot-session" if idx < 20 else "cold-session",
                    "message_id": str(idx),
                    "user_content": f"bounded archive search token-{idx}",
                    "assistant_content": f"assistant reply {idx}",
                    "tool": "memory_v2_status" if idx % 2 else "",
                }
            )
        )
    return events


def test_hot_raw_store_apis_do_not_call_unbounded_read_raw_events(tmp_path, monkeypatch):
    store = _store(tmp_path)
    events = _seed(store)

    def forbidden(*args, **kwargs):
        raise AssertionError("hot raw archive API used unbounded read_raw_events")

    monkeypatch.setattr(store, "read_raw_events", forbidden)

    assert store.get_raw_event_by_id(events[3]["id"])["id"] == events[3]["id"]
    assert [event["id"] for event in store.get_raw_events_by_ids([events[1]["id"], events[4]["id"]])] == [
        events[1]["id"],
        events[4]["id"],
    ]
    assert store.raw_event_exists(events[5]["id"])
    assert store.raw_import_key_exists(_index(store).raw_event_metadata(events[5]["id"])["import_key_sha256"])
    assert store.search_raw_events("token-7", limit=2)[0]["id"] == events[7]["id"]
    assert [event["id"] for event in store.iter_raw_events(session_id="cold-session", limit=3)] == [
        events[24]["id"],
        events[23]["id"],
        events[22]["id"],
    ]
    neighbors = store.get_raw_event_neighbors(events[10]["id"])
    assert neighbors["previous"]["id"] == events[9]["id"]
    assert neighbors["next"]["id"] == events[11]["id"]


def test_bounded_search_hydrates_only_limit_plus_one_metadata_rows(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _seed(store, count=30)

    calls = []
    original = store.read_raw_event_at_offset

    def counted(byte_offset, byte_length):
        calls.append((byte_offset, byte_length))
        return original(byte_offset, byte_length)

    monkeypatch.setattr(store, "read_raw_event_at_offset", counted)

    results = store.search_raw_events("bounded archive search", limit=5)

    assert len(results) == 5
    assert len(calls) == 5


def test_hot_raw_apis_fail_closed_when_index_missing_or_stale(tmp_path):
    store = _store(tmp_path)
    event = store.append_raw_event({"type": "turn", "session_id": "missing-index", "user_content": "needs rebuild"})

    with pytest.raises(Exception, match="raw archive index"):
        store.get_raw_event_by_id(event["id"])

    index = _index(store)
    index.rebuild_raw_archive_index(store)
    assert store.get_raw_event_by_id(event["id"])["id"] == event["id"]

    store.append_raw_event({"type": "turn", "session_id": "missing-index", "user_content": "indexed append keeps ok"})
    assert store.search_raw_events(session_id="missing-index", limit=2)
