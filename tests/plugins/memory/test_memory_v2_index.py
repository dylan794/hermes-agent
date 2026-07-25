"""SQLite FTS index/search tests for Memory v2."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from plugins.memory.memory_v2.health import MemoryHealthChecker
from plugins.memory.memory_v2.index import MemoryV2Index
from plugins.memory.memory_v2.schemas import CandidateMemory, MemoryItem, MemoryType, ProjectCard, SourceRef, ValidationError
from plugins.memory.memory_v2.store import MemoryV2Store


def _store(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    return store


def test_fts_queries_keep_single_non_stopword_mixed_with_stopwords():
    assert MemoryV2Index._fts_queries("where is ffmpeg") == ['"ffmpeg"']
    assert MemoryV2Index._fts_queries("where is") == []


def test_index_initialize_creates_sqlite_schema(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")

    index.initialize()

    assert index.db_path.is_file()
    tables = index.table_names()
    assert "memories" in tables
    assert "memories_fts" in tables
    assert "retrieval_log" in tables
    assert "source_refs" in tables


def test_index_connection_context_closes_handle_for_atomic_replacement(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    conn = index._connect()

    with conn:
        assert conn.execute("SELECT 1").fetchone() == (1,)

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")


def test_initialize_rebuilds_legacy_fts_table_with_missing_columns(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(index.db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE memories (
              id TEXT PRIMARY KEY,
              type TEXT NOT NULL,
              title TEXT,
              body TEXT,
              summary TEXT,
              status TEXT,
              updated_at TEXT NOT NULL,
              source_refs TEXT,
              tags TEXT,
              file_path TEXT
            );
            CREATE VIRTUAL TABLE memories_fts USING fts5(id UNINDEXED, title, body, summary);
            """
        )

    index.initialize()
    item = MemoryItem(
        id="pref_response_style",
        type="preference",
        subject="Alex",
        predicate="prefers_response_style",
        value="direct answers",
        summary="Alex prefers direct answers.",
        source_refs=["event_1"],
    )
    index.index_memory_item(item)

    assert index.search("direct answers", limit=5)[0]["id"] == "pref_response_style"


def test_index_project_card_is_searchable(tmp_path):
    store = _store(tmp_path)
    card = ProjectCard(
        id="Hermes Memory v2",
        name="Hermes Memory v2",
        goal="Build robust source-grounded low-compute memory for Hermes.",
        current_state="SQLite FTS search layer under implementation.",
        decisions=["Use project-card block retrieval before chunk retrieval."],
        source_refs=["source_memory_v2_thread"],
    )
    store.write_project_card(card)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    index.index_project_card(card, file_path=store.projects_dir / "hermes-memory-v2.yaml")
    results = index.search("source grounded memory", limit=5)

    assert len(results) == 1
    assert results[0]["id"] == "project:hermes-memory-v2"
    assert results[0]["type"] == "project_state"
    assert "source-grounded" in results[0]["body"]
    assert results[0]["source_refs"] == ["source_memory_v2_thread"]


def test_index_candidate_is_searchable(tmp_path):
    store = _store(tmp_path)
    candidate = CandidateMemory(
        id="cand_memory_v2_goal",
        type=MemoryType.PROJECT_STATE,
        claim="Alex wants Memory v2 to stay low-compute and source-grounded.",
        source_refs=["event_123"],
    )
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    index.index_candidate(candidate)
    results = index.search("low compute source grounded", limit=5)

    assert [result["id"] for result in results] == ["cand_memory_v2_goal"]
    assert results[0]["type"] == "candidate"
    assert results[0]["status"] == "pending"


def test_index_raw_event_is_searchable(tmp_path):
    store = _store(tmp_path)
    event = store.append_raw_event(
        {
            "type": "turn",
            "session_id": "session-1",
            "user_content": "Remember that Memory v2 needs gated writes.",
            "assistant_content": "Queued as a candidate.",
        }
    )
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    index.index_raw_event(event)
    results = index.search("gated writes", limit=5)

    assert len(results) == 1
    assert results[0]["id"] == event["id"]
    assert results[0]["type"] == "raw_event"
    assert results[0]["source_refs"] == [event["id"]]


def test_interrupted_rebuild_from_store_preserves_previous_usable_index(tmp_path, monkeypatch):
    store = _store(tmp_path)
    old_card = ProjectCard(id="Old Project", name="Old Project", goal="Original searchable goal.")
    store.write_project_card(old_card)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    index.rebuild_from_store(store)
    assert index.search("original searchable", limit=5)[0]["id"] == "project:old-project"

    store.write_project_card(ProjectCard(id="New Project", name="New Project", goal="Replacement zephyr target."))

    def fail_index_project_card(*args, **kwargs):
        raise RuntimeError("simulated rebuild crash")

    monkeypatch.setattr(index, "index_project_card", fail_index_project_card)

    try:
        index.rebuild_from_store(store)
    except RuntimeError as exc:
        assert "simulated rebuild crash" in str(exc)
    else:  # pragma: no cover - explicit assertion is clearer than pytest.raises for post-checks
        raise AssertionError("rebuild_from_store should have raised")

    assert index.search("original searchable", limit=5)[0]["id"] == "project:old-project"
    assert index.search("replacement zephyr", limit=5) == []
    assert list(index.db_path.parent.glob(f".{index.db_path.name}.*.tmp")) == []


def test_concurrent_rebuild_from_store_leaves_usable_sqlite_index(tmp_path):
    store = _store(tmp_path)
    for worker in range(4):
        store.write_project_card(
            ProjectCard(id=f"Concurrent {worker}", name=f"Concurrent {worker}", goal=f"Shared rebuild token {worker}.")
        )
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    with ThreadPoolExecutor(max_workers=3) as executor:
        counts = list(executor.map(lambda _: index.rebuild_from_store(store), range(3)))

    assert all(count["project_cards"] == 4 for count in counts)
    assert index.count_memories() == 4
    assert len(index.search("shared rebuild token", limit=10)) == 4
    with sqlite3.connect(str(index.db_path)) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_health_repair_recovers_from_partial_broken_index(tmp_path):
    store = _store(tmp_path)
    store.write_project_card(ProjectCard(id="Repair Project", name="Repair Project", goal="Recoverable index target."))
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    with sqlite3.connect(str(index.db_path)) as conn:
        conn.execute("DELETE FROM memories")
        conn.execute("DELETE FROM memories_fts")

    checker = MemoryHealthChecker(store, index)
    health = checker.check()
    assert any(issue["code"] == "index_count_mismatch" for issue in health["issues"])

    repair = checker.repair(dry_run=False)

    assert repair["actions"][0]["action"] == "rebuild_index"
    assert index.search("recoverable index target", limit=5)[0]["id"] == "project:repair-project"
    with sqlite3.connect(str(index.db_path)) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_rebuild_from_store_indexes_project_cards_candidates_and_events(tmp_path):
    store = _store(tmp_path)
    store.write_project_card(
        ProjectCard(
            id="Hermes Memory v2",
            name="Hermes Memory v2",
            goal="Build robust memory search.",
            current_state="Rebuild indexes existing files.",
        )
    )
    store.append_candidate(
        CandidateMemory(
            id="cand_eval_contract",
            type="fact",
            claim="Memory v2 should be benchmark-first and source-grounded.",
        )
    )
    store.append_raw_event(
        {"type": "turn", "session_id": "session-1", "user_content": "Need retrieval logs", "assistant_content": "Next."}
    )
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    counts = index.rebuild_from_store(store)

    assert counts == {"project_cards": 1, "candidates": 1, "raw_events": 1, "source_refs": 1, "memory_items": 0, "open_loops": 0}
    assert index.count_memories() == 3
    source = store.list_source_refs()[0]
    indexed_source = index.source_ref(source.id)
    assert indexed_source is not None
    assert indexed_source["uri"] == f"raw_event:{source.id}"
    assert index.search("benchmark first", limit=5)[0]["id"] == "cand_eval_contract"
    assert index.search("retrieval logs", limit=5)[0]["type"] == "raw_event"


def test_rebuild_from_store_indexes_memory_items_with_structured_fields(tmp_path):
    store = _store(tmp_path)
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
    store.write_memory_item(item)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    counts = index.rebuild_from_store(store)
    results = index.search("response style direct no BS", route="preference_recall", limit=5)

    assert counts == {"project_cards": 0, "candidates": 0, "raw_events": 0, "source_refs": 0, "memory_items": 1, "open_loops": 0}
    assert len(results) == 1
    result = results[0]
    assert result["id"] == "pref_response_style"
    assert result["type"] == "preference"
    assert result["subject"] == "Alex"
    assert result["predicate"] == "prefers_response_style"
    assert result["value"] == "direct no-BS tool-grounded help"
    assert result["confidence"] == 0.98
    assert result["importance"] == 0.95
    assert result["source_refs"] == ["source_user_profile"]


def test_memory_item_supersession_fields_survive_index_search(tmp_path):
    store = _store(tmp_path)
    old = MemoryItem(
        id="pref_tts_voice_old",
        type="preference",
        subject="Alex",
        predicate="preferred_tts_voice",
        value="en-US-ChristopherNeural",
        summary="Old voice preference.",
        status="superseded",
        superseded_by="pref_tts_voice_current",
        source_refs=["source_old_voice"],
        tags=["voice"],
    )
    current = MemoryItem(
        id="pref_tts_voice_current",
        type="preference",
        subject="Alex",
        predicate="preferred_tts_voice",
        value="en-US-AndrewNeural",
        summary="Current voice preference.",
        supersedes=["pref_tts_voice_old"],
        source_refs=["source_new_voice"],
        tags=["voice"],
    )
    store.write_memory_item(old)
    store.write_memory_item(current)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    index.rebuild_from_store(store)

    results = index.search("preferred tts voice", route="preference_recall", limit=5)

    by_id = {result["id"]: result for result in results}
    assert list(by_id)[0] == "pref_tts_voice_current"
    assert by_id["pref_tts_voice_current"]["supersedes"] == ["pref_tts_voice_old"]
    assert by_id["pref_tts_voice_old"]["superseded_by"] == "pref_tts_voice_current"


def test_rebuild_from_store_indexes_source_refs_for_packet_expansion(tmp_path):
    store = _store(tmp_path)
    source = SourceRef(
        id="source_memory_v2_thread",
        type="session",
        uri="discord://thread/memory-v2-thread",
        title="Memory v2 design thread",
        observed_at="2026-05-26T00:00:00Z",
        quote="Alex asked for robust low-compute memory.",
    )
    store.write_source_ref(source)
    store.write_project_card(
        ProjectCard(
            id="Hermes Memory v2",
            name="Hermes Memory v2",
            goal="Build source-backed memory packets.",
            current_state="SourceRef store and index rebuild are next.",
            source_refs=["source_memory_v2_thread"],
        )
    )
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    counts = index.rebuild_from_store(store)

    assert counts == {"project_cards": 1, "candidates": 0, "raw_events": 0, "source_refs": 1, "memory_items": 0, "open_loops": 0}
    assert index.source_ref("source_memory_v2_thread") == source.to_dict()
    result = index.search("source backed memory packets", route="project_continuity", limit=5)[0]
    assert result["id"] == "project:hermes-memory-v2"
    assert result["source_refs"] == ["source_memory_v2_thread"]


def test_search_limit_and_empty_query(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    for i in range(3):
        index.index_candidate(
            CandidateMemory(
                id=f"cand_{i}",
                type="fact",
                claim=f"Memory v2 search limit shared term {i}",
            )
        )

    assert len(index.search("shared term", limit=2)) == 2
    assert index.search("   ") == []


def test_search_logs_retrieval_decision(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    index.index_candidate(CandidateMemory(id="cand_log", type="fact", claim="Retrieval log smoke test."))

    results = index.search("retrieval log", route="project_continuity", limit=5)

    logs = index.retrieval_logs()
    assert results
    assert len(logs) == 1
    assert logs[0]["query"] == "retrieval log"
    assert logs[0]["route"] == "project_continuity"
    assert logs[0]["retrieved_ids"] == ["cand_log"]


def test_index_search_rejects_invalid_limit(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    for bad_limit in (None, "bad"):
        try:
            index.search("memory", limit=bad_limit)
        except ValueError as exc:
            assert "limit" in str(exc)
        else:
            raise AssertionError("invalid limit should raise ValueError")


def test_search_excludes_expired_and_not_yet_valid_active_memories(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    index.index_memory_item(
        MemoryItem(
            id="pref_expired",
            type="preference",
            subject="Alex",
            value="Alex prefers expired retrieval behavior.",
            summary="Expired retrieval behavior.",
            expires_at="2000-01-01T00:00:00Z",
            source_refs=["event_expired"],
        )
    )
    index.index_memory_item(
        MemoryItem(
            id="pref_future",
            type="preference",
            subject="Alex",
            value="Alex prefers future retrieval behavior.",
            summary="Future retrieval behavior.",
            valid_from="2999-01-01T00:00:00Z",
            source_refs=["event_future"],
        )
    )
    index.index_memory_item(
        MemoryItem(
            id="pref_current",
            type="preference",
            subject="Alex",
            value="Alex prefers current retrieval behavior.",
            summary="Current retrieval behavior.",
            source_refs=["event_current"],
        )
    )

    results = index.search("retrieval behavior", route="preference_recall", limit=10)

    assert [result["id"] for result in results] == ["pref_current"]


def test_historical_routes_include_past_validity_but_never_future_validity(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    for item in (
        MemoryItem(
            id="historical_expired",
            type="fact",
            subject="history",
            value="Historical routing temporal evidence expired.",
            summary="Historical routing temporal evidence expired.",
            expires_at="2000-01-01T00:00:00Z",
            source_refs=["event_expired"],
        ),
        MemoryItem(
            id="historical_past_validity",
            type="fact",
            subject="history",
            value="Historical routing temporal evidence past validity.",
            summary="Historical routing temporal evidence past validity.",
            valid_until="2000-01-01T00:00:00Z",
            source_refs=["event_past_validity"],
        ),
        MemoryItem(
            id="historical_future",
            type="fact",
            subject="history",
            value="Historical routing temporal evidence future validity.",
            summary="Historical routing temporal evidence future validity.",
            valid_from="2999-01-01T00:00:00Z",
            source_refs=["event_future"],
        ),
    ):
        index.index_memory_item(item)

    expected = {"historical_expired", "historical_past_validity"}
    for route in ("deep_recall", "past_conversation_exact", "contradiction_check"):
        results = index.search("historical routing temporal evidence", route=route, limit=10)
        assert {result["id"] for result in results} == expected

    assert index.search("historical routing temporal evidence", route="fact_recall", limit=10) == []


def test_search_orders_candidate_gate_decisions_explicitly(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    for candidate in (
        CandidateMemory(id="cand_rejected", type="fact", claim="shared candidate ordering", gate_decision="rejected", decision_reason="no"),
        CandidateMemory(id="cand_archived", type="fact", claim="shared candidate ordering", gate_decision="archived_only", decision_reason="archive"),
        CandidateMemory(id="cand_promoted", type="fact", claim="shared candidate ordering", gate_decision="promoted", decision_reason="promoted"),
        CandidateMemory(id="cand_pending", type="fact", claim="shared candidate ordering"),
    ):
        index.index_candidate(candidate)

    results = index.search("shared candidate ordering", limit=10)

    assert [result["id"] for result in results] == ["cand_pending", "cand_promoted", "cand_archived", "cand_rejected"]


def test_index_raw_event_preserves_event_timestamps(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    index.index_raw_event(
        {
            "id": "event_old",
            "type": "turn",
            "session_id": "session-1",
            "user_content": "Memory v2 historical timestamp evidence.",
            "created_at": "2001-01-01T00:00:00Z",
            "updated_at": "2001-01-02T00:00:00Z",
        }
    )

    result = index.search("historical timestamp evidence", route="past_conversation_exact", limit=1)[0]

    assert result["created_at"] == "2001-01-01T00:00:00Z"
    assert result["updated_at"] == "2001-01-02T00:00:00Z"


def test_hybrid_search_uses_field_overlap_to_rerank_relaxed_fts_matches(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    index.index_raw_event(
        {
            "id": "event_noise",
            "type": "turn",
            "session_id": "session-1",
            "user_content": "Memory retrieval routing mentioned a generic benchmark note.",
        }
    )
    index.index_raw_event(
        {
            "id": "event_target",
            "type": "turn",
            "session_id": "session-1",
            "user_content": "LoCoMo evidence retrieval recall benchmark improved hybrid Memory v2 ranking.",
        }
    )

    results = index.search("LoCoMo evidence retrieval benchmark missing", route="research_recall", limit=2)

    assert [result["id"] for result in results] == ["event_target", "event_noise"]
    assert results[0]["hybrid_score"] > results[1]["hybrid_score"]
    assert results[0]["score_components"]["token_overlap"] > results[1]["score_components"]["token_overlap"]


def test_hybrid_search_boosts_project_cards_for_project_continuity_route(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    index.index_raw_event(
        {
            "id": "event_raw_project",
            "type": "turn",
            "session_id": "session-1",
            "user_content": "Project Memory v2 status raw note: old exploratory benchmark chatter.",
        }
    )
    index.index_project_card(
        ProjectCard(
            id="Memory v2",
            name="Memory v2",
            goal="Build robust memory.",
            current_state="Project Memory v2 status: hybrid retrieval implementation is current.",
            source_refs=["source_project_current"],
        )
    )

    results = index.search("Project Memory v2 status", route="project_continuity", limit=2)

    assert results[0]["id"] == "project:memory-v2"
    assert results[0]["type"] == "project_state"
    assert results[0]["score_components"]["route_type_boost"] > 0


def test_hybrid_search_boosts_preferences_for_preference_recall_route(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    index.index_raw_event(
        {
            "id": "event_pref_raw",
            "type": "turn",
            "session_id": "session-1",
            "user_content": "Alex prefers concise answers was mentioned in raw chat.",
        }
    )
    index.index_memory_item(
        MemoryItem(
            id="pref_concise_answers",
            type="preference",
            subject="Alex",
            predicate="prefers_response_style",
            value="concise answers",
            summary="Alex prefers concise answers.",
            source_refs=["event_pref_raw"],
        )
    )

    results = index.search("what does Alex prefer for answers", route="preference_recall", limit=2)

    assert results[0]["id"] == "pref_concise_answers"
    assert results[0]["type"] == "preference"
    assert results[0]["score_components"]["route_type_boost"] > 0


def test_hybrid_search_preserves_bm25_order_for_deep_and_exact_routes(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    index.index_record(
        id="event_exact_strong",
        type="raw_event",
        title="source quote alpha beta source quote alpha beta",
        body="source quote alpha beta source quote alpha beta source quote alpha beta",
        summary="source quote alpha beta source quote alpha beta",
        status="archived",
        source_refs=["event_exact_strong"],
        tags=["source", "quote", "alpha", "beta"],
    )
    index.index_record(
        id="pref_alpha_active",
        type="preference",
        title="Active alpha preference",
        value="source quote alpha",
        summary="source quote alpha beta",
        status="active",
        source_refs=["event_pref_alpha"],
    )

    exact_results = index.search("source quote alpha beta", route="past_conversation_exact", limit=2)
    deep_results = index.search("source quote alpha beta", route="deep_recall", limit=2)

    assert exact_results[0]["id"] == "event_exact_strong"
    assert deep_results[0]["id"] == "event_exact_strong"
    assert all("hybrid_score" in result for result in exact_results + deep_results)
    assert all("score_components" in result for result in exact_results + deep_results)


def test_search_fuses_strict_and_relaxed_pools_instead_of_stopping_early(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    index.index_record(
        id="fact_strict_decoy",
        type="fact",
        title="Atlas notification digest current",
        body="Atlas notification digest current is a generic test phrase.",
        summary="Generic strict-match decoy.",
        status="active",
        source_refs=["source_decoy"],
    )
    index.index_memory_item(
        MemoryItem(
            id="pref_relaxed_target",
            type="preference",
            subject="Atlas notifications",
            predicate="prefers_notification_digest",
            value="Atlas notification preference is afternoon digest.",
            summary="Atlas uses the afternoon digest.",
            status="active",
            updated_at="2026-06-01T00:00:00Z",
            source_refs=["source_target"],
        )
    )

    results = index.search(
        "Atlas notification digest current",
        route="preference_recall",
        limit=5,
    )

    assert {result["id"] for result in results} >= {
        "fact_strict_decoy",
        "pref_relaxed_target",
    }
    assert results[0]["id"] == "pref_relaxed_target"
    assert "relaxed" in results[0]["retrieval_pools"]
    assert results[0]["score_components"]["pool_fusion"] > 0


def test_structured_alias_pool_recalls_unseen_voice_paraphrase(tmp_path):
    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    index.index_memory_item(
        MemoryItem(
            id="pref_spoken_voice",
            type="preference",
            subject="user TTS voice",
            predicate="prefers_tts_voice",
            value="Use en-US-AndrewNeural for a deeper confident male voice.",
            summary="The current TTS voice is en-US-AndrewNeural.",
            source_refs=["source_voice"],
        )
    )

    results = index.search(
        "Which narrator should read spoken replies?",
        route="preference_recall",
        limit=5,
    )

    assert results[0]["id"] == "pref_spoken_voice"
    assert "structured_alias" in results[0]["retrieval_pools"]
    assert results[0]["score_components"]["structured_match"] > 0


def test_multi_pool_search_order_is_insertion_invariant(tmp_path):
    records = [
        MemoryItem(
            id="pref_voice_a",
            type="preference",
            subject="user TTS voice",
            predicate="prefers_tts_voice",
            value="Use voice alpha for spoken replies.",
            updated_at="2025-01-01T00:00:00Z",
            source_refs=["source_a"],
        ),
        MemoryItem(
            id="pref_voice_b",
            type="preference",
            subject="user TTS voice",
            predicate="prefers_tts_voice",
            value="Use voice beta for spoken replies.",
            updated_at="2026-01-01T00:00:00Z",
            source_refs=["source_b"],
        ),
        MemoryItem(
            id="pref_style_c",
            type="preference",
            subject="user response style",
            predicate="prefers_response_style",
            value="Use concise replies.",
            source_refs=["source_c"],
        ),
    ]
    orders = []
    for name, values in (("forward", records), ("reverse", list(reversed(records)))):
        store = _store(tmp_path / name)
        index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
        index.initialize()
        for record in values:
            index.index_memory_item(record)
        results = index.search(
            "Which narrator should read spoken replies?",
            route="preference_recall",
            limit=5,
        )
        orders.append(
            [
                (result["id"], result["hybrid_score"], result["retrieval_pools"])
                for result in results
            ]
        )

    assert orders[0] == orders[1]


def test_rebuild_from_store_refuses_unverifiable_raw_events_missing_ids(tmp_path):
    store = _store(tmp_path)
    store._append_jsonl(store.raw_events_path, {"type": "turn", "user_content": "legacy missing id"})
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    with pytest.raises(ValidationError, match="integrity verification failed"):
        index.rebuild_from_store(store)

    assert index.count_memories() == 0
    assert store.read_raw_archive_manifest()["derived_index_status"] == "integrity_failed"


def test_rebuild_from_store_indexes_open_loops_and_stopword_queries_do_not_match(tmp_path):
    store = _store(tmp_path)
    store.upsert_open_loop({"id": "loop_dragonfruit", "text": "finish dragonfruit migration", "source_refs": ["event_loop"]})
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    counts = index.rebuild_from_store(store)

    assert counts["open_loops"] == 1
    assert index.search("dragonfruit migration", limit=5)[0]["id"] == "loop_dragonfruit"
    assert index.search("the and or", limit=5) == []


def test_retrieval_log_redacts_query_and_hashes_redacted_sentinel(tmp_path):
    import hashlib

    store = _store(tmp_path)
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()

    query = "Authorization:\nBearer fakebearertoken123456789"
    index.log_retrieval(query, route="current_task", retrieved_ids=[])
    log = index.retrieval_logs()[0]

    assert log["query"] == "[REDACTED sensitive query]"
    assert log["query_hash"] != hashlib.sha256(query.encode("utf-8")).hexdigest()
    assert log["query_hash"] == hashlib.sha256(b"[REDACTED sensitive query]").hexdigest()
