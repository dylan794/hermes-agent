"""Tests for offline source-grounded Memory v2 extraction."""

from __future__ import annotations

import json

from plugins.memory.memory_v2 import MemoryV2Provider
from plugins.memory.memory_v2.daily_consolidation import run_daily_consolidation_report
from plugins.memory.memory_v2.extraction import OfflineSessionExtractor
from plugins.memory.memory_v2.schemas import CandidateMemory


def _provider(tmp_path, *, agent_context: str = "primary"):
    provider = MemoryV2Provider()
    provider.initialize("session-extract", hermes_home=str(tmp_path), platform="discord", agent_context=agent_context)
    return provider


def _decision(candidate: CandidateMemory) -> str:
    return getattr(candidate.gate_decision, "value", str(candidate.gate_decision))


def _type(candidate: CandidateMemory) -> str:
    return getattr(candidate.type, "value", str(candidate.type))


def test_explicit_offline_extraction_creates_normal_session_candidates_without_promoting(tmp_path):
    provider = _provider(tmp_path)
    provider.sync_turn(
        "I prefer short Discord replies unless it is architecture work.",
        "Got it.",
        session_id="session-extract",
    )
    provider.sync_turn(
        "Memory v2 decision: keep offline extraction gated as pending candidates.",
        "I'll keep it gated.",
        session_id="session-extract",
    )
    provider.sync_turn(
        "Memory v2 next action: add source-grounded extraction tests.",
        "Noted as next action.",
        session_id="session-extract",
    )

    OfflineSessionExtractor().extract(provider.store, provider.index, session_id="session-extract")

    candidates = provider.store.list_candidates()
    assert len(candidates) == 3
    assert { _type(candidate) for candidate in candidates } == {"preference", "project_state"}
    assert all(_decision(candidate) == "pending" for candidate in candidates)
    assert provider.store.list_memory_items() == []
    assert provider.store.list_project_cards() == []
    assert provider.store.list_open_loops() == []
    assert any("prefers short Discord replies" in candidate.claim for candidate in candidates)
    assert any("pending candidates" in candidate.claim for candidate in candidates)
    assert any("add source-grounded extraction tests" in candidate.claim for candidate in candidates)
    assert all("offline_extraction:v2" in candidate.promotion_reason for candidate in candidates)
    assert all(candidate.evidence_spans for candidate in candidates)


def test_session_end_archives_without_creating_candidates_by_default(tmp_path):
    provider = _provider(tmp_path)
    provider.sync_turn(
        "I prefer session end not to create inferred candidates by default.",
        "Understood.",
        session_id="session-extract",
    )

    provider.on_session_end([
        {"role": "user", "content": "done"},
        {"role": "assistant", "content": "archiving"},
    ])

    assert provider.store.list_candidates() == []
    assert len(provider.store.list_session_archives()) == 1


def test_extracted_candidates_are_source_grounded_and_show_source_resolves(tmp_path):
    provider = _provider(tmp_path)
    provider.sync_turn(
        "Hermes environment fact: tests run inside WSL for this repo.",
        "Captured as environment context.",
        session_id="session-extract",
    )

    report = OfflineSessionExtractor().extract(provider.store, provider.index, session_id="session-extract")
    candidate = provider.store.list_candidates()[0]
    source_id = candidate.source_refs[0]
    shown = json.loads(provider.handle_tool_call("memory_v2_show_source", {"id": candidate.id}))

    assert report.created == 1
    assert _type(candidate) == "environment"
    assert source_id.startswith("event_")
    assert provider.store.read_source_ref(source_id) is not None
    assert shown["success"] is True
    assert shown["sources"][0]["id"] == source_id
    assert "tests run inside WSL" in shown["sources"][0]["quote"]


def test_extraction_skips_secrets_ephemeral_chatter_and_assistant_speculation(tmp_path):
    provider = _provider(tmp_path)
    provider.sync_turn("My API key is sk-testsecret123456789", "I will not store secrets.", session_id="session-extract")
    provider.sync_turn("Today I am debugging the eval harness for now.", "Okay.", session_id="session-extract")
    provider.sync_turn("What do you think I prefer?", "You probably prefer long essays.", session_id="session-extract")

    report = OfflineSessionExtractor().extract(provider.store, provider.index, session_id="session-extract")

    assert report.created == 0
    assert provider.store.list_candidates() == []


def test_extraction_ignores_tool_and_content_only_raw_events(tmp_path):
    provider = _provider(tmp_path)
    provider.store.append_raw_event(
        {"type": "tool", "session_id": "session-extract", "content": "I prefer tool outputs to become user memories."}
    )
    provider.store.append_raw_event(
        {"type": "message", "session_id": "session-extract", "content": "I prefer imported content to become user memories."}
    )

    report = OfflineSessionExtractor().extract(provider.store, provider.index, session_id="session-extract")

    assert report.created == 0
    assert provider.store.list_candidates() == []


def test_extraction_skips_current_reply_scoped_preferences(tmp_path):
    provider = _provider(tmp_path)
    provider.sync_turn("I prefer bullet points for this answer.", "Okay.", session_id="session-extract")
    provider.sync_turn("I prefer concise summaries in this thread.", "Okay.", session_id="session-extract")
    provider.sync_turn("I prefer source-grounded memory reviews.", "Okay.", session_id="session-extract")

    report = OfflineSessionExtractor().extract(provider.store, provider.index, session_id="session-extract")

    assert report.created == 1
    candidates = provider.store.list_candidates()
    assert len(candidates) == 1
    assert candidates[0].claim == "User prefers source-grounded memory reviews."


def test_extraction_dedupes_existing_pending_candidate_and_merges_source_refs(tmp_path):
    provider = _provider(tmp_path)
    provider.sync_turn("I prefer source-grounded memory reviews.", "Noted.", session_id="session-extract")
    first_report = OfflineSessionExtractor().extract(provider.store, provider.index, session_id="session-extract")
    provider.sync_turn("I prefer source-grounded memory reviews.", "Still noted.", session_id="session-extract")

    second_report = OfflineSessionExtractor().extract(provider.store, provider.index, session_id="session-extract")

    candidates = provider.store.list_candidates()
    assert first_report.created == 1
    assert second_report.created == 0
    assert second_report.merged == 1
    assert len(candidates) == 1
    assert len(candidates[0].source_refs) == 2


def test_daily_extraction_leaves_new_candidates_pending_until_later_review(tmp_path):
    provider = _provider(tmp_path)
    event = provider.store.append_raw_event(
        {"type": "turn", "session_id": "session-extract", "user_content": "evidence for old explicit candidate"}
    )
    provider.index.index_raw_event(event)
    source = provider.store.read_source_ref(event["id"])
    assert source is not None
    provider.index.index_source_ref(source)
    provider.store.append_candidate(
        CandidateMemory(
            id="cand_old_explicit",
            type="fact",
            claim="Old explicit candidate should consolidate.",
            source_refs=[event["id"]],
        )
    )
    provider.index.index_candidate(provider.store.list_candidates()[0])
    provider.sync_turn(
        "I prefer daily extraction to leave newly inferred candidates pending.",
        "Understood.",
        session_id="session-extract",
    )

    report = run_daily_consolidation_report(
        provider.store,
        provider.index,
            date="2026-06-24",
            allow_consolidation=True,
            authorize_mutation=True,
            allow_extraction=True,
        run_extraction=True,
    )

    candidates = provider.store.list_candidates()
    extracted = [candidate for candidate in candidates if "daily extraction" in candidate.claim]
    assert report["consolidation"]["promoted"] == 1
    assert report["extraction"]["created"] == 1
    assert provider.store.read_memory_item(report["consolidation"]["promoted_ids"][0]) is not None
    assert len(extracted) == 1
    assert _decision(extracted[0]) == "pending"
    assert extracted[0].id not in report["consolidation"]["promoted_ids"]


def test_extraction_noops_for_non_primary_provider_context(tmp_path):
    provider = _provider(tmp_path, agent_context="subagent")
    provider.sync_turn("I prefer subagents not to write normal Memory v2 candidates.", "Okay.", session_id="session-extract")
    provider.on_session_end([{"role": "user", "content": "finish"}])

    assert provider.store.list_candidates() == []


def test_extraction_uses_bounded_raw_index_and_source_validation_without_full_scan(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    provider.sync_turn("I prefer extraction without hidden archive scans.", "Okay.", session_id="session-extract")

    def forbidden_read_raw_events(*args, **kwargs):
        raise AssertionError("offline extraction must not call read_raw_events")

    monkeypatch.setattr(provider.store, "read_raw_events", forbidden_read_raw_events)

    report = OfflineSessionExtractor().extract(provider.store, provider.index, session_id="session-extract")

    assert report.created == 1
    assert provider.store.list_candidates()[0].source_refs
