"""Tests for actionable Memory v2 review plans."""

from __future__ import annotations

import json

from plugins.memory.memory_v2 import MemoryV2Provider
from plugins.memory.memory_v2.schemas import CandidateMemory, MemoryItem


def _provider(tmp_path):
    provider = MemoryV2Provider()
    provider.initialize("session-review-actions", hermes_home=str(tmp_path), platform="cli")
    return provider


def _seed_review_candidates(provider):
    event = provider.store.append_raw_event(
        {"type": "turn", "session_id": "session-review-actions", "user_content": "Alex prefers concise answers."}
    )
    provider.store.append_candidate(
        CandidateMemory(
            id="cand_safe",
            type="preference",
            claim="Alex prefers concise answers.",
            proposed_destination="core/user",
            confidence=0.92,
            source_refs=[event["id"]],
        )
    )
    provider.store.append_candidate(
        CandidateMemory(
            id="cand_ephemeral",
            type="fact",
            claim="Temporary scratch note for this answer only.",
            confidence=0.8,
            source_refs=[event["id"]],
        )
    )
    provider.store.append_candidate(
        CandidateMemory(
            id="cand_missing",
            type="fact",
            claim="Missing source fact.",
            confidence=0.9,
            source_refs=["missing_source"],
        )
    )
    provider.store.append_candidate(
        CandidateMemory(
            id="cand_skill",
            type="procedure_ref",
            claim="Procedure candidate should become a skill.",
            proposed_destination="skills/memory-review",
            confidence=0.9,
            source_refs=[event["id"]],
        )
    )
    return event


def test_review_plan_is_dry_run_stable_and_blocks_risky_candidates(tmp_path):
    provider = _provider(tmp_path)
    _seed_review_candidates(provider)
    before_candidates = [candidate.to_dict() for candidate in provider.store.list_candidates()]

    first = json.loads(provider.handle_tool_call("memory_v2_review_plan", {}))
    second = json.loads(provider.handle_tool_call("memory_v2_review_plan", {}))

    assert first["success"] is True
    assert first["mode"] == "dry_run_review_plan"
    assert first["untrusted_text"] is True
    assert first["plan_id"] == second["plan_id"]
    assert first["summary"] == {"proposed_promotions": 1, "proposed_rejections": 1, "blocked": 2}
    assert [action["candidate_id"] for action in first["actions"]] == ["cand_safe", "cand_ephemeral"]
    assert {blocked["candidate_id"] for blocked in first["blocked"]} == {"cand_missing", "cand_skill"}
    assert [candidate.to_dict() for candidate in provider.store.list_candidates()] == before_candidates
    assert provider.store.list_operation_records() == []


def test_review_apply_requires_confirm_and_explicit_action_ids(tmp_path):
    provider = _provider(tmp_path)
    _seed_review_candidates(provider)
    plan = json.loads(provider.handle_tool_call("memory_v2_review_plan", {}))

    missing_confirm = json.loads(
        provider.handle_tool_call(
            "memory_v2_review_apply",
            {"plan_id": plan["plan_id"], "action_ids": ["act_001"]},
        )
    )
    missing_actions = json.loads(
        provider.handle_tool_call(
            "memory_v2_review_apply",
            {"plan_id": plan["plan_id"], "confirm": "APPLY_MEMORY_V2_REVIEW_PLAN"},
        )
    )

    assert missing_confirm == {"success": False, "error": "confirm must equal APPLY_MEMORY_V2_REVIEW_PLAN"}
    assert missing_actions == {"success": False, "error": "action_ids must be a non-empty list"}
    assert provider.store.list_operation_records() == []


def test_review_apply_dry_run_validates_without_mutating(tmp_path):
    provider = _provider(tmp_path)
    _seed_review_candidates(provider)
    plan = json.loads(provider.handle_tool_call("memory_v2_review_plan", {}))
    before_candidates = [candidate.to_dict() for candidate in provider.store.list_candidates()]

    payload = json.loads(
        provider.handle_tool_call(
            "memory_v2_review_apply",
            {
                "plan_id": plan["plan_id"],
                "action_ids": [action["action_id"] for action in plan["actions"]],
                "confirm": "APPLY_MEMORY_V2_REVIEW_PLAN",
                "dry_run": True,
            },
        )
    )

    assert payload["success"] is True
    assert payload["dry_run"] is True
    assert len(payload["validated"] ) == 2
    assert [candidate.to_dict() for candidate in provider.store.list_candidates()] == before_candidates
    assert provider.store.list_memory_items() == []
    assert provider.store.list_rejected_candidates() == []
    assert provider.store.list_operation_records() == []


def test_review_apply_rejects_promotion_actions_but_allows_safe_rejection_actions_with_audit_records(tmp_path):
    provider = _provider(tmp_path)
    _seed_review_candidates(provider)
    plan = json.loads(provider.handle_tool_call("memory_v2_review_plan", {}))

    payload = json.loads(
        provider.handle_tool_call(
            "memory_v2_review_apply",
            {
                "plan_id": plan["plan_id"],
                "action_ids": [action["action_id"] for action in plan["actions"]],
                "confirm": "APPLY_MEMORY_V2_REVIEW_PLAN",
                "dry_run": False,
            },
        )
    )

    assert payload["success"] is False
    assert payload["summary"] == {"attempted": 2, "applied": 1, "skipped": 0, "failed": 1}
    decisions = {candidate.id: candidate.gate_decision.value for candidate in provider.store.list_candidates()}
    assert decisions["cand_safe"] == "pending"
    assert decisions["cand_ephemeral"] == "rejected"
    assert len(provider.store.list_memory_items()) == 0
    assert provider.store.list_rejected_candidates()[0].id == "cand_ephemeral"
    assert payload["failed"][0]["candidate_id"] == "cand_safe"
    assert "automatic promotion is disabled" in payload["failed"][0]["error"]
    operation_types = [record["type"] for record in provider.store.list_operation_records()]
    assert operation_types == ["reject_candidate"]


def test_review_apply_rejects_stale_plan_when_candidate_changed(tmp_path):
    provider = _provider(tmp_path)
    _seed_review_candidates(provider)
    plan = json.loads(provider.handle_tool_call("memory_v2_review_plan", {}))
    provider.store.append_candidate(
        CandidateMemory(
            id="cand_extra",
            type="fact",
            claim="A later candidate changes the plan hash.",
            confidence=0.9,
            source_refs=[provider.store.read_raw_events()[0]["id"]],
        )
    )

    payload = json.loads(
        provider.handle_tool_call(
            "memory_v2_review_apply",
            {
                "plan_id": plan["plan_id"],
                "action_ids": ["act_001"],
                "confirm": "APPLY_MEMORY_V2_REVIEW_PLAN",
            },
        )
    )

    assert payload == {"success": False, "error": "review plan is stale; regenerate memory_v2_review_plan"}
    assert provider.store.list_operation_records() == []


def test_review_plan_detects_contradiction_and_apply_blocks_it(tmp_path):
    provider = _provider(tmp_path)
    old = provider.store.append_raw_event({"type": "turn", "session_id": "old", "user_content": "Alex prefers concise answers."})
    new = provider.store.append_raw_event({"type": "turn", "session_id": "new", "user_content": "Alex prefers very long answers."})
    provider.store.write_memory_item(
        MemoryItem(
            id="mem_old",
            type="preference",
            subject="Alex",
            predicate="prefers_response_length",
            value="concise answers",
            source_refs=[old["id"]],
        )
    )
    provider.store.append_candidate(
        CandidateMemory(
            id="cand_conflict",
            type="preference",
            claim="Alex prefers very long answers.",
            confidence=0.9,
            source_refs=[new["id"]],
        )
    )

    plan = json.loads(provider.handle_tool_call("memory_v2_review_plan", {}))

    assert plan["actions"] == []
    assert plan["blocked"][0]["candidate_id"] == "cand_conflict"
    assert "possible_contradiction" in plan["blocked"][0]["blockers"]


def test_review_plan_blocks_ephemeral_candidates_with_hard_risk_flags(tmp_path):
    provider = _provider(tmp_path)
    provider.store.append_candidate(
        CandidateMemory(
            id="cand_ephemeral_missing",
            type="fact",
            claim="Temporary scratch note for this answer only with missing evidence.",
            confidence=0.9,
            source_refs=["missing_source"],
        )
    )

    plan = json.loads(provider.handle_tool_call("memory_v2_review_plan", {}))

    assert plan["actions"] == []
    assert plan["blocked"] == [
        {"candidate_id": "cand_ephemeral_missing", "blockers": ["missing_or_dangling_source"]}
    ]


def test_review_apply_accepts_same_candidate_filter_used_for_plan(tmp_path):
    provider = _provider(tmp_path)
    _seed_review_candidates(provider)
    plan = json.loads(provider.handle_tool_call("memory_v2_review_plan", {"candidate_ids": ["cand_safe"]}))

    payload = json.loads(
        provider.handle_tool_call(
            "memory_v2_review_apply",
            {
                "plan_id": plan["plan_id"],
                "candidate_ids": ["cand_safe"],
                "action_ids": ["act_001"],
                "confirm": "APPLY_MEMORY_V2_REVIEW_PLAN",
                "dry_run": False,
            },
        )
    )

    assert payload["success"] is False
    assert "automatic promotion is disabled" in payload["failed"][0]["error"]
    decisions = {candidate.id: candidate.gate_decision.value for candidate in provider.store.list_candidates()}
    assert decisions["cand_safe"] == "pending"
    assert decisions["cand_ephemeral"] == "pending"


def test_review_plan_source_check_does_not_call_unbounded_read_raw_events(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    event = _seed_review_candidates(provider)

    def forbidden_read_raw_events(*args, **kwargs):
        raise AssertionError("review plan source_check must not call read_raw_events")

    monkeypatch.setattr(provider.store, "read_raw_events", forbidden_read_raw_events)

    plan = json.loads(provider.handle_tool_call("memory_v2_review_plan", {"candidate_ids": ["cand_safe"]}))

    assert plan["success"] is True
    assert plan["actions"][0]["source_check"]["all_refs_exist"] is True
    assert plan["actions"][0]["source_check"]["sources"][0]["id"] == event["id"]
    assert plan["actions"][0]["source_check"]["sources"][0]["quote"]["present"] is True
    assert plan["actions"][0]["source_check"]["sources"][0]["quote"]["sha256"]
