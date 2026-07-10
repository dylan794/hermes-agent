"""Golden-shape tests for Memory v2 provider tool responses.

These tests intentionally validate stable JSON response contracts without taking
full value snapshots. They call tools only through MemoryV2Provider.handle_tool_call
so provider-tool integration stays covered when schemas or dispatch change.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import pytest

from hermes_state import SessionDB
from plugins.memory.memory_v2 import MemoryV2Provider
from plugins.memory.memory_v2.review_actions import CONFIRM_REVIEW_APPLY
from plugins.memory.memory_v2.schemas import CandidateMemory, MemoryItem


EXPECTED_TOOL_NAMES = {
    "memory_v2_status",
    "memory_v2_health",
    "memory_v2_repair",
    "memory_v2_search",
    "memory_v2_session_backfill",
    "memory_v2_archive_search",
    "memory_v2_archive_show",
    "memory_v2_archive_readiness",
    "memory_v2_extract_candidates",
    "memory_v2_consolidate",
    "memory_v2_daily_report",
    "memory_v2_dream_cycle",
    "memory_v2_candidates",
    "memory_v2_review_queue",
    "memory_v2_review_plan",
    "memory_v2_review_apply",
    "memory_v2_promote",
    "memory_v2_reject",
    "memory_v2_show_source",
    "memory_v2_resolve_open_loop",
    "memory_v2_contradictions",
}

TOOL_SUCCESS_SHAPES: dict[str, dict[str, Any]] = {
    "memory_v2_status": {
        "success": bool,
        "provider": str,
        "initialized": bool,
        "session_id": str,
        "platform": str,
        "base_dir": str,
        "counts": dict,
    },
    "memory_v2_health": {
        "success": bool,
        "status": str,
        "issue_count": int,
        "severity_counts": dict,
        "counts": dict,
        "issues": list,
        "repair_plan": list,
    },
    "memory_v2_repair": {"success": bool, "dry_run": bool, "actions": list},
    "memory_v2_search": {"success": bool, "count": int, "results": list},
    "memory_v2_session_backfill": {"success": bool, "mode": str, "dry_run": bool, "considered": int, "imported": int, "skipped": int},
    "memory_v2_archive_search": {"success": bool, "mode": str, "untrusted_text": bool, "policy": str, "filters": dict, "archive": dict, "count": int, "limit": int, "has_more": bool, "results": list},
    "memory_v2_archive_show": {"success": bool, "mode": str, "untrusted_text": bool, "policy": str, "event": dict},
    "memory_v2_archive_readiness": {"success": bool, "ready": bool, "rollout_step": int, "mode": str, "mutations_allowed": bool, "policy": str, "limits": dict, "checks": dict, "blockers": list, "proof": dict, "mutations": dict},
    "memory_v2_extract_candidates": {"success": bool, "ready": bool, "rollout_step": int, "mode": str, "mutations_allowed": str, "policy": str, "limits": dict, "filters": dict, "extraction": dict, "checks": dict, "counts": dict, "mutations": dict, "blockers": list},
    "memory_v2_consolidate": {
        "success": bool,
        "considered": int,
        "promoted": int,
        "rejected": int,
        "archived_only": int,
        "superseded": int,
        "promoted_ids": list,
        "rejected_ids": list,
        "archived_ids": list,
        "superseded_ids": list,
    },
    "memory_v2_daily_report": {
        "success": bool,
        "kind": str,
        "date": str,
        "created_at": str,
        "before_counts": dict,
        "after_counts": dict,
        "consolidation": dict,
        "extraction": dict,
        "open_loops": list,
        "recent_raw_event_ids": list,
        "report_path": str,
        "daily_episode_path": str,
    },
    "memory_v2_dream_cycle": {
        "success": bool,
        "kind": str,
        "date": str,
        "mode": str,
        "auto_apply": str,
        "created_at": str,
        "policy": str,
        "health": dict,
        "review_queue_summary": dict,
        "review_plan": dict,
        "review_apply": (dict, type(None)),
        "review_apply_plan": (dict, type(None)),
        "open_loops": list,
        "before_counts": dict,
        "after_counts": dict,
        "report_path": str,
        "dream_episode_path": str,
    },
    "memory_v2_candidates": {"success": bool, "count": int, "candidates": list},
    "memory_v2_review_queue": {
        "success": bool,
        "mode": str,
        "note": (str, dict),
        "untrusted_text": bool,
        "review_summary": dict,
        "groups": dict,
        "items": list,
        "stale_open_loops": list,
        "recommended_actions": dict,
        "recommendation_policy": str,
    },
    "memory_v2_review_plan": {
        "success": bool,
        "mode": str,
        "plan_id": str,
        "untrusted_text": bool,
        "summary": dict,
        "actions": list,
        "blocked": list,
    },
    "memory_v2_review_apply": {
        "success": bool,
        "dry_run": bool,
        "plan_id": str,
        "validated": list,
        "applied": list,
        "failed": list,
    },
    "memory_v2_promote": {
        "success": bool,
        "operation_id": str,
        "operation_type": str,
        "ids": list,
        "promoted": int,
        "promoted_ids": list,
        "superseded_ids": list,
        "candidate": dict,
    },
    "memory_v2_reject": {
        "success": bool,
        "operation_id": str,
        "operation_type": str,
        "ids": list,
        "candidate": dict,
    },
    "memory_v2_show_source": {"success": bool, "record": dict, "sources": list},
    "memory_v2_resolve_open_loop": {
        "success": bool,
        "operation_id": str,
        "operation_type": str,
        "ids": list,
        "loop": dict,
    },
    "memory_v2_contradictions": {
        "success": bool,
        "mode": str,
        "note": (str, dict),
        "mutated_memories": int,
        "create_candidates": bool,
        "auto_supersede": bool,
        "min_confidence": float,
        "count": int,
        "created_candidate_ids": list,
        "auto_superseded": list,
        "conflicts": list,
    },
}

IMPORTANT_ERROR_CASES: list[tuple[str, dict[str, Any], str]] = [
    ("memory_v2_search", {"query": "alpha", "limit": "bad"}, "limit must be an integer"),
    ("memory_v2_session_backfill", {"limit": "many"}, "limit, batch_size, max_batches, and message id cursors must be integers"),
    ("memory_v2_archive_search", {}, "archive search requires at least one"),
    ("memory_v2_archive_search", {"query": "alpha", "limit": "many"}, "limit and excerpt_chars must be integers"),
    ("memory_v2_archive_show", {}, "id is required"),
    ("memory_v2_archive_readiness", {"sample_size": "many"}, "sample_size must be an integer"),
    ("memory_v2_daily_report", {"date": "06/20/2026"}, "date must be YYYY-MM-DD"),
    ("memory_v2_dream_cycle", {"auto_apply": "on"}, "auto_apply must be exactly off"),
    ("memory_v2_dream_cycle", {"recent_raw_limit": True}, "recent_raw_limit must be an integer"),
    ("memory_v2_candidates", {"limit": "many"}, "limit must be an integer"),
    ("memory_v2_review_queue", {"now": "not-an-iso-timestamp"}, "now must be an ISO timestamp"),
    ("memory_v2_review_queue", {"limit": "many"}, "limit and stale_open_loop_days must be integers"),
    ("memory_v2_review_plan", {"candidate_ids": "cand_safe"}, "candidate_ids must be a list"),
    ("memory_v2_review_plan", {"max_actions": "many"}, "max_actions must be an integer"),
    ("memory_v2_review_apply", {"candidate_ids": "cand_safe"}, "candidate_ids must be a list"),
    ("memory_v2_review_apply", {"max_actions": "many"}, "max_actions must be an integer"),
    ("memory_v2_review_apply", {"plan_id": "plan_missing", "action_ids": ["act_001"]}, "confirm must equal"),
    ("memory_v2_promote", {}, "candidate_id is required"),
    ("memory_v2_reject", {"candidate_id": "cand_missing"}, "reason is required"),
    ("memory_v2_show_source", {}, "id is required"),
    ("memory_v2_resolve_open_loop", {"loop_id": "loop_schema", "status": "done"}, "status must be one of"),
    ("memory_v2_contradictions", {"limit": "many"}, "limit must be an integer"),
    ("memory_v2_contradictions", {"min_confidence": "high"}, "min_confidence must be a number"),
]

FORBIDDEN_OUTPUT_PATTERNS = [
    re.compile(r"/home/[^\s\"']+"),
    re.compile(r"/mnt/[a-z]/Users/[^\s\"']+", re.IGNORECASE),
    re.compile(r"[A-Z]:\\\\Users\\\\[^\s\"']+"),
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"ghp_[A-Za-z0-9_]{8,}"),
]


def _provider(tmp_path) -> MemoryV2Provider:
    (tmp_path / "config.yaml").write_text(
        """
memory_v2:
  archive:
    backfill_enabled: true
    include_tool_outputs: true
    search_tools_enabled: true
    show_tools_enabled: true
  extraction:
    enabled: true
    candidate_creation_enabled: true
  consolidation:
    enabled: true
  review_apply:
    enabled: true
""".lstrip(),
        encoding="utf-8",
    )
    provider = MemoryV2Provider()
    provider.initialize("session-tool-schema", hermes_home=str(tmp_path), platform="cli")
    return provider


def _tool_json(provider: MemoryV2Provider, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    raw = provider.handle_tool_call(tool_name, args)
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    # Tool results must round-trip as stable JSON objects.
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload
    return payload


def _assert_shape(payload: Mapping[str, Any], shape: Mapping[str, Any]) -> None:
    assert set(shape).issubset(payload), f"missing keys: {set(shape) - set(payload)}"
    for key, expected_type in shape.items():
        assert isinstance(payload[key], expected_type), f"{key}={payload[key]!r} is not {expected_type}"


def _assert_no_private_leaks(payload: Mapping[str, Any], tmp_path) -> None:
    serialized = json.dumps(payload, sort_keys=True)
    assert str(tmp_path) not in serialized
    for pattern in FORBIDDEN_OUTPUT_PATTERNS:
        assert not pattern.search(serialized), pattern.pattern


def _seed_tool_fixture(provider: MemoryV2Provider) -> dict[str, Any]:
    state_db = SessionDB(provider.base_dir.parent / "state.db")
    state_db.create_session("session-tool-schema", "cli")
    state_db.append_message("session-tool-schema", "user", "Alex prefers concise source-grounded answers.")
    state_db.close()
    event = provider.store.append_raw_event(
        {
            "type": "turn",
            "session_id": "session-tool-schema",
            "user_content": "Alex prefers concise source-grounded answers.",
            "assistant_content": "Noted as candidate memory.",
            "created_at": "2026-06-20T00:00:00Z",
        }
    )
    event_id = str(event["id"])
    for candidate in [
        CandidateMemory(
            id="cand_promote_schema",
            type="preference",
            claim="Alex prefers concise source-grounded answers.",
            proposed_destination="core/user",
            confidence=0.93,
            source_refs=[event_id],
        ),
        CandidateMemory(
            id="cand_reject_schema",
            type="fact",
            claim="Temporary scratch note for this response only.",
            confidence=0.72,
            source_refs=[event_id],
        ),
        CandidateMemory(
            id="cand_plan_schema",
            type="preference",
            claim="Alex prefers provider tools to return stable JSON shapes.",
            proposed_destination="core/user",
            confidence=0.91,
            source_refs=[event_id],
        ),
    ]:
        provider.store.append_candidate(candidate)
        provider.index.index_candidate(candidate)
    provider.store.write_open_loops(
        [
            {
                "id": "loop_schema",
                "text": "Follow up on Memory v2 provider-tool golden schema coverage.",
                "status": "open",
                "created_at": "2026-05-01T00:00:00Z",
                "updated_at": "2026-05-01T00:00:00Z",
                "source_refs": [event_id],
            }
        ]
    )
    provider.index.index_open_loop(
        provider.store.list_open_loops()[0], file_path=provider.store.open_loops_path
    )
    provider.store.write_memory_item(
        MemoryItem(
            id="mem_schema_old",
            type="preference",
            subject="Alex",
            predicate="answer_style",
            value="verbose answers",
            confidence=0.85,
            importance=0.8,
            source_refs=[event_id],
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
    )
    provider.store.write_memory_item(
        MemoryItem(
            id="mem_schema_new",
            type="preference",
            subject="Alex",
            predicate="answer_style",
            value="concise answers now instead",
            confidence=0.9,
            importance=0.8,
            source_refs=[event_id],
            created_at="2026-06-20T00:00:00Z",
            updated_at="2026-06-20T00:00:00Z",
        )
    )
    for item in provider.store.list_memory_items():
        provider.index.index_memory_item(item)
    plan = _tool_json(provider, "memory_v2_review_plan", {})
    return {
        "event_id": event_id,
        "review_plan_id": plan["plan_id"],
        "review_action_ids": [action["action_id"] for action in plan["actions"]],
    }


def _success_args(tool_name: str, seeded: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_v2_status": {},
        "memory_v2_health": {},
        "memory_v2_repair": {"dry_run": True},
        "memory_v2_search": {"query": "concise", "limit": 5},
        "memory_v2_session_backfill": {"dry_run": True, "limit": 5},
        "memory_v2_archive_search": {"query": "concise", "limit": 5},
        "memory_v2_archive_show": {"id": seeded["event_id"]},
        "memory_v2_archive_readiness": {},
        "memory_v2_extract_candidates": {"session_id": "session-tool-schema", "recent_raw_limit": 5},
        "memory_v2_consolidate": {},
        "memory_v2_daily_report": {"date": "2026-06-20"},
        "memory_v2_dream_cycle": {"date": "2026-06-20", "mode": "nightly", "auto_apply": "off"},
        "memory_v2_candidates": {"limit": 10},
        "memory_v2_review_queue": {"now": "2026-06-20T00:00:00Z", "limit": 10},
        "memory_v2_review_plan": {"candidate_ids": ["cand_plan_schema"], "max_actions": 5},
        "memory_v2_review_apply": {
            "plan_id": seeded["review_plan_id"],
            "action_ids": seeded["review_action_ids"],
            "confirm": CONFIRM_REVIEW_APPLY,
            "dry_run": True,
        },
        "memory_v2_promote": {"candidate_id": "cand_promote_schema"},
        "memory_v2_reject": {"candidate_id": "cand_reject_schema", "reason": "temporary example only"},
        "memory_v2_show_source": {"id": seeded["event_id"]},
        "memory_v2_resolve_open_loop": {
            "loop_id": "loop_schema",
            "status": "resolved",
            "resolution": "covered by schema test",
        },
        "memory_v2_contradictions": {"limit": 5},
    }[tool_name]


def _assert_safe_error(error: Any, expected_fragment: str) -> None:
    if isinstance(error, str):
        assert expected_fragment in error
        return
    assert isinstance(error, dict)
    assert error.get("kind") == "text"
    assert error.get("present") is True
    assert isinstance(error.get("chars"), int)
    assert isinstance(error.get("sha256"), str)


def test_registered_memory_v2_tools_match_golden_contract_table(tmp_path) -> None:
    provider = _provider(tmp_path)
    registered = [schema["name"] for schema in provider.get_tool_schemas()]

    assert len(registered) == len(set(registered))
    assert set(registered) == EXPECTED_TOOL_NAMES
    assert set(TOOL_SUCCESS_SHAPES) == set(registered)


@pytest.mark.parametrize("tool_name", sorted(EXPECTED_TOOL_NAMES))
def test_every_memory_v2_provider_tool_success_response_has_golden_shape(tool_name, tmp_path) -> None:
    provider = _provider(tmp_path)
    seeded = _seed_tool_fixture(provider)

    payload = _tool_json(provider, tool_name, _success_args(tool_name, seeded))

    _assert_shape(payload, TOOL_SUCCESS_SHAPES[tool_name])
    assert payload["success"] is True
    _assert_no_private_leaks(payload, tmp_path)


@pytest.mark.parametrize(("tool_name", "args", "error_fragment"), IMPORTANT_ERROR_CASES)
def test_memory_v2_provider_tool_error_responses_have_stable_shape(tool_name, args, error_fragment, tmp_path) -> None:
    provider = _provider(tmp_path)
    _seed_tool_fixture(provider)

    payload = _tool_json(provider, tool_name, args)

    assert set(payload) >= {"success", "error"}
    assert payload["success"] is False
    _assert_safe_error(payload["error"], error_fragment)
    _assert_no_private_leaks(payload, tmp_path)


def test_unknown_memory_v2_provider_tool_has_stable_error_shape(tmp_path) -> None:
    provider = _provider(tmp_path)

    payload = _tool_json(provider, "memory_v2_nope", {})

    assert payload == {
        "success": False,
        "error": "Unknown Memory v2 tool: memory_v2_nope",
    }
