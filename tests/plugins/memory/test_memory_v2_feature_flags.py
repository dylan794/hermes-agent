"""Feature-flag enforcement tests for Memory v2 safe rollout defaults."""

from __future__ import annotations

import json

import yaml

from hermes_state import SessionDB
from plugins.memory.memory_v2 import MemoryV2Provider
from plugins.memory.memory_v2.daily_consolidation import run_daily_consolidation_report
from plugins.memory.memory_v2.dream import SAFE_REJECTION_CANARY_CONFIRM, run_memory_dream_cycle
from plugins.memory.memory_v2.review_actions import CONFIRM_REVIEW_APPLY
from plugins.memory.memory_v2.schemas import CandidateMemory
from plugins.memory.memory_v2.session_backfill import SESSION_BACKFILL_CONFIRM


def _write_config(tmp_path, memory_v2: dict) -> None:
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"memory_v2": memory_v2}, sort_keys=False),
        encoding="utf-8",
    )


def _provider(tmp_path, memory_v2: dict | None = None) -> MemoryV2Provider:
    if memory_v2 is not None:
        _write_config(tmp_path, memory_v2)
    provider = MemoryV2Provider()
    provider.initialize("session-flags", hermes_home=str(tmp_path), platform="cli")
    return provider


def _tool_json(provider: MemoryV2Provider, name: str, args: dict) -> dict:
    payload = json.loads(provider.handle_tool_call(name, args))
    assert isinstance(payload, dict)
    return payload


def _tool_names(provider: MemoryV2Provider) -> set[str]:
    return {schema["name"] for schema in provider.get_tool_schemas()}


def test_safe_defaults_hide_mutating_and_autonomous_tools(tmp_path) -> None:
    provider = _provider(tmp_path)

    names = _tool_names(provider)

    assert "memory_v2_archive_search" in names
    assert "memory_v2_archive_show" in names
    assert "memory_v2_session_backfill" not in names
    assert "memory_v2_consolidate" not in names
    assert "memory_v2_daily_report" not in names
    assert "memory_v2_dream_cycle" not in names
    assert "memory_v2_review_apply" not in names
    assert "memory_v2_promote" not in names
    assert "memory_v2_reject" not in names
    assert "memory_v2_resolve_open_loop" not in names


def test_disabled_tools_are_rejected_even_if_called_directly(tmp_path) -> None:
    provider = _provider(tmp_path)

    payload = _tool_json(provider, "memory_v2_consolidate", {})

    assert payload == {
        "success": False,
        "error": "Memory v2 tool disabled by feature flag: memory_v2_consolidate (memory_v2.consolidation.enabled)",
    }

    promote_payload = _tool_json(provider, "memory_v2_promote", {"candidate_id": "cand_missing"})
    assert promote_payload == {
        "success": False,
        "error": "Memory v2 tool disabled by feature flag: memory_v2_promote (memory_v2.review_apply.enabled)",
    }


def test_archive_search_and_show_tool_exposure_follow_archive_flags(tmp_path) -> None:
    provider = _provider(
        tmp_path,
        {
            "archive": {
                "search_tools_enabled": False,
                "show_tools_enabled": False,
            }
        },
    )

    names = _tool_names(provider)

    assert "memory_v2_archive_search" not in names
    assert "memory_v2_archive_show" not in names
    assert _tool_json(provider, "memory_v2_archive_search", {"query": "needle"})["success"] is False
    assert "memory_v2.archive.search_tools_enabled" in _tool_json(provider, "memory_v2_archive_search", {"query": "needle"})["error"]


def test_prefetch_is_empty_until_enabled(tmp_path) -> None:
    provider = _provider(tmp_path)
    event = provider.store.append_raw_event(
        {
            "type": "turn",
            "session_id": "session-flags",
            "user_content": "prefetch gated recall needle",
            "assistant_content": "ack",
        }
    )
    provider.index.index_raw_event(event)

    assert provider.prefetch("prefetch gated recall needle", session_id="session-flags") == ""

    enabled = _provider(tmp_path, {"prefetch": {"enabled": True}})
    assert "prefetch gated recall needle" in enabled.prefetch(
        "prefetch gated recall needle", session_id="session-flags"
    )


def test_sync_turn_does_not_capture_or_extract_by_default(tmp_path) -> None:
    provider = _provider(tmp_path)

    provider.sync_turn("remember that my flag test favorite color is blue", "noted", session_id="session-flags")

    assert provider.store.count_raw_events() == 0
    assert provider.store.count_candidates() == 0
    assert provider.store.read_current_working_memory() is None


def test_session_end_does_not_archive_by_default(tmp_path) -> None:
    provider = _provider(tmp_path)

    provider.on_session_end([
        {"role": "user", "content": "session-end archive should be gated"},
        {"role": "assistant", "content": "ack"},
    ])

    assert list((tmp_path / "memory_v2" / "episodic" / "sessions").glob("*.yaml")) == []


def test_session_end_archive_requires_capture_flag(tmp_path) -> None:
    provider = _provider(tmp_path, {"archive": {"capture_enabled": True}})

    provider.on_session_end([
        {"role": "user", "content": "session-end archive enabled by config"},
        {"role": "assistant", "content": "ack"},
    ])

    assert list((tmp_path / "memory_v2" / "episodic" / "sessions").glob("*.yaml"))


def test_session_end_archive_requires_archive_enabled_too(tmp_path) -> None:
    provider = _provider(tmp_path, {"archive": {"enabled": False, "capture_enabled": True}})

    provider.on_session_end([
        {"role": "user", "content": "archive subsystem is disabled"},
        {"role": "assistant", "content": "ack"},
    ])

    assert list((tmp_path / "memory_v2" / "episodic" / "sessions").glob("*.yaml")) == []


def test_sync_turn_flags_independently_gate_capture_extraction_and_working_memory(tmp_path) -> None:
    provider = _provider(
        tmp_path,
        {
            "archive": {"capture_enabled": True},
            "extraction": {"enabled": False},
            "working_memory": {"enabled": True},
        },
    )

    provider.sync_turn("remember that my feature flag snack is pears", "noted", session_id="session-flags")

    assert provider.store.count_raw_events() == 1
    assert provider.store.count_candidates() == 0
    assert provider.store.read_current_working_memory() is not None


def test_session_backfill_non_dry_run_requires_flag_and_confirm(tmp_path) -> None:
    provider = _provider(tmp_path)
    state_db = SessionDB(tmp_path / "state.db")
    state_db.create_session("session-flags", "cli")
    state_db.append_message("session-flags", "user", "backfill gated import needle")
    state_db.close()

    disabled_payload = _tool_json(
        provider,
        "memory_v2_session_backfill",
        {"dry_run": False, "confirm": SESSION_BACKFILL_CONFIRM},
    )

    assert disabled_payload["success"] is False
    assert "memory_v2.archive.backfill_enabled" in disabled_payload["error"]

    enabled = _provider(tmp_path, {"archive": {"backfill_enabled": True}})
    wrong_confirm_payload = _tool_json(
        enabled,
        "memory_v2_session_backfill",
        {"dry_run": False, "confirm": "wrong"},
    )

    assert wrong_confirm_payload["success"] is False
    assert "confirm must equal" in wrong_confirm_payload["error"]


def test_backfill_include_tools_defaults_to_config_false(tmp_path) -> None:
    provider = _provider(tmp_path, {"archive": {"backfill_enabled": True, "include_tool_outputs": False}})
    state_db = SessionDB(tmp_path / "state.db")
    state_db.create_session("session-flags", "cli")
    state_db.append_message("session-flags", "tool", "tool output should stay excluded by default")
    state_db.close()

    payload = _tool_json(provider, "memory_v2_session_backfill", {"dry_run": True, "limit": 5})

    assert payload["success"] is True
    assert payload["considered"] == 0
    assert payload["imported"] == 0

    explicit_payload = _tool_json(
        provider,
        "memory_v2_session_backfill",
        {"dry_run": True, "include_tools": True, "limit": 5},
    )
    assert explicit_payload["success"] is True
    assert explicit_payload["considered"] == 0
    assert explicit_payload["imported"] == 0


def test_daily_report_does_not_create_extraction_candidates_when_extraction_disabled(tmp_path) -> None:
    provider = _provider(
        tmp_path,
        {
            "consolidation": {"enabled": True},
            "extraction": {"enabled": False, "candidate_creation_enabled": False},
        },
    )
    provider.store.append_raw_event(
        {
            "type": "turn",
            "session_id": "session-flags",
            "user_content": "I prefer daily-report extraction to stay disabled in this flag test.",
            "assistant_content": "noted",
        }
    )

    payload = _tool_json(provider, "memory_v2_daily_report", {"date": "2026-06-18"})

    assert payload["success"] is True
    assert payload["kind"] == "daily_memory_consolidation_report"
    assert payload["extraction"]["created"] == 0
    assert payload["extraction"]["created_ids"] == []
    assert provider.store.count_candidates() == 0


def test_extraction_enabled_does_not_default_to_candidate_creation_in_daily_report(tmp_path) -> None:
    provider = _provider(
        tmp_path,
        {
            "consolidation": {"enabled": True},
            "extraction": {"enabled": True},
        },
    )
    provider.store.append_raw_event(
        {
            "type": "turn",
            "session_id": "session-flags",
            "user_content": "I prefer daily-report extraction default candidate creation to stay off.",
            "assistant_content": "noted",
        }
    )

    payload = _tool_json(provider, "memory_v2_daily_report", {"date": "2026-06-21"})

    assert provider._config.extraction.enabled is True
    assert provider._config.extraction.candidate_creation_enabled is False
    assert payload["success"] is True
    assert payload["kind"] == "daily_memory_consolidation_report"
    assert payload["extraction"]["created"] == 0
    assert payload["extraction"]["created_ids"] == []
    assert provider.store.count_candidates() == 0


def test_daily_report_direct_api_defaults_to_no_extraction_candidates(tmp_path) -> None:
    provider = _provider(
        tmp_path,
        {
            "archive": {"capture_enabled": True},
            "extraction": {"enabled": True, "candidate_creation_enabled": True},
        },
    )
    event = provider.store.append_raw_event(
        {
            "type": "turn",
            "session_id": "session-flags",
            "user_content": "I prefer direct daily API extraction to require an explicit flag.",
            "assistant_content": "noted",
        }
    )
    provider.index.index_raw_event(event)

    report = run_daily_consolidation_report(provider.store, provider.index, date="2026-06-23")

    assert report["success"] is True
    assert report["extraction"]["created"] == 0
    assert report["extraction"]["created_ids"] == []
    assert provider.store.count_candidates() == 0


def test_dream_cycle_safe_rejection_canary_requires_review_apply_flag(tmp_path) -> None:
    provider = _provider(
        tmp_path,
        {
            "consolidation": {"enabled": True},
            "review_apply": {"enabled": False},
        },
    )
    event = provider.store.append_raw_event(
        {
            "type": "turn",
            "session_id": "session-flags",
            "user_content": "Temporary scratch preference for this answer only.",
            "assistant_content": "ack",
        }
    )
    provider.store.append_candidate(
        CandidateMemory(
            id="cand_safe_rejection_flag",
            type="preference",
            claim="Temporary scratch preference for this answer only.",
            proposed_destination="core/user",
            confidence=0.91,
            source_refs=[event["id"]],
        )
    )

    report_only = _tool_json(
        provider,
        "memory_v2_dream_cycle",
        {"date": "2026-06-19", "auto_apply": "off"},
    )
    assert report_only["success"] is True
    assert report_only["auto_apply"] == "off"
    assert provider.store.count_rejected_candidates() == 0

    payload = _tool_json(
        provider,
        "memory_v2_dream_cycle",
        {
            "date": "2026-06-20",
            "auto_apply": "safe_rejection_canary",
            "safe_rejection_canary_confirm": SAFE_REJECTION_CANARY_CONFIRM,
        },
    )

    assert payload["success"] is False
    assert "memory_v2.review_apply.enabled" in payload["error"]
    assert provider.store.count_rejected_candidates() == 0
    assert provider.store.list_candidates()[0].gate_decision.value == "pending"


def test_dream_cycle_direct_api_safe_rejection_canary_requires_review_apply_allowance(tmp_path) -> None:
    provider = _provider(tmp_path, {"review_apply": {"enabled": True}})
    event = provider.store.append_raw_event(
        {
            "type": "turn",
            "session_id": "session-flags",
            "user_content": "Temporary scratch preference for direct dream API.",
            "assistant_content": "ack",
        }
    )
    provider.store.append_candidate(
        CandidateMemory(
            id="cand_direct_dream_api_guard",
            type="preference",
            claim="Temporary scratch preference for direct dream API.",
            proposed_destination="core/user",
            confidence=0.91,
            source_refs=[event["id"]],
        )
    )

    try:
        run_memory_dream_cycle(
            provider.store,
            provider.index,
            date="2026-06-24",
            auto_apply="safe_rejection_canary",
            safe_rejection_canary_confirm=SAFE_REJECTION_CANARY_CONFIRM,
        )
    except ValueError as exc:
        assert "memory_v2.review_apply.enabled" in str(exc)
    else:  # pragma: no cover - failure branch for assertion clarity
        raise AssertionError("direct dream cycle safe rejection canary should require review-apply allowance")

    assert provider.store.count_rejected_candidates() == 0
    assert provider.store.list_candidates()[0].gate_decision.value == "pending"


def test_review_apply_direct_call_defaults_to_dry_run_until_explicit_false(tmp_path) -> None:
    provider = _provider(tmp_path, {"review_apply": {"enabled": True}})
    event = provider.store.append_raw_event(
        {
            "type": "turn",
            "session_id": "session-flags",
            "user_content": "Temporary preference for this answer only.",
            "assistant_content": "ack",
        }
    )
    provider.store.append_candidate(
        CandidateMemory(
            id="cand_review_apply_default_dry_run",
            type="preference",
            claim="Temporary preference for this answer only.",
            proposed_destination="core/user",
            confidence=0.91,
            source_refs=[event["id"]],
        )
    )
    plan = _tool_json(provider, "memory_v2_review_plan", {})
    action_ids = [action["action_id"] for action in plan["actions"]]
    assert action_ids == ["act_001"]

    default_payload = _tool_json(
        provider,
        "memory_v2_review_apply",
        {
            "plan_id": plan["plan_id"],
            "action_ids": action_ids,
            "confirm": CONFIRM_REVIEW_APPLY,
        },
    )

    assert default_payload["success"] is True
    assert default_payload["dry_run"] is True
    assert default_payload["summary"] == {
        "attempted": 1,
        "applied": 0,
        "skipped": 1,
        "failed": 0,
    }
    assert default_payload["validated"][0]["candidate_id"] == "cand_review_apply_default_dry_run"
    assert default_payload["applied"] == []
    assert provider.store.count_rejected_candidates() == 0
    assert provider.store.list_candidates()[0].gate_decision.value == "pending"

    explicit_payload = _tool_json(
        provider,
        "memory_v2_review_apply",
        {
            "plan_id": plan["plan_id"],
            "action_ids": action_ids,
            "confirm": CONFIRM_REVIEW_APPLY,
            "dry_run": False,
        },
    )

    assert explicit_payload["success"] is True
    assert explicit_payload["dry_run"] is False
    assert explicit_payload["summary"]["applied"] == 1
    assert provider.store.count_rejected_candidates() == 1
    assert provider.store.list_candidates()[0].gate_decision.value == "rejected"


def test_contradictions_auto_supersede_requires_feature_flag(tmp_path) -> None:
    provider = _provider(tmp_path)

    payload = _tool_json(provider, "memory_v2_contradictions", {"auto_supersede": True})

    assert payload["success"] is False
    assert "memory_v2.contradictions.auto_supersede" in payload["error"]
