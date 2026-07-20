#!/usr/bin/env python3
"""Run the deterministic Memory v2 Stage 8 canary in a temporary profile."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.memory.memory_v2 import MemoryV2Provider  # noqa: E402
from plugins.memory.memory_v2.health import MemoryHealthChecker  # noqa: E402
from plugins.memory.memory_v2.operations import MemoryOperationService  # noqa: E402

DEFAULT_CONFIG = (
    ROOT
    / "plugins"
    / "memory"
    / "memory_v2"
    / "stage_8_canary_config.example.yaml"
)
SESSION_ID = "stage8-synthetic-canary"
EVENT_ID = "raw_stage8_synthetic_turn_001"
CREATED_AT = "2030-01-02T03:04:05+00:00"
USER_TEXT = "I prefer canary reports with concise JSON"
ASSISTANT_TEXT = "Synthetic acknowledgement for the isolated Stage 8 canary."


def _tool(provider: MemoryV2Provider, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = json.loads(provider.handle_tool_call(name, args or {}))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} returned a non-object payload")
    return payload


def _counts(provider: MemoryV2Provider) -> dict[str, int]:
    return {
        "raw_events": provider.store.count_raw_events(),
        "candidates": provider.store.count_candidates(),
        "pending_candidates": provider.store.count_pending_candidates(),
        "memory_items": len(provider.store.list_memory_items()),
        "operation_records": len(provider.store.list_operation_records()),
    }


def run_canary(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Exercise Stage 8 without accepting or resolving any live-profile path."""
    checks: dict[str, bool] = {}
    metrics: dict[str, int] = {}

    with tempfile.TemporaryDirectory(prefix="memory-v2-stage8-canary-") as temp_dir:
        hermes_home = Path(temp_dir)
        shutil.copyfile(config_path.resolve(), hermes_home / "config.yaml")

        provider = MemoryV2Provider()
        provider.initialize(
            SESSION_ID,
            hermes_home=str(hermes_home),
            platform="stage8_canary",
            agent_context="primary",
        )
        flags = provider._config
        checks["isolated_temporary_profile"] = provider.base_dir.is_relative_to(hermes_home)
        checks["automatic_promotion_disabled"] = not flags.auto_promote.enabled
        checks["automatic_supersession_disabled"] = not flags.contradictions.auto_supersede
        checks["raw_prefetch_disabled"] = not flags.archive.prefetch_raw_enabled
        checks["model_review_apply_disabled"] = not flags.review_apply.enabled
        checks["model_extraction_disabled"] = not flags.extraction.small_model_enabled
        checks["tool_output_capture_disabled"] = not flags.archive.include_tool_outputs
        checks["semantic_prefetch_enabled"] = flags.prefetch.enabled

        provider.sync_turn(
            USER_TEXT,
            ASSISTANT_TEXT,
            session_id=SESSION_ID,
            event_id=EVENT_ID,
            created_at=CREATED_AT,
        )
        after_capture = _counts(provider)
        checks["capture_archived_one_event_only"] = after_capture == {
            "raw_events": 1,
            "candidates": 0,
            "pending_candidates": 0,
            "memory_items": 0,
            "operation_records": 0,
        }

        readiness = _tool(provider, "memory_v2_archive_readiness", {"sample_size": 1})
        search = _tool(
            provider,
            "memory_v2_archive_search",
            {"query": "canary reports", "session_id": SESSION_ID, "limit": 1},
        )
        result = (search.get("results") or [{}])[0]
        show = _tool(
            provider,
            "memory_v2_archive_show",
            {"id": result.get("event_id", "")},
        )
        checks["archive_readiness_passed"] = bool(
            readiness.get("success") and readiness.get("ready") and not readiness.get("blockers")
        )
        checks["archive_reporting_is_bounded_untrusted_evidence"] = bool(
            search.get("success")
            and search.get("count") == 1
            and search.get("untrusted_text") is True
            and search.get("can_instruct") is False
            and show.get("success")
            and show.get("untrusted_text") is True
            and show.get("can_instruct") is False
        )

        extraction = _tool(
            provider,
            "memory_v2_extract_candidates",
            {"session_id": SESSION_ID, "recent_raw_limit": 10},
        )
        after_extraction = _counts(provider)
        checks["deterministic_extraction_created_one_pending_candidate"] = bool(
            extraction.get("success")
            and extraction.get("ready")
            and extraction.get("mutations_allowed") == "pending_candidates_only"
            and extraction.get("extraction", {}).get("created") == 1
            and extraction.get("extraction", {}).get("model_created") == 0
            and after_extraction["pending_candidates"] == 1
            and after_extraction["memory_items"] == 0
            and after_extraction["operation_records"] == 0
        )

        daily_before = _counts(provider)
        daily = _tool(provider, "memory_v2_daily_report", {"date": "2030-01-02"})
        queue = _tool(provider, "memory_v2_review_queue", {"now": CREATED_AT, "limit": 20})
        candidate_id = str((extraction.get("extraction", {}).get("created_ids") or [""])[0])
        plan = _tool(provider, "memory_v2_review_plan", {"candidate_ids": [candidate_id]})
        daily_after = _counts(provider)
        actions = plan.get("actions") or []
        action = next(
            (item for item in actions if item.get("operation") == "promote_candidate"),
            {},
        )
        checks["report_and_review_are_read_only"] = bool(
            daily.get("success")
            and queue.get("success")
            and plan.get("success")
            and action
            and daily_before == daily_after
        )

        model_attempt = _tool(
            provider,
            "memory_v2_promote",
            {
                "candidate_id": candidate_id,
                "plan_id": plan.get("plan_id", ""),
                "action_id": action.get("action_id", ""),
                "candidate_fingerprint": action.get("candidate_fingerprint", ""),
            },
        )
        checks["model_promotion_rejected_without_mutation"] = bool(
            model_attempt.get("success") is False
            and "external operator authority" in str(model_attempt.get("error") or "")
            and _counts(provider) == daily_after
        )

        promotion = MemoryOperationService(provider.store, provider.index).promote_candidate(
            candidate_id,
            session_id=SESSION_ID,
            actor="stage8_canary_operator",
            expected_candidate_fingerprint=str(action.get("candidate_fingerprint") or ""),
        )
        after_promotion = _counts(provider)
        checks["trusted_host_fingerprint_promotion_succeeded"] = bool(
            promotion.success
            and after_promotion["pending_candidates"] == 0
            and after_promotion["memory_items"] == 1
            and after_promotion["operation_records"] >= 1
        )

        packet = provider.prefetch(
            "What are my preferences for canary reports?", session_id=SESSION_ID
        )
        lowered_packet = packet.lower()
        checks["semantic_prefetch_excludes_raw_archive_hydration"] = bool(
            "concise json" in lowered_packet
            and ASSISTANT_TEXT not in packet
            and "type: preference" in lowered_packet
            and "type: raw_event" not in lowered_packet
            and "event_type:" not in lowered_packet
        )

        health = MemoryHealthChecker(provider.store, provider.index).check()
        health_issues = health.get("issues") or []
        checks["health_has_no_dangling_or_recovery_issues"] = bool(
            health.get("status") == "healthy"
            and not any(
                str(issue.get("code") or "")
                in {"dangling_source_ref", "operation_recovery_required", "interrupted_operation"}
                for issue in health_issues
            )
        )

        metrics.update(
            {
                "raw_events": after_promotion["raw_events"],
                "candidates": after_promotion["candidates"],
                "pending_candidates": after_promotion["pending_candidates"],
                "memory_items": after_promotion["memory_items"],
                "review_actions": len(actions),
            }
        )

    go = bool(checks) and all(checks.values())
    return {
        "success": go,
        "go": go,
        "profile_scope": "fresh_temporary_directory_only",
        "checks": checks,
        "metrics": metrics,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the isolated deterministic Memory v2 Stage 8 canary."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Read-only canary config template copied into a fresh temporary profile.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        payload = run_canary(args.config)
    except Exception as exc:
        payload = {
            "success": False,
            "go": False,
            "profile_scope": "fresh_temporary_directory_only",
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0 if payload.get("go") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
