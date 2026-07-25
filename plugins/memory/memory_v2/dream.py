"""Dream-cycle maintenance workflow for Memory v2.

The dream cycle is an offline/cron-friendly pass. It is deliberately boring and
source-grounded: collect cheap deterministic diagnostics, optionally create
pending candidates from raw evidence when explicitly requested, and build an
actionable review plan. For this rollout it is report/draft-only by default and
does not apply review-plan actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

from .config import load_memory_v2_config
from .dream_consolidation import build_dream_consolidation_snapshot
from .file_lock import FileLockUnavailableError, release_exclusive, try_acquire_exclusive
from .health import MemoryHealthChecker
from .index import MemoryV2Index
from .operations import MemoryOperationService
from .review import MemoryReviewQueue
from .review_actions import CONFIRM_REVIEW_APPLY, MemoryReviewApplier, MemoryReviewPlanner
from .report_safety import report_safe_serialize
from .schemas import utc_now_iso
from .store import MemoryV2Store

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SAFE_REJECTION_CANARY_CONFIRM = "APPLY_MEMORY_V2_SAFE_REJECTION_CANARY"
AutoApplyMode = Literal["off", "safe_rejection_canary"]


def run_memory_dream_cycle(
    store: MemoryV2Store,
    index: MemoryV2Index,
    *,
    date: str | None = None,
    mode: str = "nightly",
    auto_apply: AutoApplyMode | str = "off",
    recent_raw_limit: int = 50,
    max_review_items: int = 200,
    max_actions: int = 20,
    run_extraction: bool = False,
    safe_rejection_canary_confirm: str = "",
    allow_review_apply: bool = False,
) -> dict[str, Any]:
    """Run an auditable offline Memory v2 maintenance pass.

    Default behavior is report/draft-only: it does not create pending
    candidates from raw events and does not promote/reject review-plan actions.
    Raw-turn extraction is disabled for this rollout because reports must not
    mutate real memory state.
    """
    if run_extraction:
        raise ValueError("run_extraction is disabled for report-only dream cycle rollout")
    report_date = _normalized_date(date)
    mode = _normalized_mode(mode)
    auto_apply = _normalized_auto_apply(auto_apply)
    if auto_apply == "safe_rejection_canary" and not allow_review_apply:
        raise ValueError(
            "Memory v2 dream cycle auto_apply=safe_rejection_canary disabled by feature flag: "
            "memory_v2.review_apply.enabled"
        )
    if auto_apply == "safe_rejection_canary" and str(safe_rejection_canary_confirm or "") != SAFE_REJECTION_CANARY_CONFIRM:
        raise ValueError(f"safe_rejection_canary_confirm must equal {SAFE_REJECTION_CANARY_CONFIRM}")
    safe_recent_raw_limit = max(0, min(int(recent_raw_limit), 500))
    safe_max_review_items = max(1, min(int(max_review_items), 500))
    safe_max_actions = max(1, min(int(max_actions), 100))

    before_counts = _counts(store)
    health = MemoryHealthChecker(store, index).check()
    extraction = {
        "considered_events": 0,
        "created": 0,
        "merged": 0,
        "skipped": 1,
        "created_ids": [],
        "merged_ids": [],
        "skipped_reasons": {"extraction_disabled": 1},
    }
    review_queue = MemoryReviewQueue(store).build(limit=safe_max_review_items)
    scoped_candidate_ids = [str(item["id"]) for item in review_queue.get("items", [])]
    review_plan = MemoryReviewPlanner(store).build(max_actions=safe_max_actions, candidate_ids=scoped_candidate_ids)
    review_apply_plan = None
    review_apply = None
    if auto_apply == "safe_rejection_canary":
        canary_action_ids = [
            action["action_id"]
            for action in review_plan.get("actions") or []
            if _is_safe_rejection_canary_action(action)
        ]
        review_apply_plan = {
            "mode": "safe_rejection_canary",
            "policy": "reject_only_no_promotions",
            "candidate_ids": [
                action["candidate_id"]
                for action in review_plan.get("actions") or []
                if action.get("action_id") in canary_action_ids
            ],
            "action_ids": canary_action_ids,
        }
        if canary_action_ids:
            review_apply = MemoryReviewApplier(
                store,
                MemoryOperationService(store, index),
            ).apply(
                plan_id=str(review_plan.get("plan_id") or ""),
                action_ids=canary_action_ids,
                confirm=CONFIRM_REVIEW_APPLY,
                dry_run=False,
                max_actions=safe_max_actions,
                candidate_ids=scoped_candidate_ids,
                allow_promotions=False,
                mode="safe_rejection_canary",
            )
        else:
            review_apply = {
                "success": True,
                "mode": "safe_rejection_canary",
                "dry_run": False,
                "plan_id": str(review_plan.get("plan_id") or ""),
                "summary": {"attempted": 0, "applied": 0, "skipped": 0, "failed": 0},
                "validated": [],
                "applied": [],
                "failed": [],
            }

    after_counts = _counts(store)
    consolidation_v1 = build_dream_consolidation_snapshot(
        store,
        extraction,
        _compact_health(health),
        review_queue,
        review_plan,
        max_pending_summaries=safe_max_review_items,
        max_active_recall_cards=min(safe_max_review_items, 20),
    )
    run_id = _run_id(report_date)
    report: dict[str, Any] = {
        "success": True,
        "kind": "memory_dream_cycle_report",
        "date": report_date,
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "mode": mode,
        "policy": "source_grounded_review_plan_first",
        "auto_apply": auto_apply,
        "run_extraction": bool(run_extraction),
        "before_counts": before_counts,
        "after_counts": after_counts,
        "health": _compact_health(health),
        "extraction": extraction,
        "review_queue_summary": review_queue.get("review_summary", {}),
        "review_plan": _compact_review_plan(review_plan),
        "review_apply_plan": review_apply_plan,
        "review_apply": review_apply,
        "consolidation_v1": consolidation_v1,
        "open_loops": [_compact_open_loop(loop) for loop in store.list_open_loops(status="open")],
        "report_path": f"reports/dream_cycles/{report_date}/{run_id}.json",
        "dream_episode_path": f"episodic/dream/{report_date}/{run_id}.yaml",
        "cron_hint": _cron_hint(),
    }
    safe_report = report_safe_serialize(report)
    _write_dream_episode(store, safe_report)
    _write_json(store.base_dir / report["report_path"], safe_report)
    return safe_report



def _sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _is_safe_rejection_canary_action(action: dict[str, Any]) -> bool:
    if action.get("operation") != "reject_candidate":
        return False
    if action.get("reason") != "ephemeral_or_task_local":
        return False
    source_check = action.get("source_check") or {}
    return bool(source_check.get("all_refs_exist")) and int(source_check.get("source_count") or 0) > 0


def _compact_review_plan(plan: dict[str, Any]) -> dict[str, Any]:
    actions = []
    for action in plan.get("actions") or []:
        source_check = dict(action.get("source_check") or {})
        sources = []
        for source in source_check.get("sources") or []:
            quote = str(source.get("quote") or "")
            sources.append(
                {
                    "id": str(source.get("id") or ""),
                    "type": str(source.get("type") or ""),
                    "uri": str(source.get("uri") or ""),
                    "title": str(source.get("title") or ""),
                    "observed_at": str(source.get("observed_at") or ""),
                    "has_quote": bool(quote),
                    "quote_sha256": _sha256_text(quote) if quote else "",
                }
            )
        actions.append(
            {
                "action_id": str(action.get("action_id") or ""),
                "candidate_id": str(action.get("candidate_id") or ""),
                "operation": str(action.get("operation") or ""),
                "reason": str(action.get("reason") or ""),
                "source_check": {
                    "valid": bool(source_check.get("valid", False)),
                    "missing_source_refs": [str(ref) for ref in source_check.get("missing_source_refs") or []],
                    "sources": sources,
                    "source_count": len(sources),
                },
            }
        )
    blocked = []
    for item in plan.get("blocked") or []:
        blocked.append(
            {
                "candidate_id": str(item.get("candidate_id") or ""),
                "reason": str(item.get("reason") or ""),
                "source_refs": [str(ref) for ref in item.get("source_refs") or []],
            }
        )
    return {
        "plan_id": str(plan.get("plan_id") or ""),
        "mode": str(plan.get("mode") or ""),
        "dry_run": bool(plan.get("dry_run", True)),
        "summary": dict(plan.get("summary") or {}),
        "actions": actions,
        "blocked": blocked,
    }


def _compact_open_loop(loop: dict[str, Any]) -> dict[str, Any]:
    text = str(loop.get("text") or "")
    return {
        "id": str(loop.get("id") or ""),
        "status": str(loop.get("status") or ""),
        "source_refs": [str(ref) for ref in loop.get("source_refs") or []],
        "session_id": str(loop.get("session_id") or ""),
        "created_at": str(loop.get("created_at") or ""),
        "updated_at": str(loop.get("updated_at") or ""),
        "has_text": bool(text),
        "text_sha256": _sha256_text(text) if text else "",
    }

def _counts(store: MemoryV2Store) -> dict[str, int]:
    return {
        "raw_events": store.count_raw_events(),
        "candidates": store.count_candidates(),
        "pending_candidates": store.count_pending_candidates(),
        "rejected_candidates": store.count_rejected_candidates(),
        "memory_items": len(store.list_memory_items()),
        "project_cards": len(store.list_project_cards()),
        "open_loops": len(store.list_open_loops(status="open")),
        "source_refs": len(store.list_source_refs()),
        "session_archives": len(store.list_session_archives()),
        "operation_records": len(store.list_operation_records()),
    }


def _compact_health(health: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": bool(health.get("success", False)),
        "status": health.get("status", "unknown"),
        "issue_count": len(health.get("issues") or []),
        "issues": health.get("issues") or [],
        "repair_plan": health.get("repair_plan") or [],
    }


def _write_dream_episode(store: MemoryV2Store, report: dict[str, Any]) -> None:
    payload = {
        "version": 1,
        "kind": "memory_dream_cycle",
        "date": report["date"],
        "run_id": report["run_id"],
        "created_at": report["created_at"],
        "mode": report["mode"],
        "policy": report["policy"],
        "auto_apply": report["auto_apply"],
        "run_extraction": report.get("run_extraction", True),
        "report_path": report["report_path"],
        "before_counts": report["before_counts"],
        "after_counts": report["after_counts"],
        "health": report["health"],
        "extraction_summary": report["extraction"],
        "review_queue_summary": report["review_queue_summary"],
        "review_plan_summary": report["review_plan"].get("summary", {}),
        "review_apply_summary": (report["review_apply"] or {}).get("summary", {}) if report.get("review_apply") else None,
        "consolidation_v1_summary": (report.get("consolidation_v1") or {}).get("summary", {}),
        "open_loops": report["open_loops"],
    }
    _write_yaml(store.base_dir / report["dream_episode_path"], payload)


def _normalized_date(date: str | None) -> str:
    if date is None or not str(date).strip():
        return datetime.now(timezone.utc).date().isoformat()
    text = str(date).strip()
    if not _DATE_RE.match(text):
        raise ValueError("date must be YYYY-MM-DD")
    datetime.strptime(text, "%Y-%m-%d")
    return text


def _run_id(report_date: str) -> str:
    timestamp = utc_now_iso().replace(":", "").replace("+", "Z")
    safe_timestamp = re.sub(r"[^0-9TZ.-]", "", timestamp)
    return f"{report_date}T{safe_timestamp}-{uuid.uuid4().hex[:8]}"


def _normalized_mode(mode: str) -> str:
    text = str(mode or "nightly").strip().lower()
    if text not in {"nightly", "awake"}:
        raise ValueError("mode must be nightly or awake")
    return text


def _normalized_auto_apply(auto_apply: str) -> AutoApplyMode:
    text = str(auto_apply or "off").strip().lower()
    if text in {"promote", "promote_all", "auto_promote", "promotions", "safe_promotions"}:
        raise ValueError("automatic promotion is disabled until stronger Memory v2 evals exist")
    if text not in {"off", "safe_rejection_canary"}:
        raise ValueError("auto_apply must be off or safe_rejection_canary")
    return text  # type: ignore[return-value]


def _cron_hint() -> dict[str, Any]:
    return {
        "recommended_schedule": "0 3 * * *",
        "module": "plugins.memory.memory_v2.dream",
        "safe_default": "--auto-apply off",
        "canary": f"--auto-apply safe_rejection_canary --safe-rejection-canary-confirm {SAFE_REJECTION_CANARY_CONFIRM}",
    }


def _locked_payload(lock_path: Path) -> dict[str, Any]:
    return {
        "success": False,
        "status": "locked",
        "skipped": True,
        "lock": str(lock_path.name if lock_path.name else lock_path),
        "error": "memory dream cycle already running",
    }


def _try_acquire_dream_lock(store: MemoryV2Store):
    lock_dir = store.base_dir / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "dream-cycle.lock"
    handle = lock_path.open("a+b")
    try:
        acquired = try_acquire_exclusive(handle)
    except FileLockUnavailableError:
        handle.close()
        raise
    if not acquired:
        handle.close()
        return None, lock_path
    handle.seek(0)
    handle.truncate()
    handle.write(
        (json.dumps({"pid": os.getpid(), "acquired_at": utc_now_iso()}, sort_keys=True) + "\n").encode("utf-8")
    )
    handle.flush()
    os.fsync(handle.fileno())
    return handle, lock_path


def _release_dream_lock(handle, lock_path: Path) -> None:
    del lock_path  # Keep the lock file in place; unlinking creates a flock inode race.
    try:
        release_exclusive(handle)
    finally:
        handle.close()


def _write_json(path: Path, payload: dict[str, Any]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Memory v2 dream-cycle maintenance pass for one profile.")
    parser.add_argument("--hermes-home", required=True, help="Profile home containing memory_v2/.")
    parser.add_argument("--date", default=None, help="Report date as YYYY-MM-DD. Defaults to current UTC date.")
    parser.add_argument("--mode", default="nightly", choices=["nightly", "awake"], help="Dream-cycle mode.")
    parser.add_argument(
        "--auto-apply",
        default="off",
        choices=["off", "safe_rejection_canary"],
        help="Apply policy. Default off is report-only; safe_rejection_canary allows only scoped low-risk rejections with confirmation.",
    )
    parser.add_argument(
        "--safe-rejection-canary-confirm",
        default="",
        help=f"Required confirmation for --auto-apply safe_rejection_canary: {SAFE_REJECTION_CANARY_CONFIRM}.",
    )
    parser.add_argument("--session-id", default="memory-dream-cycle", help="Session id used when initializing the index.")
    parser.add_argument(
        "--run-extraction",
        action="store_true",
        help="Explicitly scan raw turns and create pending candidates before writing the report (off by default).",
    )
    args = parser.parse_args(argv)

    hermes_home = Path(args.hermes_home).expanduser().resolve()
    store = MemoryV2Store(hermes_home / "memory_v2")
    store.initialize()
    lock_handle, lock_path = _try_acquire_dream_lock(store)
    if lock_handle is None:
        print(json.dumps(_locked_payload(lock_path), sort_keys=True))
        return 0
    try:
        hold_seconds = float(os.environ.get("HERMES_MEMORY_V2_DREAM_LOCK_HOLD_SECS") or 0.0)
        if hold_seconds > 0:
            time.sleep(min(hold_seconds, 30.0))
        index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
        index.initialize()
        index.rebuild_from_store(store)
        flags = load_memory_v2_config(hermes_home)
        report = run_memory_dream_cycle(
            store,
            index,
            date=args.date,
            mode=args.mode,
            auto_apply=args.auto_apply,
            run_extraction=args.run_extraction,
            safe_rejection_canary_confirm=args.safe_rejection_canary_confirm,
            allow_review_apply=flags.review_apply.enabled,
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    finally:
        _release_dream_lock(lock_handle, lock_path)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
