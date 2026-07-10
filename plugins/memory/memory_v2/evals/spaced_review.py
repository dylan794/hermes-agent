"""Local deterministic eval helper for Memory v2 spaced review."""

from __future__ import annotations

import json
from typing import Any

from ..schemas import MemoryItem, ProjectCard
from ..spaced_review import build_active_recall_review

_PRIVATE_SENTINEL = "PRIVATE_SENTINEL_SPACED_REVIEW_EVAL"


def run_spaced_review_eval() -> dict[str, Any]:
    """Run a tiny deterministic fixture eval for due coverage and privacy safety."""

    due_item = MemoryItem(
        id="eval_due_memory",
        type="preference",
        subject=f"User {_PRIVATE_SENTINEL}",
        value=f"prefers privacy-safe deterministic reports {_PRIVATE_SENTINEL}",
        status="active",
        confidence=0.9,
        importance=0.95,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        source_refs=["eval:source"],
    )
    project = ProjectCard(
        id="project:eval-spaced-review",
        name=f"Eval Project {_PRIVATE_SENTINEL}",
        current_state=f"active eval state {_PRIVATE_SENTINEL}",
        importance=0.8,
        updated_at="2026-06-20T00:00:00Z",
        source_refs=["eval:project"],
    )
    report = build_active_recall_review(items=[due_item], project_cards=[project], now="2026-06-27T00:00:00Z", max_cards=10)
    payload = json.dumps(report, sort_keys=True)
    due_hashes = {card["record_id_sha256"] for card in report["due_now"]}
    expected_due_ids = {"eval_due_memory", "project:eval-spaced-review"}
    import hashlib
    expected_due = {hashlib.sha256(value.encode("utf-8")).hexdigest() for value in expected_due_ids}
    privacy_leaks = [needle for needle in (_PRIVATE_SENTINEL, "privacy-safe deterministic reports", "active eval state") if needle in payload]
    actions = report.get("suggested_actions") or []
    report_only_actions = 1.0 if actions and all(action.get("mutation") == "none" for action in actions) else 0.0
    due_coverage = len(expected_due & due_hashes) / len(expected_due)
    metrics = {
        "due_coverage": due_coverage,
        "privacy_leak_count": len(privacy_leaks),
        "report_only_actions": report_only_actions,
    }
    return {
        "version": 1,
        "passed": due_coverage == 1.0 and not privacy_leaks and report_only_actions == 1.0,
        "metrics": metrics,
        "summary": {
            "expected_due": len(expected_due),
            "due_now": len(report["due_now"]),
            "privacy_leaks": len(privacy_leaks),
        },
        "report": report,
    }
