"""Deterministic draft/report consolidation for Memory v2 dream cycles.

This layer is intentionally non-mutating. It summarizes existing store state and
pending review candidates into compact, source-grounded draft packets that can be
reviewed later by humans or gated tools.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Any

from .belief_dashboard import build_belief_update_dashboard
from .entity_graph import build_entity_graph_draft
from .redaction import contains_sensitive_text
from .report_safety import report_safe_serialize
from .schemas import GateDecision, MemoryType
from .spaced_review import build_active_recall_review
from .store import MemoryV2Store

_CORE_DRAFT_TYPES = {
    MemoryType.PREFERENCE.value,
    MemoryType.ENVIRONMENT.value,
    MemoryType.CONSTRAINT.value,
}
_PROJECT_DEST_RE = re.compile(r"^semantic/projects/(?P<slug>[^/]+)\.ya?ml$")
_PROJECT_CLAIM_RE = re.compile(
    r"^Project\s+(?P<project>.+?)\s+(?P<kind>decision|next action|open question|current state|goal|status):\s*(?P<value>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_UNSAFE_CANDIDATE_RE = re.compile(
    r"\b(ignore\s+previous|system\s*:|developer\s+instruction|promote\s+this\s+memory\s+automatically)\b",
    re.IGNORECASE,
)
_MAX_PENDING_SUMMARIES = 200
_MAX_BLOCKED_SUMMARIES = 100
_MAX_PROJECT_DRAFTS = 20


def build_dream_consolidation_snapshot(
    store: MemoryV2Store,
    extraction: dict[str, Any],
    health: dict[str, Any],
    review_queue: dict[str, Any],
    review_plan: dict[str, Any],
    *,
    max_core_entries: int = 12,
    max_pending_summaries: int = _MAX_PENDING_SUMMARIES,
    max_active_recall_cards: int = 50,
) -> dict[str, Any]:
    """Build a stable JSON-serializable Memory v2 consolidation snapshot.

    The snapshot contains only existing memory records, candidate claims/source
    ids, compact counts, and summaries. It never reads or includes raw turn text.
    """
    pending = [candidate for candidate in store.list_candidates() if candidate.gate_decision == GateDecision.PENDING]
    pending.sort(key=lambda item: (str(getattr(item.type, "value", item.type)), item.proposed_destination, item.id))
    pending_summary_limit = _bounded_limit(
        max_pending_summaries,
        default=_MAX_PENDING_SUMMARIES,
        ceiling=_MAX_PENDING_SUMMARIES,
        minimum=0,
    )
    active_recall_card_limit = _bounded_limit(
        max_active_recall_cards,
        default=50,
        ceiling=50,
        minimum=1,
    )

    by_type = Counter(str(getattr(candidate.type, "value", candidate.type)) for candidate in pending)
    by_destination = Counter(str(candidate.proposed_destination or "unspecified") for candidate in pending)
    project_cards = store.list_project_cards()
    core_records = store.list_core_memory_records()
    open_loops = store.list_open_loops(status="open")

    memory_items = store.list_memory_items()
    source_refs = store.list_source_refs()
    core_cache_draft = _build_core_cache_draft(core_records, pending, max_entries=max_core_entries)
    project_card_drafts = _build_project_card_drafts(project_cards, pending)
    entity_graph_draft = build_entity_graph_draft(memory_items=memory_items, project_cards=project_cards, candidates=pending)
    belief_update_dashboard = build_belief_update_dashboard(items=memory_items, candidates=pending, sources=source_refs)
    active_recall_review = build_active_recall_review(
        items=memory_items,
        project_cards=project_cards,
        candidates=pending,
        open_loops=open_loops,
        now=health.get("checked_at") or None,
        max_cards=active_recall_card_limit,
    )
    blocked_candidate_summaries = [
        {**_candidate_summary(candidate), "blocked_reason": _candidate_block_reason(candidate) or "unsafe_candidate"}
        for candidate in pending
        if _candidate_block_reason(candidate)
    ][:_MAX_BLOCKED_SUMMARIES]

    counts = {
        "pending_candidates": len(pending),
        "open_loops": len(open_loops),
        "project_cards": len(project_cards),
        "core_records": len(core_records),
        "memory_items": len(memory_items),
        "entity_graph_entities": entity_graph_draft["summary"]["entity_count"],
        "belief_update_actions": belief_update_dashboard["summary"]["suggested_actions"],
        "active_recall_due_now": active_recall_review["summary"]["due_now"],
    }
    summary = {
        "status": "draft",
        "health_status": health.get("status", "unknown"),
        "pending_candidates": len(pending),
        "open_loops": len(open_loops),
        "project_card_drafts": len(project_card_drafts),
        "core_cache_entries": len(core_cache_draft["entries"]),
        "entity_graph_entities": entity_graph_draft["summary"]["entity_count"],
        "belief_update_actions": belief_update_dashboard["summary"]["suggested_actions"],
        "active_recall_due_now": active_recall_review["summary"]["due_now"],
        "extraction": _compact_extraction(extraction),
        "review_plan_summary": dict(review_plan.get("summary") or {}),
    }
    return report_safe_serialize({
        "version": 1,
        "status": "draft",
        "policy": "report_only_no_promotion",
        "health_status": health.get("status", "unknown"),
        "counts": counts,
        "pending_candidates_by_type": dict(sorted(by_type.items())),
        "pending_candidates_by_destination": dict(sorted(by_destination.items())),
        "extraction_summary": _compact_extraction(extraction),
        "review_queue_summary": dict(review_queue.get("review_summary") or {}),
        "review_plan_summary": dict(review_plan.get("summary") or {}),
        "pending_candidate_summaries": [_candidate_summary(candidate) for candidate in pending[:pending_summary_limit]],
        "blocked_candidate_summaries": blocked_candidate_summaries,
        "core_cache_draft": core_cache_draft,
        "project_card_drafts": project_card_drafts,
        "entity_graph_draft": entity_graph_draft,
        "belief_update_dashboard": belief_update_dashboard,
        "active_recall_review": active_recall_review,
        "summary": summary,
    })


def _bounded_limit(value: Any, *, default: int, ceiling: int, minimum: int) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = int(default)
    return max(int(minimum), min(limit, int(ceiling)))


def _compact_extraction(extraction: dict[str, Any]) -> dict[str, Any]:
    return {
        "considered_events": int(extraction.get("considered_events") or 0),
        "created": int(extraction.get("created") or 0),
        "merged": int(extraction.get("merged") or 0),
        "skipped": int(extraction.get("skipped") or 0),
        "created_ids": [str(value) for value in extraction.get("created_ids") or []],
        "merged_ids": [str(value) for value in extraction.get("merged_ids") or []],
        "skipped_reasons": dict(extraction.get("skipped_reasons") or {}),
    }


def _candidate_summary(candidate: Any) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "type": str(getattr(candidate.type, "value", candidate.type)),
        "proposed_destination": candidate.proposed_destination,
        "confidence": candidate.confidence,
        "importance": candidate.importance,
        "source_refs": list(candidate.source_refs),
        "claim_sha256": hashlib.sha256(str(candidate.claim or "").encode("utf-8")).hexdigest(),
    }


def _candidate_block_reason(candidate: Any) -> str:
    claim = str(getattr(candidate, "claim", "") or "")
    if contains_sensitive_text(claim):
        return "sensitive_text"
    if _UNSAFE_CANDIDATE_RE.search(claim):
        return "prompt_injection_or_system_like_text"
    return ""


def _build_core_cache_draft(core_records: list[Any], pending: list[Any], *, max_entries: int) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for record in sorted(core_records, key=lambda item: (-item.priority, -item.confidence, item.id)):
        entries.append(
            {
                "id": record.id,
                "origin": "core_record",
                "type": str(getattr(record.category, "value", record.category)),
                "statement_sha256": hashlib.sha256(str(record.statement or "").encode("utf-8")).hexdigest(),
                "priority": record.priority,
                "confidence": record.confidence,
                "source_refs": list(record.source_refs),
                "status": "active",
                "requires_review": False,
            }
        )
    for candidate in pending:
        candidate_type = str(getattr(candidate.type, "value", candidate.type))
        if candidate_type not in _CORE_DRAFT_TYPES:
            continue
        if float(candidate.confidence) < 0.70:
            continue
        if _candidate_block_reason(candidate):
            continue
        entries.append(
            {
                "id": candidate.id,
                "origin": "pending_candidate",
                "type": candidate_type,
                "claim_sha256": hashlib.sha256(str(candidate.claim or "").encode("utf-8")).hexdigest(),
                "priority": candidate.importance,
                "confidence": candidate.confidence,
                "source_refs": list(candidate.source_refs),
                "status": "draft",
                "requires_review": True,
            }
        )
    entries.sort(key=lambda item: (-float(item.get("priority") or 0), -float(item.get("confidence") or 0), str(item.get("id") or "")))
    max_entries = max(1, min(int(max_entries), 12))
    return {
        "status": "draft",
        "requires_review": True,
        "max_entries": max_entries,
        "entries": entries[:max_entries],
    }


def _build_project_card_drafts(project_cards: list[Any], pending: list[Any]) -> list[dict[str, Any]]:
    existing_by_destination = {_project_destination_from_card(card): card for card in project_cards}
    grouped: dict[str, list[Any]] = defaultdict(list)
    for candidate in pending:
        candidate_type = str(getattr(candidate.type, "value", candidate.type))
        if candidate_type != MemoryType.PROJECT_STATE.value:
            continue
        destination = str(candidate.proposed_destination or "")
        if _PROJECT_DEST_RE.match(destination):
            grouped[destination].append(candidate)

    drafts: list[dict[str, Any]] = []
    for destination in sorted(grouped):
        card = existing_by_destination.get(destination)
        fields = _base_project_fields(card)
        candidate_summaries: list[dict[str, Any]] = []
        for candidate in sorted(grouped[destination], key=lambda item: ((item.source_refs or [""])[0], item.created_at, item.id)):
            candidate_summaries.append(_candidate_summary(candidate))
            if _candidate_block_reason(candidate):
                continue
            parsed = _parse_project_claim(candidate.claim)
            if parsed is None:
                continue
            kind, value = parsed
            target = {"status": "current_state", "decision": "decisions", "next_action": "next_actions", "open_question": "open_questions"}.get(kind, kind)
            fields["candidate_count"] += 1
            fields["field_counts"][target] = fields["field_counts"].get(target, 0) + 1
            fields["value_hashes"].append(
                {
                    "field": target,
                    "value_sha256": hashlib.sha256(str(value or "").encode("utf-8")).hexdigest(),
                    "candidate_id": str(candidate.id),
                    "source_refs": [str(ref) for ref in candidate.source_refs],
                }
            )
            for ref in candidate.source_refs:
                _append_unique(fields["source_refs"], str(ref))
        drafts.append(
            {
                "status": "draft",
                "requires_review": True,
                "proposed_destination": destination,
                "project_id": card.id if card else _project_id_from_destination(destination),
                "existing_card": _compact_project_card(card) if card else None,
                "suggested_fields": fields,
                "candidate_summaries": candidate_summaries,
            }
        )
    return drafts[:_MAX_PROJECT_DRAFTS]


def _base_project_fields(card: Any | None) -> dict[str, Any]:
    source_refs = [str(ref) for ref in getattr(card, "source_refs", []) or []] if card is not None else []
    return {"candidate_count": 0, "field_counts": {}, "value_hashes": [], "source_refs": source_refs}


def _compact_project_card(card: Any) -> dict[str, Any]:
    return {
        "id": str(getattr(card, "id", "") or ""),
        "name_sha256": hashlib.sha256(str(getattr(card, "name", "") or "").encode("utf-8")).hexdigest(),
        "status": str(getattr(getattr(card, "status", ""), "value", getattr(card, "status", "")) or ""),
        "source_refs": [str(ref) for ref in getattr(card, "source_refs", []) or []],
        "field_presence": {
            "goal": bool(getattr(card, "goal", "") or ""),
            "current_state": bool(getattr(card, "current_state", "") or ""),
            "decisions": len(getattr(card, "decisions", []) or []),
            "next_actions": len(getattr(card, "next_actions", []) or []),
            "open_questions": len(getattr(card, "open_questions", []) or []),
        },
    }


def _parse_project_claim(claim: str) -> tuple[str, str] | None:
    match = _PROJECT_CLAIM_RE.match(str(claim or "").strip())
    if not match:
        return None
    kind = str(match.group("kind")).lower().replace(" ", "_")
    value = re.sub(r"\s+", " ", str(match.group("value") or "").strip())
    return kind, value


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _project_destination_from_card(card: Any) -> str:
    project_id = str(card.id)
    slug = project_id.split(":", 1)[1] if project_id.startswith("project:") else project_id
    return f"semantic/projects/{slug}.yaml"


def _project_id_from_destination(destination: str) -> str:
    match = _PROJECT_DEST_RE.match(destination)
    if not match:
        return ""
    return f"project:{match.group('slug')}"
