"""Report-only uncertainty and belief-update dashboard for Memory v2."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from .report_safety import report_safe_serialize


LOW_CONFIDENCE_THRESHOLD = 0.50
SOURCE_WEAK_CONFIDENCE_THRESHOLD = 0.85
MAX_SECTION_ITEMS = 50
MAX_CONFLICTS = 50
MAX_ACTIONS = 50
MAX_CONFLICT_GROUP_SIZE = 12
MAX_RECORDS_CONSIDERED = 500


def build_belief_update_dashboard(
    *,
    items: Iterable[Any] = (),
    candidates: Iterable[Any] = (),
    sources: Iterable[Any] = (),
    now: str | None = None,
    stale_days: int = 365,
    expiring_days: int = 30,
) -> dict[str, Any]:
    """Build a deterministic non-mutating belief/uncertainty dashboard.

    The dashboard highlights low-confidence, stale, conflicting, expiring, and
    source-weak records and proposes review actions only. It never mutates the
    store and never requires network or LLM calls.
    """

    now_dt = _parse_iso(now) or datetime.now(timezone.utc).replace(microsecond=0)
    source_by_id = {str(getattr(source, "id", "") or ""): source for source in sources}
    raw_records = [*items, *candidates][:MAX_RECORDS_CONSIDERED]
    records = [
        _record_view(record, kind="candidate" if hasattr(record, "claim") else "memory_item", source_by_id=source_by_id)
        for record in raw_records
    ]

    low_confidence = [r for r in records if r["confidence"] < LOW_CONFIDENCE_THRESHOLD or r["status"] == "uncertain"]
    stale = [r for r in records if _is_stale(r, now_dt, stale_days)]
    expiring = [r for r in records if _is_expiring(r, now_dt, expiring_days)]
    source_weak = [r for r in records if not r["source_refs"] and r["confidence"] >= SOURCE_WEAK_CONFIDENCE_THRESHOLD]
    conflicts = _detect_conflicts(records)

    suggested_actions = _suggest_actions(low_confidence, stale, expiring, source_weak, conflicts)
    return report_safe_serialize({
        "version": 1,
        "status": "draft",
        "policy": "report_only_no_mutation",
        "low_confidence": _sorted_records(low_confidence)[:MAX_SECTION_ITEMS],
        "stale": _sorted_records(stale)[:MAX_SECTION_ITEMS],
        "conflicts": conflicts[:MAX_CONFLICTS],
        "expiring": _sorted_records(expiring)[:MAX_SECTION_ITEMS],
        "source_weak": _sorted_records(source_weak)[:MAX_SECTION_ITEMS],
        "suggested_actions": suggested_actions[:MAX_ACTIONS],
        "summary": {
            "records_considered": len(records),
            "low_confidence": len(low_confidence),
            "stale": len(stale),
            "conflicts": len(conflicts),
            "expiring": len(expiring),
            "source_weak": len(source_weak),
            "suggested_actions": len(suggested_actions),
        },
    })


def _record_view(record: Any, *, kind: str, source_by_id: dict[str, Any]) -> dict[str, Any]:
    record_type = str(getattr(getattr(record, "type", ""), "value", getattr(record, "type", "")))
    status = str(getattr(getattr(record, "status", ""), "value", getattr(record, "status", "")) or getattr(getattr(record, "gate_decision", ""), "value", getattr(record, "gate_decision", "")) or "")
    source_refs = [str(ref) for ref in getattr(record, "source_refs", []) or []]
    source_observed = [_parse_iso(getattr(source_by_id.get(ref), "observed_at", "")) for ref in source_refs]
    latest_source_dt = max((dt for dt in source_observed if dt is not None), default=None)
    return {
        "id": str(getattr(record, "id", "") or ""),
        "kind": kind,
        "type": record_type,
        "subject_key": _subject_key(record),
        "status": status,
        "confidence": _safe_float(getattr(record, "confidence", 0.0), 0.0),
        "importance": _safe_float(getattr(record, "importance", 0.0), 0.0),
        "updated_at": str(getattr(record, "updated_at", "") or getattr(record, "created_at", "") or ""),
        "expires_at": str(getattr(record, "expires_at", "") or ""),
        "source_refs": source_refs,
        "source_count": len(source_refs),
        "latest_source_at": _iso_or_empty(latest_source_dt),
        "value_fingerprint": _value_fingerprint(record),
    }


def _detect_conflicts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["type"] in {"preference", "belief", "fact", "environment", "project_state"}:
            grouped[(record["type"], record["subject_key"])].append(record)
    conflicts: list[dict[str, Any]] = []
    for (_record_type, _subject), group in grouped.items():
        group = group[:MAX_CONFLICT_GROUP_SIZE]
        for left_idx, left in enumerate(group):
            for right in group[left_idx + 1 :]:
                if left["value_fingerprint"] == right["value_fingerprint"]:
                    continue
                if len(conflicts) >= MAX_CONFLICTS:
                    break
                conflicts.append(
                    {
                        "left_id": left["id"],
                        "right_id": right["id"],
                        "type": left["type"],
                        "subject_key": left["subject_key"],
                        "reason": "same_subject_different_value_or_claim",
                        "confidence_delta": abs(left["confidence"] - right["confidence"]),
                        "source_refs": sorted(set(left["source_refs"] + right["source_refs"])),
                    }
                )
            if len(conflicts) >= MAX_CONFLICTS:
                break
        if len(conflicts) >= MAX_CONFLICTS:
            break
    conflicts.sort(key=lambda row: (row["type"], row["subject_key"], row["left_id"], row["right_id"]))
    return conflicts[:MAX_CONFLICTS]


def _suggest_actions(low_confidence, stale, expiring, source_weak, conflicts) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for record in _sorted_records(low_confidence):
        actions.append(_action("review_confidence", record["id"], "Verify or archive low-confidence/uncertain record."))
    for record in _sorted_records(stale):
        actions.append(_action("refresh_stale", record["id"], "Refresh stale fact against current source before trusting it."))
    for record in _sorted_records(expiring):
        actions.append(_action("review_expiry", record["id"], "Decide whether expiring record should lapse or be renewed."))
    for record in _sorted_records(source_weak):
        actions.append(_action("attach_source", record["id"], "Find/attach source evidence before promotion or high-confidence use."))
    for conflict in conflicts:
        actions.append(_action("resolve_conflict", f"{conflict['left_id']}::{conflict['right_id']}", "Choose current belief, supersede stale one, or keep both marked uncertain."))
    # Stable de-dupe by type/target.
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for action in actions:
        key = (action["action"], action["target"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped[:MAX_ACTIONS]


def _action(action: str, target: str, reason: str) -> dict[str, Any]:
    return {"action": action, "target": target, "reason": reason, "mutation": "none"}


def _is_stale(record: dict[str, Any], now_dt: datetime, stale_days: int) -> bool:
    updated = _parse_iso(record.get("updated_at")) or _parse_iso(record.get("latest_source_at"))
    if updated is None:
        return False
    return (now_dt - updated).days > int(stale_days)


def _is_expiring(record: dict[str, Any], now_dt: datetime, expiring_days: int) -> bool:
    expires = _parse_iso(record.get("expires_at"))
    if expires is None:
        return False
    delta = expires - now_dt
    return 0 <= delta.days <= int(expiring_days)


def _subject_key(record: Any) -> str:
    subject = str(getattr(record, "subject", "") or "")
    if not subject and hasattr(record, "claim"):
        claim = str(getattr(record, "claim", "") or "")
        # Strip common copulas so related candidate claims group with semantic items,
        # then hash to avoid exposing claim fragments.
        subject = claim.split(" is ", 1)[0] if " is " in claim else claim.split(".", 1)[0]
    if not subject and hasattr(record, "name"):
        subject = str(getattr(record, "name", "") or "")
    normalized = " ".join(subject.lower().split())
    return "subject:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _value_fingerprint(record: Any) -> str:
    text = str(getattr(record, "value", "") or getattr(record, "body", "") or getattr(record, "claim", "") or getattr(record, "current_state", "") or "")
    normalized = " ".join(text.lower().split())
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sorted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda row: (row["kind"], row["type"], row["id"]))


def _parse_iso(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _iso_or_empty(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
