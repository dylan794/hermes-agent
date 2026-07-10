"""Deterministic active-recall / spaced-review reports for Memory v2.

This module is intentionally pure and report-only. It selects memory records that
would benefit from review and emits privacy-safe cards/probes using hashes and
metadata only; it never mutates stores, schedules jobs, or calls network/LLM APIs.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Iterable

_MAX_RECORDS = 1000
_DEFAULT_MAX_CARDS = 50
_ACTIVE_MEMORY_STATUSES = {"active", "uncertain"}
_ACTIVE_PROJECT_STATUSES = {"active", "paused"}
_PENDING_CANDIDATE_STATUSES = {"pending"}


def build_active_recall_review(
    *,
    items: Iterable[Any] = (),
    project_cards: Iterable[Any] = (),
    candidates: Iterable[Any] = (),
    open_loops: Iterable[Any] = (),
    retrieval_logs: Iterable[dict[str, Any]] = (),
    now: str | datetime | None = None,
    max_cards: int = _DEFAULT_MAX_CARDS,
    due_window_days: int = 7,
) -> dict[str, Any]:
    """Build a bounded, JSON-serializable spaced-review dashboard.

    Returned cards never include raw memory/user/project text. All memory-derived
    text is treated as untrusted and represented by stable hashes/fingerprints.
    """

    now_dt = _parse_iso(now) or datetime.now(timezone.utc).replace(microsecond=0)
    max_cards = max(1, min(int(max_cards or _DEFAULT_MAX_CARDS), _DEFAULT_MAX_CARDS))
    due_window_days = max(0, int(due_window_days or 0))
    logs_by_record = _logs_by_record(retrieval_logs)

    cards: list[dict[str, Any]] = []
    raw_records = (
        [(_record_id(record), "memory_item", record) for record in list(items)[:_MAX_RECORDS]]
        + [(_record_id(record), "project_card", record) for record in list(project_cards)[:_MAX_RECORDS]]
        + [(_record_id(record), "candidate", record) for record in list(candidates)[:_MAX_RECORDS]]
        + [(_record_id(record), "open_loop", record) for record in list(open_loops)[:_MAX_RECORDS]]
    )[:_MAX_RECORDS]

    for record_id, kind, record in raw_records:
        if not record_id or not _eligible(kind, record):
            continue
        card = _build_card(record, kind=kind, now_dt=now_dt, logs=logs_by_record.get(record_id, []))
        cards.append(card)

    due_now = [card for card in cards if card["due_state"] in {"due_now", "overdue"}]
    overdue = [card for card in cards if card["due_state"] == "overdue"]
    upcoming = [card for card in cards if card["due_state"] == "upcoming" and card["days_until_due"] <= due_window_days]
    not_due = [card for card in cards if card["due_state"] == "not_due"]

    due_now = _sort_cards(due_now)[:max_cards]
    overdue = _sort_cards(overdue)[:max_cards]
    upcoming = _sort_cards(upcoming)[:max_cards]
    probes = [_probe(card) for card in _sort_cards([*due_now, *overdue, *upcoming])[:max_cards]]
    actions = [_action(card) for card in _sort_cards([*overdue, *due_now, *upcoming])[:max_cards]]

    return {
        "version": 1,
        "status": "draft",
        "policy": "report_only_no_mutation",
        "generated_at": _iso(now_dt),
        "untrusted_text": True,
        "summary": {
            "records_considered": len(raw_records),
            "eligible_records": len(cards),
            "due_now": len([card for card in cards if card["due_state"] in {"due_now", "overdue"}]),
            "overdue": len([card for card in cards if card["due_state"] == "overdue"]),
            "upcoming": len([card for card in cards if card["due_state"] == "upcoming" and card["days_until_due"] <= due_window_days]),
            "not_due": len(not_due),
            "max_cards": max_cards,
        },
        "due_now": due_now,
        "overdue": overdue,
        "upcoming": upcoming,
        "retrieval_probes": probes,
        "suggested_actions": actions,
    }


def recommend_next_review(
    confidence: Any,
    importance: Any,
    age_days: Any,
    prior_reviews: int = 0,
    last_score: Any = None,
) -> dict[str, Any]:
    """Recommend a deterministic next review interval without side effects."""

    conf = _safe_float(confidence, 0.7)
    imp = _safe_float(importance, 0.5)
    age = max(0, int(_safe_float(age_days, 0)))
    reviews = max(0, int(prior_reviews or 0))
    score = None if last_score is None else _safe_float(last_score, 0.0)

    interval = 30.0
    interval *= 1.0 + (conf - 0.5) * 0.8
    interval *= 1.15 ** min(reviews, 8)
    interval *= 1.0 - (imp * 0.45)
    if age > 180:
        interval *= 0.75
    if age > 365:
        interval *= 0.65
    if score is not None:
        if score >= 0.85:
            interval *= 1.75
        elif score >= 0.55:
            interval *= 1.05
        elif score < 0.25:
            interval *= 0.35
        else:
            interval *= 0.60
    interval_days = max(1, min(365, int(round(interval))))
    return {
        "interval_days": interval_days,
        "policy": "report_only_no_mutation",
        "mutation": "none",
        "factors": {
            "confidence": conf,
            "importance": imp,
            "age_days": age,
            "prior_reviews": reviews,
            "last_score": score,
        },
    }


def score_recall_outcome(card: dict[str, Any], response: str) -> dict[str, Any]:
    """Score a recall response only when explicit expected keywords are supplied."""

    keywords = [str(word).lower() for word in (card or {}).get("expected_keywords") or [] if str(word).strip()]
    if not keywords:
        return {
            "score": None,
            "reason": "fingerprint_only_cannot_score_without_keywords",
            "matched_keywords": [],
            "policy": "report_only_no_mutation",
            "mutation": "none",
        }
    text = str(response or "").lower()
    matched = [word for word in keywords if re.search(r"\b" + re.escape(word) + r"\b", text)]
    score = len(matched) / len(keywords) if keywords else None
    return {
        "score": score,
        "matched_keywords": matched,
        "expected_keyword_count": len(keywords),
        "policy": "report_only_no_mutation",
        "mutation": "none",
    }


def _build_card(record: Any, *, kind: str, now_dt: datetime, logs: list[dict[str, Any]]) -> dict[str, Any]:
    record_id = _record_id(record)
    updated = _record_updated_at(record)
    age_days = max(0, (now_dt - updated).days) if updated else 9999
    last_review = _latest_review(logs)
    last_review_at = _parse_iso(last_review.get("reviewed_at") or last_review.get("retrieved_at") or last_review.get("created_at")) if last_review else None
    last_score = last_review.get("score") if last_review else None
    prior_reviews = len(logs)
    rec = recommend_next_review(
        confidence=_record_confidence(record),
        importance=_record_importance(record, kind),
        age_days=age_days,
        prior_reviews=prior_reviews,
        last_score=last_score,
    )
    anchor = last_review_at or updated or now_dt
    elapsed = max(0, (now_dt - anchor).days)
    interval = int(rec["interval_days"])
    days_until_due = interval - elapsed
    due_state = "not_due"
    if days_until_due < 0:
        due_state = "overdue"
    elif days_until_due == 0 or _priority_boost(kind, record) >= 0.30 or age_days >= interval:
        due_state = "due_now"
        days_until_due = min(days_until_due, 0)
    elif days_until_due <= 30:
        due_state = "upcoming"

    reasons = _review_reasons(record, kind=kind, age_days=age_days, source_count=len(_source_refs(record)), days_until_due=days_until_due)
    priority = _priority_score(record, kind=kind, age_days=age_days, due_state=due_state, reasons=reasons)
    record_ref = _stable_key("record", f"{kind}:{record_id}")
    return {
        "card_id": _stable_key("card", f"{kind}:{record_id}"),
        "id": record_ref,
        "record_ref": record_ref,
        "record_id_sha256": hashlib.sha256(str(record_id or "").encode("utf-8")).hexdigest(),
        "kind": kind,
        "type": _record_type(record, kind),
        "status": _record_status(record, kind),
        "subject_key": _subject_key(record, kind),
        "answer_fingerprint": _answer_fingerprint(record, kind),
        "source_refs": _source_refs(record),
        "source_count": len(_source_refs(record)),
        "confidence": _record_confidence(record),
        "importance": _record_importance(record, kind),
        "age_days": age_days,
        "interval_days": interval,
        "prior_reviews": prior_reviews,
        "last_reviewed_at": _iso(last_review_at) if last_review_at else "",
        "last_score": None if last_score is None else _safe_float(last_score, 0.0),
        "days_until_due": days_until_due,
        "due_state": due_state,
        "review_reasons": reasons,
        "priority": priority,
        "mutation": "none",
    }


def _eligible(kind: str, record: Any) -> bool:
    status = _record_status(record, kind)
    if kind == "memory_item":
        return status in _ACTIVE_MEMORY_STATUSES
    if kind == "project_card":
        return status in _ACTIVE_PROJECT_STATUSES
    if kind == "candidate":
        return status in _PENDING_CANDIDATE_STATUSES
    if kind == "open_loop":
        return status in {"open", ""}
    return False


def _review_reasons(record: Any, *, kind: str, age_days: int, source_count: int, days_until_due: int) -> list[str]:
    reasons = []
    if age_days >= 90:
        reasons.append("stale")
    if _record_importance(record, kind) >= 0.75:
        reasons.append("high_importance")
    if kind in {"project_card", "open_loop"}:
        reasons.append("project_or_open_loop_relevance")
    if _record_confidence(record) < 0.60 or _record_status(record, kind) == "uncertain":
        reasons.append("low_or_uncertain_confidence")
    if source_count == 0:
        reasons.append("source_weak")
    if days_until_due < 0:
        reasons.append("overdue")
    if _expires_soon(record):
        reasons.append("expiring")
    return reasons or ["scheduled_review"]


def _priority_score(record: Any, *, kind: str, age_days: int, due_state: str, reasons: list[str]) -> float:
    score = _record_importance(record, kind) * 100 + _record_confidence(record) * 20 + min(age_days, 730) / 10
    score += _priority_boost(kind, record) * 100
    score += {"overdue": 40, "due_now": 25, "upcoming": 5}.get(due_state, 0)
    score += len(reasons) * 2
    return round(score, 4)


def _priority_boost(kind: str, record: Any) -> float:
    if kind == "open_loop":
        return 0.55
    if kind == "project_card":
        return 0.45
    if kind == "candidate":
        return 0.30
    return 0.0


def _probe(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "card_id": card["card_id"],
        "target": card["id"],
        "kind": card["kind"],
        "subject_key": card["subject_key"],
        "answer_fingerprint": card["answer_fingerprint"],
        "prompt": "Recall the current safe answer for this hashed memory subject using source-grounded evidence only.",
        "untrusted_text": True,
        "mutation": "none",
    }


def _action(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": "review_active_recall_card",
        "target": card["id"],
        "card_id": card["card_id"],
        "kind": card["kind"],
        "reason_codes": list(card["review_reasons"]),
        "policy": "report_only_no_mutation",
        "mutation": "none",
    }


def _sort_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kind_rank = {"open_loop": 0, "project_card": 1, "candidate": 2, "memory_item": 3}
    due_rank = {"overdue": 0, "due_now": 1, "upcoming": 2, "not_due": 3}
    return sorted(cards, key=lambda c: (kind_rank.get(c["kind"], 9), due_rank.get(c["due_state"], 9), -float(c["priority"]), c["id"]))


def _logs_by_record(logs: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in logs or []:
        log = dict(raw or {})
        record_id = str(log.get("record_id") or log.get("id") or log.get("target") or "")
        if not record_id:
            continue
        grouped.setdefault(record_id, []).append(log)
    for values in grouped.values():
        values.sort(key=lambda item: str(item.get("reviewed_at") or item.get("retrieved_at") or item.get("created_at") or ""))
    return grouped


def _latest_review(logs: list[dict[str, Any]]) -> dict[str, Any]:
    return logs[-1] if logs else {}


def _record_id(record: Any) -> str:
    return str(_get(record, "id", "") or "")


def _record_status(record: Any, kind: str) -> str:
    if kind == "candidate":
        value = _get(record, "gate_decision", "pending")
    else:
        value = _get(record, "status", "open" if kind == "open_loop" else "active")
    return str(getattr(value, "value", value) or "").lower()


def _record_type(record: Any, kind: str) -> str:
    if kind == "project_card":
        return "project_state"
    if kind == "open_loop":
        return "open_loop"
    value = _get(record, "type", kind)
    return str(getattr(value, "value", value) or kind)


def _record_confidence(record: Any) -> float:
    return _safe_float(_get(record, "confidence", 0.7), 0.7)


def _record_importance(record: Any, kind: str) -> float:
    default = 0.75 if kind in {"project_card", "open_loop"} else 0.5
    return _safe_float(_get(record, "importance", default), default)


def _record_updated_at(record: Any) -> datetime | None:
    for key in ("updated_at", "created_at", "valid_from"):
        parsed = _parse_iso(_get(record, key, ""))
        if parsed is not None:
            return parsed
    return None


def _source_refs(record: Any) -> list[str]:
    return [str(ref) for ref in (_get(record, "source_refs", []) or [])]


def _subject_key(record: Any, kind: str) -> str:
    if kind == "project_card":
        text = _get(record, "id", "") or _get(record, "name", "")
    elif kind == "open_loop":
        text = _get(record, "id", "")
    else:
        text = _get(record, "subject", "") or _get(record, "id", "")
    normalized = " ".join(str(text or "").lower().split())
    return _stable_key("subject", normalized)


def _answer_fingerprint(record: Any, kind: str) -> str:
    fields = {
        "memory_item": ("value", "body", "summary", "predicate"),
        "project_card": ("current_state", "goal", "why_it_matters"),
        "candidate": ("claim",),
        "open_loop": ("text",),
    }.get(kind, ("id",))
    text = "\n".join(str(_get(record, field, "") or "") for field in fields)
    if kind == "project_card":
        text += "\n" + "\n".join(str(v) for v in (_get(record, "decisions", []) or []))
        text += "\n" + "\n".join(str(v) for v in (_get(record, "open_questions", []) or []))
        text += "\n" + "\n".join(str(v) for v in (_get(record, "next_actions", []) or []))
    normalized = " ".join(text.lower().split())
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _expires_soon(record: Any) -> bool:
    exp = _parse_iso(_get(record, "expires_at", ""))
    return exp is not None


def _get(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return number


def _parse_iso(value: str | datetime | None) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        if len(text) == 10 and text.count("-") == 2:
            text += "T00:00:00Z"
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stable_key(prefix: str, value: str) -> str:
    return f"{prefix}:" + hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:16]
