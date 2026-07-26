"""Owner-labeled, private diagnostic sets for Memory v2 bottleneck analysis."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from ..shadow_reranker import (
    ShadowRerankerConfig,
    ShadowUtilityReranker,
    compute_shadow_metrics,
)


SCHEMA_VERSION = "memory-v2-owner-diagnostic-set/v1"
SUMMARY_SCHEMA_VERSION = "memory-v2-owner-diagnostic-summary/v1"
SCORE_SCHEMA_VERSION = "memory-v2-owner-diagnostic-score/v1"
QUERY_CLASSES = (
    "historical_state",
    "source_rationale",
    "project_resumption",
    "current_state",
    "open_loop",
    "procedure",
    "correction_contradiction",
    "suppression_abstention",
)
ORACLE_COMPONENTS = (
    "archive_extraction",
    "candidate_recall",
    "routing",
    "ranking",
    "temporal_resolution",
    "packet_composition",
    "answer_synthesis",
)
_MEMORY_CUE = re.compile(
    r"\b(?:previous(?:ly)?|earlier|last time|history|remember|recall|"
    r"continue|resume|pick up|where (?:were|are) we|current state|"
    r"status|what(?:'s| is) next|next steps?|look at .* state|"
    r"why did|rationale|decide|correction|wrong|changed)\b",
    re.IGNORECASE,
)
_QUERY_CLASS_PATTERNS = (
    ("historical_state", re.compile(r"\b(?:previous|earlier|last time|history|used to)\b", re.I)),
    ("source_rationale", re.compile(r"\b(?:why did|rationale|reason|decide)\b", re.I)),
    ("project_resumption", re.compile(r"\b(?:continue|resume|pick up|where were we)\b", re.I)),
    ("current_state", re.compile(r"\b(?:current state|status|where are we|latest)\b", re.I)),
    ("open_loop", re.compile(r"\b(?:what(?:'s| is) next|next steps?|remaining)\b", re.I)),
    ("procedure", re.compile(r"\b(?:how did|procedure|steps did)\b", re.I)),
    ("correction_contradiction", re.compile(r"\b(?:correct|wrong|contradict|changed)\b", re.I)),
)
_LABEL_KEYS = {
    "status",
    "memory_needed",
    "required_source_refs",
    "useful_candidate_ids",
    "stale_candidate_ids",
    "irrelevant_candidate_ids",
    "minimal_bundle_candidate_ids",
    "current_answer_success",
    "citation_correct",
    "oracle_assessments",
    "notes",
}
_ORACLE_LABELS = {"pending", "rescues", "does_not_rescue", "not_applicable"}


class DiagnosticSetError(ValueError):
    """Raised when a diagnostic packet violates its strict private contract."""


ShadowRunner = Callable[
    [str, Sequence[Mapping[str, Any]], str, float],
    Mapping[str, Any],
]


def prepare_diagnostic_set(
    events: Sequence[Mapping[str, Any]],
    *,
    shadow_runner: ShadowRunner,
    created_at: str,
    episode_count: int = 30,
    control_count: int = 8,
    held_out_count: int = 10,
    participant_id: str = "single-participant-owner",
) -> dict[str, Any]:
    """Prepare an immutable private packet with pending owner labels."""

    _timestamp(created_at, "created_at")
    if not 12 <= int(episode_count) <= 100:
        raise DiagnosticSetError("episode_count must be between 12 and 100")
    if not 1 <= int(control_count) < int(episode_count):
        raise DiagnosticSetError("control_count must be within the episode count")
    if not 1 <= int(held_out_count) < int(episode_count):
        raise DiagnosticSetError("held_out_count must be within the episode count")
    normalized = _normalize_events(events)
    selected = _select_events(
        normalized,
        episode_count=int(episode_count),
        control_count=int(control_count),
    )
    held_out_ids = {
        row["id"]
        for row in selected[-int(held_out_count) :]
    }
    archive_digest = _digest(
        [
            {
                key: value
                for key, value in event.items()
                if key != "_observed"
            }
            for event in normalized
        ]
    )
    episodes: list[dict[str, Any]] = []
    for ordinal, query_event in enumerate(selected, start=1):
        cutoff = query_event["_observed"]
        evidence = [
            _private_event(event)
            for event in normalized
            if event["_observed"] < cutoff
            and event["id"] != query_event["id"]
        ]
        query = str(query_event["user_content"])
        gap_days = _gap_days(normalized, query_event)
        shadow = dict(
            shadow_runner(
                query,
                evidence,
                cutoff.isoformat(),
                gap_days,
            )
        )
        reranker_result = shadow.get("result")
        if not isinstance(reranker_result, Mapping):
            raise DiagnosticSetError("shadow runner did not return a reranker result")
        baseline = result_at_threshold(reranker_result, threshold=0.45)
        candidates = list(reranker_result.get("ranked_candidates") or [])
        episode_hash = hashlib.sha256(
            f"{query_event['id']}\0{cutoff.isoformat()}".encode()
        ).hexdigest()[:16]
        control_candidate = not bool(_MEMORY_CUE.search(query))
        episodes.append(
            {
                "episode_id": f"diagnostic-{ordinal:02d}-{episode_hash}",
                "split": (
                    "held_out"
                    if query_event["id"] in held_out_ids
                    else "development"
                ),
                "control_candidate": control_candidate,
                "query_event_id": query_event["id"],
                "session_id": query_event["session_id"],
                "query_text": query,
                "query_class": (
                    "suppression_abstention"
                    if control_candidate
                    else _query_class(query)
                ),
                "evidence_cutoff": cutoff.isoformat(),
                "gap_days": gap_days,
                "candidate_ids": [str(row.get("id") or "") for row in candidates],
                "candidate_source_refs": sorted(
                    {
                        str(source_ref)
                        for row in candidates
                        for source_ref in row.get("source_refs", [])
                    }
                ),
                "diagnostic_candidates": candidates,
                "filter_counts": dict(reranker_result.get("filter_counts") or {}),
                "filter_reason_counts": dict(
                    reranker_result.get("filter_reason_counts") or {}
                ),
                "baseline_result": baseline,
                "owner_labels": _pending_labels(),
            }
        )
    packet = {
        "schema_version": SCHEMA_VERSION,
        "study_mode": "development",
        "created_at": created_at,
        "participant_id": participant_id,
        "mutation_authority": "none",
        "live_profile_modified": False,
        "source_archive": {
            "sha256": f"sha256:{archive_digest}",
            "event_count": len(normalized),
            "first_event_at": normalized[0]["_observed"].isoformat(),
            "last_event_at": normalized[-1]["_observed"].isoformat(),
        },
        "split_policy": {
            "strategy": "chronological_tail_holdout",
            "development_count": int(episode_count) - int(held_out_count),
            "held_out_count": int(held_out_count),
            "held_out_frozen": True,
            "tuning_on_held_out_prohibited": True,
        },
        "selection": {
            "episode_count": int(episode_count),
            "control_candidate_count": int(control_count),
            "memory_cued_candidate_count": int(episode_count) - int(control_count),
        },
        "episodes": episodes,
    }
    return validate_diagnostic_set(packet)


def validate_diagnostic_set(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise DiagnosticSetError("diagnostic set must be a mapping")
    expected = {
        "schema_version",
        "study_mode",
        "created_at",
        "participant_id",
        "mutation_authority",
        "live_profile_modified",
        "source_archive",
        "split_policy",
        "selection",
        "episodes",
    }
    if set(raw) != expected:
        raise DiagnosticSetError("diagnostic set does not match its strict schema")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise DiagnosticSetError("diagnostic set schema version is invalid")
    if raw["study_mode"] != "development":
        raise DiagnosticSetError("diagnostic set must remain development-only")
    if raw["mutation_authority"] != "none" or raw["live_profile_modified"] is not False:
        raise DiagnosticSetError("diagnostic set cannot have mutation authority")
    _timestamp(raw["created_at"], "created_at")
    _bounded_text(raw["participant_id"], "participant_id", maximum=200)
    source = _strict_mapping(
        raw["source_archive"],
        {"sha256", "event_count", "first_event_at", "last_event_at"},
        "source_archive",
    )
    _digest_text(source["sha256"], "source_archive.sha256")
    _positive_int(source["event_count"], "source_archive.event_count")
    _timestamp(source["first_event_at"], "source_archive.first_event_at")
    _timestamp(source["last_event_at"], "source_archive.last_event_at")
    split = _strict_mapping(
        raw["split_policy"],
        {
            "strategy",
            "development_count",
            "held_out_count",
            "held_out_frozen",
            "tuning_on_held_out_prohibited",
        },
        "split_policy",
    )
    if (
        split["strategy"] != "chronological_tail_holdout"
        or split["held_out_frozen"] is not True
        or split["tuning_on_held_out_prohibited"] is not True
    ):
        raise DiagnosticSetError("held-out split policy is not frozen")
    selection = _strict_mapping(
        raw["selection"],
        {
            "episode_count",
            "control_candidate_count",
            "memory_cued_candidate_count",
        },
        "selection",
    )
    episodes = raw["episodes"]
    if not isinstance(episodes, list) or not episodes:
        raise DiagnosticSetError("episodes must be a nonempty list")
    if len(episodes) != selection["episode_count"]:
        raise DiagnosticSetError("episode count does not match selection")
    validated = [_validate_episode(row, index) for index, row in enumerate(episodes)]
    ids = [row["episode_id"] for row in validated]
    if len(ids) != len(set(ids)):
        raise DiagnosticSetError("episode ids must be unique")
    development = [row for row in validated if row["split"] == "development"]
    held_out = [row for row in validated if row["split"] == "held_out"]
    if len(development) != split["development_count"]:
        raise DiagnosticSetError("development split count is inconsistent")
    if len(held_out) != split["held_out_count"]:
        raise DiagnosticSetError("held-out split count is inconsistent")
    controls = sum(bool(row["control_candidate"]) for row in validated)
    if controls != selection["control_candidate_count"]:
        raise DiagnosticSetError("control candidate count is inconsistent")
    if selection["memory_cued_candidate_count"] != len(validated) - controls:
        raise DiagnosticSetError("memory-cued candidate count is inconsistent")
    return copy.deepcopy(dict(raw))


def public_summary(packet: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_diagnostic_set(packet)
    episodes = value["episodes"]
    labels = Counter(row["owner_labels"]["status"] for row in episodes)
    reasons: Counter[str] = Counter()
    filters: Counter[str] = Counter()
    for row in episodes:
        reasons.update(row["filter_reason_counts"])
        filters.update(row["filter_counts"])
    labels_complete = all(
        row["owner_labels"]["status"] == "complete"
        for row in episodes
    ) and all(
        assessment != "pending"
        for row in episodes
        for assessment in row["owner_labels"]["oracle_assessments"].values()
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "study_mode": "development",
        "participant_count": 1,
        "mutation_authority": "none",
        "live_profile_modified": False,
        "source_archive": dict(value["source_archive"]),
        "split_policy": dict(value["split_policy"]),
        "selection": dict(value["selection"]),
        "label_status_counts": dict(sorted(labels.items())),
        "filter_totals": dict(sorted(filters.items())),
        "filter_reason_totals": dict(sorted(reasons.items())),
        "owner_labels_complete": labels_complete,
        "outcome_replay_ready": False,
        "outcome_replay_blockers": [
            *([] if labels_complete else ["pending_owner_labels"]),
            "frozen_offline_variant_receipts",
        ],
        "eligible_for_pilot_go": False,
    }


def score_labeled_set(
    packet: Mapping[str, Any],
    *,
    split: str = "development",
) -> dict[str, Any]:
    value = validate_diagnostic_set(packet)
    if split not in {"development", "held_out", "all"}:
        raise DiagnosticSetError("score split must be development, held_out, or all")
    selected = [
        row
        for row in value["episodes"]
        if split == "all" or row["split"] == split
    ]
    pending = [
        row["episode_id"]
        for row in selected
        if row["owner_labels"]["status"] != "complete"
    ]
    if pending:
        raise DiagnosticSetError(
            f"{len(pending)} selected episodes still require owner labels"
        )
    records = []
    for row in selected:
        labels = row["owner_labels"]
        records.append(
            {
                "expected_memory_decision": (
                    "needed" if labels["memory_needed"] else "none"
                ),
                "useful_candidate_ids": list(labels["useful_candidate_ids"]),
                "required_source_refs": list(labels["required_source_refs"]),
                "valid_source_refs": list(labels["required_source_refs"]),
                "stale_candidate_ids": list(labels["stale_candidate_ids"]),
                "temporal_mode": "current",
                "result": copy.deepcopy(row["baseline_result"]),
            }
        )
    oracle_counts = {
        component: Counter(
            row["owner_labels"]["oracle_assessments"][component]
            for row in selected
        )
        for component in ORACLE_COMPONENTS
    }
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "study_mode": "development",
        "split": split,
        "episode_count": len(selected),
        "shadow_metrics": compute_shadow_metrics(records),
        "oracle_assessment_counts": {
            component: dict(sorted(counts.items()))
            for component, counts in oracle_counts.items()
        },
        "owner_labels_complete": True,
        "outcome_replay_ready": False,
        "outcome_replay_blockers": ["frozen_offline_variant_receipts"],
        "eligible_for_pilot_go": False,
        "mutation_authority": "none",
    }


def result_at_threshold(
    result: Mapping[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    if not 0.0 <= float(threshold) <= 1.0:
        raise DiagnosticSetError("threshold must be within [0, 1]")
    value = copy.deepcopy(dict(result))
    ranked = [
        row
        for row in value.get("ranked_candidates", [])
        if float(row.get("utility_score", 0.0)) >= float(threshold)
    ]
    reranker = ShadowUtilityReranker(
        ShadowRerankerConfig(
            enabled=True,
            minimum_utility=float(threshold),
            max_bundle_items=int(value.get("bundle", {}).get("max_items", 5)),
        )
    )
    bundle = reranker._compose_bundle(ranked)
    value["ranked_candidates"] = ranked
    value["bundle"] = bundle
    value["decision"] = {
        "memory_needed": bool(bundle["items"]),
        "selected": "candidates" if bundle["items"] else "none",
        "reason": "useful_candidates" if bundle["items"] else "below_utility_or_no_slot",
    }
    return value


def _normalize_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise DiagnosticSetError("events must be a sequence")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in events:
        if not isinstance(raw, Mapping):
            raise DiagnosticSetError("every event must be a mapping")
        event_id = _bounded_text(raw.get("id"), "event id", maximum=200)
        if event_id in seen:
            raise DiagnosticSetError("event ids must be unique")
        seen.add(event_id)
        observed = _event_time(raw)
        row = dict(raw)
        row["id"] = event_id
        row["session_id"] = _bounded_text(
            raw.get("session_id") or raw.get("provider_session_id") or event_id,
            "session id",
            maximum=300,
        )
        row["_observed"] = observed
        normalized.append(row)
    if not normalized:
        raise DiagnosticSetError("events must be nonempty")
    return sorted(normalized, key=lambda row: (row["_observed"], row["id"]))


def _select_events(
    events: Sequence[Mapping[str, Any]],
    *,
    episode_count: int,
    control_count: int,
) -> list[dict[str, Any]]:
    turns = [
        dict(row)
        for row in events
        if str(row.get("type") or "") == "turn"
        and 20 <= len(str(row.get("user_content") or "").strip()) <= 1_500
    ]
    memory = [
        row
        for row in turns
        if _MEMORY_CUE.search(str(row["user_content"]))
    ]
    controls = [
        row
        for row in turns
        if not _MEMORY_CUE.search(str(row["user_content"]))
    ]
    memory_count = episode_count - control_count
    if len(memory) < memory_count or len(controls) < control_count:
        raise DiagnosticSetError(
            "archive does not contain enough memory-cued and control candidates "
            f"(memory_cued={len(memory)}, controls={len(controls)})"
        )
    selected_controls = _evenly_spaced(controls, control_count)
    control_sessions = {row["session_id"] for row in selected_controls}
    distinct_memory = [
        row for row in memory if row["session_id"] not in control_sessions
    ]
    pool = distinct_memory if len(distinct_memory) >= memory_count else memory
    selected_memory = _evenly_spaced(pool, memory_count)
    selected = [*selected_memory, *selected_controls]
    if len({row["id"] for row in selected}) != episode_count:
        raise DiagnosticSetError("episode selection produced duplicate events")
    return sorted(selected, key=lambda row: (row["_observed"], row["id"]))


def _evenly_spaced(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = [
        math.floor(index * (len(rows) - 1) / (count - 1))
        for index in range(count)
    ]
    return [rows[index] for index in indices]


def _validate_episode(raw: Any, index: int) -> dict[str, Any]:
    where = f"episodes[{index}]"
    expected = {
        "episode_id",
        "split",
        "control_candidate",
        "query_event_id",
        "session_id",
        "query_text",
        "query_class",
        "evidence_cutoff",
        "gap_days",
        "candidate_ids",
        "candidate_source_refs",
        "diagnostic_candidates",
        "filter_counts",
        "filter_reason_counts",
        "baseline_result",
        "owner_labels",
    }
    value = _strict_mapping(raw, expected, where)
    _bounded_text(value["episode_id"], f"{where}.episode_id", maximum=200)
    if value["split"] not in {"development", "held_out"}:
        raise DiagnosticSetError(f"{where}.split is invalid")
    if not isinstance(value["control_candidate"], bool):
        raise DiagnosticSetError(f"{where}.control_candidate must be boolean")
    _bounded_text(value["query_event_id"], f"{where}.query_event_id", maximum=200)
    _bounded_text(value["session_id"], f"{where}.session_id", maximum=300)
    _bounded_text(value["query_text"], f"{where}.query_text", maximum=1_500)
    if value["query_class"] not in QUERY_CLASSES:
        raise DiagnosticSetError(f"{where}.query_class is invalid")
    _timestamp(value["evidence_cutoff"], f"{where}.evidence_cutoff")
    if (
        isinstance(value["gap_days"], bool)
        or not isinstance(value["gap_days"], (int, float))
        or not math.isfinite(float(value["gap_days"]))
        or float(value["gap_days"]) < 0
    ):
        raise DiagnosticSetError(f"{where}.gap_days is invalid")
    candidate_ids = _string_list(
        value["candidate_ids"], f"{where}.candidate_ids", maximum=30
    )
    _string_list(
        value["candidate_source_refs"],
        f"{where}.candidate_source_refs",
        maximum=360,
    )
    candidates = value["diagnostic_candidates"]
    if not isinstance(candidates, list) or len(candidates) != len(candidate_ids):
        raise DiagnosticSetError(f"{where}.diagnostic_candidates are inconsistent")
    actual_ids = [str(row.get("id") or "") for row in candidates if isinstance(row, Mapping)]
    if actual_ids != candidate_ids:
        raise DiagnosticSetError(f"{where}.candidate ids are inconsistent")
    for name in ("filter_counts", "filter_reason_counts"):
        counts = value[name]
        if not isinstance(counts, Mapping) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in counts.values()
        ):
            raise DiagnosticSetError(f"{where}.{name} is invalid")
    labels = _validate_labels(
        value["owner_labels"],
        candidate_ids=set(candidate_ids),
        source_refs=set(value["candidate_source_refs"]),
        where=f"{where}.owner_labels",
    )
    result_at_threshold(value["baseline_result"], threshold=0.45)
    result = copy.deepcopy(dict(value))
    result["owner_labels"] = labels
    return result


def _validate_labels(
    raw: Any,
    *,
    candidate_ids: set[str],
    source_refs: set[str],
    where: str,
) -> dict[str, Any]:
    value = _strict_mapping(raw, _LABEL_KEYS, where)
    if value["status"] not in {"pending", "complete"}:
        raise DiagnosticSetError(f"{where}.status is invalid")
    oracle = _strict_mapping(
        value["oracle_assessments"],
        set(ORACLE_COMPONENTS),
        f"{where}.oracle_assessments",
    )
    if any(label not in _ORACLE_LABELS for label in oracle.values()):
        raise DiagnosticSetError(f"{where}.oracle_assessments contain an invalid label")
    lists = {
        name: _string_list(value[name], f"{where}.{name}", maximum=360)
        for name in (
            "required_source_refs",
            "useful_candidate_ids",
            "stale_candidate_ids",
            "irrelevant_candidate_ids",
            "minimal_bundle_candidate_ids",
        )
    }
    if not set(lists["required_source_refs"]) <= source_refs:
        raise DiagnosticSetError(f"{where}.required_source_refs are absent from candidates")
    for name in (
        "useful_candidate_ids",
        "stale_candidate_ids",
        "irrelevant_candidate_ids",
        "minimal_bundle_candidate_ids",
    ):
        if not set(lists[name]) <= candidate_ids:
            raise DiagnosticSetError(f"{where}.{name} contains an unknown candidate")
    if not set(lists["minimal_bundle_candidate_ids"]) <= set(
        lists["useful_candidate_ids"]
    ):
        raise DiagnosticSetError(f"{where}.minimal bundle must be useful")
    classified = [
        set(lists["useful_candidate_ids"]),
        set(lists["stale_candidate_ids"]),
        set(lists["irrelevant_candidate_ids"]),
    ]
    if any(left & right for position, left in enumerate(classified) for right in classified[position + 1 :]):
        raise DiagnosticSetError(f"{where}.candidate labels must be disjoint")
    _bounded_text(value["notes"], f"{where}.notes", maximum=4_000, allow_empty=True)
    if value["status"] == "pending":
        if value["memory_needed"] is not None:
            raise DiagnosticSetError(f"{where}.pending memory_needed must be null")
    else:
        for name in ("memory_needed", "current_answer_success", "citation_correct"):
            if not isinstance(value[name], bool):
                raise DiagnosticSetError(f"{where}.{name} must be boolean when complete")
        if any(label == "pending" for label in oracle.values()):
            raise DiagnosticSetError(f"{where}.complete labels cannot have pending oracles")
    return copy.deepcopy(dict(value))


def _pending_labels() -> dict[str, Any]:
    return {
        "status": "pending",
        "memory_needed": None,
        "required_source_refs": [],
        "useful_candidate_ids": [],
        "stale_candidate_ids": [],
        "irrelevant_candidate_ids": [],
        "minimal_bundle_candidate_ids": [],
        "current_answer_success": None,
        "citation_correct": None,
        "oracle_assessments": {
            component: "pending"
            for component in ORACLE_COMPONENTS
        },
        "notes": "",
    }


def _private_event(event: Mapping[str, Any]) -> dict[str, Any]:
    value = {key: item for key, item in event.items() if key != "_observed"}
    value["profile_id"] = "single-participant-development"
    value["tenant_id"] = "private-local-development"
    value["privacy_level"] = "private"
    value["visibility"] = "private"
    return value


def _gap_days(
    events: Sequence[Mapping[str, Any]],
    query: Mapping[str, Any],
) -> float:
    prior = [
        row["_observed"]
        for row in events
        if row["_observed"] < query["_observed"]
        and row["session_id"] != query["session_id"]
    ]
    if not prior:
        return 0.0
    return round(
        max(
            0.0,
            (query["_observed"] - max(prior)).total_seconds() / 86_400,
        ),
        3,
    )


def _query_class(text: str) -> str:
    for name, pattern in _QUERY_CLASS_PATTERNS:
        if pattern.search(text):
            return name
    return "project_resumption"


def _event_time(event: Mapping[str, Any]) -> datetime:
    for key in ("observed_at", "created_at", "timestamp"):
        if value := str(event.get(key) or "").strip():
            return _timestamp(value, f"event {key}")
    raise DiagnosticSetError("event timestamp is required")


def _timestamp(value: Any, name: str) -> datetime:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DiagnosticSetError(f"{name} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise DiagnosticSetError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _strict_mapping(raw: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != keys:
        raise DiagnosticSetError(f"{name} does not match its strict schema")
    return raw


def _bounded_text(
    value: Any,
    name: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise DiagnosticSetError(f"{name} must be a string")
    text = value.strip()
    if (not text and not allow_empty) or len(text) > maximum:
        raise DiagnosticSetError(f"{name} must be bounded")
    return text


def _string_list(value: Any, name: str, *, maximum: int) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum:
        raise DiagnosticSetError(f"{name} must be a bounded list")
    result = []
    for item in value:
        result.append(_bounded_text(item, name, maximum=300))
    if len(result) != len(set(result)):
        raise DiagnosticSetError(f"{name} must contain unique values")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DiagnosticSetError(f"{name} must be a positive integer")
    return value


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _digest_text(value: Any, name: str) -> str:
    text = _bounded_text(value, name, maximum=71)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", text):
        raise DiagnosticSetError(f"{name} must be a sha256 digest")
    return text


__all__ = [
    "DiagnosticSetError",
    "ORACLE_COMPONENTS",
    "SCHEMA_VERSION",
    "prepare_diagnostic_set",
    "public_summary",
    "result_at_threshold",
    "score_labeled_set",
    "validate_diagnostic_set",
]
