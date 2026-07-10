"""Report-safe serialization helpers for Memory v2 outward-facing payloads.

These helpers are intentionally conservative. They preserve metadata that is
useful for dashboards/reports (ids, hashes, counts, statuses, lanes,
timestamps, type labels) while replacing raw text/path/source-ref values with
stable metadata or fingerprints.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

_SECRET_RE = re.compile(
    r"(?i)(?:\b(?:api[_-]?key|secret|token|password|credential|bearer)\b|sk-[a-z0-9_-]{8,}|ghp_[a-z0-9_]{8,}|xox[baprs]-[a-z0-9-]{8,})"
)
_LOCAL_PATH_RE = re.compile(
    r"(?i)(?:^~[/\\]|^/home/[^\s:;,]+|^/mnt/[a-z]/users/[^\s:;,]+|[a-z]:\\users\\[^\s:;,]+|file://(?:/home/|/mnt/[a-z]/users/|/[a-z]:/users/))"
)
_WINDOWS_PATH_RE = re.compile(r"(?i)[a-z]:\\")

_TEXT_KEYS = {
    "claim",
    "normalized_claim",
    "untrusted_claim",
    "text",
    "open_loop_text",
    "body",
    "summary",
    "value",
    "raw_value",
    "subject",
    "predicate",
    "quote",
    "content",
    "user_content",
    "assistant_content",
    "statement",
    "goal",
    "current_state",
    "resolution",
    "description",
    "title",
    "name",
}
_PATH_KEYS = {
    "path",
    "base_dir",
    "source_uri",
    "uri",
    "raw_ref",
    "local_path",
    "windows_path",
}
_RELATIVE_ARTIFACT_PATH_KEYS = {"report_path", "daily_episode_path", "dream_episode_path"}
_SOURCE_REF_KEYS = {"source_id", "source_ref", "source_refs", "missing_source_refs"}
_RELATIVE_DESTINATION_KEYS = {"proposed_destination", "destination"}

# Metadata string keys that can remain literal unless their values look like a
# local path or secret. Keep this intentionally small and obvious.
_SAFE_STRING_KEYS = {
    "id",
    "candidate_id",
    "canonical_candidate_id",
    "memory_id",
    "record_id",
    "record_type",
    "operation_id",
    "action_id",
    "plan_id",
    "run_id",
    "conflict_id",
    "superseded_id",
    "superseded_by",
    "left_id",
    "right_id",
    "provider",
    "platform",
    "kind",
    "mode",
    "type",
    "candidate_memory_type",
    "category",
    "status",
    "review_lane",
    "lane",
    "policy",
    "recommendation_policy",
    "auto_apply",
    "classification",
    "proposed_action",
    "operation",
    "action",
    "mutation",
    "date",
    "created_at",
    "updated_at",
    "observed_at",
    "checked_at",
    "expires_at",
    "latest_source_at",
    "age_bucket",
    "bucket",
    "origin",
    "privacy_level",
    "freshness_class",
    "retention_policy",
    "processing_status",
    "health_status",
    "safe_default",
    "module",
}


def report_safe_serialize(value: Any, *, max_depth: int = 32) -> Any:
    """Return a deterministic JSON-safe metadata-only representation.

    The function handles dataclasses, dicts, lists/tuples, ``None``, and scalar
    values. It is designed for outward-facing Memory v2 reports/tool responses,
    not for canonical storage.
    """

    return _serialize(value, key=None, depth=0, max_depth=max_depth)


def _serialize(value: Any, *, key: str | None, depth: int, max_depth: int) -> Any:
    if depth > max_depth:
        return {"truncated": True, "depth": depth}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Enum):
        return _serialize(value.value, key=key, depth=depth, max_depth=max_depth)
    if isinstance(value, Path):
        return _string_metadata(str(value), kind="path")
    if is_dataclass(value):
        return _serialize(asdict(value), key=key, depth=depth + 1, max_depth=max_depth)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _serialize(value.to_dict(), key=key, depth=depth + 1, max_depth=max_depth)
        except Exception:
            pass
    if isinstance(value, Mapping):
        return _serialize_mapping(value, depth=depth, max_depth=max_depth)
    if isinstance(value, (list, tuple, set, frozenset)):
        if _is_source_ref_key(key):
            refs = [str(item) for item in value]
            return {
                "source_ref_count": len(refs),
                "source_ref_fingerprints": [_fingerprint(ref) for ref in refs],
            }
        return [_serialize(item, key=key, depth=depth + 1, max_depth=max_depth) for item in value]
    if isinstance(value, str):
        return _serialize_string(value, key=key)
    return _serialize_string(str(value), key=key)


def _serialize_mapping(mapping: Mapping[Any, Any], *, depth: int, max_depth: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw_key in sorted(mapping.keys(), key=lambda item: str(item)):
        safe_key = _safe_mapping_key(str(raw_key))
        value = mapping[raw_key]
        if _is_source_ref_key(safe_key):
            if isinstance(value, Mapping) and "source_ref_count" in value and "source_ref_fingerprints" in value:
                out[safe_key] = _serialize(value, key=safe_key, depth=depth + 1, max_depth=max_depth)
                continue
            if isinstance(value, (list, tuple, set, frozenset)):
                refs = [str(item) for item in value]
            elif value in (None, ""):
                refs = []
            else:
                refs = [str(value)]
            out[safe_key] = {
                "source_ref_count": len(refs),
                "source_ref_fingerprints": [_fingerprint(ref) for ref in refs],
            }
            continue
        out[safe_key] = _serialize(value, key=safe_key, depth=depth + 1, max_depth=max_depth)
    return out


def _serialize_string(value: str, *, key: str | None) -> Any:
    text = str(value)
    if _normalize_key(key) in _RELATIVE_ARTIFACT_PATH_KEYS and not (_looks_secret(text) or _looks_like_local_path(text)):
        return text
    if _normalize_key(key) in _RELATIVE_DESTINATION_KEYS and not (_looks_secret(text) or _looks_like_local_path(text)):
        return text
    if _must_summarize_string(key, text):
        kind = "text"
        if _is_path_key(key) or _looks_like_local_path(text):
            kind = "path"
        elif _looks_secret(text):
            kind = "secret_like"
        return _string_metadata(text, kind=kind)
    if key in _SAFE_STRING_KEYS or _looks_like_hash_or_fingerprint(text):
        return text
    # Unknown report strings are data, not display text. Preserve deterministic
    # metadata only unless the value is an obviously tiny enum-ish label.
    if _looks_like_label(text):
        return text
    return _string_metadata(text, kind="text")


def _must_summarize_string(key: str | None, value: str) -> bool:
    return (
        _is_text_key(key)
        or _is_path_key(key)
        or _is_source_ref_key(key)
        or _looks_secret(value)
        or _looks_like_local_path(value)
    )


def _string_metadata(value: str, *, kind: str) -> dict[str, Any]:
    text = str(value or "")
    return {
        "kind": kind,
        "present": bool(text),
        "chars": len(text),
        "sha256": _sha256(text),
    }


def _is_text_key(key: str | None) -> bool:
    normalized = _normalize_key(key)
    return normalized in _TEXT_KEYS or normalized.endswith("_text") or normalized.endswith("_claim") or normalized.endswith("_value")


def _is_path_key(key: str | None) -> bool:
    normalized = _normalize_key(key)
    return normalized in _PATH_KEYS or normalized.endswith("_path") or normalized.endswith("_uri") or normalized.endswith("_ref")


def _is_source_ref_key(key: str | None) -> bool:
    normalized = _normalize_key(key)
    return normalized in _SOURCE_REF_KEYS or normalized.endswith("source_refs") or normalized.endswith("source_ref")


def _normalize_key(key: str | None) -> str:
    return str(key or "").strip().lower()


def _looks_secret(value: str) -> bool:
    return bool(_SECRET_RE.search(str(value or "")))


def _looks_like_local_path(value: str) -> bool:
    text = str(value or "")
    return bool(_LOCAL_PATH_RE.search(text) or _WINDOWS_PATH_RE.search(text))


def _safe_mapping_key(key: str) -> str:
    text = str(key or "")
    if _looks_secret(text) or _looks_like_local_path(text):
        return "key:" + _fingerprint(text)
    return text


def _looks_like_hash_or_fingerprint(value: str) -> bool:
    text = str(value or "")
    return bool(re.fullmatch(r"(?:sha256:)?[a-f0-9]{16,64}", text))


def _looks_like_label(value: str) -> bool:
    text = str(value or "")
    return bool(re.fullmatch(r"[A-Za-z0-9_:.+-]{0,64}", text))


def _sha256(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _fingerprint(value: str) -> str:
    return "sha256:" + _sha256(value)[:16]
