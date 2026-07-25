"""Raw-preserving workstream resolution and outcome-evidence derivation.

This module deliberately has no store or index dependency.  It turns an
immutable snapshot of canonical raw events into rebuildable derived records;
it never writes, promotes, supersedes, or grants execution authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from itertools import islice
from pathlib import PurePath
from typing import Any, Callable, Iterable, Mapping, Sequence

from .redaction import (
    contains_sensitive_text,
    escape_untrusted_evidence_text,
)


StructuredExtractor = Callable[[dict[str, Any]], dict[str, Any]]

_NODE_KINDS = {
    "request_goal",
    "proposal",
    "decision",
    "implementation_claim",
    "verified_result",
    "correction",
    "blocker",
    "open_loop",
    "next_action",
    "artifact",
    "procedure",
}
_KIND_ALIASES = {
    "goal": "request_goal",
    "request": "request_goal",
    "result": "verified_result",
    "implementation": "implementation_claim",
}
_TEXT_FIELDS = (
    ("user_content", "user"),
    ("assistant_content", "assistant"),
    ("result", "tool"),
    ("stdout", "tool"),
    ("stderr", "tool"),
    ("content", "canonical"),
)
_EVENT_TIME_FIELDS = ("observed_at", "created_at", "timestamp")
_LINKAGE_FIELDS = (
    "claim_event_id",
    "in_reply_to_event_id",
    "linked_event_ids",
    "parent_event_id",
    "related_event_id",
    "related_event_ids",
    "reply_to_event_id",
)
_DEFAULT_MAX_ANCHORS_PER_EVENT = 256
_DEFAULT_MAX_SCOPE_IDS = 16
_DEFAULT_MAX_DERIVED_NODES = 10_000
_DEFAULT_MAX_SPAN_CHARS = 4_000
_MAX_CORROBORATION_GAP_SECONDS = 7 * 24 * 60 * 60
_MEANINGFUL_TOKEN_RE = re.compile(r"[a-z0-9]{3,64}", re.IGNORECASE)
_CORROBORATION_STOPWORDS = {
    "all",
    "and",
    "built",
    "completed",
    "done",
    "exit",
    "failed",
    "fixed",
    "implemented",
    "passed",
    "result",
    "results",
    "run",
    "success",
    "successful",
    "successfully",
    "test",
    "tests",
    "that",
    "the",
    "verified",
    "with",
}


def _clean_scalar(value: Any) -> str:
    return str(value or "").strip()


def _scalar_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        cleaned = value.strip()
        return (cleaned,) if cleaned else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(
            cleaned
            for item in value
            if (cleaned := _clean_scalar(item))
        )
    cleaned = _clean_scalar(value)
    return (cleaned,) if cleaned else ()


def _bounded_scalar_values(value: Any, limit: int) -> tuple[tuple[str, ...], bool]:
    if limit <= 0:
        return (), value not in (None, "", (), [])
    if isinstance(value, str):
        cleaned = value.strip()
        return ((cleaned,) if cleaned else ()), False
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        sampled = list(islice(value, limit + 1))
        overflowed = len(sampled) > limit
        values = tuple(
            cleaned
            for item in sampled[:limit]
            if (cleaned := _clean_scalar(item))
        )
        return values, overflowed
    cleaned = _clean_scalar(value)
    return ((cleaned,) if cleaned else ()), False


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized[:96]


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _observed_at(event: Mapping[str, Any]) -> str:
    for field_name in _EVENT_TIME_FIELDS:
        value = _clean_scalar(event.get(field_name))
        if value:
            return value
    return ""


def _parse_evidence_timestamp(value: str) -> datetime | None:
    """Parse a bounded, timezone-aware ISO timestamp into UTC."""

    raw = str(value or "").strip()
    if not raw or len(raw) > 128:
        return None
    if raw.endswith(("Z", "z")):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    try:
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return None


def _timestamp_order(value: str) -> tuple[int, datetime | str]:
    parsed = _parse_evidence_timestamp(value)
    if parsed is not None:
        return (0, parsed)
    return (1, str(value or ""))


def _event_id(event: Mapping[str, Any]) -> str:
    return _clean_scalar(event.get("id"))


def _event_order(event: Mapping[str, Any]) -> tuple[Any, ...]:
    session = _clean_scalar(
        event.get("provider_session_id") or event.get("session_id")
    ).lower()
    # Prefer root/main sessions when forked transcripts contain copied turns.
    lineage_priority = 0 if re.search(r"(?:^|[-_:])(root|main)(?:$|[-_:])", session) else 1
    if "fork" in session or "subagent" in session:
        lineage_priority = 2
    try:
        chain_index = int(event.get("chain_index") or 0)
    except (TypeError, ValueError):
        chain_index = 0
    return (
        _timestamp_order(_observed_at(event)),
        chain_index,
        lineage_priority,
        session,
        _event_id(event),
    )


def _event_texts(event: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    event_type = _clean_scalar(event.get("type")).lower()
    canonical_role = _clean_scalar(event.get("role")).lower()
    values: list[tuple[str, str, str]] = []
    for field_name, default_role in _TEXT_FIELDS:
        value = event.get(field_name)
        if not isinstance(value, str) or not value.strip():
            continue
        if field_name in {"result", "stdout", "stderr"}:
            if event_type != "tool":
                continue
            role = "tool"
        elif field_name == "content":
            if event_type == "tool" and canonical_role in {"", "tool"}:
                role = "tool"
            elif canonical_role in {"user", "assistant"}:
                role = canonical_role
            else:
                # Ambiguous generic content is never upgraded to tool evidence.
                continue
        else:
            role = default_role
        values.append((field_name, role, value))
    return values


def _normalized_copy_text(event: Mapping[str, Any]) -> str:
    parts = []
    for field_name, role, text in _event_texts(event):
        normalized = re.sub(r"\s+", " ", text).strip()
        parts.append(f"{field_name}:{role}:{normalized}")
    return "\n".join(parts)


def _copy_fingerprint(event: Mapping[str, Any]) -> str:
    observed_at = _observed_at(event)
    parsed_observed_at = _parse_evidence_timestamp(observed_at)
    return "sha256:" + _stable_digest(
        {
            "type": _clean_scalar(event.get("type")).lower(),
            "text": _normalized_copy_text(event),
            # Forked transcripts preserve the original evidence time.  The
            # timestamp prevents two intentional repetitions days apart from
            # being collapsed merely because their text is identical.
            "observed_at": (
                parsed_observed_at.isoformat()
                if parsed_observed_at is not None
                else observed_at
            ),
        }
    )


def _linkage_refs(event: Mapping[str, Any]) -> tuple[str, ...]:
    refs = {
        ref
        for field_name in _LINKAGE_FIELDS
        for ref in _scalar_values(event.get(field_name))
        if len(ref) <= 256
    }
    return tuple(sorted(refs))


@dataclass(frozen=True)
class EvidenceSpan:
    source_id: str
    field: str
    role: str
    start: int
    end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "field": self.field,
            "role": self.role,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@dataclass(frozen=True)
class ResolutionEvidence:
    kind: str
    source_id: str
    field: str
    value: str
    confidence: float
    start: int | None = None
    end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_id": self.source_id,
            "field": self.field,
            "value": self.value,
            "confidence": self.confidence,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class WorkstreamResolution:
    status: str
    project_ids: tuple[str, ...] = ()
    workstream_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    evidence: tuple[ResolutionEvidence, ...] = ()
    anchor_limit_reached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "project_ids": list(self.project_ids),
            "workstream_ids": list(self.workstream_ids),
            "confidence": self.confidence,
            "evidence": [item.to_dict() for item in self.evidence],
            "anchor_limit_reached": self.anchor_limit_reached,
        }


@dataclass(frozen=True)
class HygieneRecord:
    source_id: str
    fingerprint: str
    disposition: str
    canonical_source_id: str = ""
    copied_by: tuple[str, ...] = ()
    lineage_session_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "fingerprint": self.fingerprint,
            "disposition": self.disposition,
            "canonical_source_id": self.canonical_source_id,
            "copied_by": list(self.copied_by),
            "lineage_session_id": self.lineage_session_id,
        }


@dataclass(frozen=True)
class EvidenceNode:
    id: str
    kind: str
    text: str
    role: str
    observed_at: str
    session_id: str
    source_refs: tuple[str, ...]
    evidence_spans: tuple[EvidenceSpan, ...]
    project_ids: tuple[str, ...] = ()
    workstream_ids: tuple[str, ...] = ()
    linkage_refs: tuple[str, ...] = ()
    lineage_refs: tuple[str, ...] = ()
    extraction_method: str = "deterministic"
    verified: bool = False
    corroborated_by: tuple[str, ...] = ()
    mutation_authority: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "role": self.role,
            "observed_at": self.observed_at,
            "session_id": self.session_id,
            "source_refs": list(self.source_refs),
            "evidence_spans": [span.to_dict() for span in self.evidence_spans],
            "project_ids": list(self.project_ids),
            "workstream_ids": list(self.workstream_ids),
            "linkage_refs": list(self.linkage_refs),
            "lineage_refs": list(self.lineage_refs),
            "extraction_method": self.extraction_method,
            "verified": self.verified,
            "corroborated_by": list(self.corroborated_by),
            "mutation_authority": self.mutation_authority,
        }


@dataclass(frozen=True)
class WorkstreamEvidenceBuild:
    schema_version: str = "memory-v2-workstream-evidence/v1"
    mutation_authority: str = "none"
    hygiene: tuple[HygieneRecord, ...] = ()
    resolutions: tuple[tuple[str, WorkstreamResolution], ...] = ()
    nodes: tuple[EvidenceNode, ...] = ()
    structured_rejections: tuple[str, ...] = ()
    overflow_rejections: tuple[str, ...] = ()
    rejected_node_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mutation_authority": self.mutation_authority,
            "hygiene": [item.to_dict() for item in self.hygiene],
            "resolutions": {
                source_id: resolution.to_dict()
                for source_id, resolution in self.resolutions
            },
            "nodes": [node.to_dict() for node in self.nodes],
            "structured_rejections": list(self.structured_rejections),
            "overflow_rejections": list(self.overflow_rejections),
            "rejected_node_count": self.rejected_node_count,
        }


class CorpusHygieneClassifier:
    """Classify derived-corpus exclusions while preserving raw lineage."""

    _MACHINE_PROMPT_RE = re.compile(
        r"^\s*(?:"
        r"\[SYSTEM:\s*(?:Background process|Background command|Tool process)"
        r"|You are an agent in a team of agents"
        r"|<system(?:_message)?>"
        r"|System prompt:"
        r")",
        re.IGNORECASE,
    )
    _COMPACTION_RE = re.compile(
        r"^\s*(?:"
        r"\[CONTEXT\s+COMPACTION\s*(?:-|—|–)+\s*REFERENCE\s+ONLY\]"
        r"|"
        r"(?:summary|handoff)\s+(?:from|for)\s+(?:the\s+)?previous\s+(?:model\s+)?instance"
        r"|context\s+compaction\s+(?:summary|handoff)"
        r"|compaction\s+handoff"
        r"|previous\s+model\s+instance\s+summary"
        r")",
        re.IGNORECASE,
    )

    def classify(
        self, events: Sequence[Mapping[str, Any]]
    ) -> tuple[HygieneRecord, ...]:
        sorted_events = sorted(events, key=_event_order)
        preliminary: dict[str, HygieneRecord] = {}
        by_fingerprint: dict[str, list[Mapping[str, Any]]] = {}
        for event in sorted_events:
            source_id = _event_id(event)
            if not source_id:
                continue
            text = _normalized_copy_text(event)
            fingerprint = _copy_fingerprint(event)
            session_id = _clean_scalar(
                event.get("provider_session_id") or event.get("session_id")
            )
            if self._MACHINE_PROMPT_RE.search(self._visible_text(event)):
                preliminary[source_id] = HygieneRecord(
                    source_id,
                    fingerprint,
                    "machine_prompt",
                    lineage_session_id=session_id,
                )
                continue
            if self._COMPACTION_RE.search(self._visible_text(event)):
                preliminary[source_id] = HygieneRecord(
                    source_id,
                    fingerprint,
                    "compaction_handoff",
                    lineage_session_id=session_id,
                )
                continue
            observed_at = _observed_at(event)
            if not observed_at:
                preliminary[source_id] = HygieneRecord(
                    source_id,
                    fingerprint,
                    "missing_evidence_timestamp",
                    lineage_session_id=session_id,
                )
                continue
            if _parse_evidence_timestamp(observed_at) is None:
                preliminary[source_id] = HygieneRecord(
                    source_id,
                    fingerprint,
                    "invalid_evidence_timestamp",
                    lineage_session_id=session_id,
                )
                continue
            if not text:
                preliminary[source_id] = HygieneRecord(
                    source_id,
                    fingerprint,
                    "empty",
                    lineage_session_id=session_id,
                )
                continue
            by_fingerprint.setdefault(fingerprint, []).append(event)

        for fingerprint, copies in sorted(by_fingerprint.items()):
            ordered = sorted(copies, key=_event_order)
            canonical = ordered[0]
            canonical_id = _event_id(canonical)
            copied_by = tuple(_event_id(event) for event in ordered[1:])
            preliminary[canonical_id] = HygieneRecord(
                canonical_id,
                fingerprint,
                "keep",
                canonical_source_id=canonical_id,
                copied_by=copied_by,
                lineage_session_id=_clean_scalar(
                    canonical.get("provider_session_id")
                    or canonical.get("session_id")
                ),
            )
            for event in ordered[1:]:
                source_id = _event_id(event)
                preliminary[source_id] = HygieneRecord(
                    source_id,
                    fingerprint,
                    "copied_fork_turn",
                    canonical_source_id=canonical_id,
                    lineage_session_id=_clean_scalar(
                        event.get("provider_session_id") or event.get("session_id")
                    ),
                )
        return tuple(
            preliminary[source_id]
            for source_id in sorted(preliminary)
        )

    @staticmethod
    def _visible_text(event: Mapping[str, Any]) -> str:
        text = "\n".join(
            value for _field, _role, value in _event_texts(event)
        ).strip()
        if re.match(r"^\s*\[CONTEXT\s+COMPACTION\b", text, re.IGNORECASE):
            return text
        return re.sub(
            r"^\s*\[(?!SYSTEM:)[^\]]+\]\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )


class WorkstreamResolver:
    """Resolve project/workstream scope from explicit and deterministic anchors."""

    _PROJECT_FIELDS = ("project_id", "project", "project_name")
    _WORKSTREAM_FIELDS = ("workstream_id", "workstream", "workstream_name")
    _REPO_FIELDS = ("repo_path", "repository", "cwd", "workdir")
    _ARTIFACT_FIELDS = ("artifact_refs", "artifacts", "source_paths")
    _ARTIFACT_RE = re.compile(
        r"(?P<path>(?:[A-Za-z]:)?[/\\](?:[^ \t\r\n:;,]+[/\\])*[^ \t\r\n:;,]+"
        r"|(?:[A-Za-z0-9_.-]+[/\\])+(?:[A-Za-z0-9_.-]+))"
    )

    def __init__(
        self,
        *,
        max_anchors_per_event: int = _DEFAULT_MAX_ANCHORS_PER_EVENT,
        max_scope_ids: int = _DEFAULT_MAX_SCOPE_IDS,
        max_span_chars: int = _DEFAULT_MAX_SPAN_CHARS,
    ) -> None:
        if not 1 <= int(max_anchors_per_event) <= 10_000:
            raise ValueError("max_anchors_per_event must be between 1 and 10000")
        if not 1 <= int(max_scope_ids) <= _DEFAULT_MAX_SCOPE_IDS:
            raise ValueError(
                f"max_scope_ids must be between 1 and {_DEFAULT_MAX_SCOPE_IDS}"
            )
        if not 1 <= int(max_span_chars) <= 16_000:
            raise ValueError("max_span_chars must be between 1 and 16000")
        self.max_anchors_per_event = int(max_anchors_per_event)
        self.max_scope_ids = int(max_scope_ids)
        self.max_span_chars = int(max_span_chars)

    def resolve(self, event: Mapping[str, Any]) -> WorkstreamResolution:
        source_id = _event_id(event)
        projects: dict[str, float] = {}
        workstreams: dict[str, float] = {}
        evidence: list[ResolutionEvidence] = []
        has_explicit_project = False
        anchor_limit_reached = False

        def values(raw: Any) -> tuple[str, ...]:
            nonlocal anchor_limit_reached
            remaining = self.max_anchors_per_event - len(evidence)
            bounded, overflowed = _bounded_scalar_values(raw, remaining)
            anchor_limit_reached = anchor_limit_reached or overflowed
            return bounded

        def append_evidence(item: ResolutionEvidence) -> bool:
            nonlocal anchor_limit_reached
            if len(evidence) >= self.max_anchors_per_event:
                anchor_limit_reached = True
                return False
            evidence.append(item)
            return True

        for field_name in self._PROJECT_FIELDS:
            for value in values(event.get(field_name)):
                project_id = self._project_id(value)
                if project_id:
                    has_explicit_project = True
                    projects[project_id] = 1.0
                    append_evidence(
                        ResolutionEvidence(
                            "explicit_project", source_id, field_name, value, 1.0
                        )
                    )
        for field_name in self._WORKSTREAM_FIELDS:
            for value in values(event.get(field_name)):
                workstream_id = self._workstream_id(value)
                if workstream_id:
                    workstreams[workstream_id] = 1.0
                    append_evidence(
                        ResolutionEvidence(
                            "explicit_workstream", source_id, field_name, value, 1.0
                        )
                    )

        for field_name in self._REPO_FIELDS:
            for value in values(event.get(field_name)):
                repo = self._repo_name(value)
                if not repo:
                    continue
                if not has_explicit_project:
                    projects.setdefault(f"project:{_slug(repo)}", 0.9)
                workstreams.setdefault(f"workstream:repo-{_slug(repo)}", 0.9)
                append_evidence(
                    ResolutionEvidence("repo_or_cwd", source_id, field_name, value, 0.9)
                )

        for field_name in self._ARTIFACT_FIELDS:
            for path in values(event.get(field_name)):
                repo = self._repo_name(path)
                if repo and not has_explicit_project:
                    projects.setdefault(f"project:{_slug(repo)}", 0.72)
                if repo:
                    workstreams.setdefault(
                        f"workstream:repo-{_slug(repo)}", 0.72
                    )
                append_evidence(
                    ResolutionEvidence(
                        "artifact", source_id, field_name, path, 0.72
                    )
                )

        for field_name, _role, text in _event_texts(event):
            remaining = self.max_anchors_per_event - len(evidence)
            if remaining <= 0:
                anchor_limit_reached = True
                break
            matches = list(islice(self._ARTIFACT_RE.finditer(text), remaining + 1))
            if len(matches) > remaining:
                anchor_limit_reached = True
            for match in matches[:remaining]:
                path = match.group("path").rstrip(".,)")
                if len(path) > self.max_span_chars:
                    anchor_limit_reached = True
                    continue
                repo = self._repo_name(path)
                if not repo:
                    continue
                if not has_explicit_project:
                    projects.setdefault(f"project:{_slug(repo)}", 0.72)
                workstreams.setdefault(f"workstream:repo-{_slug(repo)}", 0.72)
                append_evidence(
                    ResolutionEvidence(
                        "artifact",
                        source_id,
                        field_name,
                        path,
                        0.72,
                        match.start("path"),
                        match.start("path") + len(path),
                    )
                )

        channel = _clean_scalar(event.get("channel_id") or event.get("channel"))
        thread = _clean_scalar(event.get("thread_id") or event.get("thread"))
        if channel or thread:
            parts = []
            if channel:
                parts.extend(("channel", _slug(channel)))
            if thread:
                parts.extend(("thread", _slug(thread)))
            workstream = f"workstream:{'-'.join(parts)}"
            workstreams.setdefault(workstream, 0.65)
            append_evidence(
                ResolutionEvidence(
                    "channel_thread",
                    source_id,
                    "channel_id/thread_id",
                    f"{channel}/{thread}".strip("/"),
                    0.65,
                )
            )

        project_ids, workstream_ids, scope_ids_overflowed = (
            self._bounded_scope_ids(projects, workstreams)
        )
        anchor_limit_reached = anchor_limit_reached or scope_ids_overflowed
        if len(projects) > 1:
            status = "multi_project"
        elif projects or workstreams:
            status = "resolved"
        else:
            status = "unknown"
        confidence = max([*projects.values(), *workstreams.values()], default=0.0)
        return WorkstreamResolution(
            status=status,
            project_ids=project_ids,
            workstream_ids=workstream_ids,
            confidence=round(confidence, 3),
            evidence=tuple(
                sorted(
                    evidence,
                    key=lambda item: (
                        item.kind,
                        item.field,
                        item.start if item.start is not None else -1,
                        item.value,
                    ),
                )
            ),
            anchor_limit_reached=anchor_limit_reached,
        )

    def _bounded_scope_ids(
        self,
        projects: Mapping[str, float],
        workstreams: Mapping[str, float],
    ) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
        """Project a resolution into the retrieval index's fixed scope budget."""

        ranked_projects = sorted(
            projects,
            key=lambda item: (-projects[item], item),
        )
        ranked_workstreams = sorted(
            workstreams,
            key=lambda item: (-workstreams[item], item),
        )
        selected: list[tuple[str, str]] = []
        if ranked_projects:
            selected.append(("project", ranked_projects.pop(0)))
        if ranked_workstreams and len(selected) < self.max_scope_ids:
            selected.append(("workstream", ranked_workstreams.pop(0)))
        remaining = [
            *(
                ("project", item, projects[item])
                for item in ranked_projects
            ),
            *(
                ("workstream", item, workstreams[item])
                for item in ranked_workstreams
            ),
        ]
        remaining.sort(
            key=lambda item: (
                -item[2],
                0 if item[0] == "project" else 1,
                item[1],
            )
        )
        selected.extend(
            (kind, item)
            for kind, item, _confidence in remaining[
                : self.max_scope_ids - len(selected)
            ]
        )
        selected_projects = tuple(
            sorted(item for kind, item in selected if kind == "project")
        )
        selected_workstreams = tuple(
            sorted(item for kind, item in selected if kind == "workstream")
        )
        overflowed = len(projects) + len(workstreams) > self.max_scope_ids
        return selected_projects, selected_workstreams, overflowed

    @staticmethod
    def _project_id(value: str) -> str:
        slug = _slug(value.removeprefix("project:"))
        return f"project:{slug}" if slug else ""

    @staticmethod
    def _workstream_id(value: str) -> str:
        slug = _slug(value.removeprefix("workstream:"))
        return f"workstream:{slug}" if slug else ""

    @staticmethod
    def _repo_name(path: str) -> str:
        normalized = path.replace("\\", "/").rstrip("/")
        parts = [part for part in normalized.split("/") if part and part != "."]
        if not parts:
            return ""
        lowered = [part.lower() for part in parts]
        for marker in ("src", "repos", "projects", "work"):
            if marker in lowered:
                position = len(lowered) - 1 - lowered[::-1].index(marker)
                if position + 1 < len(parts):
                    return parts[position + 1]
        if "." in parts[-1] and len(parts) > 1:
            # A relative artifact such as docs/report.md is evidence inside an
            # already-resolved workstream, not proof that "docs" is a project.
            return ""
        name = PurePath(normalized).name
        return name if "." not in name else ""


class WorkstreamEvidenceBuilder:
    """Build immutable derived evidence records from canonical raw events."""

    _PATTERNS: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
        "user": (
            (
                "request_goal",
                re.compile(
                    r"\b(?:Goal\s*:|Objective\s*:)\s*[^.\n]+[.]?",
                    re.IGNORECASE,
                ),
            ),
            (
                "request_goal",
                re.compile(
                    r"(?:^|(?<=[.!?]\s))(?:Please\s+|Can you\s+|I want (?:you )?to\s+)"
                    r"[^.\n]+[.]?",
                    re.IGNORECASE,
                ),
            ),
            (
                "decision",
                re.compile(
                    r"\b(?:(?:We|I)\s+decided\s+to|The\s+decision\s+is\s+to|Decision\s*:)"
                    r"\s*[^.\n]+[.]?",
                    re.IGNORECASE,
                ),
            ),
            (
                "correction",
                re.compile(
                    r"\b(?:Correction\s*:|Actually\s*,?|The\s+old\s+[^.\n]{0,80}\s+was\s+wrong\b)"
                    r"[^.\n]*[.]?",
                    re.IGNORECASE,
                ),
            ),
            (
                "blocker",
                re.compile(
                    r"\b(?:Blocker\s*:|We(?:'re|\s+are)\s+blocked\s+by|"
                    r"I\s+can't\s+continue\s+because)\s*[^.\n]+[.]?",
                    re.IGNORECASE,
                ),
            ),
            (
                "open_loop",
                re.compile(
                    r"\b(?:Open\s+loop\s*:|Still\s+need\s+to|Need\s+to\s+follow\s+up\s+on)"
                    r"\s*[^.\n]+[.]?",
                    re.IGNORECASE,
                ),
            ),
            (
                "next_action",
                re.compile(
                    r"\b(?:Next\s+action\s*:|Next\s+step\s*:)\s*[^.\n]+[.]?",
                    re.IGNORECASE,
                ),
            ),
            (
                "verified_result",
                re.compile(
                    r"\bI\s+(?:verified|confirmed)\s+that\s+[^.\n]+"
                    r"(?:works?|passes?|succeeded|is\s+fixed)[^.\n]*[.]?",
                    re.IGNORECASE,
                ),
            ),
        ),
        "assistant": (
            (
                "proposal",
                re.compile(
                    r"\b(?:I\s+propose|I\s+recommend|We\s+should|Let's)\s+[^.\n]+[.]?",
                    re.IGNORECASE,
                ),
            ),
            (
                "implementation_claim",
                re.compile(
                    r"(?:^|(?<=[.!?]\s))(?:Done\b|Implemented\b|Built\b|Fixed\b|"
                    r"I\s+(?:implemented|built|fixed|completed)\b)[^.\n]*[.]?",
                    re.IGNORECASE,
                ),
            ),
        ),
        "tool": (
            (
                "verified_result",
                re.compile(
                    r"[^.\n]*(?:\b\d+\s+passed\b|\ball\s+tests\s+passed\b|"
                    r"\bcompleted\s+successfully\b|\bexit\s+code\s+0\b|"
                    r"\bsuccess(?:ful|fully)?\b)[^.\n]*[.]?",
                    re.IGNORECASE,
                ),
            ),
            (
                "blocker",
                re.compile(
                    r"[^.\n]*(?:\bfailed\b|\berror\b|\bexception\b|\bblocked\b|"
                    r"\bpermission\s+denied\b|\btimed?\s*out\b)[^.\n]*[.]?",
                    re.IGNORECASE,
                ),
            ),
        ),
    }
    _PROCEDURE_RE = re.compile(
        r"[^.\n]*(?:\brunbook\b|\bworkflow\b|\bprocedure\b|\bsteps?\s+to\b)[^.\n]*[.]?",
        re.IGNORECASE,
    )
    _VERIFIED_RESULT_RE = re.compile(
        r"\b(?:\d+\s+passed|all\s+tests\s+passed|completed\s+successfully|"
        r"exit\s+code\s+0|I\s+(?:verified|confirmed)\s+that\s+.+"
        r"(?:works?|passes?|succeeded|is\s+fixed))\b",
        re.IGNORECASE,
    )
    _FAILED_RESULT_RE = re.compile(
        r"\b(?:\d+\s+failed|failed|error|exception|permission\s+denied|timed?\s*out)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        structured_extractor: StructuredExtractor | None = None,
        resolver: WorkstreamResolver | None = None,
        hygiene_classifier: CorpusHygieneClassifier | None = None,
        max_nodes: int = _DEFAULT_MAX_DERIVED_NODES,
        max_anchors_per_event: int = _DEFAULT_MAX_ANCHORS_PER_EVENT,
        max_span_chars: int = _DEFAULT_MAX_SPAN_CHARS,
    ) -> None:
        if not 1 <= int(max_nodes) <= _DEFAULT_MAX_DERIVED_NODES:
            raise ValueError(
                f"max_nodes must be between 1 and {_DEFAULT_MAX_DERIVED_NODES}"
            )
        if not 1 <= int(max_anchors_per_event) <= 10_000:
            raise ValueError("max_anchors_per_event must be between 1 and 10000")
        if not 1 <= int(max_span_chars) <= 16_000:
            raise ValueError("max_span_chars must be between 1 and 16000")
        self.structured_extractor = structured_extractor
        self.max_nodes = int(max_nodes)
        self.max_anchors_per_event = int(max_anchors_per_event)
        self.max_span_chars = int(max_span_chars)
        self.resolver = resolver or WorkstreamResolver(
            max_anchors_per_event=self.max_anchors_per_event,
            max_span_chars=self.max_span_chars,
        )
        self.hygiene_classifier = hygiene_classifier or CorpusHygieneClassifier()

    def build(
        self, events: Iterable[Mapping[str, Any]]
    ) -> WorkstreamEvidenceBuild:
        # Copy only references into a tuple; this builder never mutates source
        # mappings or any nested source value.
        snapshot = tuple(events)
        sorted_events = tuple(sorted(snapshot, key=_event_order))
        hygiene = self.hygiene_classifier.classify(sorted_events)
        hygiene_by_id = {item.source_id: item for item in hygiene}
        eligible = tuple(
            event
            for event in sorted_events
            if (
                _event_id(event)
                and hygiene_by_id.get(_event_id(event))
                and hygiene_by_id[_event_id(event)].disposition == "keep"
            )
        )
        resolutions = {
            _event_id(event): self.resolver.resolve(event)
            for event in eligible
        }
        nodes: list[EvidenceNode] = []
        overflow_rejections: set[str] = set()
        rejected_node_count = 0
        for resolution_source_id, resolution in resolutions.items():
            if resolution.anchor_limit_reached:
                overflow_rejections.add(
                    f"{resolution_source_id}:resolution_anchor_limit_reached"
                )
        for event_position, event in enumerate(eligible):
            source_id = _event_id(event)
            lineage = hygiene_by_id[source_id]
            resolution = resolutions[source_id]
            remaining_nodes = self.max_nodes - len(nodes)
            if remaining_nodes <= 0:
                overflow_rejections.add("total_node_budget_exhausted")
                rejected_node_count += len(eligible) - event_position
                break
            event_nodes, event_rejections, event_rejected_count = (
                self._deterministic_nodes(
                    event,
                    resolution=resolution,
                    lineage_refs=lineage.copied_by,
                    node_limit=remaining_nodes,
                )
            )
            nodes.extend(event_nodes)
            overflow_rejections.update(event_rejections)
            rejected_node_count += event_rejected_count

        structured_rejections: list[str] = []
        if self.structured_extractor is not None:
            remaining_nodes = self.max_nodes - len(nodes)
            if remaining_nodes <= 0:
                overflow_rejections.add("structured_node_budget_exhausted")
                rejected_node_count += 1
            else:
                proposed, rejected, overflowed, rejected_count = (
                    self._structured_nodes(
                        eligible,
                        resolutions=resolutions,
                        hygiene_by_id=hygiene_by_id,
                        node_limit=remaining_nodes,
                    )
                )
                nodes.extend(proposed)
                structured_rejections.extend(rejected)
                overflow_rejections.update(overflowed)
                rejected_node_count += rejected_count

        nodes = self._dedupe_and_corroborate(nodes)
        return WorkstreamEvidenceBuild(
            hygiene=hygiene,
            resolutions=tuple(
                (source_id, resolutions[source_id])
                for source_id in sorted(resolutions)
            ),
            nodes=tuple(nodes),
            structured_rejections=tuple(structured_rejections),
            overflow_rejections=tuple(sorted(overflow_rejections)),
            rejected_node_count=rejected_node_count,
        )

    def _deterministic_nodes(
        self,
        event: Mapping[str, Any],
        *,
        resolution: WorkstreamResolution,
        lineage_refs: tuple[str, ...],
        node_limit: int,
    ) -> tuple[list[EvidenceNode], set[str], int]:
        nodes: list[EvidenceNode] = []
        overflow_rejections: set[str] = set()
        rejected_node_count = 0
        anchors_seen = 0
        source_id = _event_id(event)

        def bounded_matches(
            pattern: re.Pattern[str], text: str
        ) -> tuple[re.Match[str], ...]:
            nonlocal anchors_seen
            remaining = self.max_anchors_per_event - anchors_seen
            if remaining <= 0:
                overflow_rejections.add(
                    f"{source_id}:derived_anchor_limit_reached"
                )
                return ()
            matches = tuple(islice(pattern.finditer(text), remaining + 1))
            if len(matches) > remaining:
                overflow_rejections.add(
                    f"{source_id}:derived_anchor_limit_reached"
                )
            accepted = matches[:remaining]
            anchors_seen += len(accepted)
            return accepted

        def append_node(
            *,
            kind: str,
            field: str,
            role: str,
            text: str,
            start: int,
            end: int,
        ) -> bool:
            nonlocal rejected_node_count
            if end - start > self.max_span_chars:
                overflow_rejections.add(
                    f"{source_id}:derived_span_limit_exceeded"
                )
                rejected_node_count += 1
                return True
            if len(nodes) >= node_limit:
                overflow_rejections.add("total_node_budget_exhausted")
                rejected_node_count += 1
                return False
            nodes.append(
                self._node(
                    event,
                    kind=kind,
                    field=field,
                    role=role,
                    text=text,
                    start=start,
                    end=end,
                    resolution=resolution,
                    lineage_refs=lineage_refs,
                    extraction_method="deterministic",
                )
            )
            return True

        for field_name, role, text in _event_texts(event):
            if not self._safe_for_derived_evidence(text):
                continue
            for kind, pattern in self._PATTERNS.get(role, ()):
                matches = bounded_matches(pattern, text)
                if kind == "implementation_claim" and len(matches) > 1:
                    match_ranges = [(matches[0].start(), matches[-1].end())]
                else:
                    match_ranges = [
                        (match.start(), match.end()) for match in matches
                    ]
                for start, end in match_ranges:
                    exact = text[start:end]
                    if kind == "blocker" and not self._is_failure(exact):
                        continue
                    if not append_node(
                        kind=kind,
                        field=field_name,
                        role=role,
                        text=text,
                        start=start,
                        end=end,
                    ):
                        return nodes, overflow_rejections, rejected_node_count
            for match in bounded_matches(self.resolver._ARTIFACT_RE, text):
                path = match.group("path").rstrip(".,)")
                end = match.start("path") + len(path)
                if not append_node(
                    kind="artifact",
                    field=field_name,
                    role=role,
                    text=text,
                    start=match.start("path"),
                    end=end,
                ):
                    return nodes, overflow_rejections, rejected_node_count
            for match in bounded_matches(self._PROCEDURE_RE, text):
                if not append_node(
                    kind="procedure",
                    field=field_name,
                    role=role,
                    text=text,
                    start=match.start(),
                    end=match.end(),
                ):
                    return nodes, overflow_rejections, rejected_node_count
        return nodes, overflow_rejections, rejected_node_count

    def _node(
        self,
        event: Mapping[str, Any],
        *,
        kind: str,
        field: str,
        role: str,
        text: str,
        start: int,
        end: int,
        resolution: WorkstreamResolution,
        lineage_refs: tuple[str, ...],
        extraction_method: str,
    ) -> EvidenceNode:
        source_id = _event_id(event)
        exact = text[start:end]
        span = EvidenceSpan(source_id, field, role, start, end, exact)
        verified = kind == "verified_result" and self._is_verified(role, exact)
        node_payload = {
            "kind": kind,
            "source_id": source_id,
            "field": field,
            "role": role,
            "start": start,
            "end": end,
            "text": exact,
            "method": extraction_method,
        }
        return EvidenceNode(
            id=f"evidence:{_stable_digest(node_payload)[:24]}",
            kind=kind,
            text=exact,
            role=role,
            observed_at=_observed_at(event),
            session_id=_clean_scalar(
                event.get("provider_session_id") or event.get("session_id")
            ),
            source_refs=(source_id,),
            evidence_spans=(span,),
            project_ids=resolution.project_ids,
            workstream_ids=resolution.workstream_ids,
            linkage_refs=_linkage_refs(event),
            lineage_refs=tuple(sorted(lineage_refs)),
            extraction_method=extraction_method,
            verified=verified,
        )

    def _structured_nodes(
        self,
        events: Sequence[Mapping[str, Any]],
        *,
        resolutions: Mapping[str, WorkstreamResolution],
        hygiene_by_id: Mapping[str, HygieneRecord],
        node_limit: int,
    ) -> tuple[list[EvidenceNode], list[str], set[str], int]:
        event_map = {_event_id(event): event for event in events}
        try:
            response = self.structured_extractor(  # type: ignore[misc]
                self._structured_payload(events)
            )
        except Exception:
            return [], ["structured_extractor_failed_closed"], set(), 0
        if not isinstance(response, dict):
            return [], ["structured_response_not_object"], set(), 0
        if response.get("schema_version") != 1:
            return [], ["structured_schema_version_invalid"], set(), 0
        if response.get("mutation_authority", "none") != "none":
            return [], ["mutation_authority_forbidden"], set(), 0
        if set(response) - {"schema_version", "mutation_authority", "proposals"}:
            return [], ["structured_response_unknown_fields"], set(), 0
        proposals = response.get("proposals")
        if not isinstance(proposals, list) or len(proposals) > 50:
            return [], ["structured_proposals_invalid"], set(), 0

        nodes: list[EvidenceNode] = []
        rejected: list[str] = []
        overflow_rejections: set[str] = set()
        rejected_node_count = 0
        for position, proposal in enumerate(proposals):
            if len(nodes) >= node_limit:
                overflow_rejections.add("structured_node_budget_exhausted")
                rejected_node_count += len(proposals) - position
                break
            reason = f"proposal_{position}_invalid"
            try:
                if not isinstance(proposal, dict):
                    raise ValueError
                allowed = {
                    "kind",
                    "source_id",
                    "field",
                    "role",
                    "start",
                    "end",
                    "text",
                }
                if set(proposal) - allowed:
                    raise ValueError
                kind = _KIND_ALIASES.get(
                    _clean_scalar(proposal.get("kind")),
                    _clean_scalar(proposal.get("kind")),
                )
                if kind not in _NODE_KINDS:
                    raise ValueError
                source_id = _clean_scalar(proposal.get("source_id"))
                event = event_map[source_id]
                field_name = _clean_scalar(proposal.get("field"))
                role = _clean_scalar(proposal.get("role"))
                source_text = event.get(field_name)
                if not isinstance(source_text, str):
                    raise ValueError
                if not self._safe_for_structured_model(source_text):
                    raise ValueError
                expected_role = {
                    field: source_role
                    for field, source_role, _source_text in _event_texts(event)
                }.get(field_name)
                if role != expected_role:
                    raise ValueError
                start = int(proposal.get("start"))
                end = int(proposal.get("end"))
                exact = _clean_scalar(proposal.get("text"))
                if (
                    start < 0
                    or end <= start
                    or end > len(source_text)
                    or end - start > self.max_span_chars
                    or source_text[start:end] != exact
                ):
                    if end > start and end - start > self.max_span_chars:
                        overflow_rejections.add(
                            f"{source_id}:structured_span_limit_exceeded"
                        )
                        rejected_node_count += 1
                    raise ValueError
                if not self._proposal_supported(kind, role, exact):
                    raise ValueError
                lineage = hygiene_by_id[source_id]
                nodes.append(
                    self._node(
                        event,
                        kind=kind,
                        field=field_name,
                        role=role,
                        text=source_text,
                        start=start,
                        end=end,
                        resolution=resolutions[source_id],
                        lineage_refs=lineage.copied_by,
                        extraction_method="structured_model",
                    )
                )
            except (KeyError, TypeError, ValueError):
                rejected.append(reason)
        return nodes, rejected, overflow_rejections, rejected_node_count

    def _structured_payload(
        self,
        events: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for event in events:
            fields = {
                field_name: text[: self.max_span_chars]
                for field_name, _role, text in _event_texts(event)
                if WorkstreamEvidenceBuilder._safe_for_structured_model(text)
            }
            rows.append(
                {
                    "source_id": _event_id(event),
                    "type": _clean_scalar(event.get("type")),
                    "observed_at": _observed_at(event),
                    "fields": fields,
                }
            )
        return {
            "schema_version": 1,
            "instruction": (
                "Propose derived evidence nodes only. Copy exact source spans; "
                "never promote, supersede, execute, or claim mutation authority."
            ),
            "allowed_kinds": sorted(_NODE_KINDS),
            "mutation_authority": "none",
            "events": rows,
        }

    @staticmethod
    def _safe_for_derived_evidence(text: str) -> bool:
        return bool(str(text or "").strip()) and not contains_sensitive_text(text)

    @staticmethod
    def _safe_for_structured_model(text: str) -> bool:
        if not WorkstreamEvidenceBuilder._safe_for_derived_evidence(text):
            return False
        _escaped, instruction_redactions = escape_untrusted_evidence_text(text)
        return instruction_redactions == 0

    def _proposal_supported(self, kind: str, role: str, text: str) -> bool:
        if kind == "verified_result":
            return role in {"user", "tool"} and self._is_verified(role, text)
        if kind == "implementation_claim":
            return role == "assistant" and bool(
                re.search(r"\b(?:done|implemented|built|fixed|completed)\b", text, re.I)
            )
        if kind == "proposal":
            return role == "assistant" and bool(
                re.search(r"\b(?:I\s+propose|I\s+recommend|We\s+should|Let's)\b", text, re.I)
            )
        if kind == "request_goal":
            return role == "user" and bool(
                re.search(
                    r"\b(?:goal|objective|acceptance\s+criterion|please|"
                    r"can\s+you|I\s+want|need\s+to)\b",
                    text,
                    re.I,
                )
            )
        if kind == "decision":
            return role == "user" and bool(
                re.search(r"\b(?:decided|decision|choose|chose|adopt|use|keep)\b", text, re.I)
            )
        if kind == "correction":
            return role == "user" and bool(
                re.search(r"\b(?:actually|correction|wrong|instead|not)\b", text, re.I)
            )
        if kind == "open_loop":
            return role == "user" and bool(
                re.search(r"\b(?:open\s+loop|still\s+need|follow\s+up|unresolved)\b", text, re.I)
            )
        if kind == "next_action":
            return role == "user" and bool(
                re.search(r"\b(?:next\s+action|next\s+step)\b", text, re.I)
            )
        if kind == "blocker":
            return role in {"user", "tool"} and self._is_failure(text)
        if kind == "artifact":
            return role in {"user", "assistant", "tool"} and bool(
                self.resolver._ARTIFACT_RE.search(text)
            )
        if kind == "procedure":
            return role in {"user", "assistant", "tool"} and bool(
                self._PROCEDURE_RE.search(text)
            )
        return False

    def _is_verified(self, role: str, text: str) -> bool:
        if role not in {"user", "tool"}:
            return False
        if self._FAILED_RESULT_RE.search(text):
            if re.search(r"\b(?:0\s+failed|no\s+failures?)\b", text, re.I):
                return bool(self._VERIFIED_RESULT_RE.search(text))
            return False
        return bool(self._VERIFIED_RESULT_RE.search(text))

    def _is_failure(self, text: str) -> bool:
        if re.search(
            r"\b(?:Blocker\s*:|We(?:'re|\s+are)\s+blocked\s+by|"
            r"I\s+can't\s+continue\s+because)",
            text,
            re.I,
        ):
            return True
        if re.search(r"\b(?:0\s+failed|no\s+(?:test\s+)?failures?)\b", text, re.I):
            return False
        return bool(self._FAILED_RESULT_RE.search(text))

    @staticmethod
    def _meaningfully_overlaps(left: str, right: str) -> bool:
        def tokens(text: str) -> set[str]:
            bounded = text[:_DEFAULT_MAX_SPAN_CHARS].lower()
            return {
                token
                for token in islice(_MEANINGFUL_TOKEN_RE.findall(bounded), 64)
                if token not in _CORROBORATION_STOPWORDS
                and not token.isdigit()
            }

        common = tokens(left) & tokens(right)
        return len(common) >= 2 or any(len(token) >= 8 for token in common)

    @staticmethod
    def _explicitly_linked(left: EvidenceNode, right: EvidenceNode) -> bool:
        left_sources = set(left.source_refs)
        right_sources = set(right.source_refs)
        left_links = set(left.linkage_refs)
        right_links = set(right.linkage_refs)
        return bool(
            (left_sources & right_links)
            or (right_sources & left_links)
            or (left_links & right_links)
        )

    @classmethod
    def _can_corroborate(
        cls, claim: EvidenceNode, candidate: EvidenceNode
    ) -> bool:
        claim_time = _parse_evidence_timestamp(claim.observed_at)
        candidate_time = _parse_evidence_timestamp(candidate.observed_at)
        if claim_time is None or candidate_time is None or candidate_time < claim_time:
            return False
        if cls._explicitly_linked(claim, candidate):
            return True
        elapsed = (candidate_time - claim_time).total_seconds()
        if elapsed > _MAX_CORROBORATION_GAP_SECONDS:
            return False
        shared_workstream = set(claim.workstream_ids) & set(
            candidate.workstream_ids
        )
        return bool(
            shared_workstream
            and cls._meaningfully_overlaps(claim.text, candidate.text)
        )

    @classmethod
    def _dedupe_and_corroborate(
        cls, nodes: Sequence[EvidenceNode]
    ) -> list[EvidenceNode]:
        unique = {node.id: node for node in nodes}
        ordered = sorted(
            unique.values(),
            key=lambda node: (
                _timestamp_order(node.observed_at),
                node.source_refs,
                node.evidence_spans[0].start,
                node.kind,
                node.id,
            ),
        )
        verified_nodes = tuple(
            node for node in ordered if node.kind == "verified_result" and node.verified
        )
        if not verified_nodes:
            return ordered
        enriched: list[EvidenceNode] = []
        for node in ordered:
            if node.kind == "implementation_claim":
                matching = tuple(
                    candidate.id
                    for candidate in verified_nodes
                    if cls._can_corroborate(node, candidate)
                )
                enriched.append(replace(node, corroborated_by=matching[:3]))
            else:
                enriched.append(node)
        return enriched


__all__ = [
    "CorpusHygieneClassifier",
    "EvidenceNode",
    "EvidenceSpan",
    "HygieneRecord",
    "ResolutionEvidence",
    "WorkstreamEvidenceBuild",
    "WorkstreamEvidenceBuilder",
    "WorkstreamResolution",
    "WorkstreamResolver",
]
