"""Canonical schema types for the Memory v2 provider.

These dataclasses are intentionally lightweight and dependency-free. They form
Memory v2's in-process contract; later storage/index layers can serialize them
as YAML/JSON without depending on opaque dict shapes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, cast


class ValidationError(ValueError):
    """Raised when a Memory v2 schema record is invalid."""


class _StrEnum(str, Enum):
    """String enum with ergonomic coercion from raw strings."""

    @classmethod
    def coerce(cls, value: Any, field_name: str):
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except Exception as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValidationError(
                f"Invalid {field_name}: {value!r}; expected one of: {allowed}"
            ) from exc


class SourceType(_StrEnum):
    SESSION = "session"
    MESSAGE = "message"
    FILE = "file"
    TOOL_RESULT = "tool_result"
    MEMORY = "memory"
    SKILL = "skill"
    WEB = "web"
    MANUAL = "manual"


class MemoryType(_StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    BELIEF = "belief"
    CONSTRAINT = "constraint"
    ENVIRONMENT = "environment"
    PROJECT_STATE = "project_state"
    EPISODE = "episode"
    PROCEDURE_REF = "procedure_ref"


class MemoryStatus(_StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    UNCERTAIN = "uncertain"
    ARCHIVED = "archived"
    REJECTED = "rejected"


class ProjectStatus(_StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class CoreMemoryCategory(_StrEnum):
    USER = "user"
    ASSISTANT_IDENTITY = "assistant_identity"
    ENVIRONMENT = "environment"
    OPERATING_RULE = "operating_rule"


class GateDecision(_StrEnum):
    PENDING = "pending"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ARCHIVED_ONLY = "archived_only"
    SUPERSEDED = "superseded"


class CandidateClaimKind(_StrEnum):
    OTHER = "other"
    PREFERENCE = "preference"
    ENVIRONMENT_STATE = "environment_state"
    DECISION = "decision"
    NEXT_ACTION = "next_action"
    OPEN_QUESTION = "open_question"
    CURRENT_STATE = "current_state"
    GOAL = "goal"
    STATUS = "status"
    CONSTRAINT = "constraint"
    SKILL_CANDIDATE = "skill_candidate"
    OPEN_LOOP = "open_loop"
    CONTRADICTION = "contradiction"
    BLOCKER = "blocker"
    COMPLETED_ACTION = "completed_action"
    AUTHORITATIVE_ARTIFACT = "authoritative_artifact"
    ACCEPTED_PROPOSAL = "accepted_proposal"


class ArtifactModality(_StrEnum):
    TEXT = "text"
    IMAGE = "image"
    SCREENSHOT = "screenshot"
    PDF = "pdf"
    AUDIO = "audio"
    VIDEO = "video"
    REPO = "repo"
    NOTEBOOK = "notebook"
    DATASET = "dataset"
    LOG = "log"
    WEB_PAGE = "web_page"
    MIXED = "mixed"
    OTHER = "other"


class ArtifactSourceType(_StrEnum):
    UPLOAD = "upload"
    LOCAL_FILE = "local_file"
    BROWSER = "browser"
    GENERATED = "generated"
    EXTERNAL_URL = "external_url"
    REPO = "repo"
    NOTE = "note"
    TOOL_OUTPUT = "tool_output"
    MANUAL = "manual"


class PrivacyLevel(_StrEnum):
    PUBLIC = "public"
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    SECRET = "secret"


class FreshnessClass(_StrEnum):
    STATIC = "static"
    SLOW_CHANGING = "slow_changing"
    FAST_CHANGING = "fast_changing"
    EPHEMERAL = "ephemeral"
    SENSITIVE_EXTERNAL = "sensitive_external"


class ArtifactProcessingStatus(_StrEnum):
    NOT_RUN = "not_run"
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
    REDACTED = "redacted"


def utc_now_iso() -> str:
    """Return a compact UTC ISO timestamp with seconds precision."""
    return (
        datetime
        .now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _require_nonblank(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValidationError(f"{field_name} is required")
    return text


def _validate_unit_interval(value: float, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{field_name} must be a number between 0.0 and 1.0"
        ) from exc
    if number < 0.0 or number > 1.0:
        raise ValidationError(f"{field_name} must be between 0.0 and 1.0")
    return number


def _list_of_strings(values: Optional[List[Any]]) -> List[str]:
    if values is None:
        return []
    return [str(value) for value in values]


def _string_keyed_dict(values: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if values is None:
        return {}
    return {str(key): value for key, value in dict(values).items()}


def _artifact_processing_status_dict(
    values: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for key, value in dict(values or {}).items():
        result[str(key)] = ArtifactProcessingStatus.coerce(
            value, f"processing_status.{key}"
        ).value
    return result


def normalize_project_id(value: str) -> str:
    """Normalize a project name/id into ``project:<slug>`` form."""
    text = _require_nonblank(value, "project id")
    if text.startswith("project:"):
        raw = text[len("project:") :]
    else:
        raw = text
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    if not slug:
        raise ValidationError(
            "project id must contain at least one alphanumeric character"
        )
    return f"project:{slug}"


@dataclass
class ArtifactRecord:
    id: str
    content_hash: str
    modality: ArtifactModality | str
    source_type: ArtifactSourceType | str
    source_uri: str
    profile_id: str = "default"
    project_id: Optional[str] = None
    created_at: str = ""
    ingested_at: str = field(default_factory=utc_now_iso)
    last_verified_at: Optional[str] = None
    freshness_class: FreshnessClass | str = FreshnessClass.STATIC
    privacy_level: PrivacyLevel | str = PrivacyLevel.PERSONAL
    access_scope: str = "local_profile"
    retention_policy: str = "review"
    processing_status: Dict[str, Any] = field(default_factory=dict)
    injection_risk: str = "none"
    derived_refs: List[str] = field(default_factory=list)
    source_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = _require_nonblank(self.id, "id")
        self.content_hash = _require_nonblank(self.content_hash, "content_hash")
        self.modality = ArtifactModality.coerce(self.modality, "modality")
        self.source_type = ArtifactSourceType.coerce(self.source_type, "source_type")
        self.source_uri = _require_nonblank(self.source_uri, "source_uri")
        self.profile_id = str(self.profile_id or "default")
        if self.project_id is not None:
            self.project_id = str(self.project_id)
        self.created_at = str(self.created_at or "")
        self.ingested_at = str(self.ingested_at or "")
        if self.last_verified_at is not None:
            self.last_verified_at = str(self.last_verified_at)
        self.freshness_class = FreshnessClass.coerce(
            self.freshness_class, "freshness_class"
        )
        self.privacy_level = PrivacyLevel.coerce(self.privacy_level, "privacy_level")
        self.access_scope = str(self.access_scope or "local_profile")
        self.retention_policy = str(self.retention_policy or "review")
        self.processing_status = _artifact_processing_status_dict(
            self.processing_status
        )
        self.injection_risk = str(self.injection_risk or "none")
        self.derived_refs = _list_of_strings(self.derived_refs)
        self.source_refs = _list_of_strings(self.source_refs)
        self.metadata = _string_keyed_dict(self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content_hash": self.content_hash,
            "modality": cast(ArtifactModality, self.modality).value,
            "source_type": cast(ArtifactSourceType, self.source_type).value,
            "source_uri": self.source_uri,
            "profile_id": self.profile_id,
            "project_id": self.project_id,
            "created_at": self.created_at,
            "ingested_at": self.ingested_at,
            "last_verified_at": self.last_verified_at,
            "freshness_class": cast(FreshnessClass, self.freshness_class).value,
            "privacy_level": cast(PrivacyLevel, self.privacy_level).value,
            "access_scope": self.access_scope,
            "retention_policy": self.retention_policy,
            "processing_status": dict(self.processing_status),
            "injection_risk": self.injection_risk,
            "derived_refs": list(self.derived_refs),
            "source_refs": list(self.source_refs),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactRecord":
        return cls(**data)


@dataclass
class ArtifactSegment:
    id: str
    artifact_id: str
    segment_type: str
    text: str = ""
    summary: str = ""
    entities: List[str] = field(default_factory=list)
    confidence: Dict[str, Any] = field(default_factory=dict)
    location: Dict[str, Any] = field(default_factory=dict)
    privacy_level: PrivacyLevel | str = PrivacyLevel.PERSONAL
    freshness: str = "current"
    source_ref: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = _require_nonblank(self.id, "id")
        self.artifact_id = _require_nonblank(self.artifact_id, "artifact_id")
        self.segment_type = _require_nonblank(self.segment_type, "segment_type")
        self.text = str(self.text or "")
        self.summary = str(self.summary or "")
        self.entities = _list_of_strings(self.entities)
        self.confidence = _string_keyed_dict(self.confidence)
        self.location = _string_keyed_dict(self.location)
        self.privacy_level = PrivacyLevel.coerce(self.privacy_level, "privacy_level")
        self.freshness = str(self.freshness or "current")
        self.source_ref = _string_keyed_dict(self.source_ref)
        self.metadata = _string_keyed_dict(self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "artifact_id": self.artifact_id,
            "segment_type": self.segment_type,
            "text": self.text,
            "summary": self.summary,
            "entities": list(self.entities),
            "confidence": dict(self.confidence),
            "location": dict(self.location),
            "privacy_level": cast(PrivacyLevel, self.privacy_level).value,
            "freshness": self.freshness,
            "source_ref": dict(self.source_ref),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactSegment":
        return cls(**data)


@dataclass
class SourceRef:
    id: str
    type: SourceType | str
    uri: str
    title: str = ""
    observed_at: str = ""
    quote: Optional[str] = None

    def __post_init__(self) -> None:
        self.id = _require_nonblank(self.id, "id")
        self.type = SourceType.coerce(self.type, "type")
        self.uri = _require_nonblank(self.uri, "uri")
        self.title = str(self.title or "")
        self.observed_at = str(self.observed_at or "")
        if self.quote is not None:
            self.quote = str(self.quote)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "type": cast(SourceType, self.type).value,
            "uri": self.uri,
            "title": self.title,
            "observed_at": self.observed_at,
        }
        if self.quote is not None:
            data["quote"] = self.quote
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceRef":
        return cls(**data)


@dataclass
class MemoryItem:
    id: str
    type: MemoryType | str
    subject: str
    predicate: Optional[str] = None
    value: Optional[str] = None
    body: Optional[str] = None
    summary: Optional[str] = None
    status: MemoryStatus | str = MemoryStatus.ACTIVE
    confidence: float = 0.7
    importance: float = 0.5
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    expires_at: Optional[str] = None
    source_refs: List[str] = field(default_factory=list)
    supersedes: List[str] = field(default_factory=list)
    superseded_by: Optional[str] = None
    superseded_at: Optional[str] = None
    supersession_reason: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.id = _require_nonblank(self.id, "id")
        self.type = MemoryType.coerce(self.type, "type")
        self.subject = _require_nonblank(self.subject, "subject")
        self.status = MemoryStatus.coerce(self.status, "status")
        self.confidence = _validate_unit_interval(self.confidence, "confidence")
        self.importance = _validate_unit_interval(self.importance, "importance")
        self.source_refs = _list_of_strings(self.source_refs)
        self.supersedes = _list_of_strings(self.supersedes)
        self.tags = _list_of_strings(self.tags)
        if self.status == MemoryStatus.SUPERSEDED and not self.superseded_by:
            raise ValidationError("superseded_by is required when status is superseded")
        if self.status == MemoryStatus.ACTIVE and self.superseded_by:
            raise ValidationError("active memories cannot set superseded_by")
        if self.superseded_by is not None:
            self.superseded_by = str(self.superseded_by)
        if self.superseded_at is not None:
            self.superseded_at = str(self.superseded_at)
        if self.supersession_reason is not None:
            self.supersession_reason = str(self.supersession_reason)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": getattr(self.type, "value", str(self.type)),
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "body": self.body,
            "summary": self.summary,
            "status": getattr(self.status, "value", str(self.status)),
            "confidence": self.confidence,
            "importance": self.importance,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "expires_at": self.expires_at,
            "source_refs": list(self.source_refs),
            "supersedes": list(self.supersedes),
            "superseded_by": self.superseded_by,
            "superseded_at": self.superseded_at,
            "supersession_reason": self.supersession_reason,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        return cls(**data)


@dataclass
class ProjectCard:
    id: str
    name: str
    status: ProjectStatus | str = ProjectStatus.ACTIVE
    importance: float = 0.5
    updated_at: str = field(default_factory=utc_now_iso)
    goal: str = ""
    why_it_matters: str = ""
    current_state: str = ""
    decisions: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)
    source_refs: List[str] = field(default_factory=list)
    related_entities: List[str] = field(default_factory=list)
    injection_policy: Dict[str, Any] = field(default_factory=dict)
    # Append-only, source-grounded lifecycle records for individual project fields.
    # Legacy cards omit this field and continue to load unchanged.
    field_evidence: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = normalize_project_id(self.id)
        self.name = _require_nonblank(self.name, "name")
        self.status = ProjectStatus.coerce(self.status, "status")
        self.importance = _validate_unit_interval(self.importance, "importance")
        self.decisions = _list_of_strings(self.decisions)
        self.open_questions = _list_of_strings(self.open_questions)
        self.next_actions = _list_of_strings(self.next_actions)
        self.source_refs = _list_of_strings(self.source_refs)
        self.related_entities = _list_of_strings(self.related_entities)
        self.injection_policy = dict(self.injection_policy or {})
        self.field_evidence = self._validate_field_evidence(self.field_evidence)

    @staticmethod
    def _validate_field_evidence(value: Any) -> Dict[str, List[Dict[str, Any]]]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValidationError("field_evidence must be an object")
        validated: Dict[str, List[Dict[str, Any]]] = {}
        for raw_field, raw_entries in value.items():
            field_name = str(raw_field or "").strip()
            if not field_name or not isinstance(raw_entries, list):
                raise ValidationError("field_evidence entries must be lists")
            entries: List[Dict[str, Any]] = []
            for raw_entry in raw_entries:
                if not isinstance(raw_entry, dict):
                    raise ValidationError("field_evidence records must be objects")
                entry = dict(raw_entry)
                entry["value"] = str(entry.get("value") or "").strip()
                if not entry["value"]:
                    raise ValidationError("field_evidence.value must be nonblank")
                entry["status"] = str(entry.get("status") or "current").strip().lower()
                if entry["status"] not in {"current", "superseded", "resolved", "stale"}:
                    raise ValidationError("field_evidence.status is invalid")
                entry["observed_at"] = str(entry.get("observed_at") or "")
                entry["source_refs"] = sorted(set(_list_of_strings(entry.get("source_refs"))))
                candidate_id = str(entry.get("candidate_id") or "").strip()
                if candidate_id:
                    entry["candidate_id"] = candidate_id
                else:
                    entry.pop("candidate_id", None)
                entries.append(entry)
            validated[field_name] = sorted(
                entries,
                key=lambda item: (
                    str(item.get("observed_at") or ""),
                    str(item.get("candidate_id") or ""),
                    str(item.get("value") or ""),
                ),
            )
        return dict(sorted(validated.items()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": cast(ProjectStatus, self.status).value,
            "importance": self.importance,
            "updated_at": self.updated_at,
            "goal": self.goal,
            "why_it_matters": self.why_it_matters,
            "current_state": self.current_state,
            "decisions": list(self.decisions),
            "open_questions": list(self.open_questions),
            "next_actions": list(self.next_actions),
            "source_refs": list(self.source_refs),
            "related_entities": list(self.related_entities),
            "injection_policy": dict(self.injection_policy),
            "field_evidence": {
                key: [dict(entry) for entry in entries]
                for key, entries in self.field_evidence.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectCard":
        return cls(**data)


@dataclass
class CoreMemoryRecord:
    id: str
    category: CoreMemoryCategory | str
    statement: str
    layer: str = "core"
    priority: float = 0.8
    confidence: float = 0.9
    updated_at: str = field(default_factory=utc_now_iso)
    source_refs: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.id = _require_nonblank(self.id, "id")
        self.category = CoreMemoryCategory.coerce(self.category, "category")
        self.statement = _require_nonblank(self.statement, "statement")
        self.layer = str(self.layer or "core")
        if self.layer != "core":
            raise ValidationError("layer must be core")
        self.priority = _validate_unit_interval(self.priority, "priority")
        self.confidence = _validate_unit_interval(self.confidence, "confidence")
        self.source_refs = _list_of_strings(self.source_refs)
        self.tags = _list_of_strings(self.tags)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "layer": "core",
            "category": cast(CoreMemoryCategory, self.category).value,
            "statement": self.statement,
            "priority": self.priority,
            "confidence": self.confidence,
            "updated_at": self.updated_at,
            "source_refs": list(self.source_refs),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoreMemoryRecord":
        return cls(**data)


@dataclass
class CandidateMemory:
    id: str
    type: MemoryType | str
    claim: str
    created_at: str = field(default_factory=utc_now_iso)
    proposed_destination: str = ""
    importance: float = 0.5
    confidence: float = 0.7
    promotion_reason: str = ""
    source_refs: List[str] = field(default_factory=list)
    claim_kind: CandidateClaimKind | str = CandidateClaimKind.OTHER
    durability: float = 0.5
    evidence_spans: List[Dict[str, Any]] = field(default_factory=list)
    negative_evidence: List[Dict[str, Any]] = field(default_factory=list)
    extraction_method: str = "legacy"
    extractor_version: str = ""
    gate_decision: GateDecision | str = GateDecision.PENDING
    decision_reason: str = ""

    def __post_init__(self) -> None:
        self.id = _require_nonblank(self.id, "id")
        self.type = MemoryType.coerce(self.type, "type")
        self.claim = _require_nonblank(self.claim, "claim")
        self.importance = _validate_unit_interval(self.importance, "importance")
        self.confidence = _validate_unit_interval(self.confidence, "confidence")
        self.source_refs = _list_of_strings(self.source_refs)
        self.claim_kind = CandidateClaimKind.coerce(self.claim_kind, "claim_kind")
        self.durability = _validate_unit_interval(self.durability, "durability")
        self.evidence_spans = self._validate_evidence_spans(self.evidence_spans, "evidence_spans")
        self.negative_evidence = self._validate_evidence_spans(self.negative_evidence, "negative_evidence")
        self.extraction_method = _require_nonblank(self.extraction_method, "extraction_method").strip().lower()
        self.gate_decision = GateDecision.coerce(self.gate_decision, "gate_decision")
        if (
            self.gate_decision != GateDecision.PENDING
            and not str(self.decision_reason or "").strip()
        ):
            raise ValidationError(
                "decision_reason is required for non-pending gate decisions"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "type": cast(MemoryType, self.type).value,
            "claim": self.claim,
            "proposed_destination": self.proposed_destination,
            "importance": self.importance,
            "confidence": self.confidence,
            "promotion_reason": self.promotion_reason,
            "source_refs": list(self.source_refs),
            "claim_kind": cast(CandidateClaimKind, self.claim_kind).value,
            "durability": self.durability,
            "evidence_spans": [dict(span) for span in self.evidence_spans],
            "negative_evidence": [dict(span) for span in self.negative_evidence],
            "extraction_method": self.extraction_method,
            "extractor_version": self.extractor_version,
            "gate_decision": cast(GateDecision, self.gate_decision).value,
            "decision_reason": self.decision_reason,
        }

    @staticmethod
    def _validate_evidence_spans(value: Any, field_name: str) -> List[Dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValidationError(f"{field_name} must be a list")
        validated: List[Dict[str, Any]] = []
        for raw in value:
            if not isinstance(raw, dict):
                raise ValidationError(f"{field_name} entries must be objects")
            span = dict(raw)
            for key in ("source_id", "field", "role", "text"):
                span[key] = _require_nonblank(span.get(key), f"{field_name}.{key}")
            if span["role"] not in {"user", "assistant", "tool"}:
                raise ValidationError(f"{field_name}.role must be user, assistant, or tool")
            if span["field"] not in {"user_content", "assistant_content", "result", "content", "stdout", "stderr"}:
                raise ValidationError(f"{field_name}.field is not evidence-bearing")
            try:
                span["start"] = int(span.get("start"))
                span["end"] = int(span.get("end"))
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"{field_name} offsets must be integers") from exc
            if span["start"] < 0 or span["end"] <= span["start"]:
                raise ValidationError(f"{field_name} offsets are invalid")
            if span["end"] - span["start"] != len(span["text"]):
                raise ValidationError(f"{field_name} offsets must match text length")
            validated.append(span)
        return validated

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateMemory":
        return cls(**data)


@dataclass
class WorkingMemory:
    session_id: str
    updated_at: str = field(default_factory=utc_now_iso)
    focus: Dict[str, Any] = field(default_factory=dict)
    scratchpad: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.session_id = _require_nonblank(self.session_id, "session_id")
        self.focus = dict(self.focus or {})
        default_scratchpad = {
            "relevant_paths": [],
            "relevant_commands": [],
            "retrieved_memory_ids": [],
        }
        merged = dict(default_scratchpad)
        merged.update(dict(self.scratchpad or {}))
        self.scratchpad = merged

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "updated_at": self.updated_at,
            "focus": dict(self.focus),
            "scratchpad": dict(self.scratchpad),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkingMemory":
        return cls(**data)


@dataclass
class MemoryPacket:
    route: str
    confidence: str
    token_budget: int
    items: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sections: Dict[str, Any] = field(default_factory=dict)
    retrieval_plan: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.route = _require_nonblank(self.route, "route")
        self.confidence = _require_nonblank(self.confidence, "confidence")
        try:
            self.token_budget = int(self.token_budget)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "token_budget must be a non-negative integer"
            ) from exc
        if self.token_budget < 0:
            raise ValidationError("token_budget must be a non-negative integer")
        self.items = [dict(item) for item in (self.items or [])]
        self.warnings = _list_of_strings(self.warnings)
        self.sections = dict(self.sections or {})
        self.retrieval_plan = dict(self.retrieval_plan or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route,
            "confidence": self.confidence,
            "token_budget": self.token_budget,
            "items": [dict(item) for item in self.items],
            "warnings": list(self.warnings),
            "sections": dict(self.sections),
            "retrieval_plan": dict(self.retrieval_plan),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryPacket":
        return cls(**data)
