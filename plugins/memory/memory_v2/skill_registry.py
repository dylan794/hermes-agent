"""Portable, dependency-free contracts for a reviewed skill registry.

The registry stores versioned procedure bundles.  It deliberately does not
execute skills and is not part of semantic memory.  Read APIs disclose bounded
metadata first and require an exact version plus content digest before loading
a bundle.  Mutation implementations must own a trusted host authority verifier;
model-supplied values are never authority by themselves.

This module has no Hermes imports so it can be extracted into a standalone
package without carrying the Memory v2 provider or agent runtime with it.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


SKILL_REGISTRY_SCHEMA_VERSION = "memory-v2-skill-registry/v1"
MAX_SKILL_SEARCH_RESULTS = 50
MAX_SKILL_BUNDLE_FILES = 512
MAX_SKILL_BUNDLE_BYTES = 32 * 1024 * 1024
MAX_SKILL_FILE_BYTES = 8 * 1024 * 1024

_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SKILL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]{0,127}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MUTABLE_VERSION_ALIASES = {"head", "latest", "main", "stable", "current"}
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


class SkillRegistryError(ValueError):
    """Base error for invalid portable skill-registry data."""


class SkillAuthorizationError(PermissionError):
    """Raised when a skill mutation is not externally authorized."""


class SkillLifecycle(str, Enum):
    STAGED = "staged"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"


class SkillTrust(str, Enum):
    UNVERIFIED = "unverified"
    REVIEWED = "reviewed"
    TRUSTED = "trusted"


class SkillMutationAction(str, Enum):
    STAGE = "stage"
    ACTIVATE = "activate"
    SUPERSEDE = "supersede"
    REVOKE = "revoke"


class SkillExecutionOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    CANCELLED = "cancelled"


def _nonblank(value: Any, field_name: str, *, maximum: int = 4096) -> str:
    text = str(value or "").strip()
    if not text:
        raise SkillRegistryError(f"{field_name} must be nonblank")
    if len(text) > maximum:
        raise SkillRegistryError(f"{field_name} exceeds {maximum} characters")
    return text


def _timestamp(value: Any, field_name: str) -> str:
    text = _nonblank(value, field_name, maximum=64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SkillRegistryError(f"{field_name} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SkillRegistryError(f"{field_name} must include a timezone offset")
    return text


def _timestamp_value(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _digest(value: Any, field_name: str) -> str:
    text = str(value or "").strip().lower()
    if not _DIGEST_RE.fullmatch(text):
        raise SkillRegistryError(f"{field_name} must be a sha256: digest")
    return text


def _string_set(values: Sequence[str] | None, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise SkillRegistryError(f"{field_name} must be a sequence of strings")
    normalized = {_nonblank(value, field_name, maximum=512) for value in values}
    return tuple(sorted(normalized))


def normalize_skill_bundle_path(path: str) -> str:
    """Return a canonical POSIX bundle path or reject an unsafe path."""
    raw = _nonblank(path, "bundle path", maximum=1024).replace("\\", "/")
    pure = PurePosixPath(raw)
    parts = pure.parts
    if raw.startswith("/") or pure.is_absolute() or not parts:
        raise SkillRegistryError(f"unsafe bundle path: {path}")
    if any(part in {"", ".", ".."} for part in parts):
        raise SkillRegistryError(f"unsafe bundle path: {path}")
    normalized_parts: list[str] = []
    for raw_part in parts:
        part = unicodedata.normalize("NFC", raw_part)
        stem = part.split(".", 1)[0].upper()
        if (
            not part
            or part.endswith((" ", "."))
            or stem in _WINDOWS_RESERVED_NAMES
            or any(ord(char) < 32 or char in '<>:"|?*' for char in part)
        ):
            raise SkillRegistryError(f"unsafe bundle path: {path}")
        normalized_parts.append(part)
    return "/".join(normalized_parts)


def normalize_skill_bundle_files(files: Mapping[str, bytes]) -> dict[str, bytes]:
    """Validate bundle bounds and return canonical path-to-bytes content."""
    if not isinstance(files, Mapping):
        raise SkillRegistryError("files must be a mapping")
    if not files or len(files) > MAX_SKILL_BUNDLE_FILES:
        raise SkillRegistryError(
            f"bundle must contain between 1 and {MAX_SKILL_BUNDLE_FILES} files"
        )
    normalized: dict[str, bytes] = {}
    portable_paths: set[str] = set()
    total = 0
    for raw_path, raw_content in files.items():
        path = normalize_skill_bundle_path(str(raw_path))
        portable_path = path.casefold()
        if path in normalized or portable_path in portable_paths:
            raise SkillRegistryError(f"duplicate canonical bundle path: {path}")
        if not isinstance(raw_content, bytes):
            raise SkillRegistryError(f"bundle file must be bytes: {path}")
        if len(raw_content) > MAX_SKILL_FILE_BYTES:
            raise SkillRegistryError(f"bundle file exceeds size limit: {path}")
        total += len(raw_content)
        if total > MAX_SKILL_BUNDLE_BYTES:
            raise SkillRegistryError("bundle exceeds total size limit")
        normalized[path] = raw_content
        portable_paths.add(portable_path)
    if "SKILL.md" not in normalized:
        raise SkillRegistryError("bundle must contain SKILL.md")
    return normalized


def compute_skill_bundle_digest(files: Mapping[str, bytes]) -> str:
    """Hash an exact bundle tree independently of mapping iteration order."""
    normalized = normalize_skill_bundle_files(files)
    digest = hashlib.sha256()
    digest.update(b"memory-v2-skill-bundle-v1\x00")
    for path in sorted(normalized):
        path_bytes = path.encode("utf-8")
        content = normalized[path]
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return f"sha256:{digest.hexdigest()}"


def _canonical_digest(payload: Mapping[str, Any], domain: str) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256(domain.encode("utf-8") + b"\x00" + encoded).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True)
class SkillRef:
    """An immutable, fully pinned reference; name-only/latest refs are invalid."""

    skill_id: str
    version: str
    bundle_digest: str

    def __post_init__(self) -> None:
        skill_id = _nonblank(self.skill_id, "skill_id", maximum=128).lower()
        if not _SKILL_ID_RE.fullmatch(skill_id) or ".." in skill_id.split("/"):
            raise SkillRegistryError("skill_id must be a portable lowercase identifier")
        object.__setattr__(self, "skill_id", skill_id)
        version = _nonblank(self.version, "version", maximum=128)
        if version.strip().lower() in _MUTABLE_VERSION_ALIASES:
            raise SkillRegistryError("version must be immutable, not a moving alias")
        object.__setattr__(self, "version", version)
        object.__setattr__(
            self,
            "bundle_digest",
            _digest(self.bundle_digest, "bundle_digest"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "bundle_digest": self.bundle_digest,
        }


@dataclass(frozen=True)
class SkillProvenance:
    source_type: str
    source_id: str
    source_uri: str
    observed_at: str
    acquired_at: str
    evidence_refs: tuple[str, ...] = ()
    signer_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_type", _nonblank(self.source_type, "source_type", maximum=64))
        object.__setattr__(self, "source_id", _nonblank(self.source_id, "source_id", maximum=512))
        object.__setattr__(self, "source_uri", _nonblank(self.source_uri, "source_uri", maximum=4096))
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at, "observed_at"))
        object.__setattr__(self, "acquired_at", _timestamp(self.acquired_at, "acquired_at"))
        if _timestamp_value(self.acquired_at) < _timestamp_value(self.observed_at):
            raise SkillRegistryError("acquired_at cannot precede observed_at")
        object.__setattr__(self, "evidence_refs", _string_set(self.evidence_refs, "evidence_refs"))
        object.__setattr__(self, "signer_refs", _string_set(self.signer_refs, "signer_refs"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source_uri": self.source_uri,
            "observed_at": self.observed_at,
            "acquired_at": self.acquired_at,
            "evidence_refs": list(self.evidence_refs),
            "signer_refs": list(self.signer_refs),
        }


@dataclass(frozen=True)
class SkillBundle:
    ref: SkillRef
    files: Mapping[str, bytes]

    def __post_init__(self) -> None:
        normalized = normalize_skill_bundle_files(self.files)
        actual = compute_skill_bundle_digest(normalized)
        if actual != self.ref.bundle_digest:
            raise SkillRegistryError("bundle content does not match the pinned bundle_digest")
        object.__setattr__(self, "files", MappingProxyType(normalized))


@dataclass(frozen=True)
class SkillDescriptor:
    ref: SkillRef
    name: str
    description: str
    provenance: SkillProvenance
    created_at: str
    lifecycle: SkillLifecycle = SkillLifecycle.STAGED
    trust: SkillTrust = SkillTrust.UNVERIFIED
    format_id: str = "agentskills.io/v1"
    reviewed_at: str | None = None
    activated_at: str | None = None
    superseded_at: str | None = None
    superseded_by: SkillRef | None = None
    revoked_at: str | None = None
    revocation_reason: str | None = None
    compatibility: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metadata_is_untrusted: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "lifecycle", SkillLifecycle(self.lifecycle))
        object.__setattr__(self, "trust", SkillTrust(self.trust))
        name = _nonblank(self.name, "name", maximum=64).lower()
        if not _SKILL_NAME_RE.fullmatch(name):
            raise SkillRegistryError("name must use lowercase letters, digits, and hyphens")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", _nonblank(self.description, "description", maximum=1024))
        object.__setattr__(self, "format_id", _nonblank(self.format_id, "format_id", maximum=128))
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))
        for field_name in ("reviewed_at", "activated_at", "superseded_at", "revoked_at"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _timestamp(value, field_name))
        object.__setattr__(self, "compatibility", _string_set(self.compatibility, "compatibility"))
        object.__setattr__(
            self,
            "required_capabilities",
            _string_set(self.required_capabilities, "required_capabilities"),
        )
        object.__setattr__(self, "tags", _string_set(self.tags, "tags"))
        if self.metadata_is_untrusted is not True:
            raise SkillRegistryError("skill-supplied metadata must remain marked untrusted")
        if self.lifecycle == SkillLifecycle.ACTIVE:
            if self.trust == SkillTrust.UNVERIFIED or not self.reviewed_at or not self.activated_at:
                raise SkillRegistryError("active skills require review, activation time, and non-unverified trust")
        if self.lifecycle == SkillLifecycle.SUPERSEDED:
            if not self.superseded_at or self.superseded_by is None:
                raise SkillRegistryError("superseded skills require superseded_at and superseded_by")
        if self.lifecycle == SkillLifecycle.REVOKED:
            if not self.revoked_at or not str(self.revocation_reason or "").strip():
                raise SkillRegistryError("revoked skills require revoked_at and revocation_reason")

    def record_fingerprint(self) -> str:
        payload = {
            "schema": SKILL_REGISTRY_SCHEMA_VERSION,
            "ref": self.ref.to_dict(),
            "name": self.name,
            "description": self.description,
            "provenance": self.provenance.to_dict(),
            "lifecycle": self.lifecycle.value,
            "trust": self.trust.value,
            "format_id": self.format_id,
            "created_at": self.created_at,
            "reviewed_at": self.reviewed_at,
            "activated_at": self.activated_at,
            "superseded_at": self.superseded_at,
            "superseded_by": self.superseded_by.to_dict() if self.superseded_by else None,
            "revoked_at": self.revoked_at,
            "revocation_reason": self.revocation_reason,
            "compatibility": list(self.compatibility),
            "required_capabilities": list(self.required_capabilities),
            "tags": list(self.tags),
            "metadata_is_untrusted": True,
        }
        return _canonical_digest(payload, "memory-v2-skill-record-v1")


@dataclass(frozen=True)
class SkillSearchQuery:
    text: str = ""
    limit: int = 10
    lifecycle: tuple[SkillLifecycle, ...] = (SkillLifecycle.ACTIVE,)
    required_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", str(self.text or "").strip())
        if not isinstance(self.limit, int) or isinstance(self.limit, bool):
            raise SkillRegistryError("limit must be an integer")
        if not 1 <= self.limit <= MAX_SKILL_SEARCH_RESULTS:
            raise SkillRegistryError(f"limit must be between 1 and {MAX_SKILL_SEARCH_RESULTS}")
        lifecycle = tuple(SkillLifecycle(value) for value in self.lifecycle)
        if not lifecycle:
            raise SkillRegistryError("lifecycle filter must be nonempty")
        object.__setattr__(self, "lifecycle", lifecycle)
        object.__setattr__(
            self,
            "required_capabilities",
            _string_set(self.required_capabilities, "required_capabilities"),
        )


@dataclass(frozen=True)
class SkillVerification:
    ref: SkillRef
    verified: bool
    verified_at: str
    verifier: str
    failures: tuple[str, ...] = ()
    signer_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "verified_at", _timestamp(self.verified_at, "verified_at"))
        object.__setattr__(self, "verifier", _nonblank(self.verifier, "verifier", maximum=256))
        object.__setattr__(self, "failures", _string_set(self.failures, "failures"))
        object.__setattr__(self, "signer_refs", _string_set(self.signer_refs, "signer_refs"))
        if self.verified and self.failures:
            raise SkillRegistryError("verified results cannot contain failures")


@dataclass(frozen=True)
class SourceSkillCandidate:
    """Untrusted discovery metadata from an external skill source."""

    ref: SkillRef
    name: str
    description: str
    provenance: SkillProvenance

    def __post_init__(self) -> None:
        name = _nonblank(self.name, "name", maximum=64).lower()
        if not _SKILL_NAME_RE.fullmatch(name):
            raise SkillRegistryError("name must use lowercase letters, digits, and hyphens")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "description",
            _nonblank(self.description, "description", maximum=1024),
        )

    def candidate_fingerprint(self) -> str:
        return _canonical_digest(
            {
                "ref": self.ref.to_dict(),
                "name": self.name,
                "description": self.description,
                "provenance": self.provenance.to_dict(),
            },
            "memory-v2-source-skill-candidate-v1",
        )


@dataclass(frozen=True)
class SkillMutationIntent:
    action: SkillMutationAction
    subject_ref: SkillRef
    candidate_fingerprint: str
    confirm: bool
    reason: str
    requested_by: str
    requested_at: str
    expected_record_fingerprint: str | None = None
    replacement_ref: SkillRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", SkillMutationAction(self.action))
        object.__setattr__(
            self,
            "candidate_fingerprint",
            _digest(self.candidate_fingerprint, "candidate_fingerprint"),
        )
        object.__setattr__(self, "reason", _nonblank(self.reason, "reason", maximum=2048))
        object.__setattr__(self, "requested_by", _nonblank(self.requested_by, "requested_by", maximum=256))
        object.__setattr__(self, "requested_at", _timestamp(self.requested_at, "requested_at"))
        if self.expected_record_fingerprint is not None:
            object.__setattr__(
                self,
                "expected_record_fingerprint",
                _digest(self.expected_record_fingerprint, "expected_record_fingerprint"),
            )
        if self.action in {SkillMutationAction.STAGE, SkillMutationAction.ACTIVATE}:
            if self.candidate_fingerprint != self.subject_ref.bundle_digest:
                raise SkillRegistryError("stage/activate candidate_fingerprint must equal the reviewed bundle digest")
        if self.action == SkillMutationAction.SUPERSEDE:
            if self.replacement_ref is None or self.replacement_ref == self.subject_ref:
                raise SkillRegistryError("supersede requires a distinct replacement_ref")
            if self.candidate_fingerprint != self.replacement_ref.bundle_digest:
                raise SkillRegistryError("supersede candidate_fingerprint must equal the replacement bundle digest")
        if self.action == SkillMutationAction.REVOKE:
            if self.expected_record_fingerprint is None:
                raise SkillRegistryError("revoke requires expected_record_fingerprint")
            if self.candidate_fingerprint != self.expected_record_fingerprint:
                raise SkillRegistryError("revoke candidate_fingerprint must equal the reviewed record fingerprint")
        if self.action != SkillMutationAction.STAGE and self.expected_record_fingerprint is None:
            raise SkillRegistryError("canonical mutations require expected_record_fingerprint")

    def request_fingerprint(self) -> str:
        return _canonical_digest(
            {
                "action": self.action.value,
                "subject_ref": self.subject_ref.to_dict(),
                "candidate_fingerprint": self.candidate_fingerprint,
                "confirm": self.confirm,
                "reason": self.reason,
                "requested_by": self.requested_by,
                "requested_at": self.requested_at,
                "expected_record_fingerprint": self.expected_record_fingerprint,
                "replacement_ref": self.replacement_ref.to_dict() if self.replacement_ref else None,
            },
            "memory-v2-skill-mutation-intent-v1",
        )


@dataclass(frozen=True)
class SkillAuthorityDecision:
    authorized: bool
    principal_ref: str
    authority_class: str
    scopes: tuple[SkillMutationAction, ...]
    request_fingerprint: str
    issued_at: str
    expires_at: str
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "principal_ref", _nonblank(self.principal_ref, "principal_ref", maximum=256))
        authority_class = _nonblank(self.authority_class, "authority_class", maximum=32).lower()
        if authority_class not in {"host", "operator"}:
            raise SkillRegistryError("authority_class must be host or operator")
        object.__setattr__(self, "authority_class", authority_class)
        object.__setattr__(self, "scopes", tuple(SkillMutationAction(scope) for scope in self.scopes))
        object.__setattr__(
            self,
            "request_fingerprint",
            _digest(self.request_fingerprint, "request_fingerprint"),
        )
        object.__setattr__(self, "issued_at", _timestamp(self.issued_at, "issued_at"))
        object.__setattr__(self, "expires_at", _timestamp(self.expires_at, "expires_at"))
        object.__setattr__(self, "reason", str(self.reason or ""))
        if _timestamp_value(self.expires_at) < _timestamp_value(self.issued_at):
            raise SkillRegistryError("expires_at cannot precede issued_at")


@runtime_checkable
class SkillAuthorityVerifier(Protocol):
    """Host-owned verifier for opaque, non-model mutation authority."""

    def verify(self, authority: object, intent: SkillMutationIntent) -> SkillAuthorityDecision:
        ...


def require_skill_mutation_authority(
    intent: SkillMutationIntent,
    authority: object,
    verifier: SkillAuthorityVerifier,
    *,
    now: datetime | None = None,
) -> SkillAuthorityDecision:
    """Fail closed unless a host verifier binds authority to this exact intent."""
    if intent.confirm is not True:
        raise SkillAuthorizationError("explicit confirm=true is required")
    if authority is None:
        raise SkillAuthorizationError("trusted host/operator authority is required")
    try:
        decision = verifier.verify(authority, intent)
    except Exception as exc:
        raise SkillAuthorizationError("authority verification failed closed") from exc
    if not decision.authorized:
        raise SkillAuthorizationError(decision.reason or "authority denied the mutation")
    if decision.request_fingerprint != intent.request_fingerprint():
        raise SkillAuthorizationError("authority is not bound to this mutation fingerprint")
    if intent.action not in decision.scopes:
        raise SkillAuthorizationError("authority scope does not permit this mutation")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if current < _timestamp_value(decision.issued_at) or current > _timestamp_value(decision.expires_at):
        raise SkillAuthorizationError("authority is not currently valid")
    return decision


@dataclass(frozen=True)
class SkillMutationReceipt:
    operation_id: str
    action: SkillMutationAction
    ref: SkillRef
    record_fingerprint: str
    principal_ref: str
    committed_at: str
    audit_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _nonblank(self.operation_id, "operation_id", maximum=256))
        object.__setattr__(self, "action", SkillMutationAction(self.action))
        object.__setattr__(
            self,
            "record_fingerprint",
            _digest(self.record_fingerprint, "record_fingerprint"),
        )
        object.__setattr__(self, "principal_ref", _nonblank(self.principal_ref, "principal_ref", maximum=256))
        object.__setattr__(self, "committed_at", _timestamp(self.committed_at, "committed_at"))
        object.__setattr__(self, "audit_refs", _string_set(self.audit_refs, "audit_refs"))


@dataclass(frozen=True)
class SkillExecutionReceipt:
    """Non-authoritative host outcome suitable for append-only raw capture."""

    execution_id: str
    ref: SkillRef
    host_id: str
    outcome: SkillExecutionOutcome
    started_at: str
    completed_at: str
    granted_capabilities: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    output_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_id", _nonblank(self.execution_id, "execution_id", maximum=256))
        object.__setattr__(self, "host_id", _nonblank(self.host_id, "host_id", maximum=256))
        object.__setattr__(self, "outcome", SkillExecutionOutcome(self.outcome))
        object.__setattr__(self, "started_at", _timestamp(self.started_at, "started_at"))
        object.__setattr__(self, "completed_at", _timestamp(self.completed_at, "completed_at"))
        if _timestamp_value(self.completed_at) < _timestamp_value(self.started_at):
            raise SkillRegistryError("completed_at cannot precede started_at")
        object.__setattr__(
            self,
            "granted_capabilities",
            _string_set(self.granted_capabilities, "granted_capabilities"),
        )
        object.__setattr__(self, "evidence_refs", _string_set(self.evidence_refs, "evidence_refs"))
        if self.output_digest is not None:
            object.__setattr__(self, "output_digest", _digest(self.output_digest, "output_digest"))


@runtime_checkable
class SkillSourceAdapter(Protocol):
    """External catalog adapter; returned metadata is always untrusted."""

    def source_id(self) -> str:
        ...

    def discover(self, query: SkillSearchQuery) -> Sequence[SourceSkillCandidate]:
        ...

    def fetch(self, candidate: SourceSkillCandidate) -> SkillBundle:
        ...


@runtime_checkable
class SkillRegistry(Protocol):
    """Bounded read-only registry interface used by hosts and retrievers."""

    def search(self, query: SkillSearchQuery) -> Sequence[SkillDescriptor]:
        ...

    def describe(self, ref: SkillRef) -> SkillDescriptor:
        ...

    def load_bundle(self, ref: SkillRef) -> SkillBundle:
        ...

    def verify(self, ref: SkillRef) -> SkillVerification:
        ...


@runtime_checkable
class SkillRegistryWriter(Protocol):
    """Trusted-host mutation lane; implementations own their authority verifier."""

    def stage(
        self,
        bundle: SkillBundle,
        provenance: SkillProvenance,
        intent: SkillMutationIntent,
        authority: object,
    ) -> SkillMutationReceipt:
        ...

    def activate(
        self,
        ref: SkillRef,
        intent: SkillMutationIntent,
        authority: object,
    ) -> SkillMutationReceipt:
        ...

    def supersede(
        self,
        current_ref: SkillRef,
        replacement_ref: SkillRef,
        intent: SkillMutationIntent,
        authority: object,
    ) -> SkillMutationReceipt:
        ...

    def revoke(
        self,
        ref: SkillRef,
        intent: SkillMutationIntent,
        authority: object,
    ) -> SkillMutationReceipt:
        ...
