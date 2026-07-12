"""Profile-scoped file store for Memory v2 records.

The store owns Memory v2's human-readable canonical files and append-only JSONL
inbox files. Indexes are intentionally out of scope here; this layer only
provides safe local persistence for raw events, candidates, and project cards.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import fcntl

import yaml

from .redaction import redact_data, redact_text, redaction_metadata
from .schemas import (
    ArtifactRecord,
    ArtifactSegment,
    CandidateMemory,
    CoreMemoryRecord,
    GateDecision,
    MemoryItem,
    ProjectCard,
    SourceRef,
    ValidationError,
    WorkingMemory,
    normalize_project_id,
    utc_now_iso,
)


MEMORY_V2_DIRS = [
    "working",
    "core",
    "sources",
    "inbox",
    "semantic",
    "semantic/items",
    "semantic/projects",
    "semantic/environment",
    "episodic",
    "episodic/daily",
    "episodic/dream",
    "episodic/sessions",
    "graph",
    "indexes",
    "indexes/vector",
    "evals",
    "reports",
    "reports/daily_consolidation",
    "reports/dream_cycles",
    "reports/weekly_reflection",
    "audit",
    "artifacts",
    "artifacts/raw",
    "artifacts/derived",
    "artifacts/manifests",
    "artifacts/indexes",
    "privacy",
]

RAW_EVENT_SCHEMA_VERSION = 1
_RAW_EVENT_HASH_FIELDS = {
    "content_sha256",
    "record_sha256",
    "previous_record_sha256",
    "chain_index",
}
_RAW_EVENT_TRANSIENT_FIELDS = {
    "_raw_index_metadata",
}
_RAW_EVENT_CONTENT_LIMIT = 24_000
_RAW_EVENT_TOOL_CONTENT_LIMIT = 8_000
_DEFAULT_PROFILE_LOCK_TIMEOUT_SECONDS = 5.0


class MemoryV2Store:
    """Small local file store rooted at ``{hermes_home}/memory_v2``."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self._raw_event_lock = threading.Lock()

    @property
    def inbox_dir(self) -> Path:
        return self.base_dir / "inbox"

    @property
    def core_dir(self) -> Path:
        return self.base_dir / "core"

    @property
    def projects_dir(self) -> Path:
        return self.base_dir / "semantic" / "projects"

    @property
    def memory_items_dir(self) -> Path:
        return self.base_dir / "semantic" / "items"

    @property
    def sources_dir(self) -> Path:
        return self.base_dir / "sources"

    @property
    def working_dir(self) -> Path:
        return self.base_dir / "working"

    @property
    def episodic_sessions_dir(self) -> Path:
        return self.base_dir / "episodic" / "sessions"

    @property
    def artifacts_dir(self) -> Path:
        return self.base_dir / "artifacts"

    @property
    def raw_artifacts_dir(self) -> Path:
        return self.artifacts_dir / "raw"

    @property
    def derived_artifacts_dir(self) -> Path:
        return self.artifacts_dir / "derived"

    @property
    def artifact_manifests_dir(self) -> Path:
        return self.artifacts_dir / "manifests"

    @property
    def current_working_path(self) -> Path:
        return self.working_dir / "current.yaml"

    @property
    def open_loops_path(self) -> Path:
        return self.working_dir / "open_loops.yaml"

    @property
    def raw_events_path(self) -> Path:
        return self.inbox_dir / "raw_events.jsonl"

    @property
    def raw_archive_manifest_path(self) -> Path:
        return self.inbox_dir / "raw_events.manifest.yaml"

    @property
    def raw_event_tombstones_path(self) -> Path:
        return self.base_dir / "privacy" / "raw_event_tombstones.yaml"

    @property
    def default_index_path(self) -> Path:
        return self.base_dir / "indexes" / "memory.sqlite"

    @property
    def candidates_path(self) -> Path:
        return self.inbox_dir / "candidates.jsonl"

    @property
    def rejected_path(self) -> Path:
        return self.inbox_dir / "rejected.jsonl"

    @property
    def operations_path(self) -> Path:
        return self.base_dir / "audit" / "operations.jsonl"

    def initialize(self) -> None:
        """Create the Memory v2 profile-scoped directory tree and seed files."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for rel in MEMORY_V2_DIRS:
            (self.base_dir / rel).mkdir(parents=True, exist_ok=True)

        lock_path = self.base_dir / "audit" / "profile.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.touch(exist_ok=True)

        readme = self.base_dir / "README.md"
        if not readme.exists():
            self._atomic_write_text(
                readme,
                "# Memory v2\n\n"
                "Profile-scoped local memory store for Hermes. Canonical records "
                "live in human-readable files; indexes are derived and rebuildable.\n",
            )

        config = self.base_dir / "config.yaml"
        if not config.exists():
            self._atomic_write_text(
                config,
                "version: 1\n"
                "online:\n"
                "  default_packet_budget_tokens: 1500\n"
                "embeddings:\n"
                "  enabled: false\n"
                "graph:\n"
                "  enabled: true\n",
            )

        for path in (self.raw_events_path, self.candidates_path, self.rejected_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.touch()
        # Do not materialize the operations log during initialization.  Dream
        # cycle ``--auto-apply off`` may initialize/read the store on a live
        # profile, and creating an empty audit file makes a non-mutating run
        # look like it touched mutation/audit state.  The audit directory is
        # created by the layout above; the JSONL file is created lazily by the
        # first audited operation.

    @property
    def profile_lock_path(self) -> Path:
        return self.base_dir / "audit" / "profile.lock"

    @contextlib.contextmanager
    def profile_lock(self, *, timeout: float = _DEFAULT_PROFILE_LOCK_TIMEOUT_SECONDS):
        """Hold the profile-wide mutation lock with a bounded timeout.

        The lock protects all canonical Memory v2 mutations across processes.
        Callers fail closed instead of proceeding without the lock.
        """
        self.profile_lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self.profile_lock_path.open("a+b") as fh:
            while True:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if time.monotonic() >= deadline:
                        raise ValidationError("Memory v2 profile lock timeout; refusing to mutate without exclusive lock") from exc
                    time.sleep(0.01)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def append_raw_event(self, event: Dict[str, Any], *, lock_timeout: float = _DEFAULT_PROFILE_LOCK_TIMEOUT_SECONDS) -> Dict[str, Any]:
        """Append a redacted, tamper-evident raw event to ``inbox/raw_events.jsonl``.

        The raw archive is the evidence layer for Memory v2.  It is append-only,
        versioned, redacted before hashing, and linked with a cheap hash chain so
        later health checks can detect accidental or manual tampering without
        needing embeddings or an expensive database pass during normal chat.
        """
        if not isinstance(event, dict):
            raise ValidationError("raw event must be a JSON object")
        with self.profile_lock(timeout=lock_timeout), self._raw_event_lock:
            previous = self._last_raw_event_for_chain()
            payload = self._normalize_raw_event(event, previous=previous)
            byte_offset, byte_length = self._append_jsonl(self.raw_events_path, payload, fsync=True)
            self.write_source_ref(self._source_ref_from_raw_event(payload))
            self._update_raw_archive_manifest_after_append(payload)
            self._update_derived_raw_index_after_append(payload, byte_offset=byte_offset, byte_length=byte_length)
            return payload

    def read_raw_events(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Read raw archive events.

        ``limit`` is the safe/debug path and reads the bounded tail.  The
        unbounded form is retained for explicit repair/rebuild callers and
        legacy tests only; hot archive APIs must use the indexed bounded helpers
        below and must not silently full-scan this JSONL file.
        """
        if limit is None:
            return self._read_jsonl(self.raw_events_path)
        safe_limit = int(limit)
        if safe_limit < 0:
            raise ValidationError("limit must be non-negative")
        if safe_limit == 0:
            return []
        return self._read_jsonl_tail(self.raw_events_path, safe_limit)

    def read_all_raw_events_for_repair(self) -> List[Dict[str, Any]]:
        """Explicit unbounded raw archive read for verify/rebuild/repair paths."""
        return self._read_jsonl(self.raw_events_path)

    def iter_raw_events(
        self,
        *,
        query: str = "",
        session_id: str = "",
        event_type: str = "",
        source_ids: List[str] | None = None,
        created_after: str = "",
        created_before: str = "",
        limit: int = 100,
        index: Any = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield bounded raw events via SQLite metadata + JSONL byte slices."""
        for event in self.search_raw_events(
            query=query,
            session_id=session_id,
            event_type=event_type,
            source_ids=source_ids,
            created_after=created_after,
            created_before=created_before,
            limit=limit,
            index=index,
        ):
            yield event

    def get_raw_event_by_id(self, event_id: str, *, index: Any = None) -> Dict[str, Any] | None:
        if self.is_raw_event_tombstoned(event_id):
            return self._tombstoned_raw_event_stub(event_id)
        raw_index = self._require_usable_raw_index(index)
        metadata = raw_index.raw_event_metadata(str(event_id or ""))
        if metadata is None:
            return None
        return self._hydrate_raw_event_metadata(metadata)

    def get_raw_events_by_ids(self, ids: List[str], *, index: Any = None) -> List[Dict[str, Any]]:
        raw_index = self._require_usable_raw_index(index)
        return [
            event
            for metadata in raw_index.raw_event_metadata_many([str(item) for item in ids])
            if not self.is_raw_event_tombstoned(str(metadata.get("id") or ""))
            if (event := self._hydrate_raw_event_metadata(metadata)) is not None
        ]

    def read_raw_event_at_offset(self, byte_offset: int, byte_length: int) -> Dict[str, Any]:
        try:
            offset = int(byte_offset)
            length = int(byte_length)
        except (TypeError, ValueError) as exc:
            raise ValidationError("byte_offset and byte_length must be integers") from exc
        if offset < 0 or length <= 0:
            raise ValidationError("byte_offset must be non-negative and byte_length must be positive")
        if not self.raw_events_path.exists():
            raise ValidationError("raw archive JSONL is missing")
        with self.raw_events_path.open("rb") as fh:
            fh.seek(offset)
            data = fh.read(length)
        if not data:
            raise ValidationError("raw event byte slice is empty")
        if b"\n" in data.rstrip(b"\n"):
            raise ValidationError("raw event byte slice spans multiple JSONL records")
        try:
            payload = json.loads(data.strip().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError("raw event byte slice is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValidationError("raw event byte slice must decode to a JSON object")
        return payload

    def search_raw_events(
        self,
        query: str = "",
        *,
        session_id: str = "",
        event_type: str = "",
        source_ids: List[str] | None = None,
        created_after: str = "",
        created_before: str = "",
        limit: int = 5,
        index: Any = None,
    ) -> List[Dict[str, Any]]:
        raw_index = self._require_usable_raw_index(index)
        try:
            safe_limit = max(1, min(int(limit), 50))
        except (TypeError, ValueError) as exc:
            raise ValidationError("limit must be an integer") from exc
        metadata_rows = raw_index.search_raw_archive(
            query,
            session_id=session_id,
            event_type=event_type,
            source_ids=source_ids,
            created_after=created_after,
            created_before=created_before,
            limit=safe_limit + 1,
        )
        hydrated: List[Dict[str, Any]] = []
        for metadata in metadata_rows[:safe_limit]:
            if self.is_raw_event_tombstoned(str(metadata.get("id") or "")):
                continue
            event = self._hydrate_raw_event_metadata(metadata)
            if event is not None:
                hydrated.append(event)
        return hydrated

    def raw_event_exists(self, event_id: str, *, index: Any = None) -> bool:
        if self.is_raw_event_tombstoned(event_id):
            return False
        return self._require_usable_raw_index(index).raw_event_exists(event_id)

    def canonical_raw_event_exists(self, event_id: str, *, index: Any = None) -> bool:
        """Verify that indexed raw evidence resolves to an intact canonical JSONL row."""
        safe_id = str(event_id or "").strip()
        if not safe_id or self.is_raw_event_tombstoned(safe_id):
            return False
        raw_index = self._require_usable_raw_index(index)
        metadata = raw_index.raw_event_metadata(safe_id)
        if not metadata:
            return False
        try:
            event = self._hydrate_raw_event_metadata(metadata)
        except (OSError, ValueError, ValidationError, json.JSONDecodeError):
            return False
        if not event:
            return False
        recorded_content_hash = str(event.get("content_sha256") or "")
        recorded_record_hash = str(event.get("record_sha256") or "")
        if not recorded_content_hash or not recorded_record_hash:
            return False
        return (
            recorded_content_hash == self._raw_event_content_hash(event)
            and recorded_record_hash == self._raw_event_record_hash(event)
            and recorded_record_hash == str(metadata.get("record_sha256") or "")
        )

    def source_ref_exists(self, source_id: str, *, index: Any = None) -> bool:
        """Return whether a source ref is usable for grounding.

        Raw archive sidecars and derived index rows are not independently
        sufficient evidence. When a SourceRef points at ``raw_event:<id>``, the
        indexed byte slice must hydrate from canonical JSONL and pass hash checks.
        """
        safe_id = str(source_id or "").strip()
        if not safe_id:
            return False
        source = self.read_source_ref(safe_id)
        if source is not None:
            raw_id = self.raw_event_id_from_source_ref(source)
            if raw_id:
                try:
                    return self.canonical_raw_event_exists(raw_id, index=index)
                except ValidationError:
                    return False
            uri = str(source.uri or "").strip()
            if uri.lower().startswith("artifact:"):
                artifact_id = uri.split(":", 1)[1].strip()
                return bool(artifact_id and self.read_artifact_record(artifact_id) is not None)
            if uri.lower().startswith("memory:"):
                memory_id = uri.split(":", 1)[1].strip()
                return any(item.id == memory_id for item in self.list_memory_items())
            # A writable SourceRef sidecar is metadata, not proof. Manual, web,
            # file, session, and message labels require a canonical raw/artifact/
            # memory record before they can ground promotion.
            return False
        try:
            return self.canonical_raw_event_exists(safe_id, index=index)
        except ValidationError:
            return False

    @staticmethod
    def raw_event_id_from_source_ref(source: SourceRef | Dict[str, Any]) -> str:
        uri = str(source.uri if isinstance(source, SourceRef) else source.get("uri") or "").strip()
        lowered = uri.lower()
        if lowered.startswith("raw_event:"):
            return uri.split(":", 1)[1].strip()
        return ""

    def is_raw_event_tombstoned(self, event_id: str) -> bool:
        event_id = str(event_id or "").strip()
        if not event_id:
            return False
        return event_id in self.list_raw_event_tombstones()

    def list_raw_event_tombstones(self) -> Dict[str, Dict[str, Any]]:
        if not self.raw_event_tombstones_path.exists():
            return {}
        with self.raw_event_tombstones_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            return {}
        records = loaded.get("events", loaded)
        if not isinstance(records, dict):
            return {}
        return {str(key): dict(value or {}) for key, value in records.items()}

    def tombstone_raw_event(
        self,
        event_id: str,
        *,
        reason: str = "",
        actor: str = "manual",
    ) -> Dict[str, Any]:
        """Mark raw evidence unavailable without rewriting append-only JSONL.

        Tombstones are privacy-safe metadata only: ids, timestamps, actor/reason
        strings after redaction, and record hashes. Raw text is never copied into
        tombstone files or audit metadata.
        """
        safe_id = str(event_id or "").strip()
        if not safe_id:
            raise ValidationError("event_id is required")
        existing = self.list_raw_event_tombstones()
        if safe_id in existing:
            return {"event_id": safe_id, "status": "already_tombstoned", **existing[safe_id]}
        record_hash = ""
        try:
            raw_index = self._require_usable_raw_index(None)
            metadata = raw_index.raw_event_metadata(safe_id) or {}
            record_hash = str(metadata.get("record_sha256") or "")
        except Exception:
            record_hash = ""
        now = utc_now_iso()
        tombstone = {
            "status": "tombstoned",
            "tombstoned_at": now,
            "actor": redact_text(str(actor or "manual"))[:200],
            "reason": redact_text(str(reason or ""))[:500],
            "record_sha256": record_hash,
        }
        existing[safe_id] = tombstone
        payload = {"schema_version": 1, "updated_at": now, "events": existing}
        self._atomic_write_yaml(self.raw_event_tombstones_path, payload)

        source = self.read_source_ref(safe_id)
        if source is not None:
            self.write_source_ref(
                SourceRef(
                    id=safe_id,
                    type=source.type,
                    uri=f"raw_event:{safe_id}",
                    title="Raw event tombstoned / unavailable",
                    observed_at=source.observed_at,
                    quote="[TOMBSTONED RAW EVENT]",
                )
            )
        self.append_operation_record(
            {
                "operation": "raw_event_tombstoned",
                "actor": actor,
                "reason": reason,
                "before_ids": [safe_id],
                "after_ids": [safe_id],
                "metadata": {"record_sha256": record_hash, "raw_text_copied": False},
            }
        )
        return {"event_id": safe_id, **tombstone}

    def raw_import_key_exists(self, import_key_sha256: str, *, index: Any = None) -> bool:
        return self._require_usable_raw_index(index).raw_import_key_exists(import_key_sha256)

    def get_raw_event_neighbors(self, event_id: str, *, index: Any = None) -> Dict[str, Dict[str, Any] | None]:
        raw_index = self._require_usable_raw_index(index)
        metadata = raw_index.raw_event_neighbors(event_id)
        return {
            "previous": self._hydrate_raw_event_metadata(metadata["previous"]) if metadata.get("previous") else None,
            "next": self._hydrate_raw_event_metadata(metadata["next"]) if metadata.get("next") else None,
        }

    def count_raw_events(self) -> int:
        return self._count_jsonl(self.raw_events_path)

    def read_raw_archive_manifest(self) -> Dict[str, Any]:
        if not self.raw_archive_manifest_path.exists():
            return self.rebuild_raw_archive_manifest()
        with self.raw_archive_manifest_path.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        return payload if isinstance(payload, dict) else self.rebuild_raw_archive_manifest()

    def rebuild_raw_archive_manifest(self) -> Dict[str, Any]:
        verification = self.verify_raw_archive()
        manifest = {
            "schema_version": 1,
            "path": "inbox/raw_events.jsonl",
            "event_count": int(verification.get("event_count") or 0),
            "verified_event_count": int(verification.get("verified_event_count") or 0),
            "status": verification.get("status", "unknown"),
            "first_created_at": verification.get("first_created_at", ""),
            "last_created_at": verification.get("last_created_at", ""),
            "last_record_sha256": verification.get("last_record_sha256", ""),
            "derived_index_status": "not_indexed",
            "indexed_event_count": 0,
            "last_indexed_record_sha256": "",
            "raw_index_schema_version": 1,
            "byte_size": self.raw_events_path.stat().st_size if self.raw_events_path.exists() else 0,
            "updated_at": utc_now_iso(),
        }
        self._atomic_write_yaml(self.raw_archive_manifest_path, manifest)
        return manifest

    def _update_raw_archive_manifest_after_append(self, event: Dict[str, Any]) -> Dict[str, Any]:
        previous: Dict[str, Any] = {}
        manifest_valid = False
        if self.raw_archive_manifest_path.exists():
            try:
                with self.raw_archive_manifest_path.open("r", encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh) or {}
                previous = loaded if isinstance(loaded, dict) else {}
                prior_count = int(previous.get("event_count") or 0)
                manifest_valid = (
                    prior_count == max(0, int(event.get("chain_index") or 0))
                    and str(previous.get("last_record_sha256") or "")
                    == str(event.get("previous_record_sha256") or "")
                )
            except (OSError, TypeError, ValueError, yaml.YAMLError):
                previous = {}
                manifest_valid = False
        if not manifest_valid:
            # Exceptional repair path: canonical JSONL is authoritative. A
            # missing/corrupt/stale manifest must not reset counts to one.
            return self.rebuild_raw_archive_manifest()
        event_count = int(previous.get("event_count") or 0) + 1
        first_created_at = str(previous.get("first_created_at") or event.get("created_at") or "")
        manifest = {
            "schema_version": 1,
            "path": "inbox/raw_events.jsonl",
            "event_count": event_count,
            "verified_event_count": event_count,
            "status": "ok",
            "first_created_at": first_created_at,
            "last_created_at": str(event.get("created_at") or ""),
            "last_record_sha256": str(event.get("record_sha256") or ""),
            "derived_index_status": str(previous.get("derived_index_status") or "stale"),
            "indexed_event_count": int(previous.get("indexed_event_count") or 0),
            "last_indexed_record_sha256": str(previous.get("last_indexed_record_sha256") or ""),
            "raw_index_schema_version": int(previous.get("raw_index_schema_version") or 1),
            "byte_size": self.raw_events_path.stat().st_size if self.raw_events_path.exists() else 0,
            "updated_at": utc_now_iso(),
        }
        self._atomic_write_yaml(self.raw_archive_manifest_path, manifest)
        return manifest

    def update_raw_archive_index_status(
        self,
        *,
        derived_index_status: str,
        indexed_event_count: int,
        last_indexed_record_sha256: str,
        raw_index_schema_version: int,
    ) -> Dict[str, Any]:
        manifest = self.read_raw_archive_manifest()
        manifest.update(
            {
                "derived_index_status": str(derived_index_status),
                "indexed_event_count": int(indexed_event_count),
                "last_indexed_record_sha256": str(last_indexed_record_sha256 or ""),
                "raw_index_schema_version": int(raw_index_schema_version),
                "updated_at": utc_now_iso(),
            }
        )
        self._atomic_write_yaml(self.raw_archive_manifest_path, manifest)
        return manifest

    def _update_derived_raw_index_after_append(
        self,
        event: Dict[str, Any],
        *,
        byte_offset: int,
        byte_length: int,
    ) -> None:
        """Best-effort update of the derived raw index after canonical append."""
        db_path = self.base_dir / "indexes" / "memory.sqlite"
        if not db_path.exists():
            manifest = self.read_raw_archive_manifest()
            manifest["derived_index_status"] = "stale"
            manifest["updated_at"] = utc_now_iso()
            self._atomic_write_yaml(self.raw_archive_manifest_path, manifest)
            return
        last_error: Exception | None = None
        try:
            from .index import MemoryV2Index

            index = MemoryV2Index(db_path)
            indexed = index.index_raw_archive_event(
                event,
                byte_offset=byte_offset,
                byte_length=byte_length,
                line_no=int(event.get("chain_index") or 0) + 1,
            )
            if not indexed:
                raise ValidationError("incremental raw-event metadata indexing rejected the appended event")
            index.index_raw_event(event, index_archive=False)
            manifest = self.read_raw_archive_manifest()
            manifest_event_count = int(manifest.get("event_count") or 0)
            health = index.raw_event_index_health()
            indexed_event_count = int(health["raw_event_count"])
            if (
                indexed_event_count != manifest_event_count
                or int(health["raw_event_fts_count"]) != manifest_event_count
                or int(health["invalid_metadata_count"]) != 0
                or int(health["missing_fts_count"]) != 0
                or int(health["orphan_fts_count"]) != 0
                or str(health["last_record_sha256"] or "")
                != str(manifest.get("last_record_sha256") or "")
            ):
                raise ValidationError(
                    "incremental raw index is incomplete; explicit rebuild required"
                )
            self.update_raw_archive_index_status(
                derived_index_status="ok",
                indexed_event_count=indexed_event_count,
                last_indexed_record_sha256=str(event.get("record_sha256") or ""),
                raw_index_schema_version=1,
            )
            return
        except Exception as exc:
            last_error = exc
        # The JSONL archive is canonical; indexing must never make append fail.
        manifest = self.read_raw_archive_manifest()
        manifest["derived_index_status"] = "stale"
        manifest["index_error"] = str(last_error or "unknown index update failure")[:500]
        manifest["updated_at"] = utc_now_iso()
        self._atomic_write_yaml(self.raw_archive_manifest_path, manifest)

    def verify_raw_archive(self) -> Dict[str, Any]:
        """Return a privacy-safe integrity report for the raw JSONL archive."""
        issues: List[Dict[str, Any]] = []
        event_count = 0
        verified_count = 0
        previous_hash = ""
        first_created_at = ""
        last_created_at = ""
        last_record_hash = ""
        seen_ids: set[str] = set()
        if not self.raw_events_path.exists():
            return {
                "success": True,
                "status": "ok",
                "event_count": 0,
                "verified_event_count": 0,
                "issue_count": 0,
                "issues": [],
                "first_created_at": "",
                "last_created_at": "",
                "last_record_sha256": "",
            }
        with self.raw_events_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    issues.append({"code": "raw_event_malformed_json", "line": line_no})
                    continue
                if not isinstance(event, dict):
                    issues.append({"code": "raw_event_non_object", "line": line_no})
                    continue
                event_count += 1
                event_id = str(event.get("id") or "")
                if event_id in seen_ids:
                    issues.append({"code": "duplicate_raw_event_id", "line": line_no, "event_id": event_id})
                if event_id:
                    seen_ids.add(event_id)
                created_at = str(event.get("created_at") or "")
                first_created_at = first_created_at or created_at
                last_created_at = created_at or last_created_at
                stored_record_hash = str(event.get("record_sha256") or "")
                stored_previous_hash = str(event.get("previous_record_sha256") or "")
                if not stored_record_hash:
                    issues.append({"code": "legacy_raw_event_unverified", "line": line_no, "event_id": event_id})
                    previous_hash = ""
                    continue
                expected_content_hash = self._raw_event_content_hash(event)
                if str(event.get("content_sha256") or "") != expected_content_hash:
                    issues.append({"code": "raw_event_content_hash_mismatch", "line": line_no, "event_id": event_id})
                expected_record_hash = self._raw_event_record_hash(event)
                if stored_record_hash != expected_record_hash:
                    issues.append({"code": "raw_event_record_hash_mismatch", "line": line_no, "event_id": event_id})
                if stored_previous_hash != previous_hash:
                    issues.append({
                        "code": "raw_event_previous_hash_mismatch",
                        "line": line_no,
                        "event_id": event_id,
                        "expected_previous_hash": previous_hash,
                        "actual_previous_hash": stored_previous_hash,
                    })
                if (
                    stored_record_hash == expected_record_hash
                    and str(event.get("content_sha256") or "") == expected_content_hash
                    and stored_previous_hash == previous_hash
                ):
                    verified_count += 1
                previous_hash = stored_record_hash
                last_record_hash = stored_record_hash
        status = "ok" if not issues else "degraded"
        return {
            "success": True,
            "status": status,
            "event_count": event_count,
            "verified_event_count": verified_count,
            "issue_count": len(issues),
            "issues": issues,
            "first_created_at": first_created_at,
            "last_created_at": last_created_at,
            "last_record_sha256": last_record_hash,
        }

    def append_candidate(self, candidate: CandidateMemory) -> None:
        self._append_jsonl(self.candidates_path, candidate.to_dict())

    def rewrite_candidates(self, candidates: List[CandidateMemory]) -> None:
        """Rewrite ``inbox/candidates.jsonl`` with updated candidate decisions."""
        lines = "".join(
            json.dumps(candidate.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for candidate in candidates
        )
        self._atomic_write_text(self.candidates_path, lines)

    def list_candidates(self, *, limit: Optional[int] = None) -> List[CandidateMemory]:
        records = self._read_jsonl(self.candidates_path)
        if limit is not None:
            safe_limit = int(limit)
            if safe_limit < 0:
                raise ValidationError("limit must be non-negative")
            records = records[:safe_limit]
        return [CandidateMemory.from_dict(item) for item in records]

    def count_candidates(self) -> int:
        return self._count_jsonl(self.candidates_path)

    def count_pending_candidates(self) -> int:
        return sum(
            1
            for item in self._read_jsonl(self.candidates_path)
            if CandidateMemory.from_dict(item).gate_decision == GateDecision.PENDING
        )

    def append_rejected_candidate(self, candidate: CandidateMemory) -> None:
        self._append_jsonl(self.rejected_path, candidate.to_dict())

    def list_rejected_candidates(self) -> List[CandidateMemory]:
        return [
            CandidateMemory.from_dict(item)
            for item in self._read_jsonl(self.rejected_path)
        ]

    def count_rejected_candidates(self) -> int:
        return self._count_jsonl(self.rejected_path)

    def append_operation_record(self, operation: Dict[str, Any]) -> None:
        """Append an auditable Memory v2 mutation record."""
        if not isinstance(operation, dict):
            raise ValidationError("operation record must be a JSON object")
        payload = redact_data(dict(operation))
        payload.setdefault("operation_id", f"op_{uuid.uuid4().hex}")
        payload.setdefault("created_at", utc_now_iso())
        self.operations_path.parent.mkdir(parents=True, exist_ok=True)
        self._append_jsonl(self.operations_path, payload)

    def list_operation_records(
        self, *, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if limit is None:
            return self._read_jsonl(self.operations_path)
        safe_limit = int(limit)
        if safe_limit < 0:
            raise ValidationError("limit must be non-negative")
        if safe_limit == 0:
            return []
        return self._read_jsonl_tail(self.operations_path, safe_limit)

    def write_core_memory_record(self, record: CoreMemoryRecord) -> Path:
        """Write a formal core-memory record into ``core/<category>.yaml``."""
        records = [
            existing
            for existing in self.list_core_memory_records(
                category=record.category.value
            )
            if existing.id != record.id
        ]
        records.append(record)
        records.sort(key=lambda item: (-item.priority, item.id))
        path = self._core_category_path(record.category.value)
        payload = {
            "version": 1,
            "category": record.category.value,
            "records": [item.to_dict() for item in records],
        }
        self._atomic_write_yaml(path, payload)
        return path

    def read_core_memory_record(self, record_id: str) -> Optional[CoreMemoryRecord]:
        for record in self.list_core_memory_records():
            if record.id == record_id:
                return record
        return None

    def list_core_memory_records(
        self, *, category: str | None = None
    ) -> List[CoreMemoryRecord]:
        records: List[CoreMemoryRecord] = []
        if not self.core_dir.exists():
            return records
        paths = (
            [self._core_category_path(category)]
            if category
            else sorted(self.core_dir.glob("*.yaml"))
        )
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            for item in data.get("records") or []:
                records.append(CoreMemoryRecord.from_dict(item))
        records.sort(key=lambda item: (-item.priority, item.id))
        return records

    def write_project_card(self, card: ProjectCard) -> Path:
        """Write a project card to ``semantic/projects/<slug>.yaml`` atomically."""
        path = self._project_card_path(card.id)
        self._atomic_write_yaml(path, card.to_dict())
        return path

    def read_project_card(self, project_id_or_name: str) -> Optional[ProjectCard]:
        path = self._project_card_path(project_id_or_name)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return ProjectCard.from_dict(data)

    def list_project_cards(self) -> List[ProjectCard]:
        cards: List[ProjectCard] = []
        if not self.projects_dir.exists():
            return cards
        for path in sorted(self.projects_dir.glob("*.yaml")):
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            cards.append(ProjectCard.from_dict(data))
        return cards

    def write_memory_item(self, item: MemoryItem) -> Path:
        """Write a semantic memory item to ``semantic/items/<memory-id>.yaml`` atomically."""
        path = self._memory_item_path(item.id)
        self._atomic_write_yaml(path, item.to_dict())
        return path

    def read_memory_item(self, memory_id: str) -> Optional[MemoryItem]:
        path = self._memory_item_path(memory_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return MemoryItem.from_dict(data)

    def list_memory_items(
        self, *, memory_type: str | None = None, status: str | None = None
    ) -> List[MemoryItem]:
        items: List[MemoryItem] = []
        if not self.memory_items_dir.exists():
            return items
        for path in sorted(self.memory_items_dir.glob("*.yaml")):
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            item = MemoryItem.from_dict(data)
            item_type = getattr(item.type, "value", str(item.type))
            item_status = getattr(item.status, "value", str(item.status))
            if memory_type is not None and item_type != str(memory_type):
                continue
            if status is not None and item_status != str(status):
                continue
            items.append(item)
        return items

    def write_current_working_memory(self, working: WorkingMemory) -> Path:
        """Persist the mutable current working-memory snapshot."""
        self._atomic_write_yaml(self.current_working_path, working.to_dict())
        return self.current_working_path

    def read_current_working_memory(self) -> Optional[WorkingMemory]:
        if not self.current_working_path.exists():
            return None
        with self.current_working_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return WorkingMemory.from_dict(data)

    def clear_current_working_memory(self) -> None:
        if self.current_working_path.exists():
            self.current_working_path.unlink()

    def write_open_loops(self, loops: List[Dict[str, Any]]) -> Path:
        payload = {"version": 1, "updated_at": utc_now_iso(), "open_loops": loops}
        self._atomic_write_yaml(self.open_loops_path, payload)
        return self.open_loops_path

    def list_open_loops(self, *, status: str | None = None) -> List[Dict[str, Any]]:
        if not self.open_loops_path.exists():
            return []
        with self.open_loops_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        loops = list(data.get("open_loops") or [])
        if status is not None:
            loops = [
                loop for loop in loops if str(loop.get("status") or "") == str(status)
            ]
        return loops

    def upsert_open_loop(self, loop: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(loop)
        payload.setdefault("id", f"loop_{uuid.uuid4().hex}")
        payload.setdefault("status", "open")
        payload.setdefault("created_at", utc_now_iso())
        payload["updated_at"] = utc_now_iso()
        payload["text"] = redact_text(str(payload.get("text") or "").strip())
        payload["source_refs"] = [str(ref) for ref in payload.get("source_refs") or []]
        payload["session_id"] = str(payload.get("session_id") or "")
        if not payload["text"]:
            raise ValidationError("open loop text is required")
        loops = [
            existing
            for existing in self.list_open_loops()
            if existing.get("id") != payload["id"]
        ]
        loops.append(payload)
        self.write_open_loops(loops)
        return payload

    def archive_working_session(
        self, *, session_id: str, messages: List[Dict[str, Any]]
    ) -> Path:
        working = self.read_current_working_memory()
        archive = {
            "session_id": str(session_id or ""),
            "archived_at": utc_now_iso(),
            "message_count": len(messages),
            "working_memory": working.to_dict() if working else None,
            "open_loops": self.list_open_loops(status="open"),
        }
        safe_session = self._safe_yaml_stem(str(session_id or "session"), "session id")
        path = self.episodic_sessions_dir / f"{safe_session}.yaml"
        self._atomic_write_yaml(path, archive)
        self.clear_current_working_memory()
        return path

    def list_session_archives(self) -> List[Dict[str, Any]]:
        archives: List[Dict[str, Any]] = []
        if not self.episodic_sessions_dir.exists():
            return archives
        for path in sorted(self.episodic_sessions_dir.glob("*.yaml")):
            with path.open("r", encoding="utf-8") as fh:
                archives.append(yaml.safe_load(fh) or {})
        return archives

    def write_artifact_record(self, record: ArtifactRecord) -> Path:
        """Write an artifact manifest under ``artifacts/manifests/<safe-id>.yaml``."""
        path = self._ensure_under_base(self._artifact_record_path(record.id))
        self._atomic_write_yaml(path, record.to_dict())
        return path

    def read_artifact_record(self, artifact_id: str) -> Optional[ArtifactRecord]:
        path = self._artifact_record_path(artifact_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return ArtifactRecord.from_dict(data)

    def list_artifact_records(self) -> List[ArtifactRecord]:
        records: List[ArtifactRecord] = []
        if not self.artifact_manifests_dir.exists():
            return records
        for path in sorted(self.artifact_manifests_dir.glob("*.yaml")):
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            records.append(ArtifactRecord.from_dict(data))
        return records

    def write_artifact_segment(self, segment: ArtifactSegment) -> Path:
        """Write a derived segment under the safe artifact segment directory."""
        path = self._ensure_under_base(
            self._artifact_segment_path(segment.artifact_id, segment.id)
        )
        self._atomic_write_yaml(path, segment.to_dict())
        return path

    def list_artifact_segments(self, artifact_id: str) -> List[ArtifactSegment]:
        segments: List[ArtifactSegment] = []
        directory = self._artifact_segments_dir(artifact_id)
        if not directory.exists():
            return segments
        for path in sorted(directory.glob("*.yaml")):
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            segments.append(ArtifactSegment.from_dict(data))
        return segments

    def write_source_ref(self, source: SourceRef) -> Path:
        """Write source metadata to ``sources/<source-id>.yaml`` atomically."""
        path = self._source_ref_path(source.id)
        self._atomic_write_yaml(path, source.to_dict())
        return path

    def read_source_ref(self, source_id: str) -> Optional[SourceRef]:
        path = self._source_ref_path(source_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return SourceRef.from_dict(data)

    def list_source_refs(self) -> List[SourceRef]:
        sources: List[SourceRef] = []
        if not self.sources_dir.exists():
            return sources
        for path in sorted(self.sources_dir.glob("*.yaml")):
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            sources.append(SourceRef.from_dict(data))
        return sources

    def _core_category_path(self, category: str) -> Path:
        safe = self._safe_yaml_stem(str(category), "core category")
        return self.core_dir / f"{safe}.yaml"

    def _project_card_path(self, project_id_or_name: str) -> Path:
        normalized = normalize_project_id(project_id_or_name)
        slug = normalized.split(":", 1)[1]
        return self.projects_dir / f"{slug}.yaml"

    def _memory_item_path(self, memory_id: str) -> Path:
        return (
            self.memory_items_dir
            / f"{self._safe_yaml_stem(memory_id, 'memory id')}.yaml"
        )

    def _source_ref_path(self, source_id: str) -> Path:
        return self.sources_dir / f"{self._safe_yaml_stem(source_id, 'source id')}.yaml"

    def _artifact_record_path(self, artifact_id: str) -> Path:
        return (
            self.artifact_manifests_dir
            / f"{self._safe_yaml_stem(artifact_id, 'artifact id')}.yaml"
        )

    def _artifact_segments_dir(self, artifact_id: str) -> Path:
        return (
            self.derived_artifacts_dir
            / self._safe_yaml_stem(artifact_id, "artifact id")
            / "segments"
        )

    def _artifact_segment_path(self, artifact_id: str, segment_id: str) -> Path:
        return (
            self._artifact_segments_dir(artifact_id)
            / f"{self._safe_yaml_stem(segment_id, 'segment id')}.yaml"
        )

    def _ensure_under_base(self, path: Path) -> Path:
        """Return a resolved path only if it remains inside ``base_dir``."""
        resolved = Path(path).expanduser().resolve(strict=False)
        try:
            resolved.relative_to(self.base_dir)
        except ValueError as exc:
            raise ValidationError(
                "artifact write path must stay under memory_v2 base_dir"
            ) from exc
        return resolved

    def require_usable_raw_index(self, index: Any = None) -> Any:
        """Return an initialized raw index or fail instead of full-scanning.

        The raw archive JSONL is canonical, but hot paths and idempotent imports
        rely on the derived SQLite index for bounded lookups. A caller-provided
        index must be validated too; otherwise stale/corrupt indexes can make
        duplicate checks fail open and append duplicate raw evidence.
        """
        raw_index = index
        if raw_index is None:
            db_path = self.default_index_path
            if not db_path.exists():
                if self.raw_events_path.exists() and self.raw_events_path.stat().st_size > 0:
                    raise ValidationError("raw archive index is missing; rebuild Memory v2 indexes before using bounded raw archive APIs")
                from .index import MemoryV2Index

                raw_index = MemoryV2Index(db_path)
                raw_index.initialize()
            else:
                from .index import MemoryV2Index

                raw_index = MemoryV2Index(db_path)
                raw_index.initialize()
        self._validate_raw_index_manifest(raw_index)
        return raw_index

    def _require_usable_raw_index(self, index: Any = None) -> Any:
        return self.require_usable_raw_index(index)

    def _validate_raw_index_manifest(self, raw_index: Any) -> None:
        manifest = self._read_raw_archive_manifest_if_present()
        raw_has_events = self.raw_events_path.exists() and self.raw_events_path.stat().st_size > 0
        if not manifest:
            if raw_has_events:
                raise ValidationError("raw archive index status is unknown; rebuild Memory v2 indexes before using bounded raw archive APIs")
            return
        status = str(manifest.get("derived_index_status") or "")
        if status and status != "ok":
            raise ValidationError("raw archive index is stale; rebuild Memory v2 indexes before using bounded raw archive APIs")
        event_count = int(manifest.get("event_count") or 0)
        indexed_count = int(manifest.get("indexed_event_count") or 0)
        if indexed_count != event_count:
            raise ValidationError("raw archive index count does not match manifest; rebuild Memory v2 indexes before using bounded raw archive APIs")
        try:
            actual_index_count = int(raw_index.raw_event_count())
        except Exception as exc:
            raise ValidationError("raw archive index is unusable; rebuild Memory v2 indexes before using bounded raw archive APIs") from exc
        if actual_index_count != event_count:
            raise ValidationError("raw archive index row count does not match manifest; rebuild Memory v2 indexes before using bounded raw archive APIs")
        expected_last_hash = str(manifest.get("last_record_sha256") or "")
        indexed_last_hash = str(manifest.get("last_indexed_record_sha256") or "")
        if expected_last_hash != indexed_last_hash:
            raise ValidationError("raw archive index hash does not match manifest; rebuild Memory v2 indexes before using bounded raw archive APIs")

    def _read_raw_archive_manifest_if_present(self) -> Dict[str, Any]:
        if not self.raw_archive_manifest_path.exists():
            return {}
        with self.raw_archive_manifest_path.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        return payload if isinstance(payload, dict) else {}

    def _hydrate_raw_event_metadata(self, metadata: Dict[str, Any] | None) -> Dict[str, Any] | None:
        if not metadata:
            return None
        byte_offset = metadata.get("byte_offset")
        byte_length = metadata.get("byte_length")
        if byte_offset is None or byte_length is None:
            raise ValidationError("raw archive index is missing byte offsets; rebuild Memory v2 indexes")
        event = self.read_raw_event_at_offset(int(byte_offset), int(byte_length))
        if str(event.get("id") or "") != str(metadata.get("id") or ""):
            raise ValidationError("raw archive index byte slice id mismatch; rebuild Memory v2 indexes")
        indexed_hash = str(metadata.get("record_sha256") or "")
        if indexed_hash and str(event.get("record_sha256") or "") != indexed_hash:
            raise ValidationError("raw archive index record hash mismatch; rebuild Memory v2 indexes")
        return event

    def _last_raw_event_for_chain(self) -> Dict[str, Any] | None:
        events = self._read_jsonl_tail(self.raw_events_path, 1)
        return events[-1] if events else None

    def _normalize_raw_event(
        self, event: Dict[str, Any], *, previous: Dict[str, Any] | None
    ) -> Dict[str, Any]:
        payload = self._truncate_raw_event(redact_data(dict(event)))
        metadata = redaction_metadata(event)
        payload.update(metadata)
        payload.pop("content_sha256", None)
        payload.pop("record_sha256", None)
        payload.pop("previous_record_sha256", None)
        payload.pop("chain_index", None)
        payload["schema_version"] = RAW_EVENT_SCHEMA_VERSION
        payload.setdefault("id", f"event_{uuid.uuid4().hex}")
        payload.setdefault("created_at", utc_now_iso())
        payload.setdefault("observed_at", payload["created_at"])
        payload.setdefault("archive_status", "active")
        event_type = str(payload.get("type") or "").lower()
        payload["trust_level"] = "tool_output_untrusted" if "tool" in event_type else "untrusted"
        payload["can_instruct"] = False
        payload.setdefault("redaction_applied", True)
        payload.setdefault("privacy_level", "standard")
        payload.setdefault("blocked_reason", "")
        self._cap_tool_event_fields(payload)
        previous_hash = str((previous or {}).get("record_sha256") or "")
        previous_index_value = (previous or {}).get("chain_index")
        previous_index = int(previous_index_value) if previous_index_value is not None else -1
        payload["previous_record_sha256"] = previous_hash
        payload["chain_index"] = previous_index + 1
        payload["content_sha256"] = self._raw_event_content_hash(payload)
        payload["record_sha256"] = self._raw_event_record_hash(payload)
        return payload

    @staticmethod
    def _cap_tool_event_fields(payload: Dict[str, Any]) -> None:
        if str(payload.get("type") or "").lower() != "tool":
            return
        for field_name in ("content", "tool_output", "result", "stdout", "stderr", "assistant_content"):
            value = payload.get(field_name)
            if isinstance(value, str) and len(value) > _RAW_EVENT_TOOL_CONTENT_LIMIT:
                omitted = len(value) - _RAW_EVENT_TOOL_CONTENT_LIMIT
                payload[field_name] = (
                    value[:_RAW_EVENT_TOOL_CONTENT_LIMIT].rstrip()
                    + f"\n[Memory v2 raw archive omitted {omitted} tool-output chars]"
                )
        payload["tool_output_capped"] = any(
            isinstance(payload.get(field_name), str)
            and "tool-output chars" in str(payload.get(field_name))
            for field_name in ("content", "tool_output", "result", "stdout", "stderr", "assistant_content")
        )

    @classmethod
    def _tombstoned_raw_event_stub(cls, event_id: str) -> Dict[str, Any]:
        return {
            "id": str(event_id or ""),
            "type": "tombstone",
            "archive_status": "tombstoned",
            "trust_level": "untrusted",
            "privacy_level": "secret",
            "can_instruct": False,
            "user_content": "[TOMBSTONED RAW EVENT]",
            "assistant_content": "",
            "content": "[TOMBSTONED RAW EVENT]",
        }

    @classmethod
    def _truncate_raw_event(cls, value: Any, *, _depth: int = 0) -> Any:
        if _depth > 8:
            return "[Memory v2 raw archive omitted deeply nested content]"
        if isinstance(value, str):
            if "\x00" in value:
                return "[Memory v2 raw archive rejected binary-looking content]"
            if len(value) <= _RAW_EVENT_CONTENT_LIMIT:
                return value
            omitted = len(value) - _RAW_EVENT_CONTENT_LIMIT
            return value[:_RAW_EVENT_CONTENT_LIMIT].rstrip() + f"\n[Memory v2 raw archive omitted {omitted} chars]"
        if isinstance(value, (bytes, bytearray, memoryview)):
            return "[Memory v2 raw archive rejected binary content]"
        if isinstance(value, list):
            items = [cls._truncate_raw_event(item, _depth=_depth + 1) for item in value[:100]]
            if len(value) > 100:
                items.append({"omitted_items": len(value) - 100})
            return items
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for idx, (field_name, item) in enumerate(value.items()):
                if idx >= 100:
                    out["__omitted_fields__"] = len(value) - 100
                    break
                safe_field_name = redact_text(str(field_name))[:200]
                out[safe_field_name] = cls._truncate_raw_event(item, _depth=_depth + 1)
            return out
        return value

    @staticmethod
    def _canonical_json(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _sha256_json(cls, payload: Dict[str, Any]) -> str:
        digest = hashlib.sha256(cls._canonical_json(payload).encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    @classmethod
    def _raw_event_content_hash(cls, event: Dict[str, Any]) -> str:
        payload = {
            field_name: field_value
            for field_name, field_value in redact_data(dict(event)).items()
            if field_name not in _RAW_EVENT_HASH_FIELDS and field_name not in _RAW_EVENT_TRANSIENT_FIELDS
        }
        return cls._sha256_json(payload)

    @classmethod
    def _raw_event_record_hash(cls, event: Dict[str, Any]) -> str:
        payload = {
            field_name: field_value
            for field_name, field_value in redact_data(dict(event)).items()
            if field_name != "record_sha256" and field_name not in _RAW_EVENT_TRANSIENT_FIELDS
        }
        return cls._sha256_json(payload)

    @staticmethod
    def _source_ref_from_raw_event(event: Dict[str, Any]) -> SourceRef:
        event_id = str(event.get("id") or "").strip()
        event_type = str(event.get("type") or "").strip().lower()
        session_id = str(event.get("session_id") or "").strip()
        created_at = str(event.get("created_at") or "")
        if event_type == "tool":
            source_type = "tool_result"
            tool_name = str(event.get("tool") or "tool").strip() or "tool"
            title = (
                f"Raw tool evidence from session {session_id}"
                if session_id
                else "Raw tool evidence"
            )
            quote = tool_name
        else:
            source_type = "message"
            title = (
                f"Raw turn evidence from session {session_id}"
                if session_id
                else "Raw turn evidence"
            )
            quote = str(
                event.get("user_content")
                or event.get("content")
                or event.get("assistant_content")
                or ""
            ).strip()
        quote = redact_text(quote)
        if len(quote) > 500:
            quote = quote[:497].rstrip() + "..."
        return SourceRef(
            id=event_id,
            type=source_type,
            uri=f"raw_event:{event_id}",
            title=title,
            observed_at=created_at,
            quote=quote or None,
        )

    @staticmethod
    def _safe_yaml_stem(value: str, field_name: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValidationError(f"{field_name} is required")
        safe = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in text
        ).strip("-")
        if not safe:
            raise ValidationError(
                f"{field_name} must contain at least one safe filename character"
            )
        if safe == text:
            return safe
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{safe}--{digest}"

    def _append_jsonl(self, path: Path, payload: Dict[str, Any], *, fsync: bool = False) -> tuple[int, int]:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        encoded = line.encode("utf-8")
        with path.open("ab") as fh:
            byte_offset = fh.tell()
            fh.write(encoded)
            if fsync:
                fh.flush()
                os.fsync(fh.fileno())
        return byte_offset, len(encoded)

    def _read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValidationError(f"JSONL record in {path} must be an object")
                records.append(payload)
        return records

    def _read_jsonl_tail(self, path: Path, limit: int) -> List[Dict[str, Any]]:
        """Read the last ``limit`` JSONL objects without retaining the full file."""
        if not path.exists() or limit <= 0:
            return []
        records: deque[Dict[str, Any]] = deque(maxlen=limit)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValidationError(f"JSONL record in {path} must be an object")
                records.append(payload)
        return list(records)

    def _count_jsonl(self, path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())

    def _atomic_write_yaml(self, path: Path, payload: Dict[str, Any]) -> None:
        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        self._atomic_write_text(path, text)

    def _atomic_write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp_path.write_text(text, encoding="utf-8")
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
