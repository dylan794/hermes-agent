"""SQLite FTS index/search layer for Memory v2."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from .redaction import redact_data, redact_text, redacted_query_for_log, redacted_query_hash_input
from .schemas import (
    CandidateClaimKind,
    CandidateMemory,
    GateDecision,
    MemoryItem,
    MemoryType,
    ProjectCard,
    ProjectStatus,
    SourceRef,
    ValidationError,
    utc_now_iso,
)
from .store import MemoryV2Store


RAW_ARCHIVE_INDEX_SCHEMA_VERSION = 1
MAX_RAW_ARCHIVE_SOURCE_IDS = 100
HISTORICAL_ROUTES = frozenset({"deep_recall", "past_conversation_exact", "contradiction_check"})


class _ClosingConnection(sqlite3.Connection):
    """SQLite context manager that commits/rolls back and then closes."""

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc_value, traceback))
        finally:
            self.close()


class MemoryV2Index:
    """SQLite-backed keyword index for Memory v2 canonical records."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self._rebuild_lock = threading.Lock()

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                  id TEXT PRIMARY KEY,
                  type TEXT NOT NULL,
                  title TEXT,
                  subject TEXT,
                  predicate TEXT,
                  value TEXT,
                  body TEXT,
                  summary TEXT,
                  status TEXT,
                  confidence REAL,
                  importance REAL,
                  created_at TEXT,
                  updated_at TEXT NOT NULL,
                  valid_from TEXT,
                  valid_until TEXT,
                  expires_at TEXT,
                  source_refs TEXT,
                  supersedes TEXT,
                  superseded_by TEXT,
                  tags TEXT,
                  file_path TEXT
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                  id UNINDEXED,
                  title,
                  subject,
                  predicate,
                  value,
                  body,
                  summary,
                  tags
                );

                CREATE TABLE IF NOT EXISTS source_refs (
                  id TEXT PRIMARY KEY,
                  type TEXT NOT NULL,
                  uri TEXT NOT NULL,
                  title TEXT,
                  observed_at TEXT,
                  quote TEXT
                );

                CREATE TABLE IF NOT EXISTS retrieval_log (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  query TEXT NOT NULL,
                  query_hash TEXT,
                  route TEXT,
                  retrieved_ids TEXT,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS raw_events (
                  id TEXT PRIMARY KEY,
                  event_type TEXT,
                  source_system TEXT,
                  session_id TEXT,
                  provider_session_id TEXT,
                  message_id TEXT,
                  created_at TEXT,
                  observed_at TEXT,
                  archive_status TEXT,
                  trust_level TEXT,
                  privacy_level TEXT,
                  can_instruct INTEGER NOT NULL DEFAULT 0,
                  chain_index INTEGER,
                  record_sha256 TEXT,
                  previous_record_sha256 TEXT,
                  content_sha256 TEXT,
                  byte_offset INTEGER,
                  byte_length INTEGER,
                  line_no INTEGER,
                  source_ref_id TEXT,
                  import_key_sha256 TEXT,
                  indexed_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS raw_events_session_created_idx
                  ON raw_events(session_id, created_at);
                CREATE INDEX IF NOT EXISTS raw_events_type_created_idx
                  ON raw_events(event_type, created_at);
                CREATE INDEX IF NOT EXISTS raw_events_created_idx
                  ON raw_events(created_at);
                CREATE INDEX IF NOT EXISTS raw_events_import_key_idx
                  ON raw_events(import_key_sha256);
                CREATE INDEX IF NOT EXISTS raw_events_message_idx
                  ON raw_events(message_id);
                CREATE INDEX IF NOT EXISTS raw_events_record_hash_idx
                  ON raw_events(record_sha256);
                CREATE INDEX IF NOT EXISTS raw_events_chain_index_idx
                  ON raw_events(chain_index);

                CREATE TABLE IF NOT EXISTS raw_index_metadata (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS raw_events_fts USING fts5(
                  id UNINDEXED,
                  user_content,
                  assistant_content,
                  content,
                  tool,
                  session_id UNINDEXED,
                  event_type UNINDEXED
                );
                """
            )
            self._ensure_raw_events_schema(conn)
            self._ensure_column(conn, "retrieval_log", "query_hash", "TEXT")
            for column, column_type in {
                "subject": "TEXT",
                "predicate": "TEXT",
                "value": "TEXT",
                "confidence": "REAL",
                "importance": "REAL",
                "created_at": "TEXT",
                "valid_from": "TEXT",
                "valid_until": "TEXT",
                "expires_at": "TEXT",
                "supersedes": "TEXT",
                "superseded_by": "TEXT",
            }.items():
                self._ensure_column(conn, "memories", column, column_type)
            self._ensure_fts_schema(conn)

    @staticmethod
    def _ensure_fts_schema(conn: sqlite3.Connection) -> None:
        expected_columns = ["id", "title", "subject", "predicate", "value", "body", "summary", "tags"]
        existing_columns = [row[1] for row in conn.execute("PRAGMA table_info(memories_fts)").fetchall()]
        if existing_columns == expected_columns:
            return
        conn.execute("DROP TABLE IF EXISTS memories_fts")
        conn.execute(
            """
            CREATE VIRTUAL TABLE memories_fts USING fts5(
              id UNINDEXED,
              title,
              subject,
              predicate,
              value,
              body,
              summary,
              tags
            )
            """
        )
        conn.execute(
            """
            INSERT INTO memories_fts (id, title, subject, predicate, value, body, summary, tags)
            SELECT
              id,
              COALESCE(title, ''),
              COALESCE(subject, ''),
              COALESCE(predicate, ''),
              COALESCE(value, ''),
              COALESCE(body, ''),
              COALESCE(summary, ''),
              COALESCE(tags, '')
            FROM memories
            """
        )

    @staticmethod
    def _ensure_raw_events_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_index_metadata (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        expected_columns = [
            "id",
            "user_content",
            "assistant_content",
            "content",
            "tool",
            "session_id",
            "event_type",
        ]
        existing_columns = [row[1] for row in conn.execute("PRAGMA table_info(raw_events_fts)").fetchall()]
        if existing_columns != expected_columns:
            had_existing_fts = bool(existing_columns)
            conn.execute("DROP TABLE IF EXISTS raw_events_fts")
            conn.execute(
                """
                CREATE VIRTUAL TABLE raw_events_fts USING fts5(
                  id UNINDEXED,
                  user_content,
                  assistant_content,
                  content,
                  tool,
                  session_id UNINDEXED,
                  event_type UNINDEXED
                )
                """
            )
            if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='raw_events'").fetchone():
                conn.execute(
                    """
                    INSERT INTO raw_events_fts (id, user_content, assistant_content, content, tool, session_id, event_type)
                    SELECT id, '', '', '', '', COALESCE(session_id, ''), COALESCE(event_type, '')
                    FROM raw_events
                    """
                )
            if had_existing_fts:
                conn.execute(
                    """
                    INSERT INTO raw_index_metadata (key, value, updated_at)
                    VALUES ('raw_events_fts_status', 'needs_rebuild', ?)
                    ON CONFLICT(key) DO UPDATE SET
                      value=excluded.value,
                      updated_at=excluded.updated_at
                    """,
                    (utc_now_iso(),),
                )
        raw_columns = {row[1] for row in conn.execute("PRAGMA table_info(raw_events)").fetchall()}
        required = {
            "event_type": "TEXT",
            "source_system": "TEXT",
            "session_id": "TEXT",
            "provider_session_id": "TEXT",
            "message_id": "TEXT",
            "created_at": "TEXT",
            "observed_at": "TEXT",
            "archive_status": "TEXT",
            "trust_level": "TEXT",
            "privacy_level": "TEXT",
            "can_instruct": "INTEGER NOT NULL DEFAULT 0",
            "chain_index": "INTEGER",
            "record_sha256": "TEXT",
            "previous_record_sha256": "TEXT",
            "content_sha256": "TEXT",
            "byte_offset": "INTEGER",
            "byte_length": "INTEGER",
            "line_no": "INTEGER",
            "source_ref_id": "TEXT",
            "import_key_sha256": "TEXT",
            "indexed_at": "TEXT",
        }
        for column, column_type in required.items():
            if column not in raw_columns:
                conn.execute(f"ALTER TABLE raw_events ADD COLUMN {column} {column_type}")

    @staticmethod
    def raw_index_schema_version() -> int:
        return RAW_ARCHIVE_INDEX_SCHEMA_VERSION

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
        allowed = {
            "retrieval_log": {"query_hash": "TEXT"},
            "memories": {
                "subject": "TEXT",
                "predicate": "TEXT",
                "value": "TEXT",
                "confidence": "REAL",
                "importance": "REAL",
                "created_at": "TEXT",
                "valid_from": "TEXT",
                "valid_until": "TEXT",
                "expires_at": "TEXT",
                "supersedes": "TEXT",
                "superseded_by": "TEXT",
            },
        }
        if allowed.get(table, {}).get(column) != column_type:
            raise ValueError("unsupported schema migration")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(" + table + ")").fetchall()}
        if column not in columns:
            conn.execute("ALTER TABLE " + table + " ADD COLUMN " + column + " " + column_type)

    def table_names(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual') ORDER BY name").fetchall()
        return [row[0] for row in rows]

    def count_memories(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return int(row[0])

    def index_source_ref(self, source: SourceRef) -> None:
        """Index source metadata used to ground recalled memory items."""
        payload = source.to_dict()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO source_refs (id, type, uri, title, observed_at, quote)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  type=excluded.type,
                  uri=excluded.uri,
                  title=excluded.title,
                  observed_at=excluded.observed_at,
                  quote=excluded.quote
                """,
                (
                    payload["id"],
                    payload["type"],
                    payload["uri"],
                    payload.get("title", ""),
                    payload.get("observed_at", ""),
                    payload.get("quote", ""),
                ),
            )

    def source_ref(self, source_id: str) -> Dict[str, Any] | None:
        """Return indexed source metadata by id, if present."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, type, uri, title, observed_at, quote FROM source_refs WHERE id = ?",
                (str(source_id),),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "type": row[1],
            "uri": row[2],
            "title": row[3] or "",
            "observed_at": row[4] or "",
            "quote": row[5] or "",
        }

    def source_refs(self, source_ids: List[str]) -> List[Dict[str, Any]]:
        """Return source metadata for the given ids in input order."""
        sources: List[Dict[str, Any]] = []
        for source_id in source_ids:
            if source := self.source_ref(source_id):
                sources.append(source)
        return sources

    def record_exists(self, record_id: str) -> bool:
        """Return whether a canonical/index record exists by exact id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM memories WHERE id = ? LIMIT 1", (str(record_id),)
            ).fetchone()
        return row is not None

    def index_memory_item(self, item: MemoryItem, *, file_path: str | Path | None = None) -> None:
        """Index a canonical semantic/core memory item from fixture or store data."""
        payload = item.to_dict()
        body_parts = [
            payload.get("subject"),
            payload.get("predicate"),
            payload.get("value"),
            payload.get("body"),
            " ".join(payload.get("supersedes") or []),
            str(payload.get("superseded_by") or ""),
        ]
        title = " ".join(str(part) for part in (payload.get("subject"), payload.get("predicate")) if part)
        self.index_record(
            id=payload["id"],
            type=payload["type"],
            title=title,
            subject=str(payload.get("subject") or ""),
            predicate=str(payload.get("predicate") or ""),
            value=str(payload.get("value") or ""),
            body="\n".join(str(part) for part in body_parts if part),
            summary=str(payload.get("summary") or payload.get("value") or payload.get("body") or ""),
            status=payload["status"],
            confidence=payload.get("confidence"),
            importance=payload.get("importance"),
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
            valid_from=payload.get("valid_from"),
            valid_until=payload.get("valid_until"),
            expires_at=payload.get("expires_at"),
            source_refs=list(payload.get("source_refs") or []),
            supersedes=list(payload.get("supersedes") or []),
            superseded_by=payload.get("superseded_by"),
            tags=list(payload.get("tags") or []),
            file_path=str(file_path or ""),
        )

    def index_project_card(self, card: ProjectCard, *, file_path: str | Path | None = None) -> None:
        body_parts = [
            card.goal,
            card.why_it_matters,
            card.current_state,
            f"status: {cast(ProjectStatus, card.status).value}",
            "\n".join(card.decisions),
            "\n".join(card.open_questions),
            "\n".join(card.next_actions),
            "\n".join(card.related_entities),
        ]
        structured_value = {
            "goal": card.goal,
            "why_it_matters": card.why_it_matters,
            "current_state": card.current_state,
            "decisions": list(card.decisions),
            "open_questions": list(card.open_questions),
            "next_actions": list(card.next_actions),
            "related_entities": list(card.related_entities),
            "field_evidence": {
                key: [dict(entry) for entry in entries]
                for key, entries in card.field_evidence.items()
            },
        }
        self.index_record(
            id=card.id,
            type="project_state",
            title=card.name,
            body="\n".join(part for part in body_parts if part),
            summary=card.current_state or card.goal,
            status=cast(ProjectStatus, card.status).value,
            value=json.dumps(structured_value, ensure_ascii=False, sort_keys=True),
            importance=card.importance,
            updated_at=card.updated_at,
            source_refs=card.source_refs,
            tags=["project", card.id],
            file_path=str(file_path) if file_path else "",
        )

    def index_candidate(self, candidate: CandidateMemory) -> None:
        candidate_type = cast(MemoryType, candidate.type).value
        claim_kind = cast(CandidateClaimKind, candidate.claim_kind).value
        gate_decision = cast(GateDecision, candidate.gate_decision).value
        body_parts = [
            candidate.claim,
            candidate.promotion_reason,
            candidate.decision_reason,
        ]
        tags = [
            "candidate",
            candidate_type,
            gate_decision,
            claim_kind,
            candidate.extraction_method,
        ]
        successor = re.search(
            r"Promoted to (?:canonical MemoryItem|ProjectCard)\s+([^\s.]+)",
            candidate.decision_reason,
        )
        if successor:
            tags.append(f"successor:{successor.group(1)}")
        self.index_record(
            id=candidate.id,
            type="candidate",
            title=candidate.id,
            subject=candidate_type,
            predicate="gate_decision",
            value=candidate.decision_reason,
            body="\n".join(part for part in body_parts if part),
            summary=candidate.claim,
            status=gate_decision,
            confidence=candidate.confidence,
            importance=candidate.importance,
            created_at=candidate.created_at,
            source_refs=candidate.source_refs,
            tags=tags,
            file_path="inbox/candidates.jsonl",
        )

    def index_raw_archive_event(
        self,
        event: Dict[str, Any],
        *,
        byte_offset: int | None = None,
        byte_length: int | None = None,
        line_no: int | None = None,
        _conn: sqlite3.Connection | None = None,
    ) -> bool:
        """Index raw archive event metadata into rebuildable raw-event tables."""
        event = redact_data(dict(event))
        event_id = str(event.get("id") or "").strip()
        if not event_id:
            return False
        event_type = str(event.get("event_type") or event.get("type") or "").strip()
        source_system = str(event.get("source_system") or "").strip()
        session_id = str(event.get("session_id") or "").strip()
        provider_session_id = str(event.get("provider_session_id") or "").strip()
        message_id = str(event.get("message_id") or "").strip()
        created_at = str(event.get("created_at") or "")
        observed_at = str(event.get("observed_at") or created_at or "")
        archive_status = str(event.get("archive_status") or "archived")
        # Raw archive rows are evidence only. Never inherit instruction/trust
        # metadata from imported or rebuilt JSONL, even if a legacy/direct event
        # claims it is trusted.
        trust_level = "untrusted"
        privacy_level = str(event.get("privacy_level") or "standard")
        can_instruct = 0
        chain_index = self._optional_int(event.get("chain_index"))
        record_sha256 = str(event.get("record_sha256") or "")
        previous_record_sha256 = str(event.get("previous_record_sha256") or "")
        content_sha256 = str(event.get("content_sha256") or "")
        source_ref_id = str(event.get("source_ref_id") or event_id)
        import_key_sha256 = self._event_import_key_sha256(event) or self._raw_import_key_sha256(event)
        indexed_at = utc_now_iso()
        user_content = redact_text(str(event.get("user_content") or ""))
        assistant_content = redact_text(str(event.get("assistant_content") or ""))
        content = redact_text(str(event.get("content") or ""))
        tool = redact_text(str(event.get("tool") or ""))
        connection_context = nullcontext(_conn) if _conn is not None else self._connect()
        with connection_context as conn:
            conn.execute(
                """
                INSERT INTO raw_events (
                  id, event_type, source_system, session_id, provider_session_id, message_id,
                  created_at, observed_at, archive_status, trust_level, privacy_level, can_instruct,
                  chain_index, record_sha256, previous_record_sha256, content_sha256,
                  byte_offset, byte_length, line_no, source_ref_id, import_key_sha256, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  event_type=excluded.event_type,
                  source_system=excluded.source_system,
                  session_id=excluded.session_id,
                  provider_session_id=excluded.provider_session_id,
                  message_id=excluded.message_id,
                  created_at=excluded.created_at,
                  observed_at=excluded.observed_at,
                  archive_status=excluded.archive_status,
                  trust_level=excluded.trust_level,
                  privacy_level=excluded.privacy_level,
                  can_instruct=excluded.can_instruct,
                  chain_index=excluded.chain_index,
                  record_sha256=excluded.record_sha256,
                  previous_record_sha256=excluded.previous_record_sha256,
                  content_sha256=excluded.content_sha256,
                  byte_offset=COALESCE(excluded.byte_offset, raw_events.byte_offset),
                  byte_length=COALESCE(excluded.byte_length, raw_events.byte_length),
                  line_no=COALESCE(excluded.line_no, raw_events.line_no),
                  source_ref_id=excluded.source_ref_id,
                  import_key_sha256=excluded.import_key_sha256,
                  indexed_at=excluded.indexed_at
                """,
                (
                    event_id,
                    event_type,
                    source_system,
                    session_id,
                    provider_session_id,
                    message_id,
                    created_at,
                    observed_at,
                    archive_status,
                    trust_level,
                    privacy_level,
                    can_instruct,
                    chain_index,
                    record_sha256,
                    previous_record_sha256,
                    content_sha256,
                    byte_offset,
                    byte_length,
                    line_no,
                    source_ref_id,
                    import_key_sha256,
                    indexed_at,
                ),
            )
            conn.execute("DELETE FROM raw_events_fts WHERE id = ?", (event_id,))
            conn.execute(
                """
                INSERT INTO raw_events_fts (id, user_content, assistant_content, content, tool, session_id, event_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, user_content, assistant_content, content, tool, session_id, event_type),
            )
        return True

    def raw_event_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM raw_events").fetchone()
        return int(row[0])

    def raw_event_fts_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM raw_events_fts").fetchone()
        return int(row[0])

    def raw_event_index_health(self) -> Dict[str, Any]:
        """Return cheap completeness signals for incremental raw-index updates."""
        with self._connect() as conn:
            raw_count = int(conn.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0])
            fts_count = int(conn.execute("SELECT COUNT(*) FROM raw_events_fts").fetchone()[0])
            invalid_metadata_count = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM raw_events
                    WHERE byte_offset IS NULL OR byte_offset < 0
                       OR byte_length IS NULL OR byte_length <= 0
                       OR line_no IS NULL OR line_no <= 0
                    """
                ).fetchone()[0]
            )
            missing_fts_count = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM raw_events AS raw
                    LEFT JOIN raw_events_fts AS fts ON fts.id = raw.id
                    WHERE fts.id IS NULL
                    """
                ).fetchone()[0]
            )
            orphan_fts_count = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM raw_events_fts AS fts
                    LEFT JOIN raw_events AS raw ON raw.id = fts.id
                    WHERE raw.id IS NULL
                    """
                ).fetchone()[0]
            )
            last_row = conn.execute(
                """
                SELECT record_sha256 FROM raw_events
                ORDER BY line_no DESC, chain_index DESC LIMIT 1
                """
            ).fetchone()
        return {
            "raw_event_count": raw_count,
            "raw_event_fts_count": fts_count,
            "invalid_metadata_count": invalid_metadata_count,
            "missing_fts_count": missing_fts_count,
            "orphan_fts_count": orphan_fts_count,
            "last_record_sha256": str(last_row[0] or "") if last_row else "",
        }

    def raw_event_metadata(self, event_id: str) -> Dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, event_type, source_system, session_id, provider_session_id, message_id,
                       created_at, observed_at, archive_status, trust_level, privacy_level, can_instruct,
                       chain_index, record_sha256, previous_record_sha256, content_sha256,
                       byte_offset, byte_length, line_no, source_ref_id, import_key_sha256, indexed_at
                FROM raw_events WHERE id = ?
                """,
                (str(event_id),),
            ).fetchone()
        if not row:
            return None
        keys = [
            "id", "event_type", "source_system", "session_id", "provider_session_id", "message_id",
            "created_at", "observed_at", "archive_status", "trust_level", "privacy_level", "can_instruct",
            "chain_index", "record_sha256", "previous_record_sha256", "content_sha256",
            "byte_offset", "byte_length", "line_no", "source_ref_id", "import_key_sha256", "indexed_at",
        ]
        return dict(zip(keys, row))

    def raw_import_key_exists(self, import_key_sha256: str) -> bool:
        """Return whether a raw archive import-key hash is already indexed."""
        key = str(import_key_sha256 or "").strip()
        if not key:
            return False
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM raw_events WHERE import_key_sha256 = ? LIMIT 1", (key,)).fetchone()
        return row is not None

    def raw_event_metadata_many(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Return raw-event metadata rows for ids in input order."""
        rows: List[Dict[str, Any]] = []
        for event_id in ids:
            if metadata := self.raw_event_metadata(event_id):
                rows.append(metadata)
        return rows

    def raw_event_exists(self, event_id: str) -> bool:
        """Return whether a raw archive event id is indexed."""
        event_id = str(event_id or "").strip()
        if not event_id:
            return False
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM raw_events WHERE id = ? LIMIT 1", (event_id,)).fetchone()
        return row is not None

    def raw_event_neighbors(self, event_id: str) -> Dict[str, Dict[str, Any] | None]:
        """Return previous/next raw-event metadata using indexed chain_index."""
        metadata = self.raw_event_metadata(event_id)
        if not metadata or metadata.get("chain_index") is None:
            return {"previous": None, "next": None}
        chain_index = int(metadata["chain_index"])
        with self._connect() as conn:
            prev_row = conn.execute(
                self._raw_event_metadata_select_sql("chain_index = ?"),
                (chain_index - 1,),
            ).fetchone()
            next_row = conn.execute(
                self._raw_event_metadata_select_sql("chain_index = ?"),
                (chain_index + 1,),
            ).fetchone()
        return {"previous": self._raw_event_metadata_from_row(prev_row), "next": self._raw_event_metadata_from_row(next_row)}

    def search_raw_archive(
        self,
        query: str = "",
        *,
        session_id: str = "",
        event_type: str = "",
        source_ids: List[str] | None = None,
        created_after: str = "",
        created_before: str = "",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        query_text = str(query or "").strip()
        safe_limit = self._coerce_limit(limit)
        source_ids = sorted({str(item).strip() for item in (source_ids or []) if str(item).strip()})
        if len(source_ids) > MAX_RAW_ARCHIVE_SOURCE_IDS:
            raise ValidationError(f"source_ids is capped at {MAX_RAW_ARCHIVE_SOURCE_IDS} unique ids")
        filters: List[str] = []
        params: List[Any] = []
        if session_id:
            filters.append("r.session_id = ?")
            params.append(str(session_id))
        if event_type:
            filters.append("r.event_type = ?")
            params.append(str(event_type))
        if created_after:
            filters.append("julianday(r.created_at) >= julianday(?)")
            params.append(self._normalized_time_filter(created_after, field_name="created_after"))
        if created_before:
            filters.append("julianday(r.created_at) <= julianday(?)")
            params.append(self._normalized_time_filter(created_before, field_name="created_before"))
        if source_ids:
            placeholders = ", ".join("?" for _ in source_ids)
            filters.append(f"(r.id IN ({placeholders}) OR r.source_ref_id IN ({placeholders}))")
            params.extend(source_ids)
            params.extend(source_ids)
        where_suffix = (" AND " + " AND ".join(filters)) if filters else ""
        rows: List[Any] = []
        with self._connect() as conn:
            if query_text:
                raw_count = int(conn.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0])
                fts_count = int(conn.execute("SELECT COUNT(*) FROM raw_events_fts").fetchone()[0])
                if raw_count != fts_count:
                    raise ValidationError(
                        f"raw_events_fts row count mismatch; rebuild raw archive index (raw_events={raw_count}, raw_events_fts={fts_count})"
                    )
                for fts_query in self._fts_queries(query_text):
                    rows = conn.execute(
                        f"""
                        SELECT r.id, r.event_type, r.source_system, r.session_id, r.provider_session_id, r.message_id,
                               r.created_at, r.observed_at, r.archive_status, r.trust_level, r.privacy_level, r.can_instruct,
                               r.chain_index, r.record_sha256, r.previous_record_sha256, r.content_sha256,
                               r.byte_offset, r.byte_length, r.line_no, r.source_ref_id, r.import_key_sha256, r.indexed_at,
                               bm25(raw_events_fts) AS rank
                        FROM raw_events_fts
                        JOIN raw_events r ON r.id = raw_events_fts.id
                        WHERE raw_events_fts MATCH ?{where_suffix}
                        ORDER BY rank, r.created_at DESC, r.chain_index DESC
                        LIMIT ?
                        """,
                        [fts_query, *params, safe_limit],
                    ).fetchall()
                    if rows:
                        break
            else:
                rows = conn.execute(
                    f"""
                    SELECT r.id, r.event_type, r.source_system, r.session_id, r.provider_session_id, r.message_id,
                           r.created_at, r.observed_at, r.archive_status, r.trust_level, r.privacy_level, r.can_instruct,
                           r.chain_index, r.record_sha256, r.previous_record_sha256, r.content_sha256,
                           r.byte_offset, r.byte_length, r.line_no, r.source_ref_id, r.import_key_sha256, r.indexed_at,
                           0.0 AS rank
                    FROM raw_events r
                    WHERE 1=1{where_suffix}
                    ORDER BY r.created_at DESC, r.chain_index DESC
                    LIMIT ?
                    """,
                    [*params, safe_limit],
                ).fetchall()
        return [dict(item, rank=row[22]) for row in rows if (item := self._raw_event_metadata_from_row(row)) is not None]

    @staticmethod
    def _raw_event_metadata_select_sql(where: str) -> str:
        return f"""
            SELECT id, event_type, source_system, session_id, provider_session_id, message_id,
                   created_at, observed_at, archive_status, trust_level, privacy_level, can_instruct,
                   chain_index, record_sha256, previous_record_sha256, content_sha256,
                   byte_offset, byte_length, line_no, source_ref_id, import_key_sha256, indexed_at
            FROM raw_events WHERE {where}
        """

    @staticmethod
    def _raw_event_metadata_from_row(row: Any) -> Dict[str, Any] | None:
        if not row:
            return None
        keys = [
            "id", "event_type", "source_system", "session_id", "provider_session_id", "message_id",
            "created_at", "observed_at", "archive_status", "trust_level", "privacy_level", "can_instruct",
            "chain_index", "record_sha256", "previous_record_sha256", "content_sha256",
            "byte_offset", "byte_length", "line_no", "source_ref_id", "import_key_sha256", "indexed_at",
        ]
        return dict(zip(keys, row[:22]))

    def index_raw_event(self, event: Dict[str, Any], *, index_archive: bool = True) -> bool:
        event = redact_data(dict(event))
        if index_archive and not self.index_raw_archive_event(event):
            return False
        event_id = str(event.get("id") or "").strip()
        if not event_id:
            return False
        title = f"Raw event {event_id}".strip()
        body = "\n".join(
            redact_text(str(event.get(key) or ""))
            for key in (
                "type",
                "session_id",
                "user_content",
                "assistant_content",
                "tool",
                "content",
            )
            if event.get(key)
        )
        session_id = str(event.get("session_id") or "").strip()
        tags = [
            "raw_event",
            str(event.get("type") or ""),
            "trust:untrusted",
            f"privacy:{event.get('privacy_level') or 'standard'}",
        ]
        if session_id:
            tags.append(f"session:{session_id}")
        self.index_record(
            id=event_id,
            type="raw_event",
            title=title,
            body=body,
            summary=redact_text(str(event.get("user_content") or event.get("content") or "")),
            status="archived",
            created_at=str(event.get("created_at") or ""),
            updated_at=str(event.get("updated_at") or ""),
            source_refs=[event_id] if event_id else [],
            tags=tags,
            file_path="inbox/raw_events.jsonl",
        )
        self.index_source_ref(self._source_ref_from_raw_event(event))
        return True

    @staticmethod
    def _source_ref_from_raw_event(event: Dict[str, Any]) -> SourceRef:
        event_id = str(event.get("id") or "").strip()
        event_type = str(event.get("type") or "").strip().lower()
        session_id = str(event.get("session_id") or "").strip()
        created_at = str(event.get("created_at") or "")
        if event_type == "tool":
            source_type = "tool_result"
            title = f"Raw tool evidence from session {session_id}" if session_id else "Raw tool evidence"
            quote = str(event.get("tool") or "tool").strip() or "tool"
        else:
            source_type = "message"
            title = f"Raw turn evidence from session {session_id}" if session_id else "Raw turn evidence"
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

    def index_open_loop(self, loop: Dict[str, Any], *, file_path: str | Path = "") -> None:
        loop_id = str(loop.get("id") or "").strip()
        if not loop_id:
            return
        text = redact_text(str(loop.get("text") or ""))
        self.index_record(
            id=loop_id,
            type="open_loop",
            title=loop_id,
            body=text,
            summary=text,
            status=str(loop.get("status") or "open"),
            created_at=str(loop.get("created_at") or ""),
            updated_at=str(loop.get("updated_at") or ""),
            source_refs=[str(ref) for ref in loop.get("source_refs") or []],
            tags=["open_loop", str(loop.get("status") or "open")],
            file_path=str(file_path or "working/open_loops.yaml"),
        )

    def index_record(
        self,
        *,
        id: str,
        type: str,
        title: str = "",
        subject: str = "",
        predicate: str = "",
        value: str = "",
        body: str = "",
        summary: str = "",
        status: str = "active",
        confidence: float | None = None,
        importance: float | None = None,
        created_at: str = "",
        updated_at: str = "",
        valid_from: str | None = None,
        valid_until: str | None = None,
        expires_at: str | None = None,
        source_refs: Optional[List[str]] = None,
        supersedes: Optional[List[str]] = None,
        superseded_by: str | None = None,
        tags: Optional[List[str]] = None,
        file_path: str = "",
    ) -> None:
        record_id = str(id).strip()
        if not record_id:
            raise ValueError("indexed record id is required")
        source_refs = [str(ref) for ref in (source_refs or [])]
        supersedes = [str(ref) for ref in (supersedes or [])]
        tags = [str(tag) for tag in (tags or []) if str(tag)]
        now = utc_now_iso()
        stored_updated_at = str(updated_at or now)
        stored_created_at = str(created_at or stored_updated_at)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                  id, type, title, subject, predicate, value, body, summary, status,
                  confidence, importance, created_at, updated_at, valid_from, valid_until, expires_at,
                  source_refs, supersedes, superseded_by, tags, file_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  type=excluded.type,
                  title=excluded.title,
                  subject=excluded.subject,
                  predicate=excluded.predicate,
                  value=excluded.value,
                  body=excluded.body,
                  summary=excluded.summary,
                  status=excluded.status,
                  confidence=excluded.confidence,
                  importance=excluded.importance,
                  created_at=excluded.created_at,
                  updated_at=excluded.updated_at,
                  valid_from=excluded.valid_from,
                  valid_until=excluded.valid_until,
                  expires_at=excluded.expires_at,
                  source_refs=excluded.source_refs,
                  supersedes=excluded.supersedes,
                  superseded_by=excluded.superseded_by,
                  tags=excluded.tags,
                  file_path=excluded.file_path
                """,
                (
                    record_id,
                    str(type),
                    str(title or ""),
                    str(subject or ""),
                    str(predicate or ""),
                    str(value or ""),
                    str(body or ""),
                    str(summary or ""),
                    str(status or ""),
                    confidence,
                    importance,
                    stored_created_at,
                    stored_updated_at,
                    valid_from,
                    valid_until,
                    expires_at,
                    json.dumps(source_refs, ensure_ascii=False),
                    json.dumps(supersedes, ensure_ascii=False),
                    str(superseded_by or ""),
                    json.dumps(tags, ensure_ascii=False),
                    str(file_path or ""),
                ),
            )
            conn.execute("DELETE FROM memories_fts WHERE id = ?", (record_id,))
            conn.execute(
                "INSERT INTO memories_fts (id, title, subject, predicate, value, body, summary, tags) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record_id,
                    str(title or ""),
                    str(subject or ""),
                    str(predicate or ""),
                    str(value or ""),
                    str(body or ""),
                    str(summary or ""),
                    " ".join(tags),
                ),
            )

    def active_project_cards(self, *, limit: int = 5) -> List[Dict[str, Any]]:
        """Return active project-card records for broad continuity recall."""
        safe_limit = self._coerce_limit(limit)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                  id, type, title, subject, predicate, value, body, summary, status,
                  confidence, importance, created_at, updated_at, valid_from, valid_until, expires_at,
                  source_refs, supersedes, superseded_by, tags, file_path,
                  0.0 AS rank
                FROM memories
                WHERE type = 'project_state' AND status = 'active'
                ORDER BY COALESCE(importance, 0.0) DESC, updated_at DESC, title ASC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [result for result in (self._row_to_result(row) for row in rows) if self._is_temporally_visible(result)]

    def search(
        self,
        query: str,
        *,
        route: str = "",
        limit: int = 10,
        include_raw_events: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search general memory records, excluding raw archive evidence by default.

        Raw events are private evidence, not ordinary generic-search results.
        Archive access must use the provider's feature-gated, session-scoped
        archive paths instead of this broad index helper.
        """
        query_text = str(query or "").strip()
        if not query_text:
            return []
        safe_limit = self._coerce_limit(limit)
        route_key = str(route or "").strip().lower()
        include_historical = route_key in HISTORICAL_ROUTES
        preserve_bm25 = route_key in {"deep_recall", "past_conversation_exact"}
        fts_pools = (
            [("bm25", fts_query) for fts_query in self._fts_queries(query_text)]
            if preserve_bm25
            else self._fts_pools(query_text)
        )
        sql_limit = min(100, max(safe_limit * 4, safe_limit + 10))
        fused: Dict[str, Dict[str, Any]] = {}
        pool_weights = {
            "strict": 1.4,
            "field": 1.2,
            "entity": 1.1,
            "structured_alias": 1.0,
            "relaxed": 0.8,
        }
        with self._connect() as conn:
            for pool_priority, (pool_name, fts_query) in enumerate(fts_pools):
                rows = conn.execute(
                    """
                    SELECT
                      m.id, m.type, m.title, m.subject, m.predicate, m.value, m.body, m.summary, m.status,
                      m.confidence, m.importance, m.created_at, m.updated_at, m.valid_from, m.valid_until, m.expires_at,
                      m.source_refs, m.supersedes, m.superseded_by, m.tags, m.file_path,
                      bm25(memories_fts) AS rank
                    FROM memories_fts
                    JOIN memories m ON m.id = memories_fts.id
                    WHERE memories_fts MATCH ?
                      AND (? OR m.type != 'raw_event')
                    ORDER BY rank, m.id
                    LIMIT ?
                    """,
                    (fts_query, bool(include_raw_events), sql_limit),
                ).fetchall()
                for position, row in enumerate(rows, start=1):
                    result = self._row_to_result(row)
                    if not self._is_temporally_visible(result, include_historical=include_historical):
                        continue
                    item_id = str(result.get("id") or "")
                    existing = fused.get(item_id)
                    if existing is None:
                        result["retrieval_pools"] = []
                        result["rrf_score"] = 0.0
                        result["bm25_pool_priority"] = pool_priority
                        result["bm25_pool_rank"] = float(result.get("rank") or 0.0)
                        fused[item_id] = result
                        existing = result
                    existing["retrieval_pools"].append(pool_name)
                    existing["rrf_score"] += pool_weights.get(pool_name, 1.0) / (60.0 + position)
                    existing["rank"] = min(
                        float(existing.get("rank") or 0.0), float(result.get("rank") or 0.0)
                    )
                    if pool_priority < int(existing.get("bm25_pool_priority") or 0):
                        existing["bm25_pool_priority"] = pool_priority
                        existing["bm25_pool_rank"] = float(result.get("rank") or 0.0)
                if preserve_bm25 and rows:
                    break
        results = self._enrich_source_signals(list(fused.values()))
        if preserve_bm25:
            results = self._annotate_hybrid_scores(query_text, route=route, results=results)
            results.sort(
                key=lambda item: (
                    int(item.get("bm25_pool_priority") or 0),
                    float(item.get("bm25_pool_rank") or 0.0),
                    str(item.get("id") or ""),
                )
            )
            for item in results:
                item["rank"] = float(item.get("bm25_pool_rank") or 0.0)
        else:
            results = self._hybrid_rank_results(query_text, route=route, results=results)
        results = results[:safe_limit]
        self.log_retrieval(query_text, route=route, retrieved_ids=[result["id"] for result in results])
        return results

    def _enrich_source_signals(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        source_ids = sorted(
            {
                str(source_id)
                for result in results
                for source_id in (result.get("source_refs") or [])
                if str(source_id)
            }
        )
        metadata: Dict[str, tuple[str, str]] = {}
        if source_ids:
            placeholders = ", ".join("?" for _ in source_ids)
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT id, type, observed_at FROM source_refs WHERE id IN ({placeholders})",
                    source_ids,
                ).fetchall()
            metadata = {
                str(row[0]): (str(row[1] or ""), str(row[2] or "")) for row in rows
            }
        quality_by_type = {
            "manual": 1.0,
            "message": 0.95,
            "tool_result": 0.9,
            "file": 0.85,
            "session": 0.8,
            "skill": 0.8,
            "web": 0.7,
            "memory": 0.6,
        }
        enriched: List[Dict[str, Any]] = []
        for result in results:
            item = dict(result)
            refs = [str(value) for value in (item.get("source_refs") or []) if str(value)]
            source_rows = [metadata[source_id] for source_id in refs if source_id in metadata]
            if source_rows:
                item["source_quality"] = sum(
                    quality_by_type.get(source_type, 0.5)
                    for source_type, _observed_at in source_rows
                ) / len(source_rows)
                observed = [observed_at for _source_type, observed_at in source_rows if observed_at]
                if observed:
                    item["evidence_at"] = max(
                        observed, key=lambda value: self._timestamp_epoch(value)
                    )
            elif refs:
                item["source_quality"] = 0.35
            else:
                item["source_quality"] = 0.0
            item.setdefault(
                "evidence_at", item.get("updated_at") or item.get("created_at") or ""
            )
            enriched.append(item)
        return enriched

    @classmethod
    def _annotate_hybrid_scores(cls, query: str, *, route: str = "", results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        query_terms = cls._content_terms(query)
        query_phrase = " ".join(query_terms)
        annotated: List[Dict[str, Any]] = []
        for result in results:
            components = cls._score_components(result, query_terms=query_terms, query_phrase=query_phrase, route=route)
            enriched = dict(result)
            enriched["hybrid_score"] = sum(components.values())
            enriched["score_components"] = components
            annotated.append(enriched)
        return annotated

    @classmethod
    def _hybrid_rank_results(cls, query: str, *, route: str = "", results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        query_terms = cls._content_terms(query)
        query_phrase = " ".join(query_terms)
        ranked: List[Dict[str, Any]] = []
        for result in results:
            components = cls._score_components(result, query_terms=query_terms, query_phrase=query_phrase, route=route)
            score = sum(components.values())
            enriched = dict(result)
            enriched["hybrid_score"] = score
            enriched["score_components"] = components
            ranked.append(enriched)
        ranked.sort(
            key=lambda item: (
                float(item.get("hybrid_score") or 0.0),
                -cls._status_order(str(item.get("status") or "")),
                str(item.get("updated_at") or ""),
                str(item.get("id") or ""),
            ),
            reverse=True,
        )
        return ranked

    @classmethod
    def _score_components(
        cls,
        result: Dict[str, Any],
        *,
        query_terms: List[str],
        query_phrase: str,
        route: str,
    ) -> Dict[str, float]:
        rank = result.get("rank")
        try:
            # SQLite FTS5 bm25() returns smaller/more-negative scores for
            # stronger lexical matches. Keep that signal dominant for raw
            # evidence retrieval (for example LoCoMo), while still allowing
            # route/type and structured-field boosts to break close ties.
            fts_score = min(20.0, max(0.0, -float(rank or 0.0) * 5_000_000.0))
        except (TypeError, ValueError):
            fts_score = 0.0
        token_overlap = cls._token_overlap_score(result, query_terms)
        phrase_boost = cls._phrase_boost(result, query_phrase)
        route_type_boost = cls._route_type_boost(str(route or ""), str(result.get("type") or ""))
        status_boost = cls._status_boost(str(result.get("status") or ""))
        alias_terms = cls._structured_alias_terms(" ".join(query_terms))
        structured_match = cls._structured_match_score(
            result, query_terms=query_terms, alias_terms=alias_terms
        )
        evidence_time = min(
            0.75,
            max(0.0, cls._timestamp_epoch(result.get("evidence_at")) / 2_000_000_000.0)
            * 0.75,
        )
        source_quality = cls._bounded_float(result.get("source_quality"), default=0.0) * 0.75
        pool_fusion = min(3.0, max(0.0, float(result.get("rrf_score") or 0.0) * 30.0))
        confidence_boost = cls._bounded_float(result.get("confidence"), default=0.5) * 0.15
        importance_boost = cls._bounded_float(result.get("importance"), default=0.5) * 0.25
        return {
            "fts": round(fts_score, 6),
            "pool_fusion": round(pool_fusion, 6),
            "token_overlap": round(token_overlap, 6),
            "phrase": round(phrase_boost, 6),
            "route_type_boost": round(route_type_boost, 6),
            "status": round(status_boost, 6),
            "structured_match": round(structured_match, 6),
            "evidence_time": round(evidence_time, 6),
            "source_quality": round(source_quality, 6),
            "confidence": round(confidence_boost, 6),
            "importance": round(importance_boost, 6),
        }

    @classmethod
    def _structured_match_score(
        cls,
        result: Dict[str, Any],
        *,
        query_terms: List[str],
        alias_terms: List[str],
    ) -> float:
        wanted = set(query_terms) | set(alias_terms)
        if not wanted:
            return 0.0
        structured = " ".join(
            [
                str(result.get("title") or ""),
                str(result.get("subject") or ""),
                str(result.get("predicate") or ""),
                " ".join(str(tag) for tag in (result.get("tags") or [])),
            ]
        )
        matched = wanted & set(cls._content_terms(structured))
        score = min(2.0, 2.0 * len(matched) / max(1, len(wanted)))
        result_haystack = " ".join(
            str(result.get(field) or "")
            for field in ("title", "subject", "predicate", "value", "summary", "body")
        )
        result_slots = cls._semantic_slots(result_haystack)
        query_slots = cls._semantic_slots(" ".join([*query_terms, *alias_terms]))
        if result_slots & query_slots:
            score += 1.25
        return score

    @classmethod
    def _token_overlap_score(cls, result: Dict[str, Any], query_terms: List[str]) -> float:
        if not query_terms:
            return 0.0
        query_set = set(query_terms)
        field_weights = {
            "title": 1.25,
            "subject": 1.1,
            "predicate": 1.0,
            "value": 1.2,
            "summary": 1.0,
            "body": 0.75,
            "tags": 0.8,
        }
        matched_weight = 0.0
        max_weight = sum(field_weights.values())
        for field, weight in field_weights.items():
            value = result.get(field)
            if isinstance(value, list):
                text = " ".join(str(part) for part in value)
            else:
                text = str(value or "")
            field_terms = set(cls._content_terms(text))
            if field_terms:
                matched_weight += weight * (len(query_set & field_terms) / len(query_set))
        return 4.0 * (matched_weight / max_weight)

    @classmethod
    def _phrase_boost(cls, result: Dict[str, Any], query_phrase: str) -> float:
        if not query_phrase or len(query_phrase) < 6:
            return 0.0
        haystack = " ".join(
            str(result.get(field) or "")
            for field in ("title", "subject", "predicate", "value", "summary", "body")
        ).lower()
        if query_phrase in " ".join(cls._content_terms(haystack)):
            return 1.0
        return 0.0

    @staticmethod
    def _route_type_boost(route: str, memory_type: str) -> float:
        route_key = route.strip().lower()
        type_key = memory_type.strip().lower()
        preferred: Dict[str, Dict[str, float]] = {
            "project_continuity": {"project_state": 2.0, "preference": 0.25, "fact": 0.2, "raw_event": 0.1},
            "preference_recall": {"preference": 2.0, "fact": 0.35, "project_state": 0.15, "raw_event": 0.1},
            "environment_fact": {"environment": 2.0, "fact": 0.8, "raw_event": 0.1},
            "procedure_lookup": {"procedure_ref": 2.0, "fact": 0.3, "raw_event": 0.1},
            "deep_recall": {"raw_event": 0.25, "project_state": 0.15, "fact": 0.1, "preference": 0.1},
            "past_conversation_exact": {"raw_event": 0.5},
        }
        return preferred.get(route_key, {}).get(type_key, 0.0)

    @staticmethod
    def _status_boost(status: str) -> float:
        return {
            "active": 0.8,
            "uncertain": 0.45,
            "pending": 0.35,
            "promoted": 0.3,
            "archived_only": 0.1,
            "archived": 0.0,
            "resolved": -0.75,
            "stale": -0.9,
            "superseded": -1.0,
            "rejected": -1.5,
        }.get(status, 0.0)

    @staticmethod
    def _status_order(status: str) -> int:
        return {
            "active": 0,
            "uncertain": 1,
            "pending": 2,
            "promoted": 3,
            "archived_only": 4,
            "archived": 5,
            "resolved": 6,
            "stale": 6,
            "superseded": 6,
            "rejected": 7,
        }.get(status, 8)

    @staticmethod
    def _bounded_float(value: Any, *, default: float = 0.0) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = default
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _event_import_key_sha256(event: Dict[str, Any]) -> str:
        key = str(event.get("import_key_sha256") or "").strip()
        if not re.fullmatch(r"sha256:[a-fA-F0-9]{64}", key):
            return ""
        return key

    @staticmethod
    def _raw_import_key_sha256(event: Dict[str, Any]) -> str:
        payload = {
            "source_system": str(event.get("source_system") or ""),
            "session_id": str(event.get("session_id") or ""),
            "provider_session_id": str(event.get("provider_session_id") or ""),
            "message_id": str(event.get("message_id") or ""),
            "chain_index": event.get("chain_index"),
            "record_sha256": str(event.get("record_sha256") or ""),
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def _content_terms(cls, text: str) -> List[str]:
        return [term.lower() for term in cls._fts_terms(text) if term.lower() not in cls._STOPWORDS]

    def rebuild_raw_archive_index(self, store: MemoryV2Store) -> Dict[str, Any]:
        """Rebuild only derived raw archive index tables from canonical JSONL."""
        with self._rebuild_lock, store.profile_lock():
            self._require_verified_raw_archive(store)
            self.initialize()
            count = 0
            last_record_sha256 = ""
            try:
                with self._connect() as conn:
                    conn.execute("DELETE FROM raw_events")
                    conn.execute("DELETE FROM raw_events_fts")
                    if store.raw_events_path.exists():
                        with store.raw_events_path.open("rb") as fh:
                            line_no = 0
                            while True:
                                byte_offset = fh.tell()
                                line = fh.readline()
                                if not line:
                                    break
                                line_no += 1
                                stripped = line.strip()
                                if not stripped:
                                    continue
                                event = json.loads(stripped.decode("utf-8"))
                                if not isinstance(event, dict):
                                    raise ValidationError(f"raw archive line {line_no} is not a JSON object")
                                if self.index_raw_archive_event(
                                    event,
                                    byte_offset=byte_offset,
                                    byte_length=len(line),
                                    line_no=line_no,
                                    _conn=conn,
                                ):
                                    count += 1
                                    last_record_sha256 = str(event.get("record_sha256") or last_record_sha256)
                    # Recheck inside the transaction immediately before commit.
                    # This catches direct filesystem modification during the
                    # scan; normal appends are excluded by the profile lock.
                    self._require_verified_raw_archive(store)
            except ValidationError:
                self._mark_raw_index_unusable(store, status="integrity_failed")
                raise
            except Exception:
                self._mark_raw_index_unusable(store, status="stale")
                raise
            status = {
                "derived_index_status": "ok",
                "indexed_event_count": count,
                "last_indexed_record_sha256": last_record_sha256,
                "raw_index_schema_version": RAW_ARCHIVE_INDEX_SCHEMA_VERSION,
            }
            if hasattr(store, "update_raw_archive_index_status"):
                store.update_raw_archive_index_status(**status)
            return status

    def rebuild_from_store(self, store: MemoryV2Store) -> Dict[str, int]:
        """Rebuild the derived index from canonical files and atomically publish it.

        Rebuilds are staged into a temporary SQLite database next to the live DB.
        The live index is only replaced after the staged DB is fully populated and
        passes SQLite integrity checks, so interrupted/crashing rebuilds preserve
        the previous usable index.
        """
        with self._rebuild_lock, store.profile_lock():
            self._require_verified_raw_archive(store)
            original_db_path = self.db_path
            original_db_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_db_path = original_db_path.with_name(f".{original_db_path.name}.{uuid.uuid4().hex}.tmp")
            counts = {"project_cards": 0, "candidates": 0, "raw_events": 0, "source_refs": 0, "memory_items": 0, "open_loops": 0}
            try:
                self.db_path = tmp_db_path
                self.initialize()
                for source in store.list_source_refs():
                    self.index_source_ref(source)
                    counts["source_refs"] += 1
                for item in store.list_memory_items():
                    self.index_memory_item(item, file_path=store._memory_item_path(item.id))
                    counts["memory_items"] += 1
                for card in store.list_project_cards():
                    self.index_project_card(card, file_path=store._project_card_path(card.id))
                    counts["project_cards"] += 1
                for candidate in store.list_candidates():
                    self.index_candidate(candidate)
                    counts["candidates"] += 1
                for loop in store.list_open_loops():
                    self.index_open_loop(loop, file_path=store.open_loops_path)
                    counts["open_loops"] += 1
                last_record_sha256 = ""
                if store.raw_events_path.exists():
                    with store.raw_events_path.open("rb") as fh:
                        line_no = 0
                        while True:
                            byte_offset = fh.tell()
                            line = fh.readline()
                            if not line:
                                break
                            line_no += 1
                            stripped = line.strip()
                            if not stripped:
                                continue
                            event = json.loads(stripped.decode("utf-8"))
                            if not isinstance(event, dict):
                                continue
                            if self.index_raw_event(event, index_archive=False):
                                self.index_raw_archive_event(
                                    event,
                                    byte_offset=byte_offset,
                                    byte_length=len(line),
                                    line_no=line_no,
                                )
                                counts["raw_events"] += 1
                                last_record_sha256 = str(event.get("record_sha256") or last_record_sha256)
                with self._connect() as conn:
                    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
                    if integrity != "ok":
                        raise sqlite3.DatabaseError(f"rebuilt index failed integrity_check: {integrity}")
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._require_verified_raw_archive(store)
                self.db_path = original_db_path
                if original_db_path.exists():
                    with self._connect() as conn:
                        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                os.replace(tmp_db_path, original_db_path)
                for suffix in ("-wal", "-shm"):
                    path = Path(str(original_db_path) + suffix)
                    if path.exists():
                        path.unlink()
                if hasattr(store, "update_raw_archive_index_status"):
                    store.update_raw_archive_index_status(
                        derived_index_status="ok",
                        indexed_event_count=counts["raw_events"],
                        last_indexed_record_sha256=last_record_sha256,
                        raw_index_schema_version=RAW_ARCHIVE_INDEX_SCHEMA_VERSION,
                    )
                return counts
            finally:
                self.db_path = original_db_path
                for suffix in ("", "-wal", "-shm"):
                    path = Path(str(tmp_db_path) + suffix)
                    if path.exists():
                        path.unlink()

    def log_retrieval(self, query: str, *, route: str = "", retrieved_ids: Optional[List[str]] = None) -> None:
        query_text = str(query or "")
        query_hash = hashlib.sha256(redacted_query_hash_input(query_text).encode("utf-8")).hexdigest()
        query_for_log = redacted_query_for_log(query_text)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO retrieval_log (query, query_hash, route, retrieved_ids, created_at) VALUES (?, ?, ?, ?, ?)",
                (query_for_log, query_hash, route, json.dumps(retrieved_ids or [], ensure_ascii=False), utc_now_iso()),
            )

    def retrieval_logs(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, query, query_hash, route, retrieved_ids, created_at FROM retrieval_log ORDER BY id"
            ).fetchall()
        return [
            {
                "id": row[0],
                "query": row[1],
                "query_hash": row[2] or "",
                "route": row[3],
                "retrieved_ids": json.loads(row[4] or "[]"),
                "created_at": row[5],
            }
            for row in rows
        ]

    @staticmethod
    def _is_temporally_visible(result: Dict[str, Any], *, include_historical: bool = False) -> bool:
        now = datetime.now(timezone.utc)
        valid_from = MemoryV2Index._parse_time(result.get("valid_from"))
        valid_until = MemoryV2Index._parse_time(result.get("valid_until"))
        expires_at = MemoryV2Index._parse_time(result.get("expires_at"))
        if valid_from is not None and valid_from > now:
            return False
        if not include_historical and valid_until is not None and valid_until < now:
            return False
        if not include_historical and expires_at is not None and expires_at < now:
            return False
        return True

    @staticmethod
    def _parse_time(value: Any) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    @staticmethod
    def _normalized_time_filter(value: Any, *, field_name: str) -> str:
        parsed = MemoryV2Index._parse_time(value)
        if parsed is None:
            raise ValidationError(f"{field_name} must be an ISO-8601 timestamp")
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _mark_raw_index_unusable(store: MemoryV2Store, *, status: str) -> None:
        if not hasattr(store, "update_raw_archive_index_status"):
            return
        manifest = store.read_raw_archive_manifest()
        store.update_raw_archive_index_status(
            derived_index_status=status,
            indexed_event_count=int(manifest.get("indexed_event_count") or 0),
            last_indexed_record_sha256=str(manifest.get("last_indexed_record_sha256") or ""),
            raw_index_schema_version=int(
                manifest.get("raw_index_schema_version") or RAW_ARCHIVE_INDEX_SCHEMA_VERSION
            ),
        )

    @classmethod
    def _require_verified_raw_archive(cls, store: MemoryV2Store) -> Dict[str, Any]:
        verification = store.verify_raw_archive()
        if str(verification.get("status") or "").lower() != "ok":
            cls._mark_raw_index_unusable(store, status="integrity_failed")
            raise ValidationError(
                "raw archive integrity verification failed; refusing to rebuild derived index "
                f"(issues={int(verification.get('issue_count') or 0)})"
            )
        return verification

    @staticmethod
    def _timestamp_epoch(value: Any) -> float:
        parsed = MemoryV2Index._parse_time(value)
        return parsed.timestamp() if parsed is not None else 0.0

    @staticmethod
    def _redacted_query_for_log(query: str) -> str:
        sensitive_patterns = (
            r"(?i)password\s*(?:is|=|:)\s*\S+",
            r"(?i)password\s+\S+",
            r"(?i)passwd\s*(?:is|=|:)\s*\S+",
            r"(?i)passwd\s+\S+",
            r"(?i)token\s*(?:is|=|:)\s*\S+",
            r"(?i)token\s+\S+",
            r"(?i)secret\s*(?:is|=|:)\s*\S+",
            r"(?i)secret\s+\S+",
            r"(?i)client\s+secret\s*(?:is|=|:)\s*\S+",
            r"(?i)client\s+secret\s+\S+",
            r"(?i)credential\s*(?:is|=|:)\s*\S+",
            r"(?i)credential\s+\S+",
            r"(?i)private\s+key\s*(?:is|=|:)\s*\S+",
            r"(?i)private\s+key\s+\S+",
            r"(?i)api[_ -]?key\s*(?:is|=|:)\s*\S+",
            r"(?i)api[_ -]?key\s+\S+",
            r"(?i)authorization\s*:\s*bearer\s+\S+",
            r"(?i)bearer\s+\S+",
            r"(?i)[A-Z0-9_]*API[_-]?KEY\s*=\s*\S+",
            r"(?i)[A-Z0-9_]*API[_-]?KEY\s+\S+",
            r"(?i)\bsk-[A-Za-z0-9][A-Za-z0-9_-]{8,}\b",
            r"(?i)\bgh[pousr]_[A-Za-z0-9_]{8,}\b",
            r"(?i)\bxox[baprs]-[A-Za-z0-9-]{8,}\b",
            r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b",
        )
        if any(re.search(pattern, query) for pattern in sensitive_patterns):
            return "[REDACTED sensitive query]"
        return query[:500]

    @staticmethod
    def _coerce_limit(limit: Any) -> int:
        try:
            value = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit must be an integer") from exc
        return max(1, min(value, 50))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0, factory=_ClosingConnection)
        # Canonical mutations already serialize through the cross-process
        # profile lock. DELETE journaling avoids inheriting fragile WAL shared
        # state when worker processes are forked after SQLite was initialized.
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @classmethod
    def _fts_pools(cls, query: str) -> List[tuple[str, str]]:
        terms = cls._fts_terms(query)
        if not terms:
            return []
        content_terms = [
            term for term in terms if term.lower() not in cls._STOPWORDS
        ]
        if not content_terms:
            return []
        strict = " AND ".join(f'"{term}"' for term in terms)
        relaxed = " OR ".join(f'"{term}"' for term in content_terms)
        pools: List[tuple[str, str]] = [("strict", strict)]

        field_terms = content_terms[:8]
        field_query = " OR ".join(
            f'title:"{term}" OR subject:"{term}" OR predicate:"{term}" OR tags:"{term}"'
            for term in field_terms
        )
        if field_query:
            pools.append(("field", field_query))

        entity_terms = cls._entity_terms(query)
        if entity_terms:
            pools.append(
                (
                    "entity",
                    " OR ".join(f'"{term}"' for term in entity_terms),
                )
            )

        alias_terms = cls._structured_alias_terms(query)
        if alias_terms:
            pools.append(
                (
                    "structured_alias",
                    " OR ".join(f'"{term}"' for term in alias_terms),
                )
            )
        pools.append(("relaxed", relaxed))

        seen_queries: set[str] = set()
        unique: List[tuple[str, str]] = []
        for name, fts_query in pools:
            if fts_query and fts_query not in seen_queries:
                unique.append((name, fts_query))
                seen_queries.add(fts_query)
        return unique

    @classmethod
    def _entity_terms(cls, query: str) -> List[str]:
        excluded = {
            "What",
            "Where",
            "When",
            "Which",
            "Who",
            "Why",
            "How",
            "Did",
            "Does",
            "The",
            "Project",
        }
        values: List[str] = []
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_.+-]{1,}\b", str(query or "")):
            if token in excluded:
                continue
            if token.isupper() or token[:1].isupper():
                lowered = token.lower()
                if lowered not in cls._STOPWORDS and lowered not in values:
                    values.append(lowered)
        return values[:6]

    _STRUCTURED_ALIAS_GROUPS: Dict[str, tuple[set[str], tuple[str, ...]]] = {
        "voice": (
            {"voice", "tts", "spoken", "speech", "narrator", "narration", "audio"},
            ("tts", "voice", "spoken", "speech", "narrator"),
        ),
        "response_style": (
            {"answer", "answers", "reply", "replies", "response", "respond", "tone", "preface", "prefaces"},
            ("response", "style", "answers", "direct", "concise", "source", "grounded"),
        ),
        "notification": (
            {"notification", "notifications", "digest", "alert", "alerts"},
            ("notification", "digest", "preference"),
        ),
        "project_state": (
            {"project", "resume", "continue", "left", "leave", "move", "prefetch", "goal", "blocker"},
            ("project", "current", "state", "next", "action", "goal", "decision"),
        ),
        "environment": (
            {"environment", "runtime", "machine", "deploy", "path", "directory", "setup"},
            ("environment", "runtime", "deploy", "target", "path", "directory"),
        ),
        "procedure": (
            {"workflow", "steps", "process", "procedure", "troubleshoot"},
            ("procedure", "workflow", "steps", "runbook"),
        ),
        "preference": (
            {"prefer", "prefers", "preference", "preferences", "liked", "likes"},
            ("preference", "prefers", "current"),
        ),
        "history": (
            {"stale", "old", "outdated", "superseded", "replaced", "contradiction"},
            ("stale", "superseded", "current", "instead"),
        ),
    }

    @classmethod
    def _structured_alias_terms(cls, query: str) -> List[str]:
        terms = set(cls._content_terms(query))
        aliases: List[str] = []
        for _slot, (triggers, expansions) in cls._STRUCTURED_ALIAS_GROUPS.items():
            if terms & triggers:
                for expansion in expansions:
                    if expansion not in aliases:
                        aliases.append(expansion)
        return aliases[:16]

    @classmethod
    def _semantic_slots(cls, text: str) -> set[str]:
        terms = set(cls._content_terms(text))
        return {
            slot
            for slot, (triggers, _expansions) in cls._STRUCTURED_ALIAS_GROUPS.items()
            if terms & triggers
        }

    @staticmethod
    def _fts_queries(query: str) -> List[str]:
        terms = MemoryV2Index._fts_terms(query)
        if not terms:
            return []
        relaxed_terms = [term for term in terms if term not in MemoryV2Index._STOPWORDS]
        if not relaxed_terms:
            return []
        strict = " AND ".join(f'"{term}"' for term in terms)
        if len(relaxed_terms) < 2:
            if not relaxed_terms:
                return [strict] if terms == relaxed_terms else []
            relaxed_single = " AND ".join(f'"{term}"' for term in relaxed_terms)
            return [strict] if terms == relaxed_terms else [relaxed_single]
        relaxed_and = " AND ".join(f'"{term}"' for term in relaxed_terms)
        relaxed_or = " OR ".join(f'"{term}"' for term in relaxed_terms)
        queries = [strict]
        if relaxed_and != strict:
            queries.append(relaxed_and)
        queries.append(relaxed_or)
        return queries

    _STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "is",
        "it",
        "me",
        "of",
        "or",
        "should",
        "that",
        "the",
        "to",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
    }

    @staticmethod
    def _fts_terms(query: str) -> List[str]:
        return [term for term in "".join(ch if ch.isalnum() else " " for ch in query).split() if term]

    @staticmethod
    def _fts_query(query: str) -> str:
        # Backwards-compatible strict AND query used by older tests/callers.
        terms = MemoryV2Index._fts_terms(query)
        if not terms:
            return '""'
        return " AND ".join(f'"{term}"' for term in terms)

    @staticmethod
    def _row_to_result(row: sqlite3.Row | tuple) -> Dict[str, Any]:
        return {
            "id": row[0],
            "type": row[1],
            "title": row[2] or "",
            "subject": row[3] or "",
            "predicate": row[4] or "",
            "value": row[5] or "",
            "body": row[6] or "",
            "summary": row[7] or "",
            "status": row[8] or "",
            "confidence": row[9],
            "importance": row[10],
            "created_at": row[11] or "",
            "updated_at": row[12] or "",
            "valid_from": row[13] or "",
            "valid_until": row[14] or "",
            "expires_at": row[15] or "",
            "source_refs": json.loads(row[16] or "[]"),
            "supersedes": json.loads(row[17] or "[]"),
            "superseded_by": row[18] or "",
            "tags": json.loads(row[19] or "[]"),
            "file_path": row[20] or "",
            "rank": row[21],
        }
