"""Read-only SessionDB backfill into Memory v2 raw archive.

This importer deliberately talks to Hermes state.db through a read-only sqlite
connection instead of constructing ``SessionDB``.  Constructing SessionDB can
initialize/reconcile schema and mutate journal settings; backfill should only
read session history and append redacted, tamper-evident raw archive events.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .backfill_state import (
    SessionBackfillScope,
    SessionBackfillState,
    state_db_fingerprint,
    state_db_identity,
)
from .redaction import redact_data, redact_text
from .store import MemoryV2Store
from .index import MemoryV2Index

SESSION_BACKFILL_CONFIRM = "IMPORT_SESSIONDB_TO_MEMORY_V2"
_CONTENT_JSON_PREFIX = "\x00json:"
_ALLOWED_ROLES = {"user", "assistant", "tool"}


def backfill_session_db(
    *,
    state_db_path: str | Path,
    store: MemoryV2Store,
    index: MemoryV2Index,
    source: str = "",
    session_id: str = "",
    since_message_id: int | None = None,
    until_message_id: int | None = None,
    limit: int = 500,
    dry_run: bool = True,
    include_tools: bool = False,
    resume: bool = False,
    batch_size: int = 500,
    max_batches: int | None = None,
) -> Dict[str, Any]:
    """Import visible SessionDB messages as raw archive evidence.

    Returns counts and safe ids only; never returns raw message text.
    """
    db_path = Path(state_db_path).expanduser().resolve()
    if not db_path.exists():
        return {"success": False, "error": "state_db_path not found"}
    safe_limit = max(1, min(int(limit), 5000))
    safe_batch_size = max(1, min(int(batch_size), 5000, safe_limit))
    safe_max_batches = int(max_batches) if max_batches is not None else None
    if safe_max_batches is not None and safe_max_batches < 1:
        return {"success": False, "error": "max_batches must be a positive integer"}
    fingerprint = state_db_fingerprint(db_path)
    backfill_state = SessionBackfillState(store.base_dir)
    scope = SessionBackfillScope(
        source=str(source or ""),
        session_id=str(session_id or ""),
        include_tools=bool(include_tools),
        state_db_identity=state_db_identity(db_path),
        state_db_fingerprint=fingerprint,
    )
    checkpoint = backfill_state.checkpoint_for(scope)
    effective_since = since_message_id
    resumed_from_checkpoint = False
    if resume and since_message_id is None and checkpoint and checkpoint.get("last_message_id") is not None:
        effective_since = int(checkpoint["last_message_id"])
        resumed_from_checkpoint = True

    def _run() -> Dict[str, Any]:
        return _backfill_session_db_unlocked(
            db_path=db_path,
            store=store,
            index=index,
            source=source,
            session_id=session_id,
            since_message_id=effective_since,
            until_message_id=until_message_id,
            limit=safe_limit,
            dry_run=dry_run,
            include_tools=include_tools,
            resume=resume,
            resumed_from_checkpoint=resumed_from_checkpoint,
            batch_size=safe_batch_size,
            max_batches=safe_max_batches,
            backfill_state=backfill_state,
            scope=scope,
        )

    try:
        if dry_run:
            return _run()
        with backfill_state.acquire_lock():
            return _run()
    except (RuntimeError, ValueError) as exc:
        return {"success": False, "error": str(exc)}


def _backfill_session_db_unlocked(
    *,
    db_path: Path,
    store: MemoryV2Store,
    index: MemoryV2Index,
    source: str,
    session_id: str,
    since_message_id: int | None,
    until_message_id: int | None,
    limit: int,
    dry_run: bool,
    include_tools: bool,
    resume: bool,
    resumed_from_checkpoint: bool,
    batch_size: int,
    max_batches: int | None,
    backfill_state: SessionBackfillState,
    scope: SessionBackfillScope,
) -> Dict[str, Any]:
    if not dry_run:
        store.require_usable_raw_index(index)
    imported_ids: List[str] = []
    skipped_ids: List[str] = []
    skipped_reasons: Dict[str, int] = {}
    errors: List[Dict[str, Any]] = []
    last_message_id = None
    considered = 0
    batches = 0
    cursor = since_message_id
    stopped_reason = "completed"
    while considered < limit:
        if max_batches is not None and batches >= max_batches:
            stopped_reason = "max_batches"
            break
        remaining = limit - considered
        rows = _select_rows(
            db_path,
            source=source,
            session_id=session_id,
            since_message_id=cursor,
            until_message_id=until_message_id,
            limit=min(batch_size, remaining),
            include_tools=include_tools,
        )
        if not rows:
            backfill_state.update_checkpoint(
                scope,
                last_message_id=last_message_id,
                imported_delta=0,
                skipped_delta=0,
                dry_run=dry_run,
                completed=True,
            )
            break
        batches += 1
        batch_imported = 0
        batch_skipped = 0
        for row in rows:
            considered += 1
            last_message_id = int(row["message_id"])
            cursor = last_message_id
            event_id = deterministic_event_id(str(row["session_id"]), last_message_id)
            import_key = raw_import_key_sha256(str(row["session_id"]), last_message_id)
            indexed_event_exists = store.raw_event_exists(event_id, index=index)
            indexed_import_exists = store.raw_import_key_exists(import_key, index=index)
            if indexed_event_exists or indexed_import_exists:
                skipped_ids.append(event_id)
                batch_skipped += 1
                skipped_reasons["already_imported"] = skipped_reasons.get("already_imported", 0) + 1
                continue
            # Source refs are written alongside canonical raw events.  If the
            # canonical sidecar says this deterministic import already exists
            # but the raw SQLite index missed it, the index is stale/corrupt;
            # fail closed instead of appending a duplicate JSONL event.
            if not dry_run and store.read_source_ref(event_id) is not None:
                raise ValueError("raw archive index is missing existing SessionDB import; rebuild Memory v2 indexes before backfill")
            try:
                event = raw_event_from_row(row)
            except (TypeError, ValueError) as exc:
                errors.append({"message_id": last_message_id, "error": str(exc)[:120]})
                batch_skipped += 1
                skipped_reasons["malformed_row"] = skipped_reasons.get("malformed_row", 0) + 1
                continue
            if dry_run:
                imported_ids.append(event_id)
                batch_imported += 1
                continue
            persisted = store.append_raw_event(event)
            # append_raw_event() updates the raw archive index with canonical JSONL
            # byte offsets.  Only add the general semantic raw-event index here;
            # re-indexing archive metadata without offsets would clobber them to NULL.
            index.index_raw_event(persisted, index_archive=False)
            if source_ref := store.read_source_ref(str(persisted["id"])):
                index.index_source_ref(source_ref)
            imported_ids.append(event_id)
            batch_imported += 1
        completed = len(rows) < min(batch_size, remaining)
        backfill_state.update_checkpoint(
            scope,
            last_message_id=last_message_id,
            imported_delta=batch_imported,
            skipped_delta=batch_skipped,
            dry_run=dry_run,
            completed=completed,
        )
        if considered >= limit:
            stopped_reason = "limit"
            break
        if completed:
            break
    checkpoint_after = None if dry_run else backfill_state.checkpoint_for(scope)
    return {
        "success": True,
        "mode": "sessiondb_backfill",
        "dry_run": dry_run,
        "source": source,
        "session_id": session_id,
        "limit": limit,
        "batch_size": batch_size,
        "batches": batches,
        "max_batches": max_batches,
        "resume": resume,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "checkpoint_last_message_id": (checkpoint_after or {}).get("last_message_id"),
        "considered": considered,
        "imported": len(imported_ids),
        "skipped": len(skipped_ids) + sum(skipped_reasons.get(k, 0) for k in ("malformed_row",)),
        "skipped_reasons": skipped_reasons,
        "error_count": len(errors),
        "errors": errors[:10],
        "imported_ids": imported_ids[:50],
        "skipped_ids": skipped_ids[:50],
        "next_since_message_id": last_message_id,
        "stopped_reason": stopped_reason,
        "raw_archive": store._read_raw_archive_manifest_if_present() if dry_run else store.read_raw_archive_manifest(),
    }


def deterministic_event_id(session_id: str, message_id: int) -> str:
    raw_key = f"sessiondb:v1:{session_id}:{int(message_id)}"
    return "event_sessiondb_" + hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:32]


def raw_import_key_sha256(session_id: str, message_id: int) -> str:
    return "sha256:" + hashlib.sha256(f"sessiondb:v1:{session_id}:{int(message_id)}".encode("utf-8")).hexdigest()


def raw_event_from_row(row: sqlite3.Row | Dict[str, Any]) -> Dict[str, Any]:
    role = str(row["role"] or "").strip().lower()
    if role not in _ALLOWED_ROLES:
        raise ValueError("unsupported role")
    content = _content_to_text(_decode_content(row["content"]))
    tool_name = str(row["tool_name"] or "")
    if role != "tool" and not content.strip():
        raise ValueError("empty message content")
    message_id = int(row["message_id"])
    session_id = str(row["session_id"] or "")
    created_at = _unix_to_iso(row["timestamp"])
    event_id = deterministic_event_id(session_id, message_id)
    raw_key_hash = raw_import_key_sha256(session_id, message_id)
    event_type = "sessiondb_tool_message" if role == "tool" else "sessiondb_message"
    event: Dict[str, Any] = {
        "id": event_id,
        "type": event_type,
        "source_system": "hermes_sessiondb",
        "source_schema_version": 1,
        "import_key_sha256": raw_key_hash,
        "session_id": session_id,
        "provider_session_id": session_id,
        "message_id": message_id,
        "role": role,
        "platform": str(row["session_source"] or ""),
        "tool": tool_name,
        "tool_call_id": str(row["tool_call_id"] or ""),
        "platform_message_id": str(row["platform_message_id"] or ""),
        "observed": bool(row["observed"]),
        "content": content,
        "created_at": created_at,
        "observed_at": created_at,
        "trust_level": "tool_output_untrusted" if role == "tool" else "untrusted",
        "can_instruct": False,
        "privacy_level": "standard",
        "archive_status": "active",
        "session": {
            "source": str(row["session_source"] or ""),
            "model": str(row["model"] or ""),
            "parent_session_id": str(row["parent_session_id"] or ""),
            "started_at": _unix_to_iso(row["session_started_at"]),
            "ended_at": _unix_to_iso(row["session_ended_at"]),
            "end_reason": str(row["end_reason"] or ""),
            "title": str(row["title"] or ""),
        },
    }
    if role == "user":
        event["user_content"] = content
    elif role == "assistant":
        event["assistant_content"] = content
    elif role == "tool":
        event["content"] = content[:8000]
        event["tool_output_capped"] = len(content) > 8000
    return redact_data(event)


def _select_rows(
    db_path: Path,
    *,
    source: str,
    session_id: str,
    since_message_id: int | None,
    until_message_id: int | None,
    limit: int,
    include_tools: bool,
) -> List[sqlite3.Row]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON")
        roles = ("user", "assistant", "tool") if include_tools else ("user", "assistant")
        placeholders = ",".join("?" for _ in roles)
        params: List[Any] = [*roles]
        where = [f"m.role IN ({placeholders})"]
        if source:
            where.append("s.source = ?")
            params.append(source)
        if session_id:
            where.append("m.session_id = ?")
            params.append(session_id)
        if since_message_id is not None:
            where.append("m.id > ?")
            params.append(int(since_message_id))
        if until_message_id is not None:
            where.append("m.id <= ?")
            params.append(int(until_message_id))
        where.append("(COALESCE(m.content, '') != '' OR COALESCE(m.tool_name, '') != '')")
        params.append(limit)
        sql = f"""
            SELECT
              m.id AS message_id,
              m.session_id,
              m.role,
              m.content,
              m.tool_call_id,
              m.tool_name,
              m.timestamp,
              m.platform_message_id,
              m.observed,
              s.source AS session_source,
              s.model,
              s.parent_session_id,
              s.started_at AS session_started_at,
              s.ended_at AS session_ended_at,
              s.end_reason,
              s.title
            FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE {' AND '.join(where)}
            ORDER BY m.id ASC
            LIMIT ?
        """
        return list(conn.execute(sql, params).fetchall())
    finally:
        conn.close()


def _decode_content(content: Any) -> Any:
    if isinstance(content, str) and content.startswith(_CONTENT_JSON_PREFIX):
        try:
            return json.loads(content[len(_CONTENT_JSON_PREFIX):])
        except (json.JSONDecodeError, TypeError):
            return content
    return content


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    if isinstance(content, (int, float)):
        return str(content)
    if isinstance(content, str):
        return redact_text(content)
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
                else:
                    parts.append("[multimodal content omitted]")
            else:
                parts.append(str(part))
        return redact_text("\n".join(parts))
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return redact_text(content["text"])
        return redact_text(json.dumps(redact_data(content), ensure_ascii=False, sort_keys=True))
    return redact_text(str(content))


def _unix_to_iso(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError, OSError):
        return ""
