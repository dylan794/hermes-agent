"""Checkpoint and lock helpers for Memory v2 SessionDB backfill."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from .schemas import utc_now_iso


@dataclass(frozen=True)
class SessionBackfillScope:
    source: str
    session_id: str
    include_tools: bool
    state_db_identity: str
    state_db_fingerprint: str

    @property
    def key(self) -> str:
        payload = {
            "state_db_identity": self.state_db_identity,
            "source": self.source or "",
            "session_id": self.session_id or "",
            "include_tools": bool(self.include_tools),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return "scope-sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def matches_checkpoint(self, checkpoint: Dict[str, Any]) -> bool:
        """Return whether stored checkpoint metadata matches this scope.

        The checkpoint key is a hash for compactness/privacy.  Validate the
        stored fields too so legacy/damaged/colliding entries cannot steer a
        resumed import to the wrong cursor.
        """
        return (
            str(checkpoint.get("source") or "") == (self.source or "")
            and str(checkpoint.get("session_id") or "") == (self.session_id or "")
            and bool(checkpoint.get("include_tools")) == bool(self.include_tools)
            and str(checkpoint.get("state_db_identity") or "") == self.state_db_identity
        )


class SessionBackfillState:
    """Profile-local checkpoint file under memory_v2/backfill/sessiondb.yaml."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.dir = self.base_dir / "backfill"
        self.path = self.dir / "sessiondb.yaml"
        self.lock_path = self.dir / "sessiondb.lock"

    def read(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "checkpoints": {}}
        with self.path.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        if not isinstance(payload, dict):
            return {"schema_version": 1, "checkpoints": {}}
        checkpoints = payload.get("checkpoints")
        if not isinstance(checkpoints, dict):
            payload["checkpoints"] = {}
        payload.setdefault("schema_version", 1)
        return payload

    def checkpoint_for(self, scope: SessionBackfillScope) -> Dict[str, Any] | None:
        payload = self.read()
        checkpoint = (payload.get("checkpoints") or {}).get(scope.key)
        if not isinstance(checkpoint, dict):
            return None
        return checkpoint if scope.matches_checkpoint(checkpoint) else None

    def update_checkpoint(
        self,
        scope: SessionBackfillScope,
        *,
        last_message_id: int | None,
        imported_delta: int,
        skipped_delta: int,
        dry_run: bool,
        completed: bool,
    ) -> Dict[str, Any]:
        if dry_run:
            return self.checkpoint_for(scope) or {}
        payload = self.read()
        checkpoints = payload.setdefault("checkpoints", {})
        previous = checkpoints.get(scope.key) if isinstance(checkpoints.get(scope.key), dict) else {}
        imported_total = int(previous.get("imported_count") or 0) + int(imported_delta)
        skipped_total = int(previous.get("skipped_count") or 0) + int(skipped_delta)
        checkpoint = {
            "source": scope.source,
            "session_id": scope.session_id,
            "include_tools": bool(scope.include_tools),
            "state_db_identity": scope.state_db_identity,
            "state_db_fingerprint": scope.state_db_fingerprint,
            "last_message_id": int(last_message_id) if last_message_id is not None else previous.get("last_message_id"),
            "imported_count": imported_total,
            "skipped_count": skipped_total,
            "last_run_at": utc_now_iso(),
            "completed": bool(completed),
            "dry_run": False,
        }
        checkpoints[scope.key] = checkpoint
        payload["updated_at"] = utc_now_iso()
        self._atomic_write_yaml(payload)
        return checkpoint

    def acquire_lock(self) -> "SessionBackfillLock":
        return SessionBackfillLock(self.lock_path)

    def _atomic_write_yaml(self, payload: Dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + f".tmp.{os.getpid()}")
        with tmp_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)
        os.replace(tmp_path, self.path)


class SessionBackfillLock:
    """Best-effort cross-process lock based on O_EXCL lock-file creation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    def __enter__(self) -> "SessionBackfillLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            self._fd = os.open(str(self.path), flags, 0o600)
        except FileExistsError as exc:
            raise RuntimeError("sessiondb backfill is already running; remove stale memory_v2/backfill/sessiondb.lock only after verifying no import is active") from exc
        payload = f"pid: {os.getpid()}\ncreated_at: {utc_now_iso()}\n"
        os.write(self._fd, payload.encode("utf-8"))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def state_db_identity(path: str | Path) -> str:
    """Stable, privacy-preserving identity for a local SessionDB path."""
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()
    return f"path-sha256:{digest}"


def state_db_fingerprint(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    # Volatile stat fingerprint is informational only.  It intentionally omits
    # the raw path so normal DB growth cannot orphan checkpoint lookup keys and
    # reports do not expose local filesystem layout.
    return f"stat:size={stat.st_size}:mtime_ns={stat.st_mtime_ns}"
