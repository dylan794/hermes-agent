"""Rebuildable, scope-first retrieval over derived evidence units.

This module deliberately accepts and returns mappings.  The canonical archive
is owned elsewhere: rebuilding creates a complete, disposable SQLite/FTS
artifact and search opens that artifact read-only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


SCHEMA_VERSION = 1
MAX_RESULTS = 30
DEFAULT_POOL_LIMIT = 100
DEFAULT_DENSE_SCAN_LIMIT = 500
MAX_ELIGIBLE_ROWS = 10_000
MAX_EMBEDDING_DIMENSION = 8_192
RRF_K = 60
_CURRENT_INVALID_STATUSES = frozenset(
    {
        "archived_only",
        "closed",
        "deleted",
        "historical",
        "rejected",
        "resolved",
        "retracted",
        "stale",
        "superseded",
        "tombstone",
    }
)
_HISTORY_INVALID_STATUSES = frozenset(
    {"archived_only", "deleted", "rejected", "tombstone"}
)
_TOKEN_RE = re.compile(r"[\w][\w.-]*", re.UNICODE)


class ScopedIndexIntegrityError(ValueError):
    """The disposable index or embedding output cannot be trusted."""


class LocalEmbeddingAdapter(Protocol):
    """Minimal local-only adapter accepted by :class:`ScopedEvidenceIndex`."""

    model: str
    version: str
    dimension: int
    artifact_digest: str

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one fixed-width vector for each input text."""


class ScopedEvidenceIndex:
    """Derived SQLite/FTS index with hard scope filters before ranking."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        embedding_adapter: LocalEmbeddingAdapter | None = None,
        dense_scan_limit: int = DEFAULT_DENSE_SCAN_LIMIT,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.embedding_adapter = embedding_adapter
        if not 1 <= int(dense_scan_limit) <= 900:
            raise ValueError("dense_scan_limit must be between 1 and 900")
        self.dense_scan_limit = int(dense_scan_limit)

    def rebuild(
        self,
        evidence_units: Iterable[Mapping[str, Any]],
        *,
        aliases: Mapping[str, Mapping[str, Iterable[str]]] | None = None,
        edges: Iterable[Mapping[str, Any]] = (),
        tombstones: Iterable[str] = (),
        default_profile_id: str | None = None,
        default_tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Atomically replace the derived index without mutating source data."""

        default_profile = (
            _required_text(default_profile_id, "default_profile_id")
            if default_profile_id is not None
            else None
        )
        default_tenant = (
            _required_text(default_tenant_id, "default_tenant_id")
            if default_tenant_id is not None
            else None
        )
        tombstone_ids = {_required_text(value, "tombstone id") for value in tombstones}
        aliases = aliases or {}
        if not isinstance(aliases, Mapping):
            raise ScopedIndexIntegrityError("aliases must be keyed by evidence id")
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source in evidence_units:
            if not isinstance(source, Mapping):
                raise ScopedIndexIntegrityError("evidence units must be mappings")
            unit_aliases = aliases.get(str(source.get("id", "")), {})
            if not isinstance(unit_aliases, Mapping):
                raise ScopedIndexIntegrityError(
                    "per-evidence aliases must be mappings"
                )
            unit = _normalize_unit(
                source,
                unit_aliases,
                default_profile_id=default_profile,
                default_tenant_id=default_tenant,
            )
            if unit["id"] in seen:
                raise ScopedIndexIntegrityError(f"duplicate evidence id: {unit['id']}")
            seen.add(unit["id"])
            if (
                unit["id"] in tombstone_ids
                or unit["tombstone"]
                or unit["status"].lower() in {"deleted", "tombstone"}
            ):
                continue
            normalized.append(unit)

        normalized.sort(key=lambda item: item["id"])
        active_ids = {unit["id"] for unit in normalized}
        normalized_edges = _normalize_edges(edges, active_ids)
        embedding_meta = self._embedding_metadata()
        vectors: dict[str, bytes] = {}
        if self.embedding_adapter is not None and normalized:
            texts = [_embedding_text(unit) for unit in normalized]
            embedded = self._validated_embeddings(texts, embedding_meta["dimension"])
            vectors = {
                unit["id"]: _pack_vector(vector)
                for unit, vector in zip(normalized, embedded, strict=True)
            }

        digest_payload = {
            "units": normalized,
            "edges": normalized_edges,
            "tombstones": sorted(tombstone_ids),
            "embedding": embedding_meta,
        }
        rebuild_digest = _sha256_json(digest_payload)
        metadata = {
            "schema_version": str(SCHEMA_VERSION),
            "evidence_count": str(len(normalized)),
            "rebuild_digest": rebuild_digest,
            "content_digest": _content_digest(
                normalized,
                normalized_edges,
                vectors,
            ),
            "embedding_enabled": "1" if self.embedding_adapter is not None else "0",
            "embedding_model": embedding_meta["model"],
            "embedding_version": embedding_meta["version"],
            "embedding_dimension": str(embedding_meta["dimension"]),
            "embedding_digest": embedding_meta["digest"],
        }
        metadata["metadata_digest"] = _metadata_digest(metadata)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.db_path.with_name(f".{self.db_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            self._write_index(temporary, normalized, normalized_edges, vectors, metadata)
            os.replace(temporary, self.db_path)
        finally:
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    # Cleanup must not replace the primary build/publish error.
                    pass

        return {
            "indexed_units": len(normalized),
            "indexed_edges": len(normalized_edges),
            "tombstones_honored": len(tombstone_ids),
            "rebuild_digest": rebuild_digest,
            "embedding": dict(embedding_meta),
        }

    def search(
        self,
        query: str,
        *,
        profile_id: str,
        tenant_id: str,
        evidence_cutoff: str,
        workstreams: Sequence[str] = (),
        history: bool = False,
        allowed_privacy: Sequence[str] = ("public", "internal", "private"),
        allowed_visibility: Sequence[str] = ("public", "tenant", "private"),
        allow_unknown_workstream: bool = True,
        unknown_workstream_limit: int = 5,
        multi_workstream_limit: int = 20,
        neighbor_types: Sequence[str] = (),
        neighbor_limit: int = 3,
        pool_limit: int = DEFAULT_POOL_LIMIT,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return a bounded union/RRF result set from already-derived units."""

        query = _required_text(query, "query")
        profile_id = _required_text(profile_id, "profile_id")
        tenant_id = _required_text(tenant_id, "tenant_id")
        cutoff = _utc_iso(evidence_cutoff, "evidence_cutoff")
        requested_workstreams = _string_list(workstreams, "workstreams", maximum=8)
        privacy = _string_list(allowed_privacy, "allowed_privacy", maximum=16)
        visibility = _string_list(allowed_visibility, "allowed_visibility", maximum=16)
        relation_types = _string_list(neighbor_types, "neighbor_types", maximum=16)
        if not privacy or not visibility:
            return []
        if not 0 <= int(unknown_workstream_limit) <= 20:
            raise ValueError("unknown_workstream_limit must be between 0 and 20")
        if not 0 <= int(multi_workstream_limit) <= 50:
            raise ValueError("multi_workstream_limit must be between 0 and 50")
        if not 0 <= int(neighbor_limit) <= 10:
            raise ValueError("neighbor_limit must be between 0 and 10")
        if not 1 <= int(pool_limit) <= 200:
            raise ValueError("pool_limit must be between 1 and 200")
        result_limit = min(max(1, int(limit)), MAX_RESULTS)

        with closing(self._read_connection()) as conn:
            metadata = self._validated_metadata(conn)
            eligible = self._eligible_rows(
                conn,
                profile_id=profile_id,
                tenant_id=tenant_id,
                cutoff=cutoff,
                workstreams=requested_workstreams,
                history=bool(history),
                allowed_privacy=privacy,
                allowed_visibility=visibility,
                allow_unknown=bool(allow_unknown_workstream),
                unknown_limit=int(unknown_workstream_limit),
                multi_limit=int(multi_workstream_limit),
            )
            if not eligible:
                return []
            if len(eligible) > MAX_ELIGIBLE_ROWS:
                raise ScopedIndexIntegrityError(
                    "eligible evidence exceeds the scoped retrieval bound"
                )
            by_id = {row["id"]: row for row in eligible}
            scores: dict[str, dict[str, Any]] = {
                unit_id: {
                    "rrf": 0.0,
                    "bm25": None,
                    "cosine": None,
                    "alias": None,
                    "pools": set(),
                    "neighbor": [],
                }
                for unit_id in by_id
            }

            fts_ranked = self._fts_pool(conn, query, by_id, int(pool_limit))
            self._add_rrf_pool(scores, "fts", fts_ranked)
            for unit_id, value in fts_ranked:
                scores[unit_id]["bm25"] = value

            alias_ranked = _alias_pool(query, eligible, int(pool_limit))
            self._add_rrf_pool(scores, "alias", alias_ranked)
            for unit_id, value in alias_ranked:
                scores[unit_id]["alias"] = value

            dense_ranked: list[tuple[str, float]] = []
            if metadata["embedding_enabled"] == "1":
                if self.embedding_adapter is None:
                    raise ScopedIndexIntegrityError(
                        "embedding index requires its pinned local adapter"
                    )
                configured_embedding = self._embedding_metadata()
                stored_embedding = {
                    "model": metadata["embedding_model"],
                    "version": metadata["embedding_version"],
                    "dimension": int(metadata["embedding_dimension"]),
                    "digest": metadata["embedding_digest"],
                }
                if configured_embedding != stored_embedding:
                    raise ScopedIndexIntegrityError(
                        "embedding adapter identity does not match the rebuilt index"
                    )
                if len(eligible) > self.dense_scan_limit:
                    raise ScopedIndexIntegrityError(
                        "eligible embedding scan exceeds configured safety bound"
                    )
                dense_ranked = self._dense_pool(
                    conn, query, by_id, int(metadata["embedding_dimension"]), int(pool_limit)
                )
                self._add_rrf_pool(scores, "dense", dense_ranked)
                for unit_id, value in dense_ranked:
                    scores[unit_id]["cosine"] = value

            seeds = sorted(
                (
                    (unit_id, data["rrf"])
                    for unit_id, data in scores.items()
                    if data["pools"]
                ),
                key=lambda pair: (-pair[1], pair[0]),
            )
            self._add_neighbors(
                conn,
                scores,
                by_id,
                seeds,
                relation_types=relation_types,
                neighbor_limit=int(neighbor_limit),
            )

        ranked_pool = sorted(
            (unit_id for unit_id, data in scores.items() if data["pools"]),
            key=lambda unit_id: (
                -_scope_adjusted_score(scores[unit_id]["rrf"], by_id[unit_id]["scope_match"]),
                -_timestamp_number(by_id[unit_id]["evidence_at"]),
                unit_id,
            ),
        )
        ranked_ids: list[str] = []
        unknown_count = 0
        multi_count = 0
        for unit_id in ranked_pool:
            scope = str(by_id[unit_id]["scope_match"])
            if scope == "unknown":
                if unknown_count >= int(unknown_workstream_limit):
                    continue
                unknown_count += 1
            elif scope == "multi":
                if multi_count >= int(multi_workstream_limit):
                    continue
                multi_count += 1
            ranked_ids.append(unit_id)
            if len(ranked_ids) >= result_limit:
                break
        return [
            _result_mapping(by_id[unit_id], scores[unit_id])
            for unit_id in ranked_ids
        ]

    def _embedding_metadata(self) -> dict[str, Any]:
        if self.embedding_adapter is None:
            return {"model": "", "version": "", "dimension": 0, "digest": ""}
        model = _required_text(
            getattr(self.embedding_adapter, "model", None)
            or getattr(self.embedding_adapter, "model_id", None),
            "embedding model",
        )
        version = _required_text(
            getattr(self.embedding_adapter, "version", None), "embedding version"
        )
        dimension = getattr(self.embedding_adapter, "dimension", None)
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or not 1 <= dimension <= MAX_EMBEDDING_DIMENSION
        ):
            raise ScopedIndexIntegrityError(
                f"embedding dimension must be between 1 and {MAX_EMBEDDING_DIMENSION}"
            )
        artifact_digest = str(
            getattr(self.embedding_adapter, "artifact_digest", "") or ""
        )
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact_digest):
            raise ScopedIndexIntegrityError(
                "embedding adapter requires a pinned artifact digest"
            )
        return {
            "model": model,
            "version": version,
            "dimension": dimension,
            "digest": artifact_digest.removeprefix("sha256:"),
        }

    def _validated_embeddings(
        self, texts: Sequence[str], dimension: int
    ) -> list[list[float]]:
        assert self.embedding_adapter is not None
        try:
            output = self.embedding_adapter.embed(texts)
        except Exception as exc:
            raise ScopedIndexIntegrityError("local embedding adapter failed") from exc
        if not isinstance(output, Sequence) or len(output) != len(texts):
            raise ScopedIndexIntegrityError("embedding adapter returned the wrong vector count")
        return [_validate_vector(vector, dimension) for vector in output]

    @staticmethod
    def _write_index(
        path: Path,
        units: Sequence[Mapping[str, Any]],
        edges: Sequence[Mapping[str, str]],
        vectors: Mapping[str, bytes],
        metadata: Mapping[str, str],
    ) -> None:
        # sqlite3.Connection.__exit__ commits or rolls back but does not close.
        # ``closing`` exits last here, so Windows can atomically replace the
        # completed file immediately after this method returns.
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.executescript(
                """
                PRAGMA journal_mode=DELETE;
                PRAGMA synchronous=FULL;
                CREATE TABLE metadata (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE evidence (
                  id TEXT PRIMARY KEY,
                  profile_id TEXT NOT NULL,
                  tenant_id TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  title TEXT NOT NULL,
                  text TEXT NOT NULL,
                  workstreams TEXT NOT NULL,
                  evidence_at TEXT NOT NULL,
                  status TEXT NOT NULL,
                  privacy_level TEXT NOT NULL,
                  visibility TEXT NOT NULL,
                  source_refs TEXT NOT NULL,
                  entity_aliases TEXT NOT NULL,
                  artifact_aliases TEXT NOT NULL
                ) WITHOUT ROWID;
                CREATE INDEX evidence_scope_idx
                  ON evidence(profile_id, tenant_id, evidence_at);
                CREATE INDEX evidence_policy_idx
                  ON evidence(privacy_level, visibility, status);
                CREATE VIRTUAL TABLE evidence_fts USING fts5(
                  id UNINDEXED,
                  title,
                  text,
                  entity_aliases,
                  artifact_aliases,
                  workstreams
                );
                CREATE TABLE embeddings (
                  unit_id TEXT PRIMARY KEY REFERENCES evidence(id),
                  vector BLOB NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE edges (
                  source_id TEXT NOT NULL REFERENCES evidence(id),
                  target_id TEXT NOT NULL REFERENCES evidence(id),
                  relation_type TEXT NOT NULL,
                  PRIMARY KEY(source_id, target_id, relation_type)
                ) WITHOUT ROWID;
                CREATE INDEX edges_target_idx
                  ON edges(target_id, relation_type, source_id);
                """
            )
            conn.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)", sorted(metadata.items())
            )
            for unit in units:
                serialized = (
                    unit["id"],
                    unit["profile_id"],
                    unit["tenant_id"],
                    unit["kind"],
                    unit["title"],
                    unit["text"],
                    _json(unit["workstreams"]),
                    unit["evidence_at"],
                    unit["status"],
                    unit["privacy_level"],
                    unit["visibility"],
                    _json(unit["source_refs"]),
                    _json(unit["entity_aliases"]),
                    _json(unit["artifact_aliases"]),
                )
                conn.execute(
                    """
                    INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    serialized,
                )
                conn.execute(
                    "INSERT INTO evidence_fts VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        unit["id"],
                        unit["title"],
                        unit["text"],
                        " ".join(unit["entity_aliases"]),
                        " ".join(unit["artifact_aliases"]),
                        " ".join(unit["workstreams"]),
                    ),
                )
            conn.executemany(
                "INSERT INTO embeddings(unit_id, vector) VALUES (?, ?)",
                sorted(vectors.items()),
            )
            conn.executemany(
                "INSERT INTO edges(source_id, target_id, relation_type) VALUES (?, ?, ?)",
                (
                    (edge["source_id"], edge["target_id"], edge["type"])
                    for edge in edges
                ),
            )
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ScopedIndexIntegrityError("new derived index failed integrity check")
        os.chmod(path, 0o600)

    def _read_connection(self) -> sqlite3.Connection:
        if not self.db_path.is_file():
            raise ScopedIndexIntegrityError("derived index does not exist")
        try:
            conn = sqlite3.connect(f"{self.db_path.as_uri()}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as exc:
            raise ScopedIndexIntegrityError("derived index could not be opened read-only") from exc

    @staticmethod
    def _validated_metadata(conn: sqlite3.Connection) -> dict[str, str]:
        try:
            if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ScopedIndexIntegrityError("derived index integrity check failed")
            metadata = {
                str(row["key"]): str(row["value"])
                for row in conn.execute("SELECT key, value FROM metadata")
            }
        except sqlite3.Error as exc:
            raise ScopedIndexIntegrityError("derived index metadata is malformed") from exc
        required = {
            "schema_version",
            "evidence_count",
            "rebuild_digest",
            "content_digest",
            "embedding_enabled",
            "embedding_model",
            "embedding_version",
            "embedding_dimension",
            "embedding_digest",
            "metadata_digest",
        }
        if not required <= metadata.keys():
            raise ScopedIndexIntegrityError("derived index metadata is incomplete")
        supplied_digest = metadata.pop("metadata_digest")
        if (
            metadata["schema_version"] != str(SCHEMA_VERSION)
            or supplied_digest != _metadata_digest(metadata)
        ):
            raise ScopedIndexIntegrityError("derived index metadata failed integrity validation")
        try:
            expected_count = int(metadata["evidence_count"])
            dimension = int(metadata["embedding_dimension"])
        except ValueError as exc:
            raise ScopedIndexIntegrityError("derived index metadata has invalid numbers") from exc
        actual_count = int(conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0])
        embedding_count = int(conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
        enabled = metadata["embedding_enabled"]
        if expected_count != actual_count or enabled not in {"0", "1"}:
            raise ScopedIndexIntegrityError("derived index metadata does not match contents")
        if (
            enabled == "1"
            and (
                not 1 <= dimension <= MAX_EMBEDDING_DIMENSION
                or embedding_count != actual_count
            )
        ) or (
            enabled == "0" and (dimension != 0 or embedding_count != 0)
        ):
            raise ScopedIndexIntegrityError("embedding index metadata does not match contents")
        if metadata["content_digest"] != _database_content_digest(conn):
            raise ScopedIndexIntegrityError(
                "derived index content failed its integrity manifest"
            )
        metadata["metadata_digest"] = supplied_digest
        return metadata

    def _eligible_rows(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        tenant_id: str,
        cutoff: str,
        workstreams: Sequence[str],
        history: bool,
        allowed_privacy: Sequence[str],
        allowed_visibility: Sequence[str],
        allow_unknown: bool,
        unknown_limit: int,
        multi_limit: int,
    ) -> list[dict[str, Any]]:
        clauses = [
            "profile_id = ?",
            "tenant_id = ?",
            "julianday(evidence_at) <= julianday(?)",
            f"privacy_level IN ({','.join('?' for _ in allowed_privacy)})",
            f"visibility IN ({','.join('?' for _ in allowed_visibility)})",
        ]
        parameters: list[Any] = [
            profile_id,
            tenant_id,
            cutoff,
            *allowed_privacy,
            *allowed_visibility,
        ]
        invalid_statuses = (
            _HISTORY_INVALID_STATUSES if history else _CURRENT_INVALID_STATUSES
        )
        clauses.append(
            f"lower(status) NOT IN ({','.join('?' for _ in invalid_statuses)})"
        )
        parameters.extend(sorted(invalid_statuses))
        rows = conn.execute(
            f"SELECT * FROM evidence WHERE {' AND '.join(clauses)} "
            "ORDER BY evidence_at DESC, id ASC",
            parameters,
        ).fetchall()
        decoded = [_decode_row(row) for row in rows]
        if not workstreams:
            selected = decoded
        else:
            requested = set(workstreams)
            exact = [
                row
                for row in decoded
                if _workstream_scope_match(row["workstreams"], requested) == "exact"
            ]
            multi = [
                row
                for row in decoded
                if _workstream_scope_match(row["workstreams"], requested) == "multi"
            ]
            unknown = [row for row in decoded if not row["workstreams"]]
            selected = exact + multi + (
                unknown if allow_unknown and unknown_limit else []
            )
        for row in selected:
            if not workstreams:
                row["scope_match"] = "unscoped"
            elif not row["workstreams"]:
                row["scope_match"] = "unknown"
            else:
                row["scope_match"] = _workstream_scope_match(
                    row["workstreams"],
                    set(workstreams),
                )
        return selected

    @staticmethod
    def _fts_pool(
        conn: sqlite3.Connection,
        query: str,
        eligible: Mapping[str, Mapping[str, Any]],
        limit: int,
    ) -> list[tuple[str, float]]:
        tokens = _tokens(query)
        if not tokens:
            return []
        expression = " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
        try:
            conn.execute(
                "CREATE TEMP TABLE IF NOT EXISTS scoped_eligible_ids "
                "(id TEXT PRIMARY KEY) WITHOUT ROWID"
            )
            conn.execute("DELETE FROM scoped_eligible_ids")
            conn.executemany(
                "INSERT INTO scoped_eligible_ids(id) VALUES (?)",
                ((unit_id,) for unit_id in eligible),
            )
            rows = conn.execute(
                f"""
                SELECT evidence_fts.id,
                       bm25(evidence_fts, 0.0, 4.0, 1.0, 3.0, 3.0, 1.5) AS rank
                FROM evidence_fts
                JOIN scoped_eligible_ids
                  ON scoped_eligible_ids.id = evidence_fts.id
                WHERE evidence_fts MATCH ?
                ORDER BY rank ASC, evidence_fts.id ASC
                LIMIT ?
                """,
                (expression, limit),
            ).fetchall()
        except sqlite3.Error as exc:
            raise ScopedIndexIntegrityError("FTS index query failed closed") from exc
        return [(str(row["id"]), float(row["rank"])) for row in rows]

    def _dense_pool(
        self,
        conn: sqlite3.Connection,
        query: str,
        eligible: Mapping[str, Mapping[str, Any]],
        dimension: int,
        limit: int,
    ) -> list[tuple[str, float]]:
        query_vector = self._validated_embeddings([query], dimension)[0]
        if _norm(query_vector) == 0.0:
            return []
        placeholders = ",".join("?" for _ in eligible)
        rows = conn.execute(
            f"SELECT unit_id, vector FROM embeddings WHERE unit_id IN ({placeholders})",
            tuple(eligible),
        ).fetchall()
        ranked: list[tuple[str, float]] = []
        for row in rows:
            vector = _unpack_vector(row["vector"], dimension)
            score = _cosine(query_vector, vector)
            if math.isfinite(score):
                ranked.append((str(row["unit_id"]), score))
        return sorted(ranked, key=lambda pair: (-pair[1], pair[0]))[:limit]

    @staticmethod
    def _add_rrf_pool(
        scores: dict[str, dict[str, Any]],
        pool_name: str,
        ranked: Sequence[tuple[str, float]],
    ) -> None:
        for rank, (unit_id, _) in enumerate(ranked, start=1):
            if unit_id not in scores:
                raise ScopedIndexIntegrityError("retrieval pool escaped hard scope")
            scores[unit_id]["rrf"] += 1.0 / (RRF_K + rank)
            scores[unit_id]["pools"].add(pool_name)

    @staticmethod
    def _add_neighbors(
        conn: sqlite3.Connection,
        scores: dict[str, dict[str, Any]],
        eligible: Mapping[str, Mapping[str, Any]],
        seeds: Sequence[tuple[str, float]],
        *,
        relation_types: Sequence[str],
        neighbor_limit: int,
    ) -> None:
        if not seeds or not neighbor_limit:
            return
        allowed = set(relation_types)
        added = 0
        for source_id, _ in seeds:
            rows = conn.execute(
                """
                SELECT target_id, relation_type FROM edges
                WHERE source_id = ?
                ORDER BY relation_type, target_id
                """,
                (source_id,),
            ).fetchall()
            for row in rows:
                target_id = str(row["target_id"])
                relation_type = str(row["relation_type"])
                if allowed and relation_type not in allowed:
                    continue
                if target_id not in eligible or target_id == source_id:
                    continue
                diagnostic = {"from_id": source_id, "relation_type": relation_type}
                if diagnostic in scores[target_id]["neighbor"]:
                    continue
                scores[target_id]["neighbor"].append(diagnostic)
                if "neighbor" not in scores[target_id]["pools"]:
                    scores[target_id]["rrf"] += 1.0 / (RRF_K + 1)
                    scores[target_id]["pools"].add("neighbor")
                    added += 1
                if added >= neighbor_limit:
                    return


def _normalize_unit(
    source: Mapping[str, Any],
    external_aliases: Mapping[str, Iterable[str]],
    *,
    default_profile_id: str | None,
    default_tenant_id: str | None,
) -> dict[str, Any]:
    unit_id = _required_text(source.get("id"), "evidence id")
    profile = _required_text(
        source.get("profile_id", source.get("profile", default_profile_id)),
        "profile_id",
    )
    tenant = _required_text(
        source.get("tenant_id", source.get("tenant", default_tenant_id)),
        "tenant_id",
    )
    workstreams_value: Any = source.get("workstreams")
    if workstreams_value is None:
        project_ids = _string_list(
            source.get("project_ids") or [], "project_ids", maximum=16
        )
        workstream_ids = _string_list(
            source.get("workstream_ids") or [], "workstream_ids", maximum=16
        )
        if project_ids or workstream_ids:
            workstreams_value = [*project_ids, *workstream_ids]
    if workstreams_value is None:
        single = source.get("workstream_id", source.get("project_id"))
        workstreams_value = [single] if single else []
    workstreams = _string_list(workstreams_value, "workstreams", maximum=16)
    evidence_at = _utc_iso(
        source.get("evidence_at")
        or source.get("observed_at")
        or source.get("occurred_at")
        or source.get("created_at"),
        "evidence_at",
    )
    text = _first_text(
        source,
        ("text", "evidence_text", "body", "summary", "claim", "value"),
    )
    title = str(source.get("title") or source.get("subject") or "").strip()
    if not text and not title:
        raise ScopedIndexIntegrityError(f"evidence {unit_id} has no searchable text")
    source_refs = _normalize_source_refs(source.get("source_refs"), evidence_at)
    intrinsic_entities = source.get("entity_aliases", source.get("entities", []))
    intrinsic_artifacts = source.get("artifact_aliases", source.get("artifacts", []))
    entity_aliases = _merge_strings(
        intrinsic_entities, external_aliases.get("entity", external_aliases.get("entities", []))
    )
    artifact_aliases = _merge_strings(
        intrinsic_artifacts,
        external_aliases.get("artifact", external_aliases.get("artifacts", [])),
    )
    status = _required_text(source.get("status", "current"), "status")
    return {
        "id": unit_id,
        "profile_id": profile,
        "tenant_id": tenant,
        "kind": str(source.get("kind") or source.get("type") or "evidence").strip(),
        "title": title,
        "text": text,
        "workstreams": workstreams,
        "evidence_at": evidence_at,
        "status": status,
        "privacy_level": _required_text(
            source.get("privacy_level", source.get("privacy", "private")),
            "privacy_level",
        ),
        "visibility": _required_text(source.get("visibility", "private"), "visibility"),
        "source_refs": source_refs,
        "entity_aliases": entity_aliases,
        "artifact_aliases": artifact_aliases,
        "tombstone": bool(source.get("tombstone") or source.get("deleted")),
    }


def _normalize_source_refs(value: Any, evidence_at: str) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ScopedIndexIntegrityError("evidence requires at least one source reference")
    refs: list[dict[str, Any]] = []
    for raw in value:
        if isinstance(raw, str):
            refs.append({"id": _required_text(raw, "source ref id"), "observed_at": evidence_at})
            continue
        if not isinstance(raw, Mapping):
            raise ScopedIndexIntegrityError("source references must be strings or mappings")
        ref = dict(raw)
        ref["id"] = _required_text(ref.get("id"), "source ref id")
        timestamp = ref.get("observed_at") or ref.get("evidence_at") or evidence_at
        ref["observed_at"] = _utc_iso(timestamp, "source observed_at")
        refs.append(ref)
    return refs


def _normalize_edges(
    edges: Iterable[Mapping[str, Any]], active_ids: set[str]
) -> list[dict[str, str]]:
    normalized: set[tuple[str, str, str]] = set()
    for edge in edges:
        if not isinstance(edge, Mapping):
            raise ScopedIndexIntegrityError("edges must be mappings")
        source = _required_text(edge.get("source_id"), "edge source_id")
        target = _required_text(edge.get("target_id"), "edge target_id")
        relation = _required_text(
            edge.get("type", edge.get("relation_type")), "edge type"
        )
        if source in active_ids and target in active_ids and source != target:
            normalized.add((source, target, relation))
    return [
        {"source_id": source, "target_id": target, "type": relation}
        for source, target, relation in sorted(normalized)
    ]


def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        return {
            "id": str(row["id"]),
            "profile_id": str(row["profile_id"]),
            "tenant_id": str(row["tenant_id"]),
            "kind": str(row["kind"]),
            "title": str(row["title"]),
            "text": str(row["text"]),
            "workstreams": json.loads(row["workstreams"]),
            "evidence_at": str(row["evidence_at"]),
            "status": str(row["status"]),
            "privacy_level": str(row["privacy_level"]),
            "visibility": str(row["visibility"]),
            "source_refs": json.loads(row["source_refs"]),
            "entity_aliases": json.loads(row["entity_aliases"]),
            "artifact_aliases": json.loads(row["artifact_aliases"]),
        }
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ScopedIndexIntegrityError("derived evidence row is malformed") from exc


def _alias_pool(
    query: str, eligible: Sequence[Mapping[str, Any]], limit: int
) -> list[tuple[str, float]]:
    query_tokens = set(_tokens(query))
    query_lower = query.casefold()
    ranked: list[tuple[str, float]] = []
    for row in eligible:
        matches = 0.0
        for alias in [*row["entity_aliases"], *row["artifact_aliases"]]:
            alias_tokens = set(_tokens(alias))
            if alias.casefold() in query_lower:
                matches += 2.0
            elif alias_tokens and alias_tokens <= query_tokens:
                matches += 1.0
        if matches:
            ranked.append((str(row["id"]), matches))
    return sorted(ranked, key=lambda pair: (-pair[1], pair[0]))[:limit]


def _result_mapping(row: Mapping[str, Any], score: Mapping[str, Any]) -> dict[str, Any]:
    adjusted = _scope_adjusted_score(float(score["rrf"]), str(row["scope_match"]))
    return {
        "id": row["id"],
        "kind": row["kind"],
        "title": row["title"],
        "text": row["text"],
        "workstreams": list(row["workstreams"]),
        "evidence_at": row["evidence_at"],
        "status": row["status"],
        "privacy_level": row["privacy_level"],
        "visibility": row["visibility"],
        "source_refs": [dict(ref) for ref in row["source_refs"]],
        "retrieval_pools": sorted(score["pools"]),
        "score": adjusted,
        "score_diagnostics": {
            "rrf_score": float(score["rrf"]),
            "scope_adjusted_score": adjusted,
            "bm25_rank": score["bm25"],
            "cosine_score": score["cosine"],
            "alias_score": score["alias"],
        },
        "scope_diagnostics": {
            "profile_id": row["profile_id"],
            "tenant_id": row["tenant_id"],
            "workstream_match": row["scope_match"],
        },
        "neighbor_diagnostics": list(score["neighbor"]),
    }


def _scope_adjusted_score(score: float, scope: str) -> float:
    return score * {"exact": 1.0, "multi": 0.98, "unknown": 0.75, "unscoped": 1.0}[scope]


def _workstream_scope_match(
    candidate_values: Sequence[str],
    requested: set[str],
) -> str | None:
    candidate = set(candidate_values)
    if not candidate or not requested.intersection(candidate):
        return None
    requested_dimensions = {
        value.split(":", 1)[0]
        for value in requested
        if ":" in value
    }
    meaningful_extras = {
        value
        for value in candidate - requested
        if ":" not in value
        or value.split(":", 1)[0] in requested_dimensions
    }
    return "multi" if meaningful_extras else "exact"


def _embedding_text(unit: Mapping[str, Any]) -> str:
    return "\n".join(
        part
        for part in (
            unit["title"],
            unit["text"],
            " ".join(unit["entity_aliases"]),
            " ".join(unit["artifact_aliases"]),
        )
        if part
    )


def _validate_vector(value: Any, dimension: int) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ScopedIndexIntegrityError("embedding vector must be a sequence")
    if len(value) != dimension:
        raise ScopedIndexIntegrityError("embedding vector has the wrong fixed dimension")
    try:
        vector = [float(component) for component in value]
    except (TypeError, ValueError) as exc:
        raise ScopedIndexIntegrityError("embedding vector contains a non-number") from exc
    if not all(math.isfinite(component) for component in vector):
        raise ScopedIndexIntegrityError("embedding vector components must be finite")
    return vector


def _pack_vector(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}d", *vector)


def _unpack_vector(blob: Any, dimension: int) -> list[float]:
    if not isinstance(blob, bytes) or len(blob) != dimension * 8:
        raise ScopedIndexIntegrityError("stored embedding has an invalid dimension")
    vector = list(struct.unpack(f"<{dimension}d", blob))
    if not all(math.isfinite(value) for value in vector):
        raise ScopedIndexIntegrityError("stored embedding contains non-finite values")
    return vector


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    denominator = _norm(left) * _norm(right)
    if denominator == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _tokens(text: str) -> list[str]:
    return list(dict.fromkeys(token.casefold() for token in _TOKEN_RE.findall(text)))[:32]


def _merge_strings(*values: Any) -> list[str]:
    merged: list[str] = []
    for value in values:
        merged.extend(_string_list(value or [], "aliases", maximum=64))
    return list(dict.fromkeys(merged))


def _string_list(value: Any, name: str, *, maximum: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Sequence[Any] = [value]
    elif isinstance(value, Sequence):
        values = value
    else:
        raise ScopedIndexIntegrityError(f"{name} must be a sequence of strings")
    result = list(dict.fromkeys(_required_text(item, name) for item in values))
    if len(result) > maximum:
        raise ScopedIndexIntegrityError(f"{name} exceeds its safety bound")
    return result


def _first_text(source: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = source.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _required_text(value: Any, name: str) -> str:
    if value is None or not str(value).strip():
        raise ScopedIndexIntegrityError(f"{name} is required")
    return str(value).strip()


def _utc_iso(value: Any, name: str) -> str:
    text = _required_text(value, name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ScopedIndexIntegrityError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ScopedIndexIntegrityError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp_number(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _metadata_digest(metadata: Mapping[str, str]) -> str:
    return _sha256_json(dict(sorted(metadata.items())))


def _content_digest(
    units: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    vectors: Mapping[str, bytes],
) -> str:
    return _sha256_json(
        {
            "units": [dict(unit) for unit in units],
            "edges": [dict(edge) for edge in edges],
            "vectors": {
                unit_id: hashlib.sha256(bytes(vector)).hexdigest()
                for unit_id, vector in sorted(vectors.items())
            },
        }
    )


def _database_content_digest(conn: sqlite3.Connection) -> str:
    units: list[dict[str, Any]] = []
    expected_fts: dict[str, tuple[str, str, str, str, str]] = {}
    for raw in conn.execute("SELECT * FROM evidence ORDER BY id"):
        unit = _decode_row(raw)
        unit["tombstone"] = False
        units.append(unit)
        expected_fts[str(unit["id"])] = (
            str(unit["title"]),
            str(unit["text"]),
            " ".join(unit["entity_aliases"]),
            " ".join(unit["artifact_aliases"]),
            " ".join(unit["workstreams"]),
        )

    actual_fts_rows = [
        (
            str(row["id"]),
            str(row["title"]),
            str(row["text"]),
            str(row["entity_aliases"]),
            str(row["artifact_aliases"]),
            str(row["workstreams"]),
        )
        for row in conn.execute(
            "SELECT id, title, text, entity_aliases, artifact_aliases, workstreams "
            "FROM evidence_fts"
        )
    ]
    if (
        len(actual_fts_rows) != len(expected_fts)
        or len({row[0] for row in actual_fts_rows}) != len(actual_fts_rows)
    ):
        raise ScopedIndexIntegrityError(
            "derived index FTS rows are missing or duplicated"
        )
    actual_fts = {
        row[0]: row[1:]
        for row in actual_fts_rows
    }
    if actual_fts != expected_fts:
        raise ScopedIndexIntegrityError(
            "derived index FTS content does not match evidence rows"
        )

    edges = [
        {
            "source_id": str(row["source_id"]),
            "target_id": str(row["target_id"]),
            "type": str(row["relation_type"]),
        }
        for row in conn.execute(
            "SELECT source_id, target_id, relation_type FROM edges "
            "ORDER BY source_id, target_id, relation_type"
        )
    ]
    vectors = {
        str(row["unit_id"]): bytes(row["vector"])
        for row in conn.execute(
            "SELECT unit_id, vector FROM embeddings ORDER BY unit_id"
        )
    }
    return _content_digest(units, edges, vectors)
