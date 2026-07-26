"""End-to-end, read-only shadow retrieval pipeline for Memory v2 experiments.

The pipeline is intentionally separate from the live provider.  Enabling it
builds a disposable derived index from an operator-supplied event snapshot and
returns a bounded evidence bundle.  It cannot promote, supersede, or otherwise
mutate canonical memory.
"""

from __future__ import annotations

import json
import math
import queue
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .scoped_retrieval import LocalEmbeddingAdapter, ScopedEvidenceIndex
from .shadow_intent import ShadowMemoryNeedRouter
from .shadow_reranker import ShadowRerankerConfig, ShadowUtilityReranker
from .workstream_evidence import (
    EvidenceNode,
    StructuredExtractor,
    WorkstreamEvidenceBuilder,
    WorkstreamResolver,
)


SCHEMA_VERSION = "memory-v2-shadow-retrieval/v1"
MAX_RAW_EVENTS = 5_000
MAX_DERIVED_NODES = 10_000
MAX_EVENT_UTF8_BYTES = 256_000
MAX_TOTAL_EVENT_UTF8_BYTES = 8_000_000
MAX_CONTEXT_UTF8_BYTES = 64_000
MAX_GAP_DAYS = 36_500.0
MAX_QUERY_SCOPE_IDS = 8
_QUERY_SCOPE_KEYS = {
    "project_id",
    "project",
    "project_name",
    "workstream_id",
    "workstream",
    "workstream_name",
    "repo_path",
    "repository",
    "cwd",
    "workdir",
    "artifact_refs",
    "artifacts",
    "source_paths",
    "channel_id",
    "channel",
    "thread_id",
    "thread",
}
_EVIDENCE_STATUSES = {
    "active",
    "archived_only",
    "closed",
    "current",
    "historical",
    "rejected",
    "resolved",
    "retracted",
    "stale",
    "superseded",
}


class ShadowPipelineError(ValueError):
    """Raised when a shadow pipeline request fails closed."""


@dataclass(frozen=True)
class ShadowPipelineConfig:
    """Configuration for a disabled-by-default shadow run."""

    enabled: bool = False
    max_raw_events: int = MAX_RAW_EVENTS
    max_derived_nodes: int = MAX_DERIVED_NODES
    max_event_utf8_bytes: int = MAX_EVENT_UTF8_BYTES
    max_total_event_utf8_bytes: int = MAX_TOTAL_EVENT_UTF8_BYTES
    dense_scan_limit: int = 900
    structured_extractor_artifact_digest: str = ""
    structured_extractor_timeout_ms: int = 500
    embedding_timeout_ms: int = 2_000
    allowed_privacy: tuple[str, ...] = ("private",)
    allowed_visibility: tuple[str, ...] = ("private",)
    reranker: ShadowRerankerConfig = field(
        default_factory=lambda: ShadowRerankerConfig(enabled=False)
    )

    def __post_init__(self) -> None:
        if not 1 <= int(self.max_raw_events) <= MAX_RAW_EVENTS:
            raise ShadowPipelineError(
                f"max_raw_events must be between 1 and {MAX_RAW_EVENTS}"
            )
        if not 1 <= int(self.max_derived_nodes) <= MAX_DERIVED_NODES:
            raise ShadowPipelineError(
                f"max_derived_nodes must be between 1 and {MAX_DERIVED_NODES}"
            )
        if not 1 <= int(self.max_event_utf8_bytes) <= MAX_EVENT_UTF8_BYTES:
            raise ShadowPipelineError(
                f"max_event_utf8_bytes must be between 1 and {MAX_EVENT_UTF8_BYTES}"
            )
        if not (
            int(self.max_event_utf8_bytes)
            <= int(self.max_total_event_utf8_bytes)
            <= MAX_TOTAL_EVENT_UTF8_BYTES
        ):
            raise ShadowPipelineError(
                "max_total_event_utf8_bytes must include one event and stay bounded"
            )
        if not 1 <= int(self.dense_scan_limit) <= 900:
            raise ShadowPipelineError("dense_scan_limit must be between 1 and 900")
        if not 1 <= int(self.structured_extractor_timeout_ms) <= 60_000:
            raise ShadowPipelineError(
                "structured_extractor_timeout_ms must be within 1..60000"
            )
        if not 1 <= int(self.embedding_timeout_ms) <= 60_000:
            raise ShadowPipelineError("embedding_timeout_ms must be within 1..60000")
        object.__setattr__(
            self,
            "allowed_privacy",
            _normalized_policy_scope(self.allowed_privacy, "privacy"),
        )
        object.__setattr__(
            self,
            "allowed_visibility",
            _normalized_policy_scope(self.allowed_visibility, "visibility"),
        )
        if self.enabled != self.reranker.enabled:
            raise ShadowPipelineError(
                "pipeline and reranker enabled flags must agree"
            )


class ShadowRetrievalPipeline:
    """Compose intent routing, evidence derivation, retrieval, and abstention."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        config: ShadowPipelineConfig | None = None,
        structured_extractor: StructuredExtractor | None = None,
        embedding_adapter: LocalEmbeddingAdapter | None = None,
        reranker_adapter: (
            Callable[[dict[str, Any]], Mapping[str, Any]] | None
        ) = None,
        scratch_root: str | Path | None = None,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.scratch_root = (
            Path(scratch_root).expanduser().resolve()
            if scratch_root is not None
            else None
        )
        self.config = config or ShadowPipelineConfig()
        self.structured_extractor = structured_extractor
        self.embedding_adapter = embedding_adapter
        self.reranker_adapter = reranker_adapter
        self._bounded_embedding_adapter = (
            _BoundedEmbeddingAdapter(
                embedding_adapter,
                timeout_ms=int(self.config.embedding_timeout_ms),
            )
            if self.config.enabled and embedding_adapter is not None
            else None
        )
        self._reranker = ShadowUtilityReranker(self.config.reranker)
        self._structured_adapter_lock = threading.Lock()
        self._structured_adapter_circuit_open = False

    def run(
        self,
        *,
        query: str,
        raw_events: Sequence[Mapping[str, Any]],
        profile_id: str,
        tenant_id: str,
        evidence_cutoff: str,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one private shadow query without touching canonical state."""

        if not self.config.enabled:
            return self._empty("disabled")
        if self.scratch_root is None:
            raise ShadowPipelineError(
                "enabled shadow retrieval requires an explicit scratch_root"
            )
        if (
            self.db_path == self.scratch_root
            or not self.db_path.is_relative_to(self.scratch_root)
        ):
            raise ShadowPipelineError("derived index must stay inside scratch_root")
        if self.db_path.exists():
            raise ShadowPipelineError("refusing to overwrite a derived index path")

        clean_query = str(query or "").strip()
        clean_profile = str(profile_id or "").strip()
        clean_tenant = str(tenant_id or "").strip()
        clean_cutoff = str(evidence_cutoff or "").strip()
        if not clean_query or len(clean_query) > 8_000:
            raise ShadowPipelineError("query must be non-empty and bounded")
        if not clean_profile or len(clean_profile) > 200:
            raise ShadowPipelineError("profile_id must be non-empty and bounded")
        if not clean_tenant or len(clean_tenant) > 200:
            raise ShadowPipelineError("tenant_id must be non-empty and bounded")
        if not clean_cutoff:
            raise ShadowPipelineError("evidence_cutoff is required")
        cutoff_epoch = self._timestamp_epoch(clean_cutoff, "evidence_cutoff")
        if not isinstance(raw_events, Sequence) or isinstance(
            raw_events, (str, bytes)
        ):
            raise ShadowPipelineError("raw_events must be a sequence of mappings")
        if len(raw_events) > int(self.config.max_raw_events):
            raise ShadowPipelineError("raw event snapshot exceeds its safety bound")
        if any(not isinstance(event, Mapping) for event in raw_events):
            raise ShadowPipelineError("every raw event must be a mapping")
        event_ids = [str(event.get("id") or "").strip() for event in raw_events]
        if any(not event_id or len(event_id) > 200 for event_id in event_ids):
            raise ShadowPipelineError("raw event ids must be non-empty and bounded")
        if len(set(event_ids)) != len(event_ids):
            raise ShadowPipelineError("raw event ids must be unique")
        parsed_context = self._context(context)

        intent = ShadowMemoryNeedRouter(enabled=True).route(
            clean_query,
            {
                "has_current_context": parsed_context["has_current_context"],
                "gap_days": parsed_context["gap_days"],
                "workstream_ids": parsed_context["workstream_ids"],
            },
        )
        if intent.decision == "none":
            result = self._empty(intent.reason)
            result["enabled"] = True
            result["intent"] = intent.to_dict()
            return result

        eligible_events, policy_exclusions = self._prepare_events(
            raw_events,
            profile_id=clean_profile,
            tenant_id=clean_tenant,
            cutoff_epoch=cutoff_epoch,
        )
        builder = WorkstreamEvidenceBuilder(
            structured_extractor=self._validated_structured_extractor(),
            max_nodes=int(self.config.max_derived_nodes),
        )
        build = builder.build(eligible_events)
        if len(build.nodes) > int(self.config.max_derived_nodes):
            raise ShadowPipelineError("derived evidence exceeds its safety bound")

        query_resolution = WorkstreamResolver(
            max_scope_ids=MAX_QUERY_SCOPE_IDS
        ).resolve(
            self._query_event(
                clean_query,
                clean_cutoff,
                parsed_context,
            )
        )
        query_workstreams = list(
            dict.fromkeys(
                [
                    *query_resolution.project_ids,
                    *query_resolution.workstream_ids,
                ]
            )
        )
        if not query_workstreams:
            query_workstreams = list(parsed_context["workstream_ids"])

        index = ScopedEvidenceIndex(
            self.db_path,
            embedding_adapter=self._bounded_embedding_adapter,
            dense_scan_limit=int(self.config.dense_scan_limit),
        )
        edges = self._corroboration_edges(build.nodes)
        try:
            index_summary = index.rebuild(
                self._index_units(
                    build.nodes,
                    eligible_events,
                    profile_id=clean_profile,
                    tenant_id=clean_tenant,
                ),
                edges=edges,
                default_profile_id=clean_profile,
                default_tenant_id=clean_tenant,
            )
            retrieved = index.search(
                clean_query,
                profile_id=clean_profile,
                tenant_id=clean_tenant,
                evidence_cutoff=clean_cutoff,
                workstreams=query_workstreams,
                history=intent.temporal_mode == "history",
                allowed_privacy=self.config.allowed_privacy,
                allowed_visibility=self.config.allowed_visibility,
                allow_unknown_workstream=intent.allow_unknown_workstream,
                unknown_workstream_limit=min(5, intent.search_limit),
                multi_workstream_limit=min(20, intent.search_limit),
                neighbor_types=("verified_by",),
                neighbor_limit=3,
                pool_limit=max(1, min(200, intent.search_limit * 4)),
                limit=max(1, intent.search_limit),
            )
        finally:
            try:
                self.db_path.unlink(missing_ok=True)
            except OSError as exc:
                raise ShadowPipelineError(
                    "failed to remove the private derived index"
                ) from exc
        nodes_by_id = {node.id: node for node in build.nodes}
        candidates = [
            self._reranker_candidate(
                result,
                nodes_by_id[str(result["id"])],
                clean_profile,
            )
            for result in retrieved
            if str(result.get("id") or "") in nodes_by_id
        ]
        reranked = self._reranker.run(
            clean_query,
            candidates,
            {
                "memory_decision": "needed",
                "profile_id": clean_profile,
                "workstream_ids": query_workstreams[:8],
                "allow_unknown_workstream": intent.allow_unknown_workstream,
                "temporal_mode": intent.temporal_mode,
                "evidence_cutoff": clean_cutoff,
            },
            adapter=self.reranker_adapter,
            citation_sources=self._citation_sources(eligible_events),
        )
        hygiene_counts: dict[str, int] = {}
        for record in build.hygiene:
            hygiene_counts[record.disposition] = (
                hygiene_counts.get(record.disposition, 0) + 1
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": True,
            "shadow_only": True,
            "read_only": True,
            "untrusted_data": True,
            "mutation_authority": "none",
            "intent": intent.to_dict(),
            "corpus": {
                "raw_event_count": len(raw_events),
                "eligible_raw_event_count": len(eligible_events),
                "policy_exclusions": policy_exclusions,
                "derived_node_count": len(build.nodes),
                "hygiene_counts": dict(sorted(hygiene_counts.items())),
                "structured_rejection_count": len(build.structured_rejections),
                "overflow_rejections": list(build.overflow_rejections),
                "rejected_node_count": build.rejected_node_count,
            },
            "query_scope": {
                "status": query_resolution.status,
                "project_ids": list(query_resolution.project_ids),
                "workstream_ids": list(query_resolution.workstream_ids),
                "confidence": query_resolution.confidence,
            },
            "index": index_summary,
            "retrieval": {
                "candidate_count": len(retrieved),
                "candidate_ids": [str(item["id"]) for item in retrieved],
                "diagnostics": [
                    {
                        "id": str(item["id"]),
                        "retrieval_pools": list(item["retrieval_pools"]),
                        "score": float(item["score"]),
                        "scope": dict(item["scope_diagnostics"]),
                        "neighbors": list(item["neighbor_diagnostics"]),
                    }
                    for item in retrieved
                ],
            },
            "result": reranked,
        }

    @staticmethod
    def _context(value: Mapping[str, Any] | None) -> dict[str, Any]:
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise ShadowPipelineError("context must be a mapping")
        if ShadowRetrievalPipeline._utf8_size(value) > MAX_CONTEXT_UTF8_BYTES:
            raise ShadowPipelineError("context exceeds its byte bound")
        allowed_keys = {
            "has_current_context",
            "gap_days",
            "workstream_ids",
            *_QUERY_SCOPE_KEYS,
        }
        if set(value) - allowed_keys:
            raise ShadowPipelineError("context contains unknown fields")
        has_current_context = value.get("has_current_context", False)
        if not isinstance(has_current_context, bool):
            raise ShadowPipelineError("has_current_context must be boolean")
        gap_days = value.get("gap_days", 0.0)
        if isinstance(gap_days, bool):
            raise ShadowPipelineError("gap_days must be numeric")
        try:
            gap = float(gap_days)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ShadowPipelineError("gap_days must be numeric") from exc
        if not math.isfinite(gap) or not 0.0 <= gap <= MAX_GAP_DAYS:
            raise ShadowPipelineError(
                f"gap_days must be finite and within 0..{int(MAX_GAP_DAYS)}"
            )
        raw_workstreams = value.get("workstream_ids") or []
        if not isinstance(raw_workstreams, (list, tuple)):
            raise ShadowPipelineError("workstream_ids must be a sequence")
        workstreams = tuple(
            dict.fromkeys(
                str(item).strip()
                for item in raw_workstreams
                if str(item).strip()
            )
        )
        if len(workstreams) > 8:
            raise ShadowPipelineError("workstream_ids exceeds its safety bound")
        scope = {
            key: value[key]
            for key in _QUERY_SCOPE_KEYS
            if key in value
        }
        return {
            "has_current_context": has_current_context,
            "gap_days": gap,
            "workstream_ids": workstreams,
            "scope": scope,
        }

    def _prepare_events(
        self,
        events: Sequence[Mapping[str, Any]],
        *,
        profile_id: str,
        tenant_id: str,
        cutoff_epoch: float,
    ) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
        exclusions = {
            "future": 0,
            "missing_timestamp": 0,
            "privacy": 0,
            "visibility": 0,
            "retrieval_disabled": 0,
            "tombstone": 0,
        }
        eligible: list[Mapping[str, Any]] = []
        total_size = 0
        for event in events:
            event_size = self._utf8_size(event)
            if event_size > int(self.config.max_event_utf8_bytes):
                raise ShadowPipelineError("raw event exceeds its byte bound")
            total_size += event_size
            if total_size > int(self.config.max_total_event_utf8_bytes):
                raise ShadowPipelineError("raw event snapshot exceeds its byte bound")

            event_profile = str(event.get("profile_id") or "").strip()
            event_tenant = str(event.get("tenant_id") or "").strip()
            if not event_profile or not event_tenant:
                raise ShadowPipelineError(
                    "every raw event requires explicit profile_id and tenant_id"
                )
            if event_profile != profile_id:
                raise ShadowPipelineError("raw event profile does not match the request")
            if event_tenant != tenant_id:
                raise ShadowPipelineError("raw event tenant does not match the request")

            observed = next(
                (
                    str(event.get(field_name) or "").strip()
                    for field_name in ("observed_at", "created_at", "timestamp")
                    if str(event.get(field_name) or "").strip()
                ),
                "",
            )
            if not observed:
                exclusions["missing_timestamp"] += 1
                continue
            if self._timestamp_epoch(observed, "raw event timestamp") > cutoff_epoch:
                exclusions["future"] += 1
                continue

            if bool(event.get("retrieval_disabled")):
                exclusions["retrieval_disabled"] += 1
                continue
            if bool(event.get("tombstone") or event.get("deleted")):
                exclusions["tombstone"] += 1
                continue
            status = str(event.get("memory_status") or "").strip().lower()
            if status in {"deleted", "tombstone"}:
                exclusions["tombstone"] += 1
                continue
            privacy = str(
                event.get("privacy_level", event.get("privacy", "private"))
                or "private"
            ).strip()
            visibility = str(event.get("visibility", "private") or "private").strip()
            if privacy not in self.config.allowed_privacy:
                exclusions["privacy"] += 1
                continue
            if visibility not in self.config.allowed_visibility:
                exclusions["visibility"] += 1
                continue
            eligible.append(event)
        return eligible, exclusions

    @staticmethod
    def _utf8_size(value: Any) -> int:
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError) as exc:
            raise ShadowPipelineError(
                "shadow inputs must be finite JSON-compatible data"
            ) from exc
        return len(encoded)

    def _validated_structured_extractor(self) -> StructuredExtractor | None:
        if self.structured_extractor is None:
            return None
        configured = self.config.structured_extractor_artifact_digest
        actual = str(
            getattr(self.structured_extractor, "artifact_digest", "") or ""
        )
        if (
            not re.fullmatch(r"sha256:[0-9a-f]{64}", configured)
            or actual != configured
        ):
            raise ShadowPipelineError(
                "structured extractor requires its pinned artifact digest"
            )

        def bounded(payload: dict[str, Any]) -> dict[str, Any]:
            if self._structured_adapter_circuit_open:
                raise TimeoutError("structured extractor circuit is open")
            if not self._structured_adapter_lock.acquire(blocking=False):
                raise TimeoutError("structured extractor is already running")
            results: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

            def invoke() -> None:
                try:
                    results.put_nowait(("ok", self.structured_extractor(payload)))
                except Exception as exc:
                    results.put_nowait(("error", exc))
                finally:
                    self._structured_adapter_lock.release()

            worker = threading.Thread(
                target=invoke,
                name="memory-v2-shadow-structured-extractor",
                daemon=True,
            )
            worker.start()
            worker.join(int(self.config.structured_extractor_timeout_ms) / 1000.0)
            if worker.is_alive():
                self._structured_adapter_circuit_open = True
                raise TimeoutError("structured extractor timed out")
            status, value = results.get_nowait()
            if status != "ok":
                raise ShadowPipelineError("structured extractor failed") from value
            if not isinstance(value, dict):
                raise ShadowPipelineError(
                    "structured extractor response must be a mapping"
                )
            return value

        return bounded

    @staticmethod
    def _timestamp_epoch(value: str, label: str) -> float:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ShadowPipelineError(f"{label} must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise ShadowPipelineError(f"{label} must include a timezone")
        return parsed.astimezone(timezone.utc).timestamp()

    @staticmethod
    def _query_event(
        query: str,
        evidence_cutoff: str,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "id": "shadow-query",
            "type": "turn",
            "user_content": query,
            "observed_at": evidence_cutoff,
        }
        event.update(context["scope"])
        if context["workstream_ids"] and not any(
            key in event
            for key in ("workstream_id", "workstream", "workstream_name")
        ):
            event["workstream_id"] = list(context["workstream_ids"])
        return event

    @staticmethod
    def _corroboration_edges(
        nodes: Sequence[EvidenceNode],
    ) -> list[dict[str, str]]:
        node_ids = {node.id for node in nodes}
        return [
            {
                "source_id": node.id,
                "target_id": target_id,
                "type": "verified_by",
            }
            for node in nodes
            for target_id in node.corroborated_by
            if target_id in node_ids and target_id != node.id
        ]

    @staticmethod
    def _index_units(
        nodes: Sequence[EvidenceNode],
        raw_events: Sequence[Mapping[str, Any]],
        *,
        profile_id: str,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        events_by_id = {
            str(event.get("id") or "").strip(): event
            for event in raw_events
        }
        units: list[dict[str, Any]] = []
        for node in nodes:
            unit = node.to_dict()
            source_events = [
                events_by_id[source_id]
                for source_id in node.source_refs
                if source_id in events_by_id
            ]
            if len(source_events) != len(node.source_refs):
                raise ShadowPipelineError("evidence source lookup failed closed")
            unit["profile_id"] = profile_id
            unit["tenant_id"] = tenant_id
            privacy_values = {
                str(
                    event.get("privacy_level", event.get("privacy", "private"))
                    or "private"
                ).strip()
                for event in source_events
            }
            visibility_values = {
                str(event.get("visibility", "private") or "private").strip()
                for event in source_events
            }
            if len(privacy_values) != 1 or len(visibility_values) != 1:
                raise ShadowPipelineError(
                    "evidence sources contain conflicting policy values"
                )
            unit["privacy_level"] = next(iter(privacy_values))
            unit["visibility"] = next(iter(visibility_values))
            statuses = {
                str(event.get("memory_status") or "")
                .strip()
                .lower()
                for event in source_events
                if event.get("memory_status") is not None
            }
            statuses.discard("")
            if len(statuses) > 1:
                raise ShadowPipelineError(
                    "evidence sources contain conflicting memory_status values"
                )
            if statuses:
                status = next(iter(statuses))
                if status not in _EVIDENCE_STATUSES:
                    raise ShadowPipelineError("memory_status is not recognized")
                unit["status"] = status
            units.append(unit)
        return units

    @staticmethod
    def _reranker_candidate(
        retrieved: Mapping[str, Any],
        node: EvidenceNode,
        profile_id: str,
    ) -> dict[str, Any]:
        citations = [
            {
                "source_id": span.source_id,
                "field": span.field,
                "start": span.start,
                "end": span.end,
                "text": span.text,
                "evidence_at": node.observed_at,
            }
            for span in node.evidence_spans
        ]
        projects = list(dict.fromkeys(node.project_ids))
        workstream_ids = list(dict.fromkeys(node.workstream_ids))
        workstreams: list[str] = []
        if projects:
            workstreams.append(projects.pop(0))
        if workstream_ids and len(workstreams) < MAX_QUERY_SCOPE_IDS:
            workstreams.append(workstream_ids.pop(0))
        remaining = [
            item
            for pair in zip(projects, workstream_ids)
            for item in pair
        ]
        remaining.extend(projects[len(workstream_ids) :])
        remaining.extend(workstream_ids[len(projects) :])
        workstreams.extend(
            item
            for item in remaining
            if item not in workstreams
        )
        workstreams = workstreams[:MAX_QUERY_SCOPE_IDS]
        return {
            "id": node.id,
            "type": node.kind,
            "snippet": node.text,
            "source_refs": list(node.source_refs),
            "citations": citations,
            "evidence_at": node.observed_at,
            "profile_id": profile_id,
            "workstream_ids": workstreams,
            "status": str(retrieved.get("status") or "current"),
            "evidence_role": node.role,
            "verified": node.verified,
        }

    @staticmethod
    def _citation_sources(
        raw_events: Sequence[Mapping[str, Any]],
    ) -> dict[tuple[str, str], str]:
        return {
            (str(event["id"]), field_name): value
            for event in raw_events
            for field_name, value in event.items()
            if isinstance(value, str)
        }

    def _empty(self, reason: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": False,
            "shadow_only": True,
            "read_only": True,
            "untrusted_data": True,
            "mutation_authority": "none",
            "intent": {
                "decision": "none",
                "reason": reason,
                "mutation_authority": "none",
            },
            "corpus": {
                "raw_event_count": 0,
                "eligible_raw_event_count": 0,
                "policy_exclusions": {},
                "derived_node_count": 0,
                "hygiene_counts": {},
                "structured_rejection_count": 0,
            },
            "query_scope": {
                "status": "not_resolved",
                "project_ids": [],
                "workstream_ids": [],
                "confidence": 0.0,
            },
            "index": {},
            "retrieval": {
                "candidate_count": 0,
                "candidate_ids": [],
                "diagnostics": [],
            },
            "result": ShadowUtilityReranker(
                ShadowRerankerConfig(enabled=False)
            ).run("", [], {}),
        }


__all__ = [
    "SCHEMA_VERSION",
    "ShadowPipelineConfig",
    "ShadowPipelineError",
    "ShadowRetrievalPipeline",
]


class _BoundedEmbeddingAdapter:
    """Serialize one pinned embedding call and open a circuit on timeout."""

    def __init__(
        self,
        adapter: LocalEmbeddingAdapter,
        *,
        timeout_ms: int,
    ) -> None:
        self._adapter = adapter
        self._timeout_ms = timeout_ms
        self._lock = threading.Lock()
        self._circuit_open = False
        self.model = adapter.model
        self.version = adapter.version
        self.dimension = adapter.dimension
        self.artifact_digest = adapter.artifact_digest

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if self._circuit_open:
            raise TimeoutError("embedding adapter circuit is open")
        if not self._lock.acquire(blocking=False):
            raise TimeoutError("embedding adapter is already running")
        results: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def invoke() -> None:
            try:
                results.put_nowait(("ok", self._adapter.embed(texts)))
            except Exception as exc:
                results.put_nowait(("error", exc))
            finally:
                self._lock.release()

        worker = threading.Thread(
            target=invoke,
            name="memory-v2-shadow-embedding",
            daemon=True,
        )
        worker.start()
        worker.join(self._timeout_ms / 1000.0)
        if worker.is_alive():
            self._circuit_open = True
            raise TimeoutError("embedding adapter timed out")
        status, value = results.get_nowait()
        if status != "ok":
            raise ShadowPipelineError("embedding adapter failed") from value
        if not isinstance(value, Sequence):
            raise ShadowPipelineError("embedding adapter response must be a sequence")
        return value


def _normalized_policy_scope(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ShadowPipelineError(
            f"{label} scopes must be a non-string sequence"
        )
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ShadowPipelineError(
                f"{label} scopes must contain only strings"
            )
        clean = item.strip()
        if not clean or len(clean) > 80:
            raise ShadowPipelineError(
                f"{label} scopes must be non-empty bounded strings"
            )
        if clean not in normalized:
            normalized.append(clean)
    if not normalized:
        raise ShadowPipelineError(
            "privacy and visibility scopes cannot be empty"
        )
    if len(normalized) > 16:
        raise ShadowPipelineError(f"{label} scopes exceed their safety bound")
    return tuple(normalized)
