"""Read-only shadow reranking and abstention for Memory v2.

This module is deliberately not wired into the provider.  It accepts an
already-bounded candidate set, applies authority-independent hard filters,
optionally asks a local callable to score the survivors, and returns an
untrusted evidence bundle.  It performs no file, network, tool, or canonical
memory operations.
"""

from __future__ import annotations

import json
import math
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from .redaction import redact_text


REQUEST_SCHEMA_VERSION = "memory-v2-shadow-reranker-request/v1"
MODEL_RESPONSE_SCHEMA_VERSION = "memory-v2-shadow-reranker-response/v1"
RESULT_SCHEMA_VERSION = "memory-v2-shadow-reranker-result/v1"
MAX_CANDIDATES = 30

_CANDIDATE_KEYS = {
    "id",
    "type",
    "snippet",
    "source_refs",
    "citations",
    "evidence_at",
    "profile_id",
    "workstream_ids",
    "status",
    "evidence_role",
    "verified",
}
_CITATION_KEYS = {
    "source_id",
    "field",
    "start",
    "end",
    "text",
    "evidence_at",
}
_CONTEXT_KEYS = {
    "memory_decision",
    "profile_id",
    "workstream_ids",
    "allow_unknown_workstream",
    "temporal_mode",
    "evidence_cutoff",
}
_MODEL_RESPONSE_KEYS = {"schema_version", "memory_decision", "scores"}
_MODEL_SCORE_KEYS = {"candidate_id", "utility_score"}
_METRIC_RECORD_KEYS = {
    "expected_memory_decision",
    "useful_candidate_ids",
    "required_source_refs",
    "valid_source_refs",
    "stale_candidate_ids",
    "temporal_mode",
    "result",
}
_METRIC_RESULT_KEYS = {
    "schema_version",
    "enabled",
    "shadow_only",
    "read_only",
    "decision",
    "ranking_source",
    "ranked_candidates",
    "bundle",
    "filter_counts",
    "adapter",
    "latency_ms",
}
_METRIC_DECISION_KEYS = {"memory_needed", "selected", "reason"}
_METRIC_BUNDLE_KEYS = {"untrusted_data", "bounded", "max_items", "slots", "items"}
_METRIC_SLOT_KEYS = {
    "current_state",
    "goal_constraints",
    "decision_rationale",
    "verified_result",
    "blocker_open_loop_next_action",
    "procedure_artifact",
    "implementation_provenance",
}
_METRIC_CANDIDATE_KEYS = {
    "id",
    "type",
    "status",
    "evidence_at",
    "workstream_ids",
    "source_refs",
    "citations",
    "evidence_role",
    "verified",
    "utility_score",
    "deterministic_score",
}
_METRIC_CITATION_KEYS = {
    *_CITATION_KEYS,
    "well_formed_span",
    "exact_source_span",
}
_METRIC_FILTER_KEYS = {
    "accepted",
    "invalid",
    "profile",
    "workstream",
    "future",
    "temporal",
    "overflow",
}
_METRIC_ADAPTER_KEYS = {
    "status",
    "artifact_digest",
    "timeout_ms",
    "elapsed_ms",
}
_MAX_METRIC_RECORDS = 10_000
_MAX_METRIC_SOURCE_IDS = MAX_CANDIDATES * 12
_CURRENTLY_INVALID_STATUSES = {
    "superseded",
    "stale",
    "resolved",
    "closed",
    "rejected",
    "archived_only",
}
_HISTORY_INVALID_STATUSES = {"rejected", "archived_only"}
_VERIFIABLE_ROLES = {"tool", "operator", "user", "deterministic_harness"}
_VERIFIED_RESULT_TYPES = {
    "verified_result",
    "tool_result",
    "completed_action",
    "outcome",
}
_INSTRUCTION_SHAPED = re.compile(
    r"(?i)"
    r"(?:ignore\s+(?:all\s+)?previous\s+instructions|"
    r"ignore\s+(?:all\s+)?prior\s+instructions|"
    r"\b(?:system|developer|assistant|tool)\s*:|"
    r"\b(?:tool_call|function_call)\b|"
    r"promote\s+this\s+memory\s+automatically|"
    r"reveal\s+(?:the\s+)?(?:hidden\s+)?system\s+prompt)"
)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.+-]*")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "for",
    "from",
    "have",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "was",
    "we",
    "what",
}


class ShadowRerankerError(ValueError):
    """Raised for invalid local shadow-reranker configuration."""


@dataclass(frozen=True)
class ShadowRerankerConfig:
    """Configuration for a disabled-by-default shadow execution."""

    enabled: bool = False
    max_candidates: int = MAX_CANDIDATES
    max_bundle_items: int = 5
    unknown_workstream_limit: int = 10
    minimum_utility: float = 0.45
    model_artifact_digest: str = ""
    adapter_timeout_ms: int = 250
    adapter_failure_mode: str = "deterministic"

    def __post_init__(self) -> None:
        if not 1 <= int(self.max_candidates) <= MAX_CANDIDATES:
            raise ShadowRerankerError(
                f"max_candidates must be between 1 and {MAX_CANDIDATES}"
            )
        if not 1 <= int(self.max_bundle_items) <= 8:
            raise ShadowRerankerError("max_bundle_items must be between 1 and 8")
        if not 1 <= int(self.unknown_workstream_limit) <= int(self.max_candidates):
            raise ShadowRerankerError(
                "unknown_workstream_limit must be within max_candidates"
            )
        if not 0.0 <= float(self.minimum_utility) <= 1.0:
            raise ShadowRerankerError("minimum_utility must be within [0, 1]")
        if not 1 <= int(self.adapter_timeout_ms) <= 60_000:
            raise ShadowRerankerError("adapter_timeout_ms must be within 1..60000")
        if self.adapter_failure_mode not in {"deterministic", "none"}:
            raise ShadowRerankerError(
                "adapter_failure_mode must be deterministic or none"
            )


class ShadowUtilityReranker:
    """Bounded, read-only utility reranker with explicit abstention."""

    def __init__(self, config: ShadowRerankerConfig | None = None) -> None:
        self.config = config or ShadowRerankerConfig()
        self._adapter_lock = threading.Lock()
        self._adapter_circuit_open = False

    def run(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        *,
        adapter: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
        citation_sources: Mapping[tuple[str, str], str] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if not self.config.enabled:
            return self._empty_result(
                reason="disabled",
                started=started,
                enabled=False,
                adapter_status="not_called",
            )

        validated_context = self._validate_context(context)
        if validated_context["memory_decision"] == "none":
            return self._empty_result(
                reason="memory_not_needed",
                started=started,
                enabled=True,
                adapter_status="not_called",
            )

        accepted, filter_counts = self._filter_candidates(
            candidates,
            validated_context,
            citation_sources=citation_sources,
        )
        if not accepted:
            result = self._empty_result(
                reason="no_eligible_candidates",
                started=started,
                enabled=True,
                adapter_status="not_called",
            )
            result["filter_counts"] = filter_counts
            return result

        deterministic_scores = {
            item["id"]: self._deterministic_score(str(query or ""), item)
            for item in accepted
        }
        scores = deterministic_scores
        ranking_source = "deterministic"
        adapter_status = "not_configured" if adapter is None else "not_called"
        adapter_elapsed_ms = 0.0
        adapter_decision = "needed"

        if adapter is not None:
            adapter_digest = str(getattr(adapter, "artifact_digest", "") or "")
            if (
                not self._valid_artifact_digest(self.config.model_artifact_digest)
                or adapter_digest != self.config.model_artifact_digest
            ):
                adapter_status = "invalid_configuration"
                if self.config.adapter_failure_mode == "none":
                    result = self._empty_result(
                        reason="adapter_invalid_configuration",
                        started=started,
                        enabled=True,
                        adapter_status=adapter_status,
                    )
                    result["filter_counts"] = filter_counts
                    return result
                ranking_source = "deterministic_fallback"
            else:
                request = self._adapter_request(
                    str(query or ""), accepted, validated_context
                )
                call = self._call_adapter(adapter, request)
                adapter_status = call["status"]
                adapter_elapsed_ms = call["elapsed_ms"]
                if adapter_status == "ok":
                    try:
                        adapter_decision, scores = self._validate_model_response(
                            call["response"], accepted
                        )
                        ranking_source = "local_adapter"
                    except (ShadowRerankerError, TypeError, ValueError):
                        adapter_status = "malformed"
                if adapter_status != "ok":
                    if self.config.adapter_failure_mode == "none":
                        result = self._empty_result(
                            reason=f"adapter_{adapter_status}",
                            started=started,
                            enabled=True,
                            adapter_status=adapter_status,
                            adapter_elapsed_ms=adapter_elapsed_ms,
                        )
                        result["filter_counts"] = filter_counts
                        return result
                    scores = deterministic_scores
                    ranking_source = "deterministic_fallback"

        if adapter_decision == "none":
            result = self._empty_result(
                reason="adapter_abstained",
                started=started,
                enabled=True,
                adapter_status=adapter_status,
                adapter_elapsed_ms=adapter_elapsed_ms,
            )
            result["filter_counts"] = filter_counts
            result["ranking_source"] = ranking_source
            return result

        ranked_internal = sorted(
            accepted,
            key=lambda item: (
                float(scores.get(item["id"], 0.0)),
                float(deterministic_scores.get(item["id"], 0.0)),
                self._timestamp_epoch(item["evidence_at"]),
                item["id"],
            ),
            reverse=True,
        )
        ranked_internal = [
            item
            for item in ranked_internal
            if float(scores.get(item["id"], 0.0))
            >= float(self.config.minimum_utility)
        ]
        ranked_public = [
            self._public_candidate(
                item,
                utility_score=float(scores[item["id"]]),
                deterministic_score=float(deterministic_scores[item["id"]]),
            )
            for item in ranked_internal
        ]
        bundle = self._compose_bundle(ranked_public)
        selected = "candidates" if bundle["items"] else "none"
        reason = "useful_candidates" if bundle["items"] else "below_utility_or_no_slot"
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "enabled": True,
            "shadow_only": True,
            "read_only": True,
            "decision": {
                "memory_needed": bool(bundle["items"]),
                "selected": selected,
                "reason": reason,
            },
            "ranking_source": ranking_source,
            "ranked_candidates": ranked_public,
            "bundle": bundle,
            "filter_counts": filter_counts,
            "adapter": {
                "status": adapter_status,
                "artifact_digest": (
                    self.config.model_artifact_digest if adapter is not None else ""
                ),
                "timeout_ms": int(self.config.adapter_timeout_ms),
                "elapsed_ms": round(adapter_elapsed_ms, 3),
            },
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }

    def _filter_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        *,
        citation_sources: Mapping[tuple[str, str], str] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        counts = {
            "accepted": 0,
            "invalid": 0,
            "profile": 0,
            "workstream": 0,
            "future": 0,
            "temporal": 0,
            "overflow": max(0, len(candidates) - int(self.config.max_candidates)),
        }
        accepted: list[dict[str, Any]] = []
        query_workstreams = set(context["workstream_ids"])
        allow_unknown = bool(context["allow_unknown_workstream"])
        cutoff_epoch = self._timestamp_epoch(context["evidence_cutoff"])
        temporal_mode = str(context["temporal_mode"])

        for raw in list(candidates)[: int(self.config.max_candidates)]:
            try:
                item = self._validate_candidate(
                    raw,
                    citation_sources=citation_sources,
                )
            except (ShadowRerankerError, TypeError, ValueError):
                counts["invalid"] += 1
                continue
            if item["profile_id"] != context["profile_id"]:
                counts["profile"] += 1
                continue
            if query_workstreams:
                candidate_workstreams = set(item["workstream_ids"])
                if candidate_workstreams:
                    if not query_workstreams & candidate_workstreams:
                        counts["workstream"] += 1
                        continue
                elif not allow_unknown:
                    counts["workstream"] += 1
                    continue
            if self._timestamp_epoch(item["evidence_at"]) > cutoff_epoch:
                counts["future"] += 1
                continue
            invalid_statuses = (
                _HISTORY_INVALID_STATUSES
                if temporal_mode == "history"
                else _CURRENTLY_INVALID_STATUSES
            )
            if item["status"] in invalid_statuses:
                counts["temporal"] += 1
                continue
            accepted.append(item)

        unknown = (
            accepted
            if not query_workstreams
            else [item for item in accepted if not item["workstream_ids"]]
        )
        if len(unknown) > self.config.unknown_workstream_limit:
            permitted = {
                item["id"]
                for item in unknown[: int(self.config.unknown_workstream_limit)]
            }
            overflow = len(unknown) - int(self.config.unknown_workstream_limit)
            accepted = [
                item
                for item in accepted
                if (
                    (query_workstreams and item["workstream_ids"])
                    or item["id"] in permitted
                )
            ]
            counts["overflow"] += overflow
        counts["accepted"] = len(accepted)
        return accepted, counts

    @staticmethod
    def _validate_context(context: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(context, Mapping) or set(context) != _CONTEXT_KEYS:
            raise ShadowRerankerError("context does not match its strict schema")
        memory_decision = str(context.get("memory_decision") or "")
        if memory_decision not in {"needed", "none"}:
            raise ShadowRerankerError("memory_decision must be needed or none")
        profile_id = str(context.get("profile_id") or "").strip()
        if not profile_id or len(profile_id) > 200:
            raise ShadowRerankerError("profile_id must be a bounded opaque value")
        workstreams = context.get("workstream_ids")
        if (
            not isinstance(workstreams, list)
            or len(workstreams) > 8
            or any(not isinstance(value, str) or not value.strip() for value in workstreams)
        ):
            raise ShadowRerankerError("workstream_ids must be a bounded string list")
        allow_unknown = context.get("allow_unknown_workstream")
        if not isinstance(allow_unknown, bool):
            raise ShadowRerankerError("allow_unknown_workstream must be boolean")
        temporal_mode = str(context.get("temporal_mode") or "")
        if temporal_mode not in {"current", "history", "any"}:
            raise ShadowRerankerError(
                "temporal_mode must be current, history, or any"
            )
        cutoff = str(context.get("evidence_cutoff") or "")
        ShadowUtilityReranker._timestamp_epoch(cutoff)
        return {
            "memory_decision": memory_decision,
            "profile_id": profile_id,
            "workstream_ids": list(dict.fromkeys(value.strip() for value in workstreams)),
            "allow_unknown_workstream": allow_unknown,
            "temporal_mode": temporal_mode,
            "evidence_cutoff": cutoff,
        }

    @staticmethod
    def _validate_candidate(
        raw: Mapping[str, Any],
        *,
        citation_sources: Mapping[tuple[str, str], str] | None,
    ) -> dict[str, Any]:
        if not isinstance(raw, Mapping) or set(raw) != _CANDIDATE_KEYS:
            raise ShadowRerankerError("candidate does not match its strict schema")
        candidate_id = str(raw.get("id") or "").strip()
        candidate_type = str(raw.get("type") or "").strip()
        snippet = str(raw.get("snippet") or "")
        profile_id = str(raw.get("profile_id") or "").strip()
        status = str(raw.get("status") or "").strip().lower()
        evidence_role = str(raw.get("evidence_role") or "").strip().lower()
        if not candidate_id or len(candidate_id) > 200:
            raise ShadowRerankerError("candidate id is invalid")
        if not candidate_type or len(candidate_type) > 80:
            raise ShadowRerankerError("candidate type is invalid")
        if not snippet or len(snippet) > 8_000:
            raise ShadowRerankerError("candidate snippet is invalid")
        if not profile_id or len(profile_id) > 200:
            raise ShadowRerankerError("candidate profile is invalid")
        if not status or len(status) > 40:
            raise ShadowRerankerError("candidate status is invalid")
        if evidence_role not in {
            "user",
            "assistant",
            "tool",
            "operator",
            "deterministic_harness",
        }:
            raise ShadowRerankerError("candidate evidence role is invalid")
        if not isinstance(raw.get("verified"), bool):
            raise ShadowRerankerError("candidate verified must be boolean")

        source_refs = raw.get("source_refs")
        workstreams = raw.get("workstream_ids")
        citations = raw.get("citations")
        if (
            not isinstance(source_refs, list)
            or not 1 <= len(source_refs) <= 12
            or any(not isinstance(value, str) or not value.strip() for value in source_refs)
        ):
            raise ShadowRerankerError("candidate source_refs are invalid")
        if (
            not isinstance(workstreams, list)
            or len(workstreams) > 8
            or any(not isinstance(value, str) or not value.strip() for value in workstreams)
        ):
            raise ShadowRerankerError("candidate workstreams are invalid")
        if not isinstance(citations, list) or not 1 <= len(citations) <= 12:
            raise ShadowRerankerError("candidate citations are invalid")

        evidence_at = str(raw.get("evidence_at") or "")
        ShadowUtilityReranker._timestamp_epoch(evidence_at)
        validated_citations = [
            ShadowUtilityReranker._validate_citation(
                citation,
                source_refs=source_refs,
                evidence_at=evidence_at,
                citation_sources=citation_sources,
            )
            for citation in citations
        ]
        return {
            "id": candidate_id,
            "type": candidate_type,
            "snippet": snippet,
            "source_refs": list(dict.fromkeys(value.strip() for value in source_refs)),
            "citations": validated_citations,
            "evidence_at": evidence_at,
            "profile_id": profile_id,
            "workstream_ids": list(
                dict.fromkeys(value.strip() for value in workstreams)
            ),
            "status": status,
            "evidence_role": evidence_role,
            "verified": bool(raw["verified"]),
        }

    @staticmethod
    def _validate_citation(
        raw: Mapping[str, Any],
        *,
        source_refs: Sequence[str],
        evidence_at: str,
        citation_sources: Mapping[tuple[str, str], str] | None,
    ) -> dict[str, Any]:
        if not isinstance(raw, Mapping) or set(raw) != _CITATION_KEYS:
            raise ShadowRerankerError("citation does not match its strict schema")
        source_id = str(raw.get("source_id") or "").strip()
        field = str(raw.get("field") or "").strip()
        text = str(raw.get("text") or "")
        start = raw.get("start")
        end = raw.get("end")
        citation_time = str(raw.get("evidence_at") or "")
        if source_id not in source_refs:
            raise ShadowRerankerError("citation source is absent from source_refs")
        if not field or not text or len(text) > 4_000:
            raise ShadowRerankerError("citation field or text is invalid")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end - start != len(text)
        ):
            raise ShadowRerankerError("citation is not an exact bounded span")
        ShadowUtilityReranker._timestamp_epoch(citation_time)
        if citation_time != evidence_at:
            raise ShadowRerankerError(
                "citation and candidate evidence timestamps disagree"
            )
        exact = False
        if citation_sources is not None:
            source_text = citation_sources.get((source_id, field))
            if (
                not isinstance(source_text, str)
                or end > len(source_text)
                or source_text[start:end] != text
            ):
                raise ShadowRerankerError(
                    "citation does not match its trusted source text"
                )
            exact = True
        return {
            "source_id": source_id,
            "field": field,
            "start": start,
            "end": end,
            "text": text,
            "evidence_at": citation_time,
            "well_formed_span": True,
            "exact_source_span": exact,
        }

    def _adapter_request(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "query_untrusted_data": self._escape_untrusted(query, limit=2_000),
            "context": {
                "memory_decision": context["memory_decision"],
                "profile_id": context["profile_id"],
                "workstream_ids": list(context["workstream_ids"]),
                "allow_unknown_workstream": context["allow_unknown_workstream"],
                "temporal_mode": context["temporal_mode"],
                "evidence_cutoff": context["evidence_cutoff"],
            },
            "candidates": [
                {
                    "candidate_id": item["id"],
                    "type": item["type"],
                    "snippet_untrusted_data": self._escape_untrusted(
                        item["snippet"], limit=1_500
                    ),
                    "source_refs": list(item["source_refs"]),
                    "evidence_at": item["evidence_at"],
                    "status": item["status"],
                    "evidence_role": item["evidence_role"],
                    "verified": (
                        bool(item["verified"])
                        and item["evidence_role"] in _VERIFIABLE_ROLES
                    ),
                }
                for item in candidates[:MAX_CANDIDATES]
            ],
            "model": {
                "artifact_digest": self.config.model_artifact_digest,
                "timeout_ms": int(self.config.adapter_timeout_ms),
            },
        }

    def _call_adapter(
        self,
        adapter: Callable[[dict[str, Any]], Mapping[str, Any]],
        request: dict[str, Any],
    ) -> dict[str, Any]:
        if self._adapter_circuit_open:
            return {"status": "circuit_open", "response": None, "elapsed_ms": 0.0}
        if not self._adapter_lock.acquire(blocking=False):
            return {"status": "busy", "response": None, "elapsed_ms": 0.0}
        results: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def invoke() -> None:
            try:
                results.put_nowait(("ok", adapter(request)))
            except TimeoutError:
                results.put_nowait(("timeout", None))
            except Exception:
                results.put_nowait(("error", None))
            finally:
                self._adapter_lock.release()

        started = time.perf_counter()
        worker = threading.Thread(
            target=invoke, name="memory-v2-shadow-reranker", daemon=True
        )
        worker.start()
        worker.join(int(self.config.adapter_timeout_ms) / 1000.0)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if worker.is_alive():
            self._adapter_circuit_open = True
            return {"status": "timeout", "response": None, "elapsed_ms": elapsed_ms}
        try:
            status, response = results.get_nowait()
        except queue.Empty:
            return {"status": "error", "response": None, "elapsed_ms": elapsed_ms}
        return {"status": status, "response": response, "elapsed_ms": elapsed_ms}

    @staticmethod
    def _validate_model_response(
        raw: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
    ) -> tuple[str, dict[str, float]]:
        if not isinstance(raw, Mapping) or set(raw) != _MODEL_RESPONSE_KEYS:
            raise ShadowRerankerError("model response does not match strict schema")
        if raw.get("schema_version") != MODEL_RESPONSE_SCHEMA_VERSION:
            raise ShadowRerankerError("model response schema version is invalid")
        memory_decision = str(raw.get("memory_decision") or "")
        if memory_decision not in {"needed", "none"}:
            raise ShadowRerankerError("model memory decision is invalid")
        rows = raw.get("scores")
        if not isinstance(rows, list) or len(rows) != len(candidates):
            raise ShadowRerankerError("model must score every eligible candidate")
        expected = {str(item["id"]) for item in candidates}
        scores: dict[str, float] = {}
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != _MODEL_SCORE_KEYS:
                raise ShadowRerankerError("model score does not match strict schema")
            candidate_id = str(row.get("candidate_id") or "")
            value = row.get("utility_score")
            if (
                candidate_id not in expected
                or candidate_id in scores
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ShadowRerankerError("model score id or value is invalid")
            scores[candidate_id] = float(value)
        if set(scores) != expected:
            raise ShadowRerankerError("model score candidate set is invalid")
        return memory_decision, scores

    @staticmethod
    def _deterministic_score(query: str, item: Mapping[str, Any]) -> float:
        query_terms = ShadowUtilityReranker._terms(query)
        item_terms = ShadowUtilityReranker._terms(
            " ".join(
                [
                    str(item.get("type") or ""),
                    str(item.get("snippet") or ""),
                    " ".join(str(value) for value in item.get("workstream_ids", [])),
                ]
            )
        )
        overlap = len(query_terms & item_terms) / max(1, len(query_terms))
        type_name = str(item.get("type") or "").lower()
        type_terms = set(type_name.split("_"))
        query_type_overlap = len(query_terms & type_terms) / max(1, len(type_terms))
        verified_boost = (
            0.08
            if bool(item.get("verified"))
            and str(item.get("evidence_role") or "") in _VERIFIABLE_ROLES
            else 0.0
        )
        active_boost = 0.05 if str(item.get("status") or "") == "active" else 0.0
        score = 0.03 + (0.72 * overlap) + (0.12 * query_type_overlap)
        score += verified_boost + active_boost
        return round(min(1.0, max(0.0, score)), 6)

    def _compose_bundle(
        self, ranked: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        slots = {
            "current_state": [],
            "goal_constraints": [],
            "decision_rationale": [],
            "verified_result": [],
            "blocker_open_loop_next_action": [],
            "procedure_artifact": [],
            "implementation_provenance": [],
        }
        selected: list[dict[str, Any]] = []
        seen_citations: set[tuple[str, str, int, int]] = set()

        for wanted_slot in slots:
            for item in ranked:
                candidate_id = str(item["id"])
                if any(existing["id"] == candidate_id for existing in selected):
                    continue
                if self._slot_for(item) != wanted_slot:
                    continue
                citations = self._citation_identities(item)
                if citations and citations <= seen_citations:
                    continue
                selected.append(dict(item))
                slots[wanted_slot].append(candidate_id)
                seen_citations.update(citations)
                break
            if len(selected) >= int(self.config.max_bundle_items):
                break

        if len(selected) < int(self.config.max_bundle_items):
            for item in ranked:
                if len(selected) >= int(self.config.max_bundle_items):
                    break
                candidate_id = str(item["id"])
                if any(existing["id"] == candidate_id for existing in selected):
                    continue
                slot = self._slot_for(item)
                if slot is None:
                    continue
                citations = self._citation_identities(item)
                if citations and citations <= seen_citations:
                    continue
                selected.append(dict(item))
                slots[slot].append(candidate_id)
                seen_citations.update(citations)

        return {
            "untrusted_data": True,
            "bounded": True,
            "max_items": int(self.config.max_bundle_items),
            "slots": slots,
            "items": selected,
        }

    @staticmethod
    def _slot_for(item: Mapping[str, Any]) -> str | None:
        kind = str(item.get("type") or "").strip().lower()
        if kind in _VERIFIED_RESULT_TYPES:
            if (
                bool(item.get("verified"))
                and str(item.get("evidence_role") or "") in _VERIFIABLE_ROLES
                and str(item.get("evidence_role") or "") != "assistant"
            ):
                return "verified_result"
            return None
        if kind in {"decision", "rationale", "accepted_proposal"}:
            return "decision_rationale"
        if kind == "request_goal":
            return "goal_constraints"
        if kind in {"blocker", "open_loop", "next_action"}:
            return "blocker_open_loop_next_action"
        if kind in {"procedure", "artifact"}:
            return "procedure_artifact"
        if kind in {"implementation_claim", "proposal"}:
            return "implementation_provenance"
        if kind in {
            "current_state",
            "project_state",
            "fact",
            "preference",
            "environment",
            "correction",
            "episode",
        }:
            return "current_state"
        return None

    @staticmethod
    def _citation_identities(
        item: Mapping[str, Any],
    ) -> set[tuple[str, str, int, int]]:
        return {
            (
                str(citation.get("source_id") or ""),
                str(citation.get("field") or ""),
                int(citation.get("start") or 0),
                int(citation.get("end") or 0),
            )
            for citation in item.get("citations", [])
            if isinstance(citation, Mapping)
        }

    @staticmethod
    def _public_candidate(
        item: Mapping[str, Any],
        *,
        utility_score: float,
        deterministic_score: float,
    ) -> dict[str, Any]:
        return {
            "id": item["id"],
            "type": item["type"],
            "status": item["status"],
            "evidence_at": item["evidence_at"],
            "workstream_ids": list(item["workstream_ids"]),
            "source_refs": list(item["source_refs"]),
            "citations": [dict(citation) for citation in item["citations"]],
            "evidence_role": item["evidence_role"],
            "verified": (
                bool(item["verified"])
                and item["evidence_role"] in _VERIFIABLE_ROLES
                and item["evidence_role"] != "assistant"
            ),
            "utility_score": round(float(utility_score), 6),
            "deterministic_score": round(float(deterministic_score), 6),
        }

    def _empty_result(
        self,
        *,
        reason: str,
        started: float,
        enabled: bool,
        adapter_status: str,
        adapter_elapsed_ms: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "enabled": enabled,
            "shadow_only": True,
            "read_only": True,
            "decision": {
                "memory_needed": False,
                "selected": "none",
                "reason": reason,
            },
            "ranking_source": "none",
            "ranked_candidates": [],
            "bundle": {
                "untrusted_data": True,
                "bounded": True,
                "max_items": int(self.config.max_bundle_items),
                "slots": {
                    "current_state": [],
                    "goal_constraints": [],
                    "decision_rationale": [],
                    "verified_result": [],
                    "blocker_open_loop_next_action": [],
                    "procedure_artifact": [],
                    "implementation_provenance": [],
                },
                "items": [],
            },
            "filter_counts": {
                "accepted": 0,
                "invalid": 0,
                "profile": 0,
                "workstream": 0,
                "future": 0,
                "temporal": 0,
                "overflow": 0,
            },
            "adapter": {
                "status": adapter_status,
                "artifact_digest": "",
                "timeout_ms": int(self.config.adapter_timeout_ms),
                "elapsed_ms": round(float(adapter_elapsed_ms), 3),
            },
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }

    @staticmethod
    def _escape_untrusted(text: str, *, limit: int) -> str:
        redacted = redact_text(str(text or ""))
        escaped = _INSTRUCTION_SHAPED.sub("[ESCAPED_INSTRUCTION_SHAPED_TEXT]", redacted)
        escaped = escaped[:limit]
        # A JSON string is used as an additional data boundary.  The adapter
        # receives the quotes and escaping as part of the field value.
        return json.dumps(escaped, ensure_ascii=False)

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            token
            for token in _TOKEN_RE.findall(str(text or "").lower())
            if token not in _STOP_WORDS and len(token) > 1
        }

    @staticmethod
    def _timestamp_epoch(value: Any) -> float:
        text = str(value or "").strip()
        if not text:
            raise ShadowRerankerError("timestamp is required")
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ShadowRerankerError("timestamp must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise ShadowRerankerError("timestamp must include a timezone")
        return parsed.astimezone(timezone.utc).timestamp()

    @staticmethod
    def _valid_artifact_digest(value: str) -> bool:
        return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")))


def compute_shadow_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute bounded development diagnostics from externally labeled rows.

    Labels are inputs to this helper; the reranker never creates its own gold
    judgments.  Empty denominators return ``None`` so uncovered cases cannot
    manufacture a perfect score; the accompanying counts and eligibility
    flags make missing coverage visible.
    """

    validated_records = _validate_metric_records(records)
    top1_hits = 0
    top1_total = 0
    predicted_none = 0
    true_none = 0
    true_none_predicted_none = 0
    irrelevant_injections = 0
    evidence_completeness: list[float] = []
    stale_selected = 0
    current_selected = 0
    valid_citations = 0
    citation_count = 0
    latency_values: list[float] = []

    for record in validated_records:
        result = record["result"]
        expected = record["expected_memory_decision"]
        ranked = result["ranked_candidates"]
        bundle = result["bundle"]
        decision = result["decision"]
        selected_none = decision.get("selected") == "none"
        if selected_none:
            predicted_none += 1
        if expected == "none":
            true_none += 1
            if selected_none:
                true_none_predicted_none += 1
            elif bundle["items"]:
                irrelevant_injections += 1
        else:
            top1_total += 1
            useful = set(record["useful_candidate_ids"])
            if ranked and str(ranked[0].get("id") or "") in useful:
                top1_hits += 1

        required_refs = set(record["required_source_refs"])
        selected_refs = {
            str(source_id)
            for item in bundle["items"]
            for source_id in item["source_refs"]
        }
        if required_refs:
            evidence_completeness.append(
                len(required_refs & selected_refs) / len(required_refs)
            )

        stale_ids = set(record["stale_candidate_ids"])
        if record["temporal_mode"] == "current":
            for item in bundle["items"]:
                current_selected += 1
                if str(item.get("id") or "") in stale_ids:
                    stale_selected += 1

        valid_refs = set(record["valid_source_refs"])
        for item in bundle["items"]:
            for citation in item["citations"]:
                citation_count += 1
                if (
                    citation.get("exact_source_span") is True
                    and str(citation.get("source_id") or "") in valid_refs
                ):
                    valid_citations += 1

        latency_values.append(float(result["latency_ms"]))

    latency_values.sort()
    return {
        "records": len(validated_records),
        "top1_useful": _safe_rate(top1_hits, top1_total),
        "none_precision": _safe_rate(true_none_predicted_none, predicted_none),
        "none_recall": _safe_rate(true_none_predicted_none, true_none),
        "irrelevant_injection": _safe_rate(irrelevant_injections, true_none),
        "evidence_set_completeness": (
            sum(evidence_completeness) / len(evidence_completeness)
            if evidence_completeness
            else None
        ),
        "stale_as_current": _safe_rate(stale_selected, current_selected),
        "citation_validity": _safe_rate(valid_citations, citation_count),
        "latency": {
            "count": len(latency_values),
            "p50_ms": _percentile(latency_values, 0.50),
            "p95_ms": _percentile(latency_values, 0.95),
            "max_ms": max(latency_values, default=0.0),
        },
        "coverage": {
            "memory_needed": top1_total,
            "expected_none": true_none,
            "predicted_none": predicted_none,
            "citations": citation_count,
            "current_selected_items": current_selected,
            "required_evidence_sets": len(evidence_completeness),
        },
        "eligible_gates": {
            "top1_useful": top1_total > 0,
            "none_precision": predicted_none > 0,
            "none_recall": true_none > 0,
            "irrelevant_injection": true_none > 0,
            "evidence_set_completeness": bool(evidence_completeness),
            "stale_as_current": current_selected > 0,
            "citation_validity": citation_count > 0,
            "latency": bool(latency_values),
        },
    }


def _validate_metric_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ShadowRerankerError("metric records must be a sequence")
    if len(records) > _MAX_METRIC_RECORDS:
        raise ShadowRerankerError("metric records exceed their safety bound")
    validated: list[dict[str, Any]] = []
    for raw in records:
        if not isinstance(raw, Mapping) or set(raw) != _METRIC_RECORD_KEYS:
            raise ShadowRerankerError(
                "metric record does not match its strict schema"
            )
        expected = raw.get("expected_memory_decision")
        if expected not in {"needed", "none"}:
            raise ShadowRerankerError("metric expected decision is invalid")
        temporal_mode = raw.get("temporal_mode")
        if temporal_mode not in {"current", "history"}:
            raise ShadowRerankerError("metric temporal_mode is invalid")
        useful = _metric_id_list(
            raw.get("useful_candidate_ids"),
            "useful_candidate_ids",
            maximum=MAX_CANDIDATES,
        )
        stale = _metric_id_list(
            raw.get("stale_candidate_ids"),
            "stale_candidate_ids",
            maximum=MAX_CANDIDATES,
        )
        required = _metric_id_list(
            raw.get("required_source_refs"),
            "required_source_refs",
            maximum=_MAX_METRIC_SOURCE_IDS,
        )
        valid = _metric_id_list(
            raw.get("valid_source_refs"),
            "valid_source_refs",
            maximum=_MAX_METRIC_SOURCE_IDS,
        )
        validated.append(
            {
                "expected_memory_decision": expected,
                "useful_candidate_ids": useful,
                "required_source_refs": required,
                "valid_source_refs": valid,
                "stale_candidate_ids": stale,
                "temporal_mode": temporal_mode,
                "result": _validate_metric_result(raw.get("result")),
            }
        )
    return validated


def _validate_metric_result(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _METRIC_RESULT_KEYS:
        raise ShadowRerankerError("metric result does not match its strict schema")
    if raw.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ShadowRerankerError("metric result schema version is invalid")
    for key in ("enabled", "shadow_only", "read_only"):
        if not isinstance(raw.get(key), bool):
            raise ShadowRerankerError(f"metric result {key} must be boolean")
    if raw.get("shadow_only") is not True or raw.get("read_only") is not True:
        raise ShadowRerankerError("metric result must be shadow-only and read-only")

    ranked = raw.get("ranked_candidates")
    if not isinstance(ranked, list) or len(ranked) > MAX_CANDIDATES:
        raise ShadowRerankerError("metric ranked candidates are invalid")
    validated_ranked = [_validate_metric_candidate(item) for item in ranked]
    ranked_by_id = {item["id"]: item for item in validated_ranked}
    if len(ranked_by_id) != len(validated_ranked):
        raise ShadowRerankerError("metric ranked candidate ids must be unique")

    bundle = _validate_metric_bundle(raw.get("bundle"), ranked_by_id)
    decision = raw.get("decision")
    if not isinstance(decision, Mapping) or set(decision) != _METRIC_DECISION_KEYS:
        raise ShadowRerankerError("metric decision does not match its strict schema")
    if not isinstance(decision.get("memory_needed"), bool):
        raise ShadowRerankerError("metric memory_needed must be boolean")
    selected = decision.get("selected")
    if selected not in {"none", "candidates"}:
        raise ShadowRerankerError("metric selected decision is invalid")
    reason = decision.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 200:
        raise ShadowRerankerError("metric decision reason is invalid")
    has_items = bool(bundle["items"])
    if (
        bool(decision["memory_needed"]) != has_items
        or (selected == "candidates") != has_items
    ):
        raise ShadowRerankerError("metric decision and bundle disagree")

    ranking_source = raw.get("ranking_source")
    if ranking_source not in {
        "none",
        "deterministic",
        "deterministic_fallback",
        "local_adapter",
    }:
        raise ShadowRerankerError("metric ranking_source is invalid")
    _validate_metric_counts(raw.get("filter_counts"))
    _validate_metric_adapter(raw.get("adapter"))
    latency = _metric_number(raw.get("latency_ms"), "latency_ms", minimum=0.0)
    return {
        **dict(raw),
        "decision": dict(decision),
        "ranked_candidates": validated_ranked,
        "bundle": bundle,
        "latency_ms": latency,
    }


def _validate_metric_bundle(
    raw: Any,
    ranked_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _METRIC_BUNDLE_KEYS:
        raise ShadowRerankerError("metric bundle does not match its strict schema")
    if raw.get("untrusted_data") is not True or raw.get("bounded") is not True:
        raise ShadowRerankerError("metric bundle safety flags are invalid")
    max_items = raw.get("max_items")
    if (
        isinstance(max_items, bool)
        or not isinstance(max_items, int)
        or not 1 <= max_items <= 8
    ):
        raise ShadowRerankerError("metric bundle max_items is invalid")
    items = raw.get("items")
    if not isinstance(items, list) or len(items) > max_items:
        raise ShadowRerankerError("metric bundle items exceed their bound")
    validated_items = [_validate_metric_candidate(item) for item in items]
    item_ids = [item["id"] for item in validated_items]
    if len(set(item_ids)) != len(item_ids):
        raise ShadowRerankerError("metric bundle candidate ids must be unique")
    for item in validated_items:
        if item["id"] not in ranked_by_id or item != ranked_by_id[item["id"]]:
            raise ShadowRerankerError(
                "metric bundle item is absent from ranked candidates"
            )

    slots = raw.get("slots")
    if not isinstance(slots, Mapping) or set(slots) != _METRIC_SLOT_KEYS:
        raise ShadowRerankerError("metric bundle slots do not match strict schema")
    slot_ids: list[str] = []
    validated_slots: dict[str, list[str]] = {}
    for slot_name in sorted(_METRIC_SLOT_KEYS):
        ids = _metric_id_list(
            slots.get(slot_name),
            f"bundle slot {slot_name}",
            maximum=max_items,
        )
        slot_ids.extend(ids)
        validated_slots[slot_name] = ids
    if len(slot_ids) != len(set(slot_ids)) or set(slot_ids) != set(item_ids):
        raise ShadowRerankerError("metric bundle slots and items disagree")
    return {
        "untrusted_data": True,
        "bounded": True,
        "max_items": max_items,
        "slots": validated_slots,
        "items": validated_items,
    }


def _validate_metric_candidate(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _METRIC_CANDIDATE_KEYS:
        raise ShadowRerankerError(
            "metric candidate does not match its strict schema"
        )
    candidate_id = _metric_text(raw.get("id"), "candidate id", maximum=200)
    candidate_type = _metric_text(raw.get("type"), "candidate type", maximum=80)
    status = _metric_text(raw.get("status"), "candidate status", maximum=40)
    evidence_at = _metric_text(
        raw.get("evidence_at"), "candidate evidence_at", maximum=100
    )
    ShadowUtilityReranker._timestamp_epoch(evidence_at)
    evidence_role = raw.get("evidence_role")
    if evidence_role not in {
        "user",
        "assistant",
        "tool",
        "operator",
        "deterministic_harness",
    }:
        raise ShadowRerankerError("metric candidate evidence role is invalid")
    if not isinstance(raw.get("verified"), bool):
        raise ShadowRerankerError("metric candidate verified must be boolean")
    source_refs = _metric_id_list(
        raw.get("source_refs"), "candidate source_refs", maximum=12
    )
    if not source_refs:
        raise ShadowRerankerError("metric candidate requires source_refs")
    workstreams = _metric_id_list(
        raw.get("workstream_ids"), "candidate workstream_ids", maximum=8
    )
    citations = raw.get("citations")
    if not isinstance(citations, list) or not 1 <= len(citations) <= 12:
        raise ShadowRerankerError("metric candidate citations are invalid")
    validated_citations = [
        _validate_metric_citation(
            citation,
            source_refs=source_refs,
            evidence_at=evidence_at,
        )
        for citation in citations
    ]
    return {
        "id": candidate_id,
        "type": candidate_type,
        "status": status,
        "evidence_at": evidence_at,
        "workstream_ids": workstreams,
        "source_refs": source_refs,
        "citations": validated_citations,
        "evidence_role": evidence_role,
        "verified": raw["verified"],
        "utility_score": _metric_number(
            raw.get("utility_score"), "utility_score", minimum=0.0, maximum=1.0
        ),
        "deterministic_score": _metric_number(
            raw.get("deterministic_score"),
            "deterministic_score",
            minimum=0.0,
            maximum=1.0,
        ),
    }


def _validate_metric_citation(
    raw: Any,
    *,
    source_refs: Sequence[str],
    evidence_at: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _METRIC_CITATION_KEYS:
        raise ShadowRerankerError(
            "metric citation does not match its strict schema"
        )
    source_id = _metric_text(raw.get("source_id"), "citation source_id", maximum=200)
    field = _metric_text(raw.get("field"), "citation field", maximum=200)
    text = raw.get("text")
    if not isinstance(text, str) or not text or len(text) > 4_000:
        raise ShadowRerankerError("metric citation text is invalid")
    start = raw.get("start")
    end = raw.get("end")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
        or end - start != len(text)
    ):
        raise ShadowRerankerError("metric citation span is invalid")
    citation_at = _metric_text(
        raw.get("evidence_at"), "citation evidence_at", maximum=100
    )
    ShadowUtilityReranker._timestamp_epoch(citation_at)
    if source_id not in source_refs or citation_at != evidence_at:
        raise ShadowRerankerError(
            "metric citation provenance disagrees with its candidate"
        )
    if (
        raw.get("well_formed_span") is not True
        or not isinstance(raw.get("exact_source_span"), bool)
    ):
        raise ShadowRerankerError("metric citation attestation is invalid")
    return {
        "source_id": source_id,
        "field": field,
        "start": start,
        "end": end,
        "text": text,
        "evidence_at": citation_at,
        "well_formed_span": True,
        "exact_source_span": raw["exact_source_span"],
    }


def _validate_metric_counts(raw: Any) -> None:
    if not isinstance(raw, Mapping) or set(raw) != _METRIC_FILTER_KEYS:
        raise ShadowRerankerError(
            "metric filter counts do not match strict schema"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _MAX_METRIC_SOURCE_IDS
        for value in raw.values()
    ):
        raise ShadowRerankerError("metric filter counts are invalid")


def _validate_metric_adapter(raw: Any) -> None:
    if not isinstance(raw, Mapping) or set(raw) != _METRIC_ADAPTER_KEYS:
        raise ShadowRerankerError("metric adapter does not match strict schema")
    if raw.get("status") not in {
        "not_called",
        "not_configured",
        "invalid_configuration",
        "ok",
        "malformed",
        "timeout",
        "circuit_open",
        "busy",
        "error",
    }:
        raise ShadowRerankerError("metric adapter status is invalid")
    digest = raw.get("artifact_digest")
    if not isinstance(digest, str) or (
        digest and not ShadowUtilityReranker._valid_artifact_digest(digest)
    ):
        raise ShadowRerankerError("metric adapter digest is invalid")
    timeout = raw.get("timeout_ms")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int)
        or not 1 <= timeout <= 60_000
    ):
        raise ShadowRerankerError("metric adapter timeout is invalid")
    _metric_number(raw.get("elapsed_ms"), "adapter elapsed_ms", minimum=0.0)


def _metric_id_list(value: Any, label: str, *, maximum: int) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ShadowRerankerError(f"metric {label} must be a bounded list")
    normalized = [
        _metric_text(item, label, maximum=200)
        for item in value
    ]
    if len(set(normalized)) != len(normalized):
        raise ShadowRerankerError(f"metric {label} ids must be unique")
    return normalized


def _metric_text(value: Any, label: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ShadowRerankerError(f"metric {label} must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > maximum:
        raise ShadowRerankerError(f"metric {label} is invalid")
    return normalized


def _metric_number(
    value: Any,
    label: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
        or (maximum is not None and float(value) > maximum)
    ):
        raise ShadowRerankerError(f"metric {label} is invalid")
    return float(value)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)
