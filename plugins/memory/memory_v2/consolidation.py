"""Rule-based promotion/consolidation for Memory v2 candidates.

The v0 consolidator is deliberately cheap and deterministic. It promotes only
candidate shapes that can become source-grounded canonical ``MemoryItem`` records
without an LLM, archives/rejects candidates that belong in other systems, and
handles explicit supersession by updating old records rather than silently
letting contradictions coexist as active memories.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, cast

from .index import MemoryV2Index
from .schemas import (
    CandidateMemory,
    GateDecision,
    MemoryItem,
    MemoryStatus,
    MemoryType,
    ProjectCard,
    ProjectStatus,
    ValidationError,
    normalize_project_id,
    utc_now_iso,
)
from .store import MemoryV2Store


@dataclass
class ConsolidationReport:
    """Summary of one consolidation pass."""

    considered: int = 0
    promoted: int = 0
    rejected: int = 0
    archived_only: int = 0
    superseded: int = 0
    skipped: int = 0
    mutation_authorized: bool = False
    promoted_ids: List[str] = field(default_factory=list)
    rejected_ids: List[str] = field(default_factory=list)
    archived_ids: List[str] = field(default_factory=list)
    superseded_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        promoted_ids, promoted_omitted = self._bounded_ids(self.promoted_ids)
        rejected_ids, rejected_omitted = self._bounded_ids(self.rejected_ids)
        archived_ids, archived_omitted = self._bounded_ids(self.archived_ids)
        superseded_ids, superseded_omitted = self._bounded_ids(self.superseded_ids)
        return {
            "considered": self.considered,
            "promoted": self.promoted,
            "rejected": self.rejected,
            "archived_only": self.archived_only,
            "superseded": self.superseded,
            "skipped": self.skipped,
            "mutation_authorized": self.mutation_authorized,
            "promoted_ids": promoted_ids,
            "rejected_ids": rejected_ids,
            "archived_ids": archived_ids,
            "superseded_ids": superseded_ids,
            "omitted_ids": {
                "promoted": promoted_omitted,
                "rejected": rejected_omitted,
                "archived": archived_omitted,
                "superseded": superseded_omitted,
            },
        }

    @staticmethod
    def _bounded_ids(ids: List[str], limit: int = 50) -> tuple[List[str], int]:
        safe_limit = max(0, int(limit))
        values = [str(item) for item in ids]
        return values[:safe_limit], max(0, len(values) - safe_limit)


class RuleBasedConsolidator:
    """Low-compute v0 candidate promoter.

    This class intentionally does not perform semantic LLM summarization. It only
    promotes explicit, already-gated candidates and keeps source refs attached so
    later richer consolidation can audit or improve the canonical records.
    """

    def consolidate(
        self,
        store: MemoryV2Store,
        index: MemoryV2Index,
        *,
        authorize_mutation: bool = False,
        safe_auto_only: bool = False,
    ) -> ConsolidationReport:
        """Run an authorized pass, or return a non-mutating pending-candidate report."""
        if not authorize_mutation:
            pending = sum(
                candidate.gate_decision == GateDecision.PENDING
                for candidate in store.list_candidates()
            )
            return ConsolidationReport(
                considered=pending,
                skipped=pending,
                mutation_authorized=False,
            )
        # Local import avoids the module cycle: operations imports this class's
        # deterministic conversion helpers for manual promotion.
        from .operations import MemoryOperationService

        operation_id = f"op_{uuid.uuid4().hex}"
        operation_type = "rule_based_consolidation"
        with store.profile_lock():
            interrupted = MemoryOperationService(store, index)._interrupted_operations()
            if interrupted:
                raise ValidationError(
                    "interrupted operation blocks canonical mutation: "
                    f"{interrupted[0]['operation_id']}"
                )
            eligible_candidate_ids: set[str] | None = None
            if safe_auto_only:
                from .review import MemoryReviewQueue

                queue = MemoryReviewQueue(store).build(limit=500)
                eligible_candidate_ids = {
                    str(item["id"])
                    for item in queue.get("items") or []
                    if item.get("review_lane") == "probably_promotable"
                }
            store.append_operation_record(
                {
                    "operation_id": operation_id,
                    "type": operation_type,
                    "status": "prepared",
                    "actor": "rule_based_consolidator",
                    "before_ids": [],
                    "after_ids": [],
                }
            )
            try:
                report = self._consolidate_locked(
                    store,
                    index,
                    eligible_candidate_ids=eligible_candidate_ids,
                )
                report.mutation_authorized = True
            except Exception as exc:
                store.append_operation_record(
                    {
                        "operation_id": operation_id,
                        "type": operation_type,
                        "status": "recovery_required",
                        "actor": "rule_based_consolidator",
                        "reason": str(exc),
                        "before_ids": [],
                        "after_ids": [],
                        "metadata": {"error": str(exc)},
                    }
                )
                raise
            store.append_operation_record(
                {
                    "operation_id": operation_id,
                    "type": operation_type,
                    "status": "committed",
                    "actor": "rule_based_consolidator",
                    "before_ids": [],
                    "after_ids": report.promoted_ids + report.rejected_ids + report.archived_ids,
                }
            )
            return report

    def _consolidate_locked(
        self,
        store: MemoryV2Store,
        index: MemoryV2Index,
        *,
        eligible_candidate_ids: set[str] | None = None,
    ) -> ConsolidationReport:
        candidates = store.list_candidates()
        report = ConsolidationReport()
        updated_candidates: List[CandidateMemory] = []

        for candidate in candidates:
            if candidate.gate_decision != GateDecision.PENDING:
                updated_candidates.append(candidate)
                continue

            report.considered += 1
            if eligible_candidate_ids is not None and candidate.id not in eligible_candidate_ids:
                updated_candidates.append(candidate)
                report.skipped += 1
                continue
            if not candidate.source_refs and not self._should_reject(candidate) and not self._should_archive_only(candidate):
                rejected = self._with_decision(
                    candidate,
                    GateDecision.REJECTED,
                    "Semantic promotion requires at least one source_refs evidence id.",
                )
                store.append_rejected_candidate(rejected)
                updated_candidates.append(rejected)
                report.rejected += 1
                report.rejected_ids.append(candidate.id)
                index.index_candidate(rejected)
                self._audit(
                    store,
                    "consolidate_candidate_rejected",
                    reason=rejected.decision_reason,
                    source_refs=list(rejected.source_refs),
                    before_ids=[candidate.id],
                    after_ids=[candidate.id],
                    metadata={"gate_decision": "rejected"},
                )
                continue

            dangling_source_refs = self._dangling_source_refs(candidate, store, index)
            if dangling_source_refs and not self._should_reject(candidate) and not self._should_archive_only(candidate):
                rejected = self._with_decision(
                    candidate,
                    GateDecision.REJECTED,
                    f"Semantic promotion has dangling source_refs: {', '.join(dangling_source_refs)}.",
                )
                store.append_rejected_candidate(rejected)
                updated_candidates.append(rejected)
                report.rejected += 1
                report.rejected_ids.append(candidate.id)
                index.index_candidate(rejected)
                self._audit(
                    store,
                    "consolidate_candidate_rejected",
                    reason=rejected.decision_reason,
                    source_refs=list(rejected.source_refs),
                    before_ids=[candidate.id],
                    after_ids=[candidate.id],
                    metadata={"gate_decision": "rejected", "dangling_source_refs": dangling_source_refs},
                )
                continue

            if self._should_reject(candidate):
                rejected = self._with_decision(
                    candidate,
                    GateDecision.REJECTED,
                    "Procedure candidates require skill authoring/review; not promoted as semantic memory.",
                )
                store.append_rejected_candidate(rejected)
                updated_candidates.append(rejected)
                report.rejected += 1
                report.rejected_ids.append(candidate.id)
                index.index_candidate(rejected)
                self._audit(
                    store,
                    "consolidate_candidate_rejected",
                    reason=rejected.decision_reason,
                    source_refs=list(rejected.source_refs),
                    before_ids=[candidate.id],
                    after_ids=[candidate.id],
                    metadata={"gate_decision": "rejected", "destination": candidate.proposed_destination},
                )
                continue

            if self._is_open_loop_candidate(candidate):
                loop = store.upsert_open_loop(
                    {
                        "id": f"loop_{candidate.id}",
                        "text": candidate.claim,
                        "source_refs": candidate.source_refs,
                        "session_id": self._session_id_from_source(candidate),
                        "candidate_id": candidate.id,
                    }
                )
                archived = self._with_decision(
                    candidate,
                    GateDecision.ARCHIVED_ONLY,
                    f"Routed to working/open_loops.yaml as {loop['id']}; semantic promotion skipped.",
                )
                updated_candidates.append(archived)
                report.archived_only += 1
                report.archived_ids.append(candidate.id)
                index.index_open_loop(loop, file_path=store.open_loops_path)
                index.index_candidate(archived)
                self._audit(
                    store,
                    "consolidate_candidate_to_open_loop",
                    reason=archived.decision_reason,
                    source_refs=list(archived.source_refs),
                    before_ids=[candidate.id],
                    after_ids=[candidate.id, str(loop.get("id") or "")],
                    metadata={"gate_decision": "archived_only", "open_loop_id": str(loop.get("id") or "")},
                )
                continue

            if self._should_archive_only(candidate):
                archived = self._with_decision(
                    candidate,
                    GateDecision.ARCHIVED_ONLY,
                    f"Candidate belongs in {candidate.proposed_destination}; semantic promotion skipped in v0.",
                )
                updated_candidates.append(archived)
                report.archived_only += 1
                report.archived_ids.append(candidate.id)
                index.index_candidate(archived)
                self._audit(
                    store,
                    "consolidate_candidate_archived_only",
                    reason=archived.decision_reason,
                    source_refs=list(archived.source_refs),
                    before_ids=[candidate.id],
                    after_ids=[candidate.id],
                    metadata={"gate_decision": "archived_only", "destination": candidate.proposed_destination},
                )
                continue

            if self._is_project_card_candidate(candidate):
                card = self._merge_project_card(candidate, store)
                path = store.write_project_card(card)
                index.index_project_card(card, file_path=path)
                promoted = self._with_decision(candidate, GateDecision.PROMOTED, f"Promoted to ProjectCard {card.id}.")
                updated_candidates.append(promoted)
                index.index_candidate(promoted)
                report.promoted += 1
                report.promoted_ids.append(card.id)
                self._audit(
                    store,
                    "consolidate_candidate_to_project_card",
                    reason=promoted.decision_reason,
                    source_refs=list(promoted.source_refs),
                    before_ids=[candidate.id],
                    after_ids=[candidate.id, card.id],
                    metadata={"gate_decision": "promoted", "project_id": card.id},
                )
                continue

            item = self._memory_item_from_candidate(candidate)
            superseded = self._superseded_items_for(candidate, item, store)
            if superseded:
                item.supersedes = [old.id for old in superseded]
                for old in superseded:
                    old.status = MemoryStatus.SUPERSEDED
                    old.superseded_by = item.id
                    old.updated_at = utc_now_iso()
                    path = store.write_memory_item(old)
                    index.index_memory_item(old, file_path=path)
                    report.superseded += 1
                    report.superseded_ids.append(old.id)

            path = store.write_memory_item(item)
            index.index_memory_item(item, file_path=path)
            promoted = self._with_decision(candidate, GateDecision.PROMOTED, f"Promoted to canonical MemoryItem {item.id}.")
            updated_candidates.append(promoted)
            index.index_candidate(promoted)
            report.promoted += 1
            report.promoted_ids.append(item.id)
            self._audit(
                store,
                "consolidate_candidate_to_memory_item",
                reason=promoted.decision_reason,
                source_refs=list(promoted.source_refs),
                before_ids=[candidate.id] + [old.id for old in superseded],
                after_ids=[candidate.id, item.id],
                metadata={"gate_decision": "promoted", "memory_item_id": item.id, "superseded_ids": [old.id for old in superseded]},
            )

        store.rewrite_candidates(updated_candidates)
        return report

    @staticmethod
    def _audit(
        store: MemoryV2Store,
        operation_type: str,
        *,
        reason: str,
        source_refs: List[str],
        before_ids: List[str],
        after_ids: List[str],
        metadata: dict | None = None,
    ) -> None:
        store.append_operation_record(
            {
                "type": operation_type,
                "actor": "rule_based_consolidator",
                "reason": reason,
                "source_refs": RuleBasedConsolidator._unique(source_refs),
                "before_ids": RuleBasedConsolidator._unique(before_ids),
                "after_ids": RuleBasedConsolidator._unique(after_ids),
                "metadata": dict(metadata or {}),
            }
        )

    @staticmethod
    def _unique(values: List[str]) -> List[str]:
        seen: set[str] = set()
        unique: List[str] = []
        for value in values:
            text = str(value or "").strip()
            if text and text not in seen:
                unique.append(text)
                seen.add(text)
        return unique

    @staticmethod
    def _with_decision(candidate: CandidateMemory, decision: GateDecision, reason: str) -> CandidateMemory:
        data = candidate.to_dict()
        data["gate_decision"] = cast(GateDecision, decision).value
        data["decision_reason"] = reason
        return CandidateMemory.from_dict(data)

    @staticmethod
    def _should_reject(candidate: CandidateMemory) -> bool:
        candidate_type = cast(MemoryType, candidate.type).value
        return candidate_type == MemoryType.PROCEDURE_REF.value or candidate.proposed_destination == "skills"

    @staticmethod
    def _source_exists(store: MemoryV2Store, index: MemoryV2Index, source_id: str) -> bool:
        return store.source_ref_exists(source_id, index=index)

    @classmethod
    def _dangling_source_refs(cls, candidate: CandidateMemory, store: MemoryV2Store, index: MemoryV2Index) -> List[str]:
        missing: List[str] = []
        for source_id in candidate.source_refs:
            if not cls._source_exists(store, index, str(source_id)):
                missing.append(source_id)
        return missing

    @staticmethod
    def _should_archive_only(candidate: CandidateMemory) -> bool:
        destination = str(candidate.proposed_destination or "")
        return destination.startswith("working/") or destination.startswith("episodic/")

    @staticmethod
    def _is_open_loop_candidate(candidate: CandidateMemory) -> bool:
        destination = str(candidate.proposed_destination or "")
        reason = str(candidate.promotion_reason or "").lower()
        return destination == "working/open_loops.yaml" or reason.startswith("open_loop")

    @staticmethod
    def _session_id_from_source(candidate: CandidateMemory) -> str:
        # Raw event metadata can resolve this later; keep the open-loop record cheap and source-grounded.
        return ""

    @staticmethod
    def _is_project_card_candidate(candidate: CandidateMemory) -> bool:
        candidate_type = cast(MemoryType, candidate.type).value
        destination = str(candidate.proposed_destination or "")
        return candidate_type == MemoryType.PROJECT_STATE.value and destination.startswith("semantic/projects/")

    def _merge_project_card(self, candidate: CandidateMemory, store: MemoryV2Store) -> ProjectCard:
        project_id = self._project_id_for(candidate)
        existing = store.read_project_card(project_id)
        card = existing or ProjectCard(
            id=project_id,
            name=self._project_name_for(project_id),
            importance=candidate.importance,
            updated_at="",
        )
        update_kind = self._project_update_kind(candidate)
        update_text = self._project_update_text(candidate, update_kind)
        evidence_time = self._project_evidence_time(candidate, store)
        evidence = self._bootstrap_project_evidence(card)
        if update_kind in {"next_action_resolved", "next_action_stale"}:
            lifecycle = "resolved" if update_kind.endswith("resolved") else "stale"
            self._close_project_values(
                evidence,
                "next_actions",
                update_text,
                lifecycle,
                evidence_time,
                candidate,
            )
        else:
            field_name = {
                "goal": "goal",
                "why_it_matters": "why_it_matters",
                "decision": "decisions",
                "open_question": "open_questions",
                "next_action": "next_actions",
                "status": "status",
                "current_state": "current_state",
            }.get(update_kind, "current_state")
            if field_name == "status":
                update_text = self._project_status_from_text(update_text).value
            self._add_project_evidence(
                evidence,
                field_name,
                update_text,
                evidence_time,
                candidate,
            )
        self._apply_project_lifecycle(evidence, "next_actions")
        card.field_evidence = self._normalize_project_evidence(evidence)
        self._materialize_project_fields(card)
        card.importance = max(card.importance, candidate.importance)
        card.source_refs = sorted(
            {
                str(source_ref)
                for entries in card.field_evidence.values()
                for entry in entries
                for source_ref in (entry.get("source_refs") or [])
                if str(source_ref)
            }
        )
        observed_times = [
            str(entry.get("observed_at") or "")
            for entries in card.field_evidence.values()
            for entry in entries
            if entry.get("observed_at")
        ]
        card.updated_at = max(observed_times, key=self._timestamp_key, default=card.updated_at)
        return ProjectCard.from_dict(card.to_dict())

    @classmethod
    def _project_evidence_time(cls, candidate: CandidateMemory, store: MemoryV2Store) -> str:
        source_timestamps: List[str] = []
        for source_id in candidate.source_refs:
            source = store.read_source_ref(source_id)
            if source and source.observed_at:
                source_timestamps.append(str(source.observed_at))
        if source_timestamps:
            return max(source_timestamps, key=cls._timestamp_key)
        return str(candidate.created_at or utc_now_iso())

    @staticmethod
    def _timestamp_key(value: str) -> tuple[float, str]:
        text = str(value or "")
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp(), text
        except ValueError:
            return 0.0, text

    @staticmethod
    def _bootstrap_project_evidence(card: ProjectCard) -> dict[str, List[dict]]:
        if card.field_evidence:
            return {
                field_name: [dict(entry) for entry in entries]
                for field_name, entries in card.field_evidence.items()
            }
        evidence: dict[str, List[dict]] = {}
        values = {
            "goal": [card.goal],
            "why_it_matters": [card.why_it_matters],
            "current_state": [card.current_state],
            "status": [cast(ProjectStatus, card.status).value],
            "decisions": list(card.decisions),
            "open_questions": list(card.open_questions),
            "next_actions": list(card.next_actions),
        }
        for field_name, field_values in values.items():
            for value in field_values:
                text = str(value or "").strip()
                if not text:
                    continue
                evidence.setdefault(field_name, []).append(
                    {
                        "value": text,
                        "status": "current",
                        "observed_at": card.updated_at,
                        "source_refs": sorted(set(card.source_refs)),
                    }
                )
        return evidence

    @classmethod
    def _add_project_evidence(
        cls,
        evidence: dict[str, List[dict]],
        field_name: str,
        value: str,
        observed_at: str,
        candidate: CandidateMemory,
    ) -> None:
        text = str(value or "").strip()
        if not text:
            return
        entries = evidence.setdefault(field_name, [])
        for entry in entries:
            if (
                str(entry.get("value") or "") == text
                and str(entry.get("observed_at") or "") == observed_at
            ):
                entry["source_refs"] = sorted(
                    set(entry.get("source_refs") or []) | set(candidate.source_refs)
                )
                entry["candidate_id"] = min(
                    str(entry.get("candidate_id") or candidate.id), candidate.id
                )
                return
        entries.append(
            {
                "value": text,
                "status": "current",
                "observed_at": observed_at,
                "source_refs": sorted(set(candidate.source_refs)),
                "candidate_id": candidate.id,
            }
        )
        if field_name in {"goal", "why_it_matters", "current_state", "status"}:
            newest = max(
                entries,
                key=lambda entry: (
                    cls._timestamp_key(str(entry.get("observed_at") or "")),
                    str(entry.get("candidate_id") or ""),
                    str(entry.get("value") or ""),
                ),
            )
            for entry in entries:
                entry["status"] = "current" if entry is newest else "superseded"

    @staticmethod
    def _close_project_values(
        evidence: dict[str, List[dict]],
        field_name: str,
        target: str,
        lifecycle: str,
        observed_at: str,
        candidate: CandidateMemory,
    ) -> None:
        evidence.setdefault(field_name, []).append(
            {
                "value": str(target or "").strip() or "*",
                "status": lifecycle,
                "observed_at": observed_at,
                "source_refs": sorted(set(candidate.source_refs)),
                "candidate_id": candidate.id,
                "lifecycle_event": True,
            }
        )

    @classmethod
    def _apply_project_lifecycle(
        cls, evidence: dict[str, List[dict]], field_name: str
    ) -> None:
        entries = evidence.get(field_name, [])
        values = [entry for entry in entries if not entry.get("lifecycle_event")]
        events = sorted(
            (entry for entry in entries if entry.get("lifecycle_event")),
            key=lambda entry: (
                cls._timestamp_key(str(entry.get("observed_at") or "")),
                str(entry.get("candidate_id") or ""),
            ),
        )
        for entry in values:
            entry["status"] = "current"
            for key in (
                "resolved_at",
                "resolved_source_refs",
                "resolved_candidate_id",
                "stale_at",
                "stale_source_refs",
                "stale_candidate_id",
            ):
                entry.pop(key, None)
        for event in events:
            needle = re.sub(r"\s+", " ", str(event.get("value") or "").lower())
            event_time = str(event.get("observed_at") or "")
            lifecycle = str(event.get("status") or "resolved")
            for entry in values:
                haystack = re.sub(r"\s+", " ", str(entry.get("value") or "").lower())
                if cls._timestamp_key(str(entry.get("observed_at") or "")) > cls._timestamp_key(event_time):
                    continue
                if needle not in {"", "*"} and needle not in haystack and haystack not in needle:
                    continue
                entry["status"] = lifecycle
                entry[f"{lifecycle}_at"] = event_time
                entry[f"{lifecycle}_source_refs"] = list(event.get("source_refs") or [])
                entry[f"{lifecycle}_candidate_id"] = str(event.get("candidate_id") or "")

    @staticmethod
    def _normalize_project_evidence(evidence: dict[str, List[dict]]) -> dict[str, List[dict]]:
        return {
            field_name: sorted(
                (dict(entry) for entry in entries),
                key=lambda entry: (
                    str(entry.get("observed_at") or ""),
                    str(entry.get("candidate_id") or ""),
                    str(entry.get("value") or ""),
                ),
            )
            for field_name, entries in sorted(evidence.items())
        }

    @staticmethod
    def _materialize_project_fields(card: ProjectCard) -> None:
        def current_values(field_name: str) -> List[str]:
            return [
                str(entry.get("value") or "")
                for entry in card.field_evidence.get(field_name, [])
                if str(entry.get("status") or "") == "current"
                and not entry.get("lifecycle_event")
                and str(entry.get("value") or "")
            ]

        card.goal = (current_values("goal") or [""])[-1]
        card.why_it_matters = (current_values("why_it_matters") or [""])[-1]
        card.current_state = (current_values("current_state") or [""])[-1]
        status = (current_values("status") or [cast(ProjectStatus, card.status).value])[-1]
        card.status = ProjectStatus.coerce(status, "status")
        card.decisions = current_values("decisions")
        card.open_questions = current_values("open_questions")
        card.next_actions = current_values("next_actions")

    @staticmethod
    def _project_id_for(candidate: CandidateMemory) -> str:
        destination = str(candidate.proposed_destination or "")
        match = re.search(r"semantic/projects/([^/]+?)(?:\.ya?ml)?$", destination)
        if match:
            return normalize_project_id(match.group(1))
        match = re.search(r"\bproject\s+(.+?)\s+(?:(?:resolved|stale|completed)\s+)?(?:current state|decision|open question|next action|status|goal|why it matters)(?:\s+(?:resolved|stale|completed))?\s*:", candidate.claim, re.IGNORECASE)
        if match:
            return normalize_project_id(match.group(1))
        return normalize_project_id("project")

    @staticmethod
    def _project_name_for(project_id: str) -> str:
        slug = normalize_project_id(project_id).split(":", 1)[1]
        return " ".join(part.upper() if part in {"v2", "api", "ui"} else part.capitalize() for part in slug.split("-"))

    @staticmethod
    def _project_update_kind(candidate: CandidateMemory) -> str:
        text = f"{candidate.promotion_reason}\n{candidate.claim}".lower()
        if "next action" in text or "next_action" in text:
            if any(term in text for term in ("resolved", "completed", "done")):
                return "next_action_resolved"
            if any(term in text for term in ("stale", "obsolete", "cancelled", "canceled")):
                return "next_action_stale"
        if "open_question" in text or "open question" in text:
            return "open_question"
        if "next_action" in text or "next action" in text:
            return "next_action"
        if "why_it_matters" in text or "why it matters" in text:
            return "why_it_matters"
        for kind in ("decision", "status", "goal"):
            if re.search(rf"\b{kind}\b", text):
                return kind
        return "current_state"

    @staticmethod
    def _project_update_text(candidate: CandidateMemory, update_kind: str) -> str:
        if update_kind in {"next_action_resolved", "next_action_stale"}:
            patterns = [
                r"^\s*project\s+.+?\s+(?:resolved|completed|stale|obsolete|cancelled|canceled)\s+next\s+action\s*:\s*(.*?)\s*$",
                r"^\s*project\s+.+?\s+next\s+action\s+(?:resolved|completed|stale|obsolete|cancelled|canceled)\s*:\s*(.*?)\s*$",
                r"^\s*(?:resolved|completed|stale|obsolete|cancelled|canceled)\s+next\s+action\s*:\s*(.*?)\s*$",
            ]
            for pattern in patterns:
                match = re.match(pattern, candidate.claim, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            return ""
        label = update_kind.replace("_", r"[ _]")
        patterns = [
            rf"^\s*project\s+.+?\s+{label}\s*:\s*(.+?)\s*$",
            rf"^\s*{label}\s*:\s*(.+?)\s*$",
        ]
        if update_kind == "current_state":
            patterns.insert(0, r"^\s*project\s+.+?\s+current\s+state\s*:\s*(.+?)\s*$")
        for pattern in patterns:
            match = re.match(pattern, candidate.claim, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return candidate.claim.strip()

    @staticmethod
    def _project_status_from_text(text: str) -> ProjectStatus:
        lowered = text.lower()
        if "archive" in lowered:
            return ProjectStatus.ARCHIVED
        if "pause" in lowered:
            return ProjectStatus.PAUSED
        return ProjectStatus.ACTIVE

    @staticmethod
    def _append_unique(values: List[str], *new_values: str) -> List[str]:
        merged = list(values)
        seen = {value for value in merged}
        for value in new_values:
            text = str(value or "").strip()
            if text and text not in seen:
                merged.append(text)
                seen.add(text)
        return merged

    def _memory_item_from_candidate(self, candidate: CandidateMemory) -> MemoryItem:
        memory_type = cast(MemoryType, candidate.type).value
        subject, predicate = self._subject_predicate_for(candidate)
        item_id = self._memory_id(memory_type, subject, predicate, candidate.claim)
        return MemoryItem(
            id=item_id,
            type=memory_type,
            subject=subject,
            predicate=predicate,
            value=candidate.claim,
            body=candidate.claim,
            summary=candidate.claim,
            confidence=candidate.confidence,
            importance=candidate.importance,
            source_refs=list(candidate.source_refs),
            tags=[memory_type, "promoted_from_candidate", candidate.id],
        )

    @staticmethod
    def _subject_predicate_for(candidate: CandidateMemory) -> tuple[str, str]:
        memory_type = cast(MemoryType, candidate.type).value
        claim = candidate.claim.strip()
        if memory_type == MemoryType.PREFERENCE.value:
            subject_match = re.match(r"^([A-Z][\w.-]{1,40})\s+prefers\b", claim)
            subject = subject_match.group(1) if subject_match else "user"
            return subject, "prefers"
        if memory_type == MemoryType.ENVIRONMENT.value:
            return "Hermes runtime", "has_environment_fact"
        if memory_type == MemoryType.PROJECT_STATE.value:
            return "project", "has_current_state"
        if memory_type == MemoryType.EPISODE.value:
            return "session", "has_episode"
        return "memory", "states"

    @staticmethod
    def _memory_id(memory_type: str, subject: str, predicate: str, claim: str) -> str:
        normalized = " ".join([memory_type, subject, predicate, claim]).lower().encode("utf-8")
        digest = hashlib.sha256(normalized).hexdigest()[:12]
        return f"mem_{RuleBasedConsolidator._safe_slug(memory_type)}_{digest}"

    @staticmethod
    def _safe_slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        return slug or "item"

    def _superseded_items_for(self, candidate: CandidateMemory, new_item: MemoryItem, store: MemoryV2Store) -> List[MemoryItem]:
        if not self._candidate_requests_supersession(candidate):
            return []
        superseded: List[MemoryItem] = []
        new_type = cast(MemoryType, new_item.type).value
        candidate_tokens = self._supersession_tokens(candidate.claim)
        for item in store.list_memory_items(memory_type=new_type, status=MemoryStatus.ACTIVE.value):
            if item.id == new_item.id:
                continue
            if item.subject != new_item.subject or item.predicate != new_item.predicate:
                continue
            item_text = " ".join(str(part or "") for part in (item.value, item.summary, item.body, " ".join(item.tags)))
            if candidate_tokens.intersection(self._supersession_tokens(item_text)):
                superseded.append(item)
        return superseded

    @staticmethod
    def _supersession_tokens(text: str) -> set[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "alex",
            "for",
            "i",
            "is",
            "it",
            "not",
            "now",
            "prefers",
            "prefer",
            "the",
            "to",
            "user",
            "with",
        }
        return {token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text.lower()) if token not in stopwords}

    @staticmethod
    def _candidate_requests_supersession(candidate: CandidateMemory) -> bool:
        text = f"{candidate.promotion_reason}\n{candidate.claim}".lower()
        return any(marker in text for marker in ("supersede_existing", " no longer ", " not ", " instead of "))
