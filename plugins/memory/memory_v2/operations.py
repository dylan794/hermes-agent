"""Audited state transitions for Memory v2 canonical records.

This module is the small "mutation kernel" for Memory v2.  Higher-level
consolidators and tools should route durable state changes through here instead
of hand-editing candidates, memory items, and working-memory records in several
places.  The goals are deliberately boring and important:

* validate lifecycle invariants before writing records;
* keep source-grounding gates near promotion logic;
* write an append-only audit event for every successful mutation;
* keep index updates colocated with canonical file writes.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .consolidation import RuleBasedConsolidator
from .index import MemoryV2Index
from .schemas import CandidateMemory, GateDecision, MemoryItem, MemoryStatus, ValidationError, utc_now_iso
from .store import MemoryV2Store


@dataclass
class MemoryOperationResult:
    """Result returned by one audited Memory v2 operation."""

    success: bool
    operation_id: str = ""
    operation_type: str = ""
    error: str = ""
    ids: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "success": self.success,
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "ids": list(self.ids),
        }
        if self.error:
            data["error"] = self.error
        data.update(dict(self.payload))
        return data


class MemoryOperationService:
    """State-machine layer for mutating Memory v2 records."""

    def __init__(self, store: MemoryV2Store, index: MemoryV2Index) -> None:
        self.store = store
        self.index = index

    def reject_candidate(self, candidate_id: str, reason: str, *, actor: str = "manual_tool") -> MemoryOperationResult:
        candidate_id = str(candidate_id or "").strip()
        reason = str(reason or "").strip()
        if not candidate_id:
            return self._error("reject_candidate", "candidate_id is required")
        if not reason:
            return self._error("reject_candidate", "reason is required")

        operation_id = f"op_{uuid.uuid4().hex}"
        operation_type = "reject_candidate"
        canonical_write_started = False
        target: Optional[CandidateMemory] = None
        try:
            with self.store.profile_lock():
                interrupted = self._interrupted_operations()
                if interrupted:
                    return self._error(operation_type, f"interrupted operation blocks canonical mutation: {interrupted[0]['operation_id']}")
                candidates = self.store.list_candidates()
                target = next((candidate for candidate in candidates if candidate.id == candidate_id), None)
                if target is None:
                    return self._error(operation_type, f"candidate not found: {candidate_id}")
                if target.gate_decision == GateDecision.REJECTED:
                    return MemoryOperationResult(
                        success=True,
                        operation_type=operation_type,
                        ids=[candidate_id],
                        payload={"already_rejected": True, "candidate": target.to_dict()},
                    )
                self._audit(
                    operation_type,
                    operation_id=operation_id,
                    status="prepared",
                    actor=actor,
                    reason=reason,
                    source_refs=list(target.source_refs),
                    before_ids=[candidate_id],
                    after_ids=[candidate_id],
                    metadata={"gate_decision": "rejected"},
                )
                self._maybe_failpoint("after_operation_prepare")
                rejected = self._candidate_with_decision(target, GateDecision.REJECTED, reason)
                updated_candidates = [rejected if candidate.id == candidate_id else candidate for candidate in candidates]
                canonical_write_started = True
                self.store.rewrite_candidates(updated_candidates)
                self._maybe_failpoint("after_reject_candidates_rewrite")
                self.store.append_rejected_candidate(rejected)
                self.index.index_candidate(rejected)
                self._audit(
                    operation_type,
                    operation_id=operation_id,
                    status="committed",
                    actor=actor,
                    reason=reason,
                    source_refs=list(rejected.source_refs),
                    before_ids=[candidate_id],
                    after_ids=[candidate_id],
                    metadata={"gate_decision": "rejected"},
                )
        except Exception as exc:
            self._audit(
                operation_type,
                operation_id=operation_id,
                status="recovery_required" if canonical_write_started else "failed",
                actor=actor,
                reason=reason,
                source_refs=list(getattr(target, "source_refs", [])),
                before_ids=[candidate_id],
                after_ids=[candidate_id],
                metadata={"error": str(exc)},
            )
            return self._error(operation_type, str(exc))
        return MemoryOperationResult(
            success=True,
            operation_id=operation_id,
            operation_type=operation_type,
            ids=[candidate_id],
            payload={"candidate": rejected.to_dict()},
        )

    def promote_candidate(
        self,
        candidate_id: str,
        *,
        force: bool = False,
        force_reason: str = "",
        session_id: str = "",
        actor: str = "manual_tool",
    ) -> MemoryOperationResult:
        candidate_id = str(candidate_id or "").strip()
        if not candidate_id:
            return self._error("promote_candidate", "candidate_id is required")
        consolidator = RuleBasedConsolidator()
        promoted_ids: List[str] = []
        superseded_ids: List[str] = []
        operation_id = f"op_{uuid.uuid4().hex}"
        operation_type = "promote_candidate"
        canonical_write_started = False
        target: Optional[CandidateMemory] = None
        try:
            with self.store.profile_lock():
                interrupted = self._interrupted_operations()
                if interrupted:
                    return self._error(operation_type, f"interrupted operation blocks canonical mutation: {interrupted[0]['operation_id']}")
                candidates = self.store.list_candidates()
                target = next((candidate for candidate in candidates if candidate.id == candidate_id), None)
                if target is None:
                    return self._error(operation_type, f"candidate not found: {candidate_id}")
                if target.proposed_destination.strip().lower() == "skills" or str(getattr(target.type, "value", target.type)) == "procedure_ref":
                    return self._error(
                        operation_type,
                        "procedure/skills candidates require skill authoring or manual rejection, not semantic promotion",
                    )
                if target.gate_decision != GateDecision.PENDING:
                    decision = getattr(target.gate_decision, "value", str(target.gate_decision))
                    return self._error(
                        operation_type,
                        f"candidate is already {decision}; only pending candidates can be promoted",
                        payload={"candidate": target.to_dict()},
                    )
                gated = self._validate_candidate_sources(target, force=force, force_reason=force_reason)
                if not gated.success:
                    return gated
                target = CandidateMemory.from_dict(gated.payload["candidate"])
                if consolidator._is_open_loop_candidate(target):
                    operation_type = "route_candidate_to_open_loop"
                    self._audit(operation_type, operation_id=operation_id, status="prepared", actor=actor, reason=target.claim, source_refs=list(target.source_refs), before_ids=[candidate_id], after_ids=[candidate_id], metadata={"candidate_id": candidate_id})
                    existing_loop = next((loop for loop in self.store.list_open_loops() if loop.get("candidate_id") == target.id), None)
                    canonical_write_started = True
                    loop = existing_loop or self.store.upsert_open_loop(
                        {"text": target.claim, "source_refs": target.source_refs, "session_id": session_id, "candidate_id": target.id}
                    )
                    self._maybe_failpoint("after_promote_open_loop_write")
                    self.index.index_open_loop(loop, file_path=self.store.open_loops_path)
                    updated_target = self._candidate_with_decision(
                        target, GateDecision.ARCHIVED_ONLY, f"Manually routed to working/open_loops.yaml as {loop['id']}."
                    )
                    promoted_ids.append(str(loop["id"]))
                elif consolidator._is_project_card_candidate(target):
                    operation_type = "promote_candidate_to_project_card"
                    self._audit(operation_type, operation_id=operation_id, status="prepared", actor=actor, reason=target.claim, source_refs=list(target.source_refs), before_ids=[candidate_id], after_ids=[candidate_id], metadata={"candidate_id": candidate_id})
                    card = consolidator._merge_project_card(target, self.store)
                    canonical_write_started = True
                    path = self.store.write_project_card(card)
                    self._maybe_failpoint("after_promote_project_card_write")
                    self.index.index_project_card(card, file_path=path)
                    updated_target = self._candidate_with_decision(target, GateDecision.PROMOTED, f"Manually promoted to ProjectCard {card.id}.")
                    promoted_ids.append(card.id)
                else:
                    operation_type = "promote_candidate_to_memory_item"
                    item = consolidator._memory_item_from_candidate(target)
                    superseded = consolidator._superseded_items_for(target, item, self.store)
                    after_ids_preview = [item.id, candidate_id]
                    self._audit(operation_type, operation_id=operation_id, status="prepared", actor=actor, reason=target.claim, source_refs=list(target.source_refs), before_ids=[candidate_id] + [old.id for old in superseded], after_ids=after_ids_preview, metadata={"candidate_id": candidate_id})
                    if superseded:
                        item.supersedes = [old.id for old in superseded]
                        for old in superseded:
                            self._apply_supersession_fields(
                                old,
                                superseded_by=item.id,
                                reason=f"Superseded by candidate promotion {target.id}.",
                                tag="candidate_superseded",
                            )
                            canonical_write_started = True
                            path = self.store.write_memory_item(old)
                            self._maybe_failpoint("after_promote_memory_item_write")
                            self.index.index_memory_item(old, file_path=path)
                            superseded_ids.append(old.id)
                    canonical_write_started = True
                    path = self.store.write_memory_item(item)
                    self._maybe_failpoint("after_promote_memory_item_write")
                    self.index.index_memory_item(item, file_path=path)
                    updated_target = self._candidate_with_decision(target, GateDecision.PROMOTED, f"Manually promoted to MemoryItem {item.id}.")
                    promoted_ids.append(item.id)

                self._validate_memory_items_invariants()
                updated_candidates = [updated_target if candidate.id == candidate_id else candidate for candidate in candidates]
                self.store.rewrite_candidates(updated_candidates)
                self.index.index_candidate(updated_target)
                op = self._audit(
                    operation_type,
                    operation_id=operation_id,
                    status="committed",
                    actor=actor,
                    reason=updated_target.decision_reason,
                    source_refs=list(updated_target.source_refs),
                    before_ids=[candidate_id] + superseded_ids,
                    after_ids=promoted_ids + [candidate_id],
                    metadata={"candidate_id": candidate_id, "promoted_ids": promoted_ids, "superseded_ids": superseded_ids},
                )
        except Exception as exc:
            self._audit(operation_type, operation_id=operation_id, status="recovery_required" if canonical_write_started else "failed", actor=actor, reason=str(exc), source_refs=list(getattr(target, "source_refs", [])), before_ids=[candidate_id] + superseded_ids, after_ids=promoted_ids + [candidate_id], metadata={"error": str(exc)})
            return self._error(operation_type, str(exc))
        return MemoryOperationResult(
            success=True,
            operation_id=op["operation_id"],
            operation_type=operation_type,
            ids=promoted_ids,
            payload={
                "promoted": len(promoted_ids),
                "promoted_ids": promoted_ids,
                "superseded_ids": superseded_ids,
                "candidate": updated_target.to_dict(),
            },
        )

    def supersede_memory(
        self,
        old_id: str,
        new_id: str,
        *,
        reason: str,
        actor: str = "manual_tool",
        tag: str = "superseded",
    ) -> MemoryOperationResult:
        old_id = str(old_id or "").strip()
        new_id = str(new_id or "").strip()
        reason = str(reason or "").strip()
        if not old_id or not new_id:
            return self._error("supersede_memory", "old_id and new_id are required")
        if old_id == new_id:
            return self._error("supersede_memory", "old_id and new_id must differ")
        if not reason:
            return self._error("supersede_memory", "reason is required")
        old_item = self.store.read_memory_item(old_id)
        new_item = self.store.read_memory_item(new_id)
        if old_item is None or new_item is None:
            return self._error("supersede_memory", f"memory item not found: {old_id if old_item is None else new_id}")
        self._apply_supersession_fields(old_item, superseded_by=new_id, reason=reason, tag=tag)
        if old_id not in new_item.supersedes:
            new_item.supersedes.append(old_id)
        new_item.updated_at = utc_now_iso()
        old_path = self.store.write_memory_item(old_item)
        new_path = self.store.write_memory_item(new_item)
        self.index.index_memory_item(old_item, file_path=old_path)
        self.index.index_memory_item(new_item, file_path=new_path)
        self._validate_memory_items_invariants()
        op = self._audit(
            "supersede_memory",
            actor=actor,
            reason=reason,
            source_refs=self._unique(list(old_item.source_refs) + list(new_item.source_refs)),
            before_ids=[old_id, new_id],
            after_ids=[old_id, new_id],
            metadata={"superseded_id": old_id, "superseded_by": new_id},
        )
        return MemoryOperationResult(
            success=True,
            operation_id=op["operation_id"],
            operation_type="supersede_memory",
            ids=[old_id, new_id],
            payload={"superseded_id": old_id, "superseded_by": new_id, "reason": reason},
        )

    def resolve_open_loop(
        self,
        loop_id: str,
        status: str,
        *,
        resolution: str = "",
        actor: str = "manual_tool",
    ) -> MemoryOperationResult:
        loop_id = str(loop_id or "").strip()
        status = str(status or "").strip()
        resolution = str(resolution or "").strip()
        allowed = {"open", "resolved", "abandoned", "blocked", "snoozed"}
        if not loop_id:
            return self._error("resolve_open_loop", "loop_id is required")
        if status not in allowed:
            return self._error("resolve_open_loop", f"status must be one of: {sorted(allowed)}")
        loops = self.store.list_open_loops()
        updated_loop: Optional[Dict[str, Any]] = None
        now = utc_now_iso()
        for loop in loops:
            if loop.get("id") == loop_id:
                history = list(loop.get("history") or [])
                history.append(
                    {
                        "updated_at": now,
                        "from_status": str(loop.get("status") or ""),
                        "to_status": status,
                        "resolution": resolution,
                        "actor": actor,
                    }
                )
                loop["status"] = status
                loop["updated_at"] = now
                loop["history"] = history
                if resolution:
                    loop["resolution"] = resolution
                if status in {"resolved", "abandoned"}:
                    loop["resolved_at"] = now
                updated_loop = loop
                break
        if updated_loop is None:
            return self._error("resolve_open_loop", f"open loop not found: {loop_id}")
        self.store.write_open_loops(loops)
        self.index.index_open_loop(updated_loop, file_path=self.store.open_loops_path)
        op = self._audit(
            "resolve_open_loop",
            actor=actor,
            reason=resolution or f"status changed to {status}",
            source_refs=[str(ref) for ref in updated_loop.get("source_refs") or []],
            before_ids=[loop_id],
            after_ids=[loop_id],
            metadata={"status": status},
        )
        return MemoryOperationResult(
            success=True,
            operation_id=op["operation_id"],
            operation_type="resolve_open_loop",
            ids=[loop_id],
            payload={"loop": updated_loop},
        )

    def _validate_candidate_sources(self, candidate: CandidateMemory, *, force: bool, force_reason: str) -> MemoryOperationResult:
        if force:
            force_reason = str(force_reason or "").strip()
            if not force_reason:
                return self._error("promote_candidate", "force_reason is required when force=true")
            valid_refs = [source_id for source_id in candidate.source_refs if self._source_exists(source_id)]
            if valid_refs != list(candidate.source_refs):
                data = candidate.to_dict()
                data["source_refs"] = valid_refs
                data["confidence"] = min(float(data.get("confidence") or 0.0), 0.5)
                candidate = CandidateMemory.from_dict(data)
            return MemoryOperationResult(success=True, payload={"candidate": candidate.to_dict()})
        if not candidate.source_refs:
            return self._error("promote_candidate", "candidate source_refs are required for manual promotion")
        dangling = [source_id for source_id in candidate.source_refs if not self._source_exists(source_id)]
        if dangling:
            return self._error("promote_candidate", f"candidate has dangling source_refs: {dangling}")
        return MemoryOperationResult(success=True, payload={"candidate": candidate.to_dict()})

    def _source_exists(self, source_id: str) -> bool:
        return self.store.source_ref_exists(source_id, index=self.index)

    def _validate_memory_items_invariants(self) -> None:
        for item in self.store.list_memory_items():
            status = getattr(item.status, "value", str(item.status))
            if status == MemoryStatus.ACTIVE.value and item.superseded_by:
                raise ValidationError(f"active memory {item.id} cannot set superseded_by")
            if status == MemoryStatus.SUPERSEDED.value and not item.superseded_by:
                raise ValidationError(f"superseded memory {item.id} requires superseded_by")
            if item.superseded_by and self.store.read_memory_item(item.superseded_by) is None:
                raise ValidationError(f"memory {item.id} superseded_by target is missing: {item.superseded_by}")

    @staticmethod
    def _apply_supersession_fields(item: MemoryItem, *, superseded_by: str, reason: str, tag: str) -> None:
        now = utc_now_iso()
        item.status = MemoryStatus.SUPERSEDED
        item.superseded_by = superseded_by
        item.superseded_at = now
        item.supersession_reason = reason
        item.updated_at = now
        if tag and tag not in item.tags:
            item.tags.append(tag)

    @staticmethod
    def _candidate_with_decision(candidate: CandidateMemory, decision: GateDecision, reason: str) -> CandidateMemory:
        data = candidate.to_dict()
        data["gate_decision"] = decision.value
        data["decision_reason"] = reason
        return CandidateMemory.from_dict(data)

    def _audit(
        self,
        operation_type: str,
        *,
        actor: str,
        reason: str,
        source_refs: List[str],
        before_ids: List[str],
        after_ids: List[str],
        metadata: Dict[str, Any] | None = None,
        operation_id: str = "",
        status: str = "committed",
    ) -> Dict[str, Any]:
        operation = {
            "operation_id": operation_id or f"op_{uuid.uuid4().hex}",
            "type": operation_type,
            "operation": operation_type,
            "status": str(status or "committed"),
            "actor": str(actor or ""),
            "reason": str(reason or ""),
            "source_refs": self._unique(source_refs),
            "before_ids": self._unique(before_ids),
            "after_ids": self._unique(after_ids),
            "metadata": dict(metadata or {}),
            "created_at": utc_now_iso(),
        }
        self.store.append_operation_record(operation)
        return operation

    def _interrupted_operations(self) -> List[Dict[str, Any]]:
        latest: Dict[str, Dict[str, Any]] = {}
        for record in self.store.list_operation_records():
            operation_id = str(record.get("operation_id") or "").strip()
            if operation_id:
                latest[operation_id] = record
        return [record for record in latest.values() if str(record.get("status") or "") in {"prepared", "recovery_required"}]

    @staticmethod
    def _maybe_failpoint(name: str) -> None:
        if os.environ.get("MEMORY_V2_FAILPOINT") == name:
            raise RuntimeError(f"Memory v2 failpoint triggered: {name}")

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
    def _error(operation_type: str, error: str, *, payload: Dict[str, Any] | None = None) -> MemoryOperationResult:
        return MemoryOperationResult(success=False, operation_type=operation_type, error=error, payload=dict(payload or {}))
