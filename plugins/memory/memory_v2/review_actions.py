"""Action planning/apply workflow for Memory v2 candidate review.

The queue stays read-only. This module adds a two-phase workflow:
plan first, then explicitly apply selected action ids with confirmation. Durable
mutations still go through MemoryOperationService.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from .operations import MemoryOperationService
from .redaction import redact_text
from .review import MemoryReviewQueue
from .schemas import GateDecision
from .store import MemoryV2Store

CONFIRM_REVIEW_APPLY = "APPLY_MEMORY_V2_REVIEW_PLAN"


class MemoryReviewPlanner:
    """Build a deterministic, non-mutating review action plan."""

    def __init__(self, store: MemoryV2Store) -> None:
        self.store = store

    def build(self, *, max_actions: int = 20, candidate_ids: List[str] | None = None) -> Dict[str, Any]:
        safe_max = max(1, min(int(max_actions), 100))
        explicit_ids = {str(candidate_id) for candidate_id in (candidate_ids or []) if str(candidate_id).strip()}
        queue = MemoryReviewQueue(self.store).build(limit=500)
        items = [item for item in queue["items"] if not explicit_ids or item["id"] in explicit_ids]
        lane_order = {"probably_promotable": 0, "reject_or_archive": 1}
        items.sort(key=lambda item: (lane_order.get(str(item.get("review_lane") or ""), 9), str(item["id"])))

        actions: List[Dict[str, Any]] = []
        blocked: List[Dict[str, Any]] = []
        for item in items:
            lane = str(item.get("review_lane") or "")
            candidate_id = str(item["id"])
            blockers = self._blockers_for_item(item)
            hard_blockers = {
                "missing_or_dangling_source",
                "likely_duplicate",
                "possible_contradiction",
                "skill_candidate",
                "adversarial_or_policy_bait",
                "secret_or_config_bait",
                "unsafe_identifier",
            }
            if hard_blockers & set(blockers):
                blocked.append({"candidate_id": candidate_id, "blockers": blockers})
            elif lane == "probably_promotable" and len(actions) < safe_max:
                actions.append(self._action(len(actions) + 1, item, operation="promote_candidate"))
            elif lane == "reject_or_archive" and len(actions) < safe_max:
                actions.append(
                    self._action(
                        len(actions) + 1,
                        item,
                        operation="reject_candidate",
                        reason="ephemeral_or_task_local",
                    )
                )
            else:
                blocked.append({"candidate_id": candidate_id, "blockers": blockers})

        body = {
            "actions": actions,
            "blocked": blocked,
            "candidate_state": self._candidate_state_payload(),
        }
        plan_id = "plan_" + hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return {
            "success": True,
            "mode": "dry_run_review_plan",
            "untrusted_text": True,
            "recommendation_policy": "triage_only_validate_sources_before_mutation",
            "plan_id": plan_id,
            "summary": {
                "proposed_promotions": sum(1 for action in actions if action["operation"] == "promote_candidate"),
                "proposed_rejections": sum(1 for action in actions if action["operation"] == "reject_candidate"),
                "blocked": len(blocked),
            },
            "actions": actions,
            "blocked": blocked,
            "apply_contract": {
                "tool": "memory_v2_review_apply",
                "requires_plan_id": True,
                "requires_action_ids": True,
                "requires_confirm": True,
                "confirm_value": CONFIRM_REVIEW_APPLY,
            },
        }

    def _action(self, ordinal: int, item: Dict[str, Any], *, operation: str, reason: str = "") -> Dict[str, Any]:
        candidate = self._candidate_payload(str(item["id"]))
        return {
            "action_id": f"act_{ordinal:03d}",
            "candidate_id": str(item["id"]),
            "operation": operation,
            "eligible": True,
            "requires_source_confirmation": operation == "promote_candidate",
            "reason": reason or "source-backed low-risk review candidate",
            "source_refs": list(item.get("source_refs") or []),
            "source_check": self._source_check(list(item.get("source_refs") or [])),
            "candidate_fingerprint": self._candidate_fingerprint(candidate),
        }

    def _candidate_payload(self, candidate_id: str) -> Dict[str, Any]:
        for candidate in self.store.list_candidates():
            if candidate.id == candidate_id:
                return candidate.to_dict()
        return {}

    def _candidate_state_payload(self) -> List[Dict[str, Any]]:
        return [candidate.to_dict() for candidate in self.store.list_candidates()]

    @staticmethod
    def _candidate_fingerprint(candidate: Dict[str, Any]) -> str:
        fields = {
            "id": candidate.get("id"),
            "type": candidate.get("type"),
            "claim": candidate.get("claim"),
            "proposed_destination": candidate.get("proposed_destination"),
            "confidence": candidate.get("confidence"),
            "source_refs": candidate.get("source_refs") or [],
            "gate_decision": candidate.get("gate_decision"),
        }
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def _source_check(self, source_refs: List[str]) -> Dict[str, Any]:
        """Return source metadata only; review plans never hydrate raw archive text."""
        sources = []
        for source_id in source_refs:
            source = self.store.read_source_ref(source_id)
            payload = source.to_dict() if source is not None else {}
            uri = str(payload.get("uri") or "").strip().lower()
            raw_evidence = uri.startswith("raw_event:")
            sources.append(
                {
                    "id": str(source_id),
                    "exists": source is not None,
                    "raw_evidence_omitted": raw_evidence,
                    "quote": "" if raw_evidence else self._bounded(redact_text(str(payload.get("quote") or ""))),
                }
            )
        return {
            "all_refs_exist": bool(source_refs) and all(source["exists"] for source in sources),
            "source_count": len(source_refs),
            "sources": sources,
        }

    @staticmethod
    def _bounded(text: str, limit: int = 240) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _blockers_for_item(item: Dict[str, Any]) -> List[str]:
        flags = item.get("flags") or {}
        blockers: List[str] = []
        if flags.get("missing_or_dangling_source"):
            blockers.append("missing_or_dangling_source")
        if flags.get("likely_duplicate"):
            blockers.append("likely_duplicate")
        if flags.get("possible_contradictions"):
            blockers.append("possible_contradiction")
        if flags.get("skill_candidate"):
            blockers.append("skill_candidate")
        if flags.get("adversarial_or_policy_bait"):
            blockers.append("adversarial_or_policy_bait")
        if flags.get("secret_or_config_bait"):
            blockers.append("secret_or_config_bait")
        if flags.get("unsafe_identifier"):
            blockers.append("unsafe_identifier")
        if flags.get("low_confidence"):
            blockers.append("low_confidence")
        if not blockers:
            blockers.append(str(item.get("review_lane") or "inspect_manually"))
        return blockers


class MemoryReviewApplier:
    """Apply explicit selected actions from the current deterministic plan."""

    def __init__(self, store: MemoryV2Store, operations: MemoryOperationService) -> None:
        self.store = store
        self.operations = operations

    def apply(
        self,
        *,
        plan_id: str,
        action_ids: List[str],
        confirm: str,
        dry_run: bool = True,
        max_actions: int = 20,
        candidate_ids: List[str] | None = None,
        allow_promotions: bool = False,
        mode: str = "apply_review_plan",
    ) -> Dict[str, Any]:
        plan_id = str(plan_id or "").strip()
        if str(confirm or "") != CONFIRM_REVIEW_APPLY:
            return {"success": False, "error": f"confirm must equal {CONFIRM_REVIEW_APPLY}"}
        if not isinstance(action_ids, list) or not action_ids:
            return {"success": False, "error": "action_ids must be a non-empty list"}
        selected_ids = [str(action_id) for action_id in action_ids]
        if len(selected_ids) != len(set(selected_ids)):
            return {"success": False, "error": "action_ids must be unique"}

        plan = MemoryReviewPlanner(self.store).build(max_actions=max_actions, candidate_ids=candidate_ids)
        if plan.get("plan_id") != plan_id:
            return {"success": False, "error": "review plan is stale; regenerate memory_v2_review_plan"}

        actions_by_id = {action["action_id"]: action for action in plan["actions"]}
        missing = [action_id for action_id in selected_ids if action_id not in actions_by_id]
        if missing:
            return {"success": False, "error": f"unknown or blocked action_ids: {missing}"}

        validated: List[Dict[str, Any]] = []
        applied: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        service = self.operations
        for action_id in selected_ids:
            action = actions_by_id[action_id]
            candidate = self._candidate(action["candidate_id"])
            if candidate is None:
                failed.append({"action_id": action_id, "candidate_id": action["candidate_id"], "error": "candidate not found"})
                continue
            if candidate.gate_decision != GateDecision.PENDING:
                failed.append({"action_id": action_id, "candidate_id": candidate.id, "error": "candidate is no longer pending"})
                continue
            if MemoryReviewPlanner._candidate_fingerprint(candidate.to_dict()) != action["candidate_fingerprint"]:
                failed.append({"action_id": action_id, "candidate_id": candidate.id, "error": "candidate changed since plan"})
                continue
            if dry_run:
                validated.append({"action_id": action_id, "candidate_id": candidate.id, "operation": action["operation"]})
                continue
            if action["operation"] == "promote_candidate":
                if not allow_promotions:
                    failed.append({
                        "action_id": action_id,
                        "candidate_id": candidate.id,
                        "operation": action["operation"],
                        "error": "automatic promotion is disabled until stronger Memory v2 evals exist",
                    })
                    continue
                result = service.promote_candidate(candidate.id, actor="review_apply")
            elif action["operation"] == "reject_candidate":
                result = service.reject_candidate(candidate.id, action.get("reason") or "review_apply_reject", actor="review_apply")
            else:
                failed.append({"action_id": action_id, "candidate_id": candidate.id, "error": "unsupported operation"})
                continue
            if result.success:
                applied.append({"action_id": action_id, "candidate_id": candidate.id, **result.to_dict()})
            else:
                failed.append({"action_id": action_id, "candidate_id": candidate.id, **result.to_dict()})

        summary = {
            "attempted": len(selected_ids),
            "applied": len(applied),
            "skipped": len(validated) if dry_run else 0,
            "failed": len(failed),
        }
        return {
            "success": not failed,
            "mode": mode,
            "dry_run": dry_run,
            "plan_id": plan_id,
            "summary": summary,
            "validated": validated,
            "applied": applied,
            "failed": failed,
        }

    def _candidate(self, candidate_id: str):
        for candidate in self.store.list_candidates():
            if candidate.id == candidate_id:
                return candidate
        return None
