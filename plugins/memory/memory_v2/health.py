"""Health checks and dry-run repair planning for Memory v2."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .index import MemoryV2Index
from .schemas import GateDecision, MemoryStatus
from .store import MemoryV2Store


@dataclass
class MemoryHealthIssue:
    code: str
    severity: str
    message: str
    record_id: str = ""
    repair: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "record_id": self.record_id,
            "repair": self.repair,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


class MemoryHealthChecker:
    """Cheap deterministic checker for Memory v2 canonical state."""

    def __init__(self, store: MemoryV2Store, index: MemoryV2Index) -> None:
        self.store = store
        self.index = index

    def check(self) -> Dict[str, Any]:
        issues: List[MemoryHealthIssue] = []
        issues.extend(self._source_ref_issues())
        issues.extend(self._memory_lifecycle_issues())
        issues.extend(self._candidate_issues())
        issues.extend(self._raw_archive_issues())
        issues.extend(self._index_issues())
        severity_counts: Dict[str, int] = {}
        for issue in issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
        return {
            "success": True,
            "status": "healthy" if not issues else "degraded",
            "issue_count": len(issues),
            "severity_counts": severity_counts,
            "counts": self._counts(),
            "issues": [issue.to_dict() for issue in issues],
            "repair_plan": self.repair(dry_run=True)["actions"],
        }

    def repair(self, *, dry_run: bool = True) -> Dict[str, Any]:
        """Return safe repair actions; only rebuilds derived indexes when dry_run=false."""
        actions: List[Dict[str, Any]] = []
        expected_index_records = self._expected_index_records()
        indexed_records = self.index.count_memories()
        manifest = self.store.read_raw_archive_manifest()
        raw_event_count = self.store.count_raw_events()
        if int(manifest.get("event_count") or 0) != raw_event_count:
            action = {
                "action": "rebuild_raw_archive_manifest",
                "safe": True,
                "reason": "raw archive manifest differs from current archive scan",
                "mutates": "derived manifest only",
            }
            if not dry_run:
                action["result"] = self.store.rebuild_raw_archive_manifest()
            actions.append(action)
        if indexed_records != expected_index_records:
            action = {
                "action": "rebuild_index",
                "safe": True,
                "reason": f"index has {indexed_records} records but canonical store has {expected_index_records} indexable records",
                "mutates": "derived index only",
            }
            if not dry_run:
                action["result"] = self.index.rebuild_from_store(self.store)
            actions.append(action)
        for issue in self._memory_lifecycle_issues() + self._source_ref_issues() + self._candidate_issues() + self._raw_archive_issues():
            if issue.repair and issue.repair != "rebuild_index":
                actions.append(
                    {
                        "action": issue.repair,
                        "safe": False,
                        "reason": issue.message,
                        "record_id": issue.record_id,
                        "mutates": "canonical memory files; manual review required",
                    }
                )
        return {"success": True, "dry_run": dry_run, "actions": actions}

    def _source_ref_issues(self) -> List[MemoryHealthIssue]:
        issues: List[MemoryHealthIssue] = []
        existing_sources = self._existing_source_ids()
        for record_type, record_id, refs in self._records_with_source_refs():
            for source_id in refs:
                if str(source_id) not in existing_sources:
                    issues.append(
                        MemoryHealthIssue(
                            code="dangling_source_ref",
                            severity="error",
                            message=f"{record_type} {record_id} references missing source {source_id}",
                            record_id=record_id,
                            repair="manual_source_review",
                            details={"source_id": source_id, "record_type": record_type},
                        )
                    )
        return issues

    def _memory_lifecycle_issues(self) -> List[MemoryHealthIssue]:
        issues: List[MemoryHealthIssue] = []
        memory_ids = {item.id for item in self.store.list_memory_items()}
        for item in self.store.list_memory_items():
            status = getattr(item.status, "value", str(item.status))
            if status == MemoryStatus.ACTIVE.value and item.superseded_by:
                issues.append(
                    MemoryHealthIssue(
                        code="active_memory_has_superseded_by",
                        severity="error",
                        message=f"active memory {item.id} sets superseded_by={item.superseded_by}",
                        record_id=item.id,
                        repair="manual_lifecycle_review",
                    )
                )
            if status == MemoryStatus.SUPERSEDED.value and not item.superseded_by:
                issues.append(
                    MemoryHealthIssue(
                        code="superseded_memory_missing_target",
                        severity="error",
                        message=f"superseded memory {item.id} is missing superseded_by",
                        record_id=item.id,
                        repair="manual_lifecycle_review",
                    )
                )
            if item.superseded_by and item.superseded_by not in memory_ids:
                issues.append(
                    MemoryHealthIssue(
                        code="superseded_by_missing_target",
                        severity="error",
                        message=f"memory {item.id} superseded_by target is missing: {item.superseded_by}",
                        record_id=item.id,
                        repair="manual_lifecycle_review",
                        details={"superseded_by": item.superseded_by},
                    )
                )
            for old_id in item.supersedes:
                if old_id not in memory_ids:
                    issues.append(
                        MemoryHealthIssue(
                            code="supersedes_missing_target",
                            severity="warning",
                            message=f"memory {item.id} supersedes missing target {old_id}",
                            record_id=item.id,
                            repair="manual_lifecycle_review",
                            details={"supersedes": old_id},
                        )
                    )
        return issues

    def _candidate_issues(self) -> List[MemoryHealthIssue]:
        issues: List[MemoryHealthIssue] = []
        rejected_ids = {candidate.id for candidate in self.store.list_rejected_candidates()}
        candidate_ids: set[str] = set()
        for candidate in self.store.list_candidates():
            if candidate.id in candidate_ids:
                issues.append(
                    MemoryHealthIssue(
                        code="duplicate_candidate_id",
                        severity="error",
                        message=f"duplicate candidate id in inbox: {candidate.id}",
                        record_id=candidate.id,
                        repair="manual_candidate_review",
                    )
                )
            candidate_ids.add(candidate.id)
            decision = getattr(candidate.gate_decision, "value", str(candidate.gate_decision))
            if decision != GateDecision.PENDING.value and not candidate.decision_reason:
                issues.append(
                    MemoryHealthIssue(
                        code="candidate_decision_missing_reason",
                        severity="error",
                        message=f"candidate {candidate.id} has decision {decision} but no decision_reason",
                        record_id=candidate.id,
                        repair="manual_candidate_review",
                    )
                )
            if decision == GateDecision.REJECTED.value and candidate.id not in rejected_ids:
                issues.append(
                    MemoryHealthIssue(
                        code="rejected_candidate_missing_rejected_log",
                        severity="warning",
                        message=f"candidate {candidate.id} is rejected but absent from rejected.jsonl",
                        record_id=candidate.id,
                        repair="manual_candidate_review",
                    )
                )
        for candidate in self.store.list_rejected_candidates():
            decision = getattr(candidate.gate_decision, "value", str(candidate.gate_decision))
            if decision != GateDecision.REJECTED.value:
                issues.append(
                    MemoryHealthIssue(
                        code="rejected_log_contains_non_rejected_candidate",
                        severity="warning",
                        message=f"rejected log candidate {candidate.id} has decision {decision}",
                        record_id=candidate.id,
                        repair="manual_candidate_review",
                    )
                )
        return issues

    def _raw_archive_issues(self) -> List[MemoryHealthIssue]:
        issues: List[MemoryHealthIssue] = []
        manifest = self.store.read_raw_archive_manifest()
        raw_event_count = self.store.count_raw_events()
        if str(manifest.get("status") or "") not in {"", "ok"}:
            issues.append(
                MemoryHealthIssue(
                    code="raw_archive_manifest_degraded",
                    severity="warning",
                    message="raw archive manifest reports degraded integrity",
                    repair="manual_raw_archive_review",
                    details={"status": str(manifest.get("status") or "")},
                )
            )
        if int(manifest.get("event_count") or 0) != raw_event_count:
            issues.append(
                MemoryHealthIssue(
                    code="raw_archive_manifest_count_mismatch",
                    severity="warning",
                    message="raw archive manifest event_count differs from current archive line count",
                    repair="rebuild_raw_archive_manifest",
                )
            )
        return issues

    def _index_issues(self) -> List[MemoryHealthIssue]:
        issues: List[MemoryHealthIssue] = []
        expected = self._expected_index_records()
        indexed = self.index.count_memories()
        if indexed != expected:
            issues.append(
                MemoryHealthIssue(
                    code="index_count_mismatch",
                    severity="warning",
                    message=f"index has {indexed} records but canonical store has {expected} indexable records",
                    repair="rebuild_index",
                    details={"indexed": indexed, "expected": expected},
                )
            )
        source_ref_count = self._indexed_source_ref_count()
        canonical_source_count = len(self.store.list_source_refs())
        if source_ref_count != canonical_source_count:
            issues.append(
                MemoryHealthIssue(
                    code="source_index_count_mismatch",
                    severity="warning",
                    message=f"source index has {source_ref_count} records but canonical store has {canonical_source_count} source refs",
                    repair="rebuild_index",
                    details={"indexed": source_ref_count, "expected": canonical_source_count},
                )
            )
        return issues

    def _records_with_source_refs(self) -> List[tuple[str, str, List[str]]]:
        rows: List[tuple[str, str, List[str]]] = []
        for item in self.store.list_memory_items():
            rows.append(("memory_item", item.id, list(item.source_refs)))
        for card in self.store.list_project_cards():
            rows.append(("project_card", card.id, list(card.source_refs)))
        for candidate in self.store.list_candidates():
            rows.append(("candidate", candidate.id, list(candidate.source_refs)))
        for loop in self.store.list_open_loops():
            rows.append(("open_loop", str(loop.get("id") or ""), [str(ref) for ref in loop.get("source_refs") or []]))
        return rows

    def _existing_source_ids(self) -> set[str]:
        """Return all resolvable source ids for explicit health/repair audits.

        This is intentionally unbounded and must stay fenced to health/repair
        flows; normal review/extraction/promotion paths use read_source_ref plus
        raw-index point lookups instead of archive inventory scans.
        """
        ids = {str(source.id) for source in self.store.list_source_refs()}
        ids.update(str(event.get("id") or "") for event in self.store.read_all_raw_events_for_repair())
        ids.discard("")
        return ids

    def _counts(self) -> Dict[str, int]:
        return {
            "raw_events": self.store.count_raw_events(),
            "source_refs": len(self.store.list_source_refs()),
            "pending_candidates": self.store.count_pending_candidates(),
            "rejected_candidates": self.store.count_rejected_candidates(),
            "project_cards": len(self.store.list_project_cards()),
            "memory_items": len(self.store.list_memory_items()),
            "open_loops": len(self.store.list_open_loops()),
            "operations": len(self.store.list_operation_records()),
            "indexed_memories": self.index.count_memories(),
        }

    def _expected_index_records(self) -> int:
        return (
            len(self.store.list_memory_items())
            + len(self.store.list_project_cards())
            + len(self.store.list_candidates())
            + len(self.store.list_open_loops())
            + self.store.count_raw_events()
        )

    def _indexed_source_ref_count(self) -> int:
        path = Path(self.index.db_path)
        if not path.exists():
            return 0
        with sqlite3.connect(str(path)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM source_refs").fetchone()
        return int(row[0])
