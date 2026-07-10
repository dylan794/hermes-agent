"""Tests for Memory v2 audited operations and health checks."""

from __future__ import annotations

import json

from plugins.memory.memory_v2 import MemoryV2Provider
from plugins.memory.memory_v2.health import MemoryHealthChecker
from plugins.memory.memory_v2.operations import MemoryOperationService
from plugins.memory.memory_v2.schemas import CandidateMemory, MemoryItem


def _provider(tmp_path):
    provider = MemoryV2Provider()
    provider.initialize("session-ops", hermes_home=str(tmp_path), platform="cli")
    return provider


def test_manual_promote_candidate_routes_through_audited_operation(tmp_path):
    provider = _provider(tmp_path)
    provider.sync_turn(
        "Remember that Alex prefers memory mutations to be audited.",
        "Queued as a Memory v2 candidate.",
        session_id="session-ops",
    )
    candidate = provider.store.list_candidates()[0]

    result = json.loads(provider.handle_tool_call("memory_v2_promote", {"candidate_id": candidate.id}))

    assert result["success"] is True
    assert result["operation_id"].startswith("op_")
    assert result["operation_type"] == "promote_candidate_to_memory_item"
    promoted_id = result["promoted_ids"][0]
    promoted = provider.store.read_memory_item(promoted_id)
    assert promoted is not None
    assert promoted.source_refs == candidate.source_refs
    assert getattr(provider.store.list_candidates()[0].gate_decision, "value", provider.store.list_candidates()[0].gate_decision) == "promoted"

    operations = provider.store.list_operation_records()
    assert len(operations) == 1
    assert operations[0]["operation_id"] == result["operation_id"]
    assert operations[0]["type"] == "promote_candidate_to_memory_item"
    assert operations[0]["source_refs"] == candidate.source_refs
    assert candidate.id in operations[0]["before_ids"]
    assert promoted_id in operations[0]["after_ids"]


def test_reject_candidate_operation_updates_rejected_log_and_audit(tmp_path):
    provider = _provider(tmp_path)
    provider.store.append_candidate(
        CandidateMemory(id="cand_reject", type="fact", claim="Temporary scratch detail", source_refs=[])
    )

    result = MemoryOperationService(provider.store, provider.index).reject_candidate(
        "cand_reject", "too ephemeral", actor="test"
    )

    assert result.success is True
    candidate = provider.store.list_candidates()[0]
    assert getattr(candidate.gate_decision, "value", candidate.gate_decision) == "rejected"
    assert candidate.decision_reason == "too ephemeral"
    assert provider.store.list_rejected_candidates()[0].id == "cand_reject"
    operations = provider.store.list_operation_records()
    assert operations[0]["type"] == "reject_candidate"
    assert operations[0]["actor"] == "test"


def test_resolve_open_loop_preserves_history_and_audit(tmp_path):
    provider = _provider(tmp_path)
    event = provider.store.append_raw_event({"type": "turn", "session_id": "session-ops", "user_content": "ship health layer"})
    loop = provider.store.upsert_open_loop({"text": "Ship Memory v2 health layer", "source_refs": [event["id"]]})

    result = json.loads(
        provider.handle_tool_call(
            "memory_v2_resolve_open_loop",
            {"loop_id": loop["id"], "status": "resolved", "resolution": "implemented and tested"},
        )
    )

    assert result["success"] is True
    updated = provider.store.list_open_loops()[0]
    assert updated["status"] == "resolved"
    assert updated["resolution"] == "implemented and tested"
    assert updated["history"][-1]["from_status"] == "open"
    assert updated["history"][-1]["to_status"] == "resolved"
    assert provider.store.list_operation_records()[0]["type"] == "resolve_open_loop"


def test_health_detects_dangling_sources_and_index_mismatch(tmp_path):
    provider = _provider(tmp_path)
    provider.store.append_candidate(
        CandidateMemory(id="cand_dangling", type="fact", claim="Dangling evidence", source_refs=["missing_source"])
    )
    item = MemoryItem(
        id="mem_dangling_target",
        type="fact",
        subject="memory",
        predicate="states",
        value="old fact",
        status="superseded",
        superseded_by="mem_missing_newer",
        source_refs=["missing_source"],
    )
    provider.store.write_memory_item(item)

    health = MemoryHealthChecker(provider.store, provider.index).check()

    codes = {issue["code"] for issue in health["issues"]}
    assert health["status"] == "degraded"
    assert "dangling_source_ref" in codes
    assert "superseded_by_missing_target" in codes
    assert "index_count_mismatch" in codes
    assert any(action["action"] == "rebuild_index" for action in health["repair_plan"])


def test_repair_non_dry_run_rebuilds_derived_index_only(tmp_path):
    provider = _provider(tmp_path)
    event = provider.store.append_raw_event({"type": "turn", "session_id": "session-ops", "user_content": "evidence"})
    provider.store.append_candidate(
        CandidateMemory(id="cand_unindexed", type="fact", claim="Needs index rebuild", source_refs=[event["id"]])
    )
    assert provider.index.count_memories() == 0

    repair = MemoryHealthChecker(provider.store, provider.index).repair(dry_run=False)

    assert repair["success"] is True
    assert repair["dry_run"] is False
    assert repair["actions"][0]["action"] == "rebuild_index"
    assert provider.index.count_memories() == 2  # raw event + candidate
    assert getattr(provider.store.list_candidates()[0].gate_decision, "value", provider.store.list_candidates()[0].gate_decision) == "pending"
