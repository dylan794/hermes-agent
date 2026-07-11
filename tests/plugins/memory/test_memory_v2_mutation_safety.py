"""P0 mutation-safety tests for Memory v2 canonical state."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from plugins.memory.memory_v2 import MemoryV2Provider
from plugins.memory.memory_v2.health import MemoryHealthChecker
from plugins.memory.memory_v2.index import MemoryV2Index
from plugins.memory.memory_v2.operations import MemoryOperationService
from plugins.memory.memory_v2.review_actions import CONFIRM_REVIEW_APPLY
from plugins.memory.memory_v2.schemas import CandidateMemory, ValidationError
from plugins.memory.memory_v2.store import MemoryV2Store


def _write_enabled_config(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        """
memory_v2:
  archive:
    enabled: true
    capture_enabled: true
    search_tools_enabled: true
    show_tools_enabled: true
  extraction:
    enabled: true
    candidate_creation_enabled: true
  review_apply:
    enabled: true
  working_memory:
    enabled: true
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _provider(tmp_path: Path) -> MemoryV2Provider:
    _write_enabled_config(tmp_path)
    provider = MemoryV2Provider()
    provider.initialize("session-safety", hermes_home=str(tmp_path), platform="cli")
    return provider


def _append_raw_event_worker(args: tuple[str, int]) -> str:
    base_dir, idx = args
    store = MemoryV2Store(Path(base_dir))
    store.initialize()
    return store.append_raw_event({"type": "turn", "content": f"multiprocess event {idx}"})["id"]


def test_multiprocess_raw_event_appends_preserve_hash_chain_manifest_and_index(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    index = MemoryV2Index(store.default_index_path)
    index.initialize()

    with ProcessPoolExecutor(max_workers=6) as executor:
        ids = list(executor.map(_append_raw_event_worker, [(str(store.base_dir), idx) for idx in range(24)]))

    assert len(set(ids)) == 24
    assert store.count_raw_events() == 24
    report = store.verify_raw_archive()
    assert report["status"] == "ok"
    assert report["verified_event_count"] == 24
    manifest = store.read_raw_archive_manifest()
    assert manifest["status"] == "ok"
    assert manifest["event_count"] == 24
    assert manifest["derived_index_status"] == "ok"
    assert manifest["indexed_event_count"] == 24
    assert index.raw_event_count() == 24
    assert store.search_raw_events("multiprocess", limit=24, index=index)


def test_profile_lock_timeout_fails_closed_when_held_by_another_process(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys, time
                from pathlib import Path
                from plugins.memory.memory_v2.store import MemoryV2Store
                store = MemoryV2Store(Path(sys.argv[1]))
                store.initialize()
                with store.profile_lock(timeout=5):
                    print('locked', flush=True)
                    time.sleep(2)
                """
            ),
            str(store.base_dir),
        ],
        cwd=str(Path(__file__).resolve().parents[3]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        start = time.monotonic()
        with pytest.raises(ValidationError, match="profile lock"):
            store.append_raw_event({"type": "turn", "content": "must fail closed"}, lock_timeout=0.1)
        assert time.monotonic() - start < 1.0
        assert store.count_raw_events() == 0
    finally:
        holder.terminate()
        holder.wait(timeout=5)


def test_operation_journal_records_prepared_then_committed_for_canonical_mutation(tmp_path):
    provider = _provider(tmp_path)
    event = provider.store.append_raw_event({"type": "turn", "session_id": "session-safety", "user_content": "Temporary scratch note."})
    provider.store.append_candidate(CandidateMemory(id="cand_journal", type="fact", claim="Temporary scratch note.", source_refs=[event["id"]]))

    result = MemoryOperationService(provider.store, provider.index).reject_candidate("cand_journal", "ephemeral", actor="test")

    assert result.success is True
    records = provider.store.list_operation_records()
    assert [record["status"] for record in records] == ["prepared", "committed"]
    assert records[0]["operation_id"] == records[1]["operation_id"] == result.operation_id
    assert records[0]["type"] == records[1]["type"] == "reject_candidate"


def test_interrupted_operation_is_detected_and_fail_closed_until_recovered(tmp_path):
    provider = _provider(tmp_path)
    event = provider.store.append_raw_event({"type": "turn", "session_id": "session-safety", "user_content": "source"})
    provider.store.append_candidate(CandidateMemory(id="cand_interrupted", type="fact", claim="Needs review", source_refs=[event["id"]]))
    provider.store.append_operation_record(
        {
            "operation_id": "op_interrupted",
            "type": "reject_candidate",
            "status": "prepared",
            "actor": "test",
            "before_ids": ["cand_interrupted"],
            "after_ids": ["cand_interrupted"],
        }
    )

    health = MemoryHealthChecker(provider.store, provider.index).check()
    assert any(issue["code"] == "interrupted_operation" for issue in health["issues"])

    blocked = MemoryOperationService(provider.store, provider.index).reject_candidate("cand_interrupted", "try while interrupted")
    assert blocked.success is False
    assert "interrupted operation" in blocked.error

    repair = MemoryHealthChecker(provider.store, provider.index).repair(dry_run=False)
    assert any(action["action"] == "mark_interrupted_operation_failed" for action in repair["actions"])
    records = provider.store.list_operation_records()
    assert records[-1]["operation_id"] == "op_interrupted"
    assert records[-1]["status"] == "failed"

    allowed = MemoryOperationService(provider.store, provider.index).reject_candidate("cand_interrupted", "after recovery")
    assert allowed.success is True


def test_direct_provider_mutation_tools_require_review_plan_action_fingerprint(tmp_path):
    provider = _provider(tmp_path)
    event = provider.store.append_raw_event({"type": "turn", "session_id": "session-safety", "user_content": "Temporary scratch note for this answer only."})
    provider.store.append_candidate(CandidateMemory(id="cand_direct", type="fact", claim="Temporary scratch note for this answer only.", source_refs=[event["id"]]))
    plan = json.loads(provider.handle_tool_call("memory_v2_review_plan", {"candidate_ids": ["cand_direct"]}))
    assert plan["actions"]
    action = plan["actions"][0]

    bypass = json.loads(provider.handle_tool_call("memory_v2_reject", {"candidate_id": "cand_direct", "reason": "direct bypass"}))
    assert bypass["success"] is False
    assert "review plan" in bypass["error"]
    assert provider.store.list_operation_records() == []

    applied = json.loads(
        provider.handle_tool_call(
            "memory_v2_reject",
            {
                "candidate_id": "cand_direct",
                "reason": "reviewed rejection",
                "plan_id": plan["plan_id"],
                "action_id": action["action_id"],
                "candidate_fingerprint": action["candidate_fingerprint"],
                "confirm": CONFIRM_REVIEW_APPLY,
            },
        )
    )
    assert applied["success"] is True
    assert provider.store.list_candidates()[0].gate_decision.value == "rejected"


def test_operation_failpoint_after_prepare_leaves_failed_journal_and_no_partial_mutation(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    event = provider.store.append_raw_event({"type": "turn", "session_id": "session-safety", "user_content": "Temporary failpoint note."})
    provider.store.append_candidate(CandidateMemory(id="cand_failpoint", type="fact", claim="Temporary failpoint note.", source_refs=[event["id"]]))
    monkeypatch.setenv("MEMORY_V2_FAILPOINT", "after_operation_prepare")

    result = MemoryOperationService(provider.store, provider.index).reject_candidate("cand_failpoint", "failpoint", actor="test")

    assert result.success is False
    assert "failpoint" in result.error
    assert provider.store.list_candidates()[0].gate_decision.value == "pending"
    records = provider.store.list_operation_records()
    assert [record["status"] for record in records] == ["prepared", "failed"]
    assert records[0]["operation_id"] == records[1]["operation_id"]


def test_reject_post_write_failpoint_requires_manual_recovery_and_keeps_future_mutations_blocked(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    event = provider.store.append_raw_event({"type": "turn", "session_id": "session-safety", "user_content": "Partial reject note."})
    provider.store.append_candidate(CandidateMemory(id="cand_partial_reject", type="fact", claim="Partial reject note.", source_refs=[event["id"]]))
    provider.store.append_candidate(CandidateMemory(id="cand_other", type="fact", claim="Other note.", source_refs=[event["id"]]))
    monkeypatch.setenv("MEMORY_V2_FAILPOINT", "after_reject_candidates_rewrite")

    result = MemoryOperationService(provider.store, provider.index).reject_candidate("cand_partial_reject", "post-write fail", actor="test")

    assert result.success is False
    assert provider.store.list_candidates()[0].gate_decision.value == "rejected"
    assert provider.store.list_rejected_candidates() == []
    records = provider.store.list_operation_records()
    assert [record["status"] for record in records] == ["prepared", "recovery_required"]
    health = MemoryHealthChecker(provider.store, provider.index).check()
    issue = next(issue for issue in health["issues"] if issue["code"] == "operation_recovery_required")
    assert issue["severity"] == "critical"

    monkeypatch.delenv("MEMORY_V2_FAILPOINT")
    blocked = MemoryOperationService(provider.store, provider.index).reject_candidate("cand_other", "must stay blocked", actor="test")
    assert blocked.success is False
    assert "recovery" in blocked.error or "interrupted operation" in blocked.error

    repair = MemoryHealthChecker(provider.store, provider.index).repair(dry_run=False)
    recovery_actions = [action for action in repair["actions"] if action.get("record_id") == records[0]["operation_id"]]
    assert recovery_actions == [
        {
            "action": "manual_operation_recovery_required",
            "safe": False,
            "reason": issue["message"],
            "record_id": records[0]["operation_id"],
            "mutates": "canonical memory files; manual rollback or completion required",
        }
    ]
    assert provider.store.list_operation_records()[-1]["status"] == "recovery_required"


def test_promote_post_write_failpoint_requires_manual_recovery_and_keeps_partial_memory_blocked(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    event = provider.store.append_raw_event({"type": "turn", "session_id": "session-safety", "user_content": "Alex prefers partial recovery tests."})
    provider.store.append_candidate(
        CandidateMemory(
            id="cand_partial_promote",
            type="preference",
            claim="Alex prefers partial recovery tests.",
            proposed_destination="core/user",
            confidence=0.95,
            source_refs=[event["id"]],
        )
    )
    provider.store.append_candidate(CandidateMemory(id="cand_other", type="fact", claim="Other note.", source_refs=[event["id"]]))
    monkeypatch.setenv("MEMORY_V2_FAILPOINT", "after_promote_memory_item_write")

    result = MemoryOperationService(provider.store, provider.index).promote_candidate("cand_partial_promote", actor="test")

    assert result.success is False
    assert len(provider.store.list_memory_items()) == 1
    assert provider.store.list_candidates()[0].gate_decision.value == "pending"
    records = provider.store.list_operation_records()
    assert [record["status"] for record in records] == ["prepared", "recovery_required"]
    health = MemoryHealthChecker(provider.store, provider.index).check()
    assert any(issue["code"] == "operation_recovery_required" and issue["severity"] == "critical" for issue in health["issues"])

    monkeypatch.delenv("MEMORY_V2_FAILPOINT")
    blocked = MemoryOperationService(provider.store, provider.index).reject_candidate("cand_other", "must stay blocked", actor="test")
    assert blocked.success is False
    repair = MemoryHealthChecker(provider.store, provider.index).repair(dry_run=False)
    assert any(action["action"] == "manual_operation_recovery_required" and action["safe"] is False for action in repair["actions"])
    assert provider.store.list_operation_records()[-1]["status"] == "recovery_required"
