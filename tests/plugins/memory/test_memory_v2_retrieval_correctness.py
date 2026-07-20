"""Correctness invariants for current-state retrieval and project lifecycle data."""

from __future__ import annotations

import pytest
import yaml

from plugins.memory.memory_v2.consolidation import RuleBasedConsolidator
from plugins.memory.memory_v2.index import MemoryV2Index
from plugins.memory.memory_v2.retrieval import MemoryPacketComposer
from plugins.memory.memory_v2.schemas import (
    CandidateMemory,
    GateDecision,
    MemoryItem,
    ProjectCard,
    SourceRef,
)
from plugins.memory.memory_v2.store import MemoryV2Store
from plugins.memory.memory_v2.write_gate import RuleBasedWriteGate, WriteGateOutcome


def _store_and_index(root):
    store = MemoryV2Store(root / "memory_v2")
    store.initialize()
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    return store, index


@pytest.mark.parametrize("source_count", [3, 10, 100])
def test_project_packet_keeps_a_useful_result_without_section_duplication(
    tmp_path, source_count
):
    store, index = _store_and_index(tmp_path)
    source_ids = []
    for number in range(source_count):
        source = SourceRef(
            id=f"source_budget_{number:03d}",
            type="file",
            uri=f"memory://evidence/budget-{number:03d}.txt",
            title=f"Budget evidence {number}",
            observed_at="2026-01-01T00:00:00Z",
            quote="Budget Atlas evidence.",
        )
        store.write_source_ref(source)
        source_ids.append(source.id)
    store.write_project_card(
        ProjectCard(
            id="Budget Atlas",
            name="Budget Atlas",
            goal="Keep retrieval useful under bounded packet budgets.",
            current_state="The bounded project result is eligible and useful.",
            next_actions=["Verify the 3, 10, and 100 source cases."],
            source_refs=source_ids,
        )
    )
    index.rebuild_from_store(store)

    packet = MemoryPacketComposer(index).compose("Where did we leave Budget Atlas?")
    rendered = MemoryPacketComposer.render(packet)
    parsed = yaml.safe_load(rendered)

    assert packet.items
    assert packet.items[0]["id"] == "project:budget-atlas"
    assert packet.items[0]["project"]["current_state"]
    assert MemoryPacketComposer._estimate_tokens(rendered) <= packet.token_budget
    assert parsed["sections"]["active_project_state"] == [
        {"item_ref": "project:budget-atlas"}
    ]
    if source_count > 8:
        assert packet.items[0]["source_ref_count"] == source_count
        assert packet.items[0]["source_refs_truncated"] is True


def test_candidates_are_route_compatible_and_promoted_successors_hide_them(tmp_path):
    _store, index = _store_and_index(tmp_path)
    index.index_candidate(
        CandidateMemory(
            id="cand_atlas_project",
            type="project_state",
            claim="Project Atlas current state: route-compatible candidate.",
            proposed_destination="semantic/projects/atlas.yaml",
            source_refs=["source_atlas"],
        )
    )
    index.index_candidate(
        CandidateMemory(
            id="cand_atlas_preference",
            type="preference",
            claim="Atlas route-compatible preference candidate.",
            source_refs=["source_atlas"],
        )
    )
    canonical = MemoryItem(
        id="pref_atlas_canonical",
        type="preference",
        subject="user",
        predicate="prefers",
        value="Atlas canonical preference",
        summary="Atlas canonical preference",
        source_refs=["source_atlas"],
    )
    index.index_memory_item(canonical)
    index.index_candidate(
        CandidateMemory(
            id="cand_atlas_promoted",
            type="preference",
            claim="Atlas canonical preference",
            source_refs=["source_atlas"],
            gate_decision=GateDecision.PROMOTED,
            decision_reason="Promoted to canonical MemoryItem pref_atlas_canonical.",
        )
    )

    project_packet = MemoryPacketComposer(index).compose(
        "Where did we leave Project Atlas?"
    )
    assert "cand_atlas_project" in {item["id"] for item in project_packet.items}
    assert "cand_atlas_preference" not in {item["id"] for item in project_packet.items}

    preference_packet = MemoryPacketComposer(index).compose(
        "What do I prefer about Atlas canonical preference?"
    )
    assert "pref_atlas_canonical" in {item["id"] for item in preference_packet.items}
    assert "cand_atlas_promoted" not in {item["id"] for item in preference_packet.items}


def test_structured_project_field_wins_before_generic_conflict_rule():
    decision = RuleBasedWriteGate().classify(
        "Remember that Project Atlas decision: use SQLite instead of JSON files."
    )

    assert decision.outcome == WriteGateOutcome.PROJECT_UPDATE
    assert decision.memory_type == "project_state"
    assert decision.proposed_destination == "semantic/projects/atlas.yaml"
    assert decision.reason == "project_update: decision"


def _consolidate_project_history(root, candidate_order):
    store, index = _store_and_index(root)
    candidates = {
        "old_state": CandidateMemory(
            id="cand_old_state",
            type="project_state",
            claim="Project Atlas current state: old state.",
            created_at="2025-01-01T00:00:00Z",
            proposed_destination="semantic/projects/atlas.yaml",
            promotion_reason="project_update: current_state",
            source_refs=["event_old_state"],
        ),
        "new_state": CandidateMemory(
            id="cand_new_state",
            type="project_state",
            claim="Project Atlas current state: current state.",
            created_at="2026-01-01T00:00:00Z",
            proposed_destination="semantic/projects/atlas.yaml",
            promotion_reason="project_update: current_state",
            source_refs=["event_new_state"],
        ),
        "action": CandidateMemory(
            id="cand_action",
            type="project_state",
            claim="Project Atlas next action: ship the retrieval invariant.",
            created_at="2025-06-01T00:00:00Z",
            proposed_destination="semantic/projects/atlas.yaml",
            promotion_reason="project_update: next_action",
            source_refs=["event_action"],
        ),
        "resolved": CandidateMemory(
            id="cand_resolved",
            type="project_state",
            claim="Project Atlas resolved next action: ship the retrieval invariant.",
            created_at="2026-02-01T00:00:00Z",
            proposed_destination="semantic/projects/atlas.yaml",
            promotion_reason="project_update: next_action resolved",
            source_refs=["event_resolved"],
        ),
    }
    event_times = {
        "event_old_state": "2025-01-01T00:00:00Z",
        "event_new_state": "2026-01-01T00:00:00Z",
        "event_action": "2025-06-01T00:00:00Z",
        "event_resolved": "2026-02-01T00:00:00Z",
    }
    for event_id, observed_at in event_times.items():
        store.append_raw_event(
            {
                "id": event_id,
                "type": "turn",
                "observed_at": observed_at,
                "created_at": observed_at,
                "user_content": f"evidence {event_id}",
            }
        )
    for key in candidate_order:
        store.append_candidate(candidates[key])
    RuleBasedConsolidator().consolidate(store, index)
    index.rebuild_from_store(store)
    card = store.read_project_card("Atlas")
    assert card is not None
    return card, MemoryPacketComposer(index).compose("Where did we leave Project Atlas?")


def test_project_lifecycle_is_evidence_time_ordered_and_restart_invariant(tmp_path):
    chronological_card, chronological_packet = _consolidate_project_history(
        tmp_path / "chronological", ["old_state", "action", "new_state", "resolved"]
    )
    shuffled_card, shuffled_packet = _consolidate_project_history(
        tmp_path / "shuffled", ["resolved", "new_state", "action", "old_state"]
    )

    assert chronological_card.to_dict() == shuffled_card.to_dict()
    assert chronological_card.current_state == "current state."
    assert chronological_card.next_actions == []
    action_records = [
        entry
        for entry in chronological_card.field_evidence["next_actions"]
        if not entry.get("lifecycle_event")
    ]
    assert action_records[0]["status"] == "resolved"
    assert chronological_packet.items[0]["project"].get("next_actions", []) == []
    assert chronological_packet.items == shuffled_packet.items
