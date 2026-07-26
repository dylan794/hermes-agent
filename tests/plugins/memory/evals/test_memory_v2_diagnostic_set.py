from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

import pytest

from plugins.memory.memory_v2.evals.diagnostic_set import (
    DiagnosticSetError,
    ORACLE_COMPONENTS,
    prepare_diagnostic_set,
    public_summary,
    score_labeled_set,
    validate_diagnostic_set,
)
from plugins.memory.memory_v2.shadow_reranker import (
    ShadowRerankerConfig,
    ShadowUtilityReranker,
)


def _events() -> list[dict]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(40):
        memory_cued = index < 25
        content = (
            f"Continue the previous project work for item {index}."
            if memory_cued
            else f"Please format the table for item {index} into three columns."
        )
        rows.append(
            {
                "id": f"event-{index:02d}",
                "type": "turn",
                "user_content": content,
                "observed_at": (start + timedelta(days=index)).isoformat(),
                "session_id": f"session-{index:02d}",
            }
        )
    return rows


def _candidate() -> dict:
    return {
        "id": "candidate-good",
        "type": "decision",
        "snippet": "Previous project decision evidence.",
        "source_refs": ["source-good"],
        "citations": [
            {
                "source_id": "source-good",
                "field": "content",
                "start": 0,
                "end": len("verified evidence"),
                "text": "verified evidence",
                "evidence_at": "2025-12-01T00:00:00+00:00",
            }
        ],
        "evidence_at": "2025-12-01T00:00:00+00:00",
        "profile_id": "single-participant-development",
        "workstream_ids": [],
        "status": "current",
        "evidence_role": "user",
        "verified": False,
    }


def _runner(query, evidence, cutoff, gap_days):
    del evidence, cutoff, gap_days
    result = ShadowUtilityReranker(
        ShadowRerankerConfig(enabled=True, minimum_utility=0.0)
    ).run(
        query,
        [_candidate()],
        {
            "memory_decision": "needed",
            "profile_id": "single-participant-development",
            "workstream_ids": [],
            "allow_unknown_workstream": True,
            "temporal_mode": "history",
            "evidence_cutoff": "2026-12-31T00:00:00+00:00",
        },
        citation_sources={
            ("source-good", "content"): "verified evidence",
        },
    )
    return {"result": result}


def _packet() -> dict:
    return prepare_diagnostic_set(
        _events(),
        shadow_runner=_runner,
        created_at="2026-07-25T12:00:00+00:00",
        episode_count=30,
        control_count=8,
        held_out_count=10,
    )


def test_prepare_freezes_exact_control_and_temporal_holdout_counts():
    packet = _packet()

    assert packet["selection"] == {
        "episode_count": 30,
        "control_candidate_count": 8,
        "memory_cued_candidate_count": 22,
    }
    assert packet["split_policy"]["development_count"] == 20
    assert packet["split_policy"]["held_out_count"] == 10
    assert packet["split_policy"]["held_out_frozen"] is True
    assert packet["split_policy"]["tuning_on_held_out_prohibited"] is True
    assert sum(row["control_candidate"] for row in packet["episodes"]) == 8
    assert all(
        row["owner_labels"]["status"] == "pending"
        for row in packet["episodes"]
    )
    development_times = [
        row["evidence_cutoff"]
        for row in packet["episodes"]
        if row["split"] == "development"
    ]
    held_out_times = [
        row["evidence_cutoff"]
        for row in packet["episodes"]
        if row["split"] == "held_out"
    ]
    assert max(development_times) < min(held_out_times)


def test_public_summary_is_content_free_and_reports_pending_handoff():
    packet = _packet()

    summary = public_summary(packet)
    serialized = repr(summary)

    assert summary["label_status_counts"] == {"pending": 30}
    assert summary["owner_labels_complete"] is False
    assert summary["outcome_replay_ready"] is False
    assert summary["outcome_replay_blockers"] == [
        "pending_owner_labels",
        "frozen_offline_variant_receipts",
    ]
    assert summary["eligible_for_pilot_go"] is False
    assert "Continue the previous" not in serialized
    assert "verified evidence" not in serialized


def test_validation_rejects_heldout_or_label_contract_tampering():
    packet = _packet()
    packet["split_policy"]["held_out_frozen"] = False
    with pytest.raises(DiagnosticSetError, match="split policy"):
        validate_diagnostic_set(packet)

    packet = _packet()
    packet["episodes"][0]["owner_labels"]["useful_candidate_ids"] = ["unknown"]
    with pytest.raises(DiagnosticSetError, match="unknown candidate"):
        validate_diagnostic_set(packet)


def test_scoring_requires_owner_labels_and_accepts_completed_development_set():
    packet = _packet()
    with pytest.raises(DiagnosticSetError, match="require owner labels"):
        score_labeled_set(packet, split="development")

    completed = copy.deepcopy(packet)
    for episode in completed["episodes"]:
        labels = episode["owner_labels"]
        memory_needed = not episode["control_candidate"]
        labels.update(
            {
                "status": "complete",
                "memory_needed": memory_needed,
                "required_source_refs": ["source-good"] if memory_needed else [],
                "useful_candidate_ids": (
                    ["candidate-good"] if memory_needed else []
                ),
                "stale_candidate_ids": [],
                "irrelevant_candidate_ids": (
                    [] if memory_needed else ["candidate-good"]
                ),
                "minimal_bundle_candidate_ids": (
                    ["candidate-good"] if memory_needed else []
                ),
                "current_answer_success": False,
                "citation_correct": False,
                "oracle_assessments": {
                    component: "not_applicable"
                    for component in ORACLE_COMPONENTS
                },
                "notes": "",
            }
        )

    score = score_labeled_set(completed, split="development")

    assert score["episode_count"] == 20
    assert score["shadow_metrics"]["records"] == 20
    assert score["owner_labels_complete"] is True
    assert score["outcome_replay_ready"] is False
    assert score["outcome_replay_blockers"] == [
        "frozen_offline_variant_receipts"
    ]
    assert score["eligible_for_pilot_go"] is False
    assert score["mutation_authority"] == "none"
