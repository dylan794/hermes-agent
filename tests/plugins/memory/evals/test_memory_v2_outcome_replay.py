"""Behavior contracts for the offline Memory v2 outcome-replay lab."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from plugins.memory.memory_v2.evals.outcome_replay import (
    DATASET_SCHEMA_VERSION,
    PRIVATE_INTAKE_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    ValidationError,
    analyze_dataset,
    audit_dataset_disjointness,
    collect_private_intake,
    load_dataset,
    validate_dataset,
)


REPO_ROOT = Path(__file__).parents[4]
FIXTURE = (
    REPO_ROOT
    / "plugins"
    / "memory"
    / "memory_v2"
    / "evals"
    / "fixtures"
    / "outcome_replay_synthetic_v1.yaml"
)


def _dataset() -> dict:
    return load_dataset(FIXTURE)


def _variants(dataset: dict, ablation: str):
    for episode in dataset["episodes"]:
        yield next(row for row in episode["variants"] if row["ablation"] == ablation)


def _make_disjoint(dataset: dict) -> dict:
    result = copy.deepcopy(dataset)
    result["lab_id"] = "disjoint_synthetic_rehearsal_v1"
    for index, episode in enumerate(result["episodes"], start=101):
        episode["episode_id"] = f"opaque:{index:016x}"
        episode["participant_ref"] = f"opaque:{index + 100:016x}"
        episode["project_ref"] = f"opaque:{index + 200:016x}"
        episode["workstream_ref"] = f"opaque:{index + 300:016x}"
        episode["query_ref"] = f"opaque:{index + 400:016x}"
        episode["query_hash"] = f"sha256:{index + 500:064x}"
        episode["snapshot"]["archive_digest"] = f"sha256:{index + 600:064x}"
    return result


def _private_intake() -> dict:
    dataset = _dataset()
    episodes = []
    for index, episode in enumerate(dataset["episodes"], start=1):
        snapshot = episode["snapshot"]
        episodes.append(
            {
                "participant_id": f"private participant {index}",
                "project_id": f"private project {index}",
                "workstream_id": f"private workstream {index}",
                "query_text": f"What changed in private project {index}?",
                "query_class": episode["query_class"],
                "checkpoint_days": episode["checkpoint_days"],
                "evidence_cutoff": episode["evidence_cutoff"],
                "snapshot": {
                    "corpus_id": f"private corpus {index}",
                    **{
                        field_name: snapshot[field_name]
                        for field_name in (
                            "archive_digest",
                            "index_digest",
                            "config_digest",
                            "code_digest",
                            "answerer_digest",
                        )
                    },
                },
                "variants": episode["variants"],
            }
        )
    return {
        "schema_version": PRIVATE_INTAKE_SCHEMA_VERSION,
        "lab_id": "private_opt_in_pilot_v1",
        "study_mode": "pilot",
        "created_at": dataset["created_at"],
        "consent": {
            "collection_mode": "opt_in_shadow",
            "consent_granted": True,
            "consent_ref": "policy:private-opt-in-consent",
            "retention_until": dataset["privacy"]["retention_until"],
            "revocation_policy_ref": "policy:private-opt-in-revocation",
            "profile_scope_id": "private profile scope",
        },
        "thresholds": dataset["thresholds"],
        "episodes": episodes,
    }


def test_fixture_is_minimized_and_strictly_valid():
    dataset = _dataset()

    assert dataset["schema_version"] == DATASET_SCHEMA_VERSION
    assert dataset["study_mode"] == "development"
    assert len(dataset["episodes"]) == 4
    assert dataset["privacy"]["raw_query_stored"] is False
    assert dataset["privacy"]["raw_answer_stored"] is False
    assert dataset["privacy"]["raw_tool_output_stored"] is False
    assert all(row["participant_ref"].startswith("opaque:") for row in dataset["episodes"])


def test_private_intake_is_keyed_minimized_and_deterministic():
    intake = _private_intake()
    key = bytes(range(32))

    first = collect_private_intake(intake, token_material=key)
    second = collect_private_intake(intake, token_material=key)
    other_key = collect_private_intake(intake, token_material=b"z" * 32)

    assert first == second
    assert first["schema_version"] == DATASET_SCHEMA_VERSION
    assert first["study_mode"] == "pilot"
    assert first["privacy"]["raw_query_stored"] is False
    assert first["episodes"][0]["participant_ref"].startswith("opaque:")
    assert first["episodes"][0]["participant_ref"] != other_key["episodes"][0][
        "participant_ref"
    ]
    serialized = repr(first)
    assert "private participant" not in serialized
    assert "What changed" not in serialized


def test_private_intake_requires_consent_and_a_full_strength_key():
    intake = _private_intake()
    intake["consent"]["consent_granted"] = False
    with pytest.raises(ValidationError, match="consent_granted"):
        collect_private_intake(intake, token_material=bytes(range(32)))

    with pytest.raises(ValidationError, match="exactly 32 bytes"):
        collect_private_intake(_private_intake(), token_material=b"short")


def test_analysis_identifies_engineered_ranking_gap_deterministically():
    dataset = _dataset()

    first = analyze_dataset(dataset)
    second = analyze_dataset(dataset)

    assert first == second
    assert first["schema_version"] == RESULT_SCHEMA_VERSION
    assert first["diagnostic_only"] is True
    assert first["eligible_for_human_superiority_claim"] is False
    assert first["mutation_authority"] == "none"
    assert first["diagnosis"]["status"] == "decisive_bottleneck"
    assert first["diagnosis"]["component"] == "ranking"
    assert first["oracle_panel_complete_and_comparable"] is True
    ranking = next(
        row for row in first["ablation_diagnostics"] if row["ablation"] == "oracle_rerank"
    )
    assert ranking["paired_episodes"] == 4
    assert ranking["participant_clusters"] == 4
    assert ranking["paired_success_delta"] == 1.0
    assert ranking["confidence_interval"]["lower"] == 1.0
    assert ranking["safety_regression_count"] == 0


def test_underpowered_ablations_are_not_presented_as_bottlenecks():
    dataset = _dataset()
    dataset["thresholds"]["minimum_participant_clusters"] = 5

    report = analyze_dataset(dataset)

    assert report["diagnosis"]["status"] == "insufficient_evidence"
    assert report["diagnosis"]["component"] is None


def test_partial_oracle_panels_cannot_select_the_largest_bottleneck():
    dataset = _dataset()
    for episode in dataset["episodes"]:
        episode["variants"] = [
            row
            for row in episode["variants"]
            if row["ablation"] in {"current", "oracle_rerank"}
        ]

    report = analyze_dataset(dataset)

    assert report["diagnosis"]["status"] == "partial_bottleneck_signal"
    assert report["diagnosis"]["component"] == "ranking"
    assert report["oracle_panel_complete_and_comparable"] is False
    assert report["missing_episode_counts_by_ablation"]["oracle_temporal"] == 4


def test_safety_regression_blocks_a_positive_recommendation():
    dataset = _dataset()
    for variant in _variants(dataset, "current"):
        variant["outcome"]["safety_failures"] = ["gate:synthetic-old-failure"]
    for variant in _variants(dataset, "oracle_rerank"):
        variant["outcome"]["work_continuity_success"] = False
        variant["outcome"]["citation_correct"] = False
        # Equal failure counts still regress when the variant introduces a
        # different safety failure.
        variant["outcome"]["safety_failures"] = ["gate:synthetic-new-failure"]

    report = analyze_dataset(dataset)

    ranking = next(
        row for row in report["ablation_diagnostics"] if row["ablation"] == "oracle_rerank"
    )
    assert ranking["safety_regression_count"] == 4
    assert ranking["decisive_positive_gap"] is False
    assert report["diagnosis"]["status"] == "insufficient_evidence"


def test_disjointness_reports_only_overlap_fingerprints():
    dataset = _dataset()

    overlapping = audit_dataset_disjointness(dataset, [dataset])
    disjoint = audit_dataset_disjointness(_make_disjoint(dataset), [dataset])

    assert overlapping["disjoint"] is False
    assert overlapping["comparisons"][0]["overlaps"]["participant_refs"]["count"] == 4
    assert all(
        value.startswith("sha256:")
        for value in overlapping["comparisons"][0]["overlaps"]["participant_refs"][
            "fingerprints"
        ]
    )
    assert disjoint["disjoint"] is True


def test_raw_content_and_confirmatory_mode_are_rejected():
    dataset = _dataset()
    dataset["privacy"]["raw_query_stored"] = True
    with pytest.raises(ValidationError, match="raw_query_stored"):
        validate_dataset(dataset)

    dataset = _dataset()
    dataset["study_mode"] = "confirmatory"
    with pytest.raises(ValidationError, match="study_mode"):
        validate_dataset(dataset)


def test_raw_prompt_fields_fail_the_exact_schema():
    dataset = _dataset()
    dataset["episodes"][0]["raw_prompt"] = "must never enter this artifact"

    with pytest.raises(ValidationError, match="extra=.*raw_prompt"):
        validate_dataset(dataset)


def test_model_self_report_cannot_be_an_outcome_label():
    dataset = _dataset()
    next(_variants(dataset, "oracle_rerank"))["outcome"]["label_source"] = "model_self_report"

    with pytest.raises(ValidationError, match="label_source"):
        validate_dataset(dataset)


def test_offline_replays_must_disable_side_effects():
    dataset = _dataset()
    next(_variants(dataset, "oracle_route"))["offline_side_effects_disabled"] = False

    with pytest.raises(ValidationError, match="offline_side_effects_disabled"):
        validate_dataset(dataset)


def test_success_cannot_hide_safety_temporal_or_citation_failures():
    for field_name, value in (
        ("safety_failures", ["gate:failure"]),
        ("stale_conflict_error", True),
        ("citation_correct", False),
        ("citation_correct", None),
    ):
        dataset = _dataset()
        outcome = next(_variants(dataset, "oracle_rerank"))["outcome"]
        outcome[field_name] = value
        with pytest.raises(ValidationError, match="cannot be successful"):
            validate_dataset(dataset)


def test_current_and_single_change_component_contracts_are_required():
    dataset = _dataset()
    dataset["episodes"][0]["variants"] = [
        row for row in dataset["episodes"][0]["variants"] if row["ablation"] != "current"
    ]
    with pytest.raises(ValidationError, match="must include current"):
        validate_dataset(dataset)

    dataset = _dataset()
    next(_variants(dataset, "oracle_rerank"))["changed_component"] = "routing"
    with pytest.raises(ValidationError, match="changed_component"):
        validate_dataset(dataset)


def test_trace_counts_and_absolute_evidence_times_are_enforced():
    dataset = _dataset()
    current = next(_variants(dataset, "current"))
    current["trace"]["retrieved_count"] = 1
    with pytest.raises(ValidationError, match="retrieved_count"):
        validate_dataset(dataset)

    dataset = _dataset()
    current = next(_variants(dataset, "current"))
    current["trace"]["evidence_timestamps"][0] = "2026-06-01T12:00:00"
    with pytest.raises(ValidationError, match="timezone"):
        validate_dataset(dataset)

    dataset = _dataset()
    current = next(_variants(dataset, "current"))
    current["trace"]["evidence_timestamps"][0] = "2026-07-01T12:00:00Z"
    with pytest.raises(ValidationError, match="evidence_cutoff"):
        validate_dataset(dataset)


def test_citations_and_judgment_times_cannot_escape_the_frozen_replay():
    dataset = _dataset()
    current = next(_variants(dataset, "current"))
    current["trace"]["cited_ref_fingerprints"] = [f"sha256:{'f' * 64}"]
    with pytest.raises(ValidationError, match="subset of ranked"):
        validate_dataset(dataset)

    dataset = _dataset()
    current = next(_variants(dataset, "current"))
    current["outcome"]["verified_at"] = "2026-07-21T12:00:00Z"
    with pytest.raises(ValidationError, match="dataset.created_at"):
        validate_dataset(dataset)


def test_replay_artifacts_are_unique_within_an_episode():
    dataset = _dataset()
    variants = dataset["episodes"][0]["variants"]
    variants[1]["replay_artifact_digest"] = variants[0]["replay_artifact_digest"]

    with pytest.raises(ValidationError, match="replay_artifact_digest"):
        validate_dataset(dataset)


def test_privacy_retention_and_opaque_references_fail_closed():
    dataset = _dataset()
    dataset["privacy"]["retention_until"] = "2025-01-01T00:00:00Z"
    with pytest.raises(ValidationError, match="retention_until"):
        validate_dataset(dataset)

    dataset = _dataset()
    dataset["episodes"][0]["participant_ref"] = "real-person-name"
    with pytest.raises(ValidationError, match="opaque"):
        validate_dataset(dataset)
