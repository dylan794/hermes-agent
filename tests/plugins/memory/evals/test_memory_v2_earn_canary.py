"""Behavior contracts for the four-arm Memory v2 canary study."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from plugins.memory.memory_v2.evals.earn_canary import (
    CONDITIONS,
    DIMENSIONS,
    JUDGMENTS_SCHEMA_VERSION,
    QUERY_CLASSES,
    SLOTS,
    ValidationError,
    _coverage,
    _fingerprint,
    _packet_cluster_agreement_interval,
    _packet_cluster_guess_interval,
    _shadow_gate_status,
    _validate_thresholds,
    _wilson_interval,
    audit_study_disjointness,
    load_judgments,
    load_protocol,
    load_responses,
    prepare_blinded_packets,
    score_study,
    validate_judgments,
    validate_protocol,
)


REPO_ROOT = Path(__file__).parents[4]
FIXTURES = REPO_ROOT / "plugins" / "memory" / "memory_v2" / "evals" / "fixtures"
PROTOCOL = FIXTURES / "earn_canary_protocol_synthetic_v1.yaml"
RESPONSES = FIXTURES / "earn_canary_responses_synthetic_v1.yaml"
JUDGMENTS = FIXTURES / "earn_canary_judgments_synthetic_v1.yaml"


def _study() -> dict:
    return load_protocol(PROTOCOL)


def _bundle(seed: int = 17) -> dict:
    return prepare_blinded_packets(_study(), load_responses(RESPONSES), seed=seed)


def _make_disjoint(study: dict) -> dict:
    result = copy.deepcopy(study)
    result["study_id"] = "earn_canary_disjoint_synthetic_v1"
    for index, episode in enumerate(result["episodes"], start=101):
        episode["episode_id"] = f"opaque:{index:016x}"
        episode["participant_ref"] = f"opaque:{index + 100:016x}"
        episode["project_ref"] = f"opaque:{index + 200:016x}"
        episode["workstream_ref"] = f"opaque:{index + 300:016x}"
        episode["query_ref"] = f"opaque:{index + 400:016x}"
        episode["query_hash"] = f"sha256:{index + 500:064x}"
        episode["snapshot"]["corpus_ref"] = f"opaque:{index + 600:016x}"
        episode["snapshot"]["archive_digest"] = f"sha256:{index + 700:064x}"
    return result


def _references(study: dict) -> list[dict]:
    return [_make_disjoint(study)]


def _taxonomy() -> dict:
    return {
        "primary": "candidate_recall",
        "secondary": None,
        "label_source": "independent_judge",
        "judgment_ref": "judgment:synthetic",
    }


def _judgments(
    bundle: dict,
    *,
    disagree_all_slots: bool = False,
    include_adjudicator: bool = False,
) -> dict:
    rows = []
    assignments = {
        row["packet_id"]: row["slots"] for row in bundle["private"]["assignment_key"]
    }
    for packet_index, packet in enumerate(bundle["public"]["judge_packets"], start=1):
        for judge_number in (1, 2):
            scores = {}
            taxonomy = {}
            for slot in SLOTS:
                condition = assignments[packet["packet_id"]][slot]["condition"]
                success = condition in {"memory_v2", "oracle_evidence"}
                if disagree_all_slots and judge_number == 2:
                    success = not success
                score = 5 if success else 2
                scores[slot] = {dimension: score for dimension in DIMENSIONS}
                taxonomy[slot] = None if success else _taxonomy()
            rows.append({
                "packet_id": packet["packet_id"],
                "judge_id": f"opaque:{judge_number:08x}{packet_index:08x}",
                "judge_role": "primary",
                "scores": scores,
                "gate_failures": {slot: [] for slot in SLOTS},
                "condition_guesses": {
                    slot: assignments[packet["packet_id"]][slot]["condition"]
                    for slot in SLOTS
                },
                "guess_confidence": {slot: "low" for slot in SLOTS},
                "material_error_notes": {slot: None for slot in SLOTS},
                "potentially_identifiable": {slot: False for slot in SLOTS},
                "failure_taxonomy": taxonomy,
            })
        if include_adjudicator:
            rows.append({
                "packet_id": packet["packet_id"],
                "judge_id": f"opaque:{3:08x}{packet_index:08x}",
                "judge_role": "adjudicator",
                "scores": {
                    slot: {dimension: 5 for dimension in DIMENSIONS} for slot in SLOTS
                },
                "gate_failures": {slot: [] for slot in SLOTS},
                "condition_guesses": {
                    slot: assignments[packet["packet_id"]][slot]["condition"]
                    for slot in SLOTS
                },
                "guess_confidence": {slot: "low" for slot in SLOTS},
                "material_error_notes": {slot: None for slot in SLOTS},
                "potentially_identifiable": {slot: False for slot in SLOTS},
                "failure_taxonomy": {slot: None for slot in SLOTS},
            })
    return {
        "schema_version": JUDGMENTS_SCHEMA_VERSION,
        "study_id": bundle["public"]["study_id"],
        "packet_set_fingerprint": bundle["public"]["packet_set_fingerprint"],
        "judgments": rows,
    }


def test_fixture_is_strict_and_development_only():
    study = _study()

    assert study["study_mode"] == "development"
    assert len(study["episodes"]) == 4
    assert study["design"]["conditions"] == list(CONDITIONS)
    assert study["privacy"]["raw_work_evidence_stored"] is False
    assert len(load_judgments(JUDGMENTS)["judgments"]) == 1

    extra = copy.deepcopy(study)
    extra["raw_prompt_dump"] = "forbidden"
    with pytest.raises(ValidationError, match="extra=.*raw_prompt_dump"):
        validate_protocol(extra)


def test_prepare_is_deterministic_balanced_and_condition_blind():
    first = _bundle(seed=17)
    second = _bundle(seed=17)

    assert first == second
    with pytest.raises(ValidationError, match="seed does not match"):
        _bundle(seed=18)
    assignments = first["private"]["assignment_key"]
    assert len(assignments) == len(CONDITIONS)
    for slot in SLOTS:
        assert {row["slots"][slot]["condition"] for row in assignments} == set(
            CONDITIONS
        )

    serialized_public = repr(first["public"])
    assert "participant_ref" not in serialized_public
    assert "assignment_key" not in serialized_public
    assert "condition" not in serialized_public


def test_scoring_reports_paired_uplift_but_development_cannot_go():
    study = _study()
    bundle = _bundle()
    report = score_study(
        study,
        bundle,
        _judgments(bundle),
        responses=load_responses(RESPONSES),
        reference_protocols=_references(study),
        untouched_pool_attested=True,
        consent_active_attested=True,
    )

    assert report["mutation_authority"] == "none"
    assert report["diagnostic_only"] is True
    assert (
        report["comparisons"]["memory_v2_vs_no_memory"]["paired_success_delta"] == 1.0
    )
    assert report["comparisons"]["memory_v2_vs_raw_fts"]["paired_success_delta"] == 1.0
    assert (
        report["comparisons"]["oracle_evidence_vs_memory_v2"]["paired_success_delta"]
        == 0.0
    )
    assert report["judge_agreement"]["raw_agreement"] == 1.0
    assert report["gates"]["pilot_design"] is False
    assert report["go"] is False


def test_packet_tampering_is_rejected_before_scoring():
    bundle = _bundle()
    judgments = _judgments(bundle)
    bundle["public"]["judge_packets"][0]["responses"]["A"]["text"] += " tampered"

    with pytest.raises(ValidationError, match="fingerprint"):
        score_study(
            _study(),
            bundle,
            judgments,
            responses=load_responses(RESPONSES),
            reference_protocols=_references(_study()),
            untouched_pool_attested=True,
            consent_active_attested=True,
        )


def test_disagreement_requires_registered_adjudication():
    bundle = _bundle()
    disagreements = _judgments(bundle, disagree_all_slots=True)

    with pytest.raises(ValidationError, match="requires exactly one adjudicator"):
        score_study(
            _study(),
            bundle,
            disagreements,
            responses=load_responses(RESPONSES),
            reference_protocols=_references(_study()),
            untouched_pool_attested=True,
            consent_active_attested=True,
        )

    adjudicated = _judgments(
        bundle,
        disagree_all_slots=True,
        include_adjudicator=True,
    )
    report = score_study(
        _study(),
        bundle,
        adjudicated,
        responses=load_responses(RESPONSES),
        reference_protocols=_references(_study()),
        untouched_pool_attested=True,
        consent_active_attested=True,
    )
    assert report["judge_agreement"]["adjudicated_slot_count"] == 16


def test_disjointness_hides_values_and_rejects_overlap():
    study = _study()
    overlap = audit_study_disjointness(study, [study])

    disjoint = audit_study_disjointness(_make_disjoint(study), [study])

    assert overlap["disjoint"] is False
    assert overlap["comparisons"][0]["overlaps"]["participants"]["count"] == 4
    assert all(
        value.startswith("sha256:")
        for value in overlap["comparisons"][0]["overlaps"]["participants"][
            "fingerprints"
        ]
    )
    assert "opaque:a000000000000001" not in repr(overlap)
    assert disjoint["disjoint"] is True


def test_private_condition_relabel_and_duplicate_packet_are_rejected():
    study = _study()
    responses = load_responses(RESPONSES)
    bundle = _bundle()
    judgments = _judgments(bundle)

    relabeled = copy.deepcopy(bundle)
    conditions = [
        relabeled["private"]["assignment_key"][0]["slots"][slot]["condition"]
        for slot in SLOTS
    ]
    for index, slot in enumerate(SLOTS):
        relabeled["private"]["assignment_key"][0]["slots"][slot]["condition"] = (
            conditions[(index + 1) % len(conditions)]
        )
    with pytest.raises(ValidationError, match="frozen protocol, responses, or seed"):
        score_study(
            study,
            relabeled,
            judgments,
            responses=responses,
            reference_protocols=_references(study),
            untouched_pool_attested=True,
            consent_active_attested=True,
        )

    duplicated = copy.deepcopy(bundle)
    duplicated["public"]["judge_packets"].append(
        copy.deepcopy(duplicated["public"]["judge_packets"][0])
    )
    duplicated["private"]["assignment_key"].append(
        copy.deepcopy(duplicated["private"]["assignment_key"][0])
    )
    with pytest.raises(ValidationError, match="public packet_id|private episode_id"):
        score_study(
            study,
            duplicated,
            judgments,
            responses=responses,
            reference_protocols=_references(study),
            untouched_pool_attested=True,
            consent_active_attested=True,
        )


def test_response_input_manifest_and_judgment_replay_are_rejected():
    study = _study()
    responses = load_responses(RESPONSES)
    bad_responses = copy.deepcopy(responses)
    bad_responses["responses"][0]["input_packet_digest"] = f"sha256:{'f' * 64}"
    with pytest.raises(ValidationError, match="input digest differs"):
        prepare_blinded_packets(study, bad_responses, seed=17)

    old_bundle = _bundle()
    old_judgments = _judgments(old_bundle)
    changed_study = copy.deepcopy(study)
    changed_study["thresholds"]["max_mean_tokens"] += 1
    changed_study = validate_protocol(changed_study)
    changed_responses = copy.deepcopy(responses)
    changed_responses["protocol_fingerprint"] = _fingerprint(changed_study)
    new_bundle = prepare_blinded_packets(changed_study, changed_responses, seed=17)
    with pytest.raises(ValidationError, match="different packet set"):
        score_study(
            changed_study,
            new_bundle,
            old_judgments,
            responses=changed_responses,
            reference_protocols=_references(changed_study),
            untouched_pool_attested=True,
            consent_active_attested=True,
        )


def test_guess_bijection_namespace_and_shadow_panel_fail_closed():
    study = _study()
    bundle = _bundle()
    judgments = _judgments(bundle)
    judgments["judgments"][0]["condition_guesses"] = {
        slot: "memory_v2" for slot in SLOTS
    }
    with pytest.raises(ValidationError, match="assign every condition once"):
        validate_judgments(judgments)

    reference = _make_disjoint(study)
    reference["privacy"]["identity_namespace_digest"] = f"sha256:{'e' * 64}"
    assert audit_study_disjointness(study, [reference])["disjoint"] is False

    shadow_bundle = {
        "schema_version": "memory-v2-earn-canary-shadow-metrics/v1",
        "study_id": study["study_id"],
        "protocol_fingerprint": _fingerprint(study),
        "response_set_fingerprint": _fingerprint(load_responses(RESPONSES)),
        "records": [],
    }
    with pytest.raises(ValidationError, match="exact episode panel"):
        score_study(
            study,
            bundle,
            _judgments(bundle),
            responses=load_responses(RESPONSES),
            reference_protocols=_references(study),
            untouched_pool_attested=True,
            consent_active_attested=True,
            shadow_metric_bundle=shadow_bundle,
        )


def test_yaml_aliases_and_permissive_pilot_thresholds_are_rejected(tmp_path):
    alias_path = tmp_path / "alias.yaml"
    alias_path.write_text("x: &x [1, 2]\ny: *x\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="aliases are forbidden"):
        load_protocol(alias_path)

    pilot = copy.deepcopy(_study())
    pilot["study_mode"] = "pilot"
    with pytest.raises(ValidationError, match="at least 500"):
        validate_protocol(pilot)

    vacuous_rubric = copy.deepcopy(_study())
    vacuous_rubric["study_mode"] = "pilot"
    vacuous_rubric["design"]["minimum_dimension_score"] = 1
    with pytest.raises(ValidationError, match="fixed value 3"):
        validate_protocol(vacuous_rubric)

    thresholds = copy.deepcopy(_study()["thresholds"])
    thresholds.update({
        "minimum_episodes": 500,
        "minimum_participant_clusters": 30,
        "minimum_project_clusters": 20,
        "minimum_effective_participant_clusters": 20,
        "minimum_effective_project_clusters": 15,
        "max_participant_episode_share": 0.10,
        "max_project_episode_share": 0.20,
        "minimum_episodes_per_checkpoint": 25,
        "minimum_episodes_per_query_class": 25,
        "minimum_participant_clusters_per_stratum": 10,
        "minimum_project_clusters_per_stratum": 8,
        "minimum_shadow_eligible_records": 50,
        "required_query_classes": list(QUERY_CLASSES),
        "bootstrap_samples": 10_000,
        "confidence_level": 0.95,
        "stratum_noninferiority_margin": -1.0,
        "min_judge_agreement": 0.80,
        "min_judge_kappa": 0.60,
        "max_condition_guess_accuracy": 0.35,
    })
    with pytest.raises(ValidationError, match="noninferiority margin"):
        _validate_thresholds(thresholds, study_mode="pilot")


def test_false_go_cluster_shadow_and_blinding_shortcuts_are_closed():
    study = _study()
    thresholds = study["thresholds"]
    thresholds.update({
        "minimum_episodes": 500,
        "minimum_participant_clusters": 30,
        "minimum_project_clusters": 20,
        "minimum_effective_participant_clusters": 20,
        "minimum_effective_project_clusters": 15,
        "max_participant_episode_share": 0.10,
        "max_project_episode_share": 0.20,
        "minimum_episodes_per_checkpoint": 25,
        "minimum_episodes_per_query_class": 25,
        "minimum_participant_clusters_per_stratum": 10,
        "minimum_project_clusters_per_stratum": 8,
        "minimum_shadow_eligible_records": 50,
    })
    rows = [
        {
            "participant_ref": "opaque:aaaaaaaaaaaaaaaa",
            "project_ref": "opaque:bbbbbbbbbbbbbbbb",
            "checkpoint_days": (30, 90, 365)[index % 3],
            "query_class": thresholds["required_query_classes"][
                index % len(thresholds["required_query_classes"])
            ],
        }
        for index in range(500)
    ]
    coverage = _coverage(study, rows)
    assert coverage["participant_effective_coverage"] is False
    assert coverage["project_effective_coverage"] is False
    assert coverage["participant_dominance"] is False
    assert coverage["project_dominance"] is False

    two_row_metrics = {
        "top1_useful": 1.0,
        "evidence_set_completeness": 1.0,
        "none_precision": 1.0,
        "none_recall": 1.0,
        "irrelevant_injection": 0.0,
        "stale_as_current": 0.0,
        "citation_validity": 1.0,
        "eligible_gates": {
            name: True
            for name in (
                "top1_useful",
                "evidence_set_completeness",
                "none_precision",
                "none_recall",
                "irrelevant_injection",
                "stale_as_current",
                "citation_validity",
            )
        },
        "coverage": {
            "memory_needed": 2,
            "required_evidence_sets": 2,
            "predicted_none": 2,
            "expected_none": 2,
            "current_selected_items": 2,
            "citations": 2,
        },
    }
    assert not any(_shadow_gate_status(two_row_metrics, thresholds).values())
    assert _wilson_interval(2_000, 2_000)["upper"] > 0.35

    guess_packets = [(8, 8)] * 333 + [(0, 8)] * 667
    clustered_guess = _packet_cluster_guess_interval(
        guess_packets,
        samples=3_000,
        seed=7,
        confidence=0.95,
    )
    slot_wilson = _wilson_interval(333 * 8, 1_000 * 8)
    assert slot_wilson["upper"] < 0.35
    assert clustered_guess["upper"] > 0.35

    agreement_packets = [
        [(True, True), (False, False), (True, True), (False, False)]
    ] * 405 + [[(True, False), (False, True), (True, False), (False, True)]] * 95
    clustered_agreement = _packet_cluster_agreement_interval(
        agreement_packets,
        samples=3_000,
        seed=11,
        confidence=0.95,
    )
    assert clustered_agreement["raw_agreement"]["lower"] < 0.80
    assert clustered_agreement["cohen_kappa"]["lower"] < 0.60


def test_consent_and_retention_block_processing_before_private_artifacts():
    study = _study()
    with pytest.raises(ValidationError, match="active consent"):
        score_study(
            study,
            {},
            {},
            responses={},
            reference_protocols=[],
            untouched_pool_attested=False,
            consent_active_attested=False,
        )

    expired = copy.deepcopy(study)
    expired["created_at"] = "2020-07-24T12:00:00Z"
    expired["privacy"]["retention_until"] = "2021-07-24T12:00:00Z"
    query_as_of = datetime(2020, 7, 1, 12, tzinfo=timezone.utc)
    for episode in expired["episodes"]:
        episode["query_as_of"] = query_as_of.isoformat().replace("+00:00", "Z")
        episode["evidence_cutoff"] = episode["query_as_of"]
        episode["target_evidence_at"] = (
            (query_as_of - timedelta(days=episode["checkpoint_days"]))
            .isoformat()
            .replace("+00:00", "Z")
        )
    expired = validate_protocol(expired)
    with pytest.raises(ValidationError, match="retention expired"):
        score_study(
            expired,
            {},
            {},
            responses={},
            reference_protocols=[],
            untouched_pool_attested=False,
            consent_active_attested=True,
        )
