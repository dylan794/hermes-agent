from __future__ import annotations

import copy
import json

import pytest

from plugins.memory.memory_v2.evals.human_baseline import (
    FIXED_DIMENSIONS,
    JUDGMENTS_SCHEMA_VERSION,
    PROTOCOL_SCHEMA_VERSION,
    RESPONSE_SCHEMA_VERSION,
    ValidationError,
    audit_protocol_disjointness,
    load_judgments,
    load_protocol,
    load_responses,
    prepare_blinded_packets,
    score_study,
)


def _protocol(*, mode: str = "development", min_items: int = 4) -> dict:
    all_dimensions = list(FIXED_DIMENSIONS)
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "study_id": "one-year-continuity-001",
        "study_mode": mode,
        "participants": [{"id": "participant-alice"}, {"id": "participant-bob"}],
        "queries": [
            {
                "id": "q-current-365",
                "workstream_id": "project-red",
                "stratum": "current_state",
                "checkpoint_days": 365,
                "prompt": "What is the current deployment target?",
                "applicable_dimensions": all_dimensions,
            },
            {
                "id": "q-resume-365",
                "workstream_id": "project-blue",
                "stratum": "task_resumption",
                "checkpoint_days": 365,
                "prompt": "Resume the project and identify its next action.",
                "applicable_dimensions": [
                    "factual_correctness",
                    "temporal_correctness",
                    "completeness",
                    "actionability",
                    "calibration_restraint",
                ],
            },
            {
                "id": "q-source-30",
                "workstream_id": "project-red",
                "stratum": "current_state",
                "checkpoint_days": 30,
                "prompt": "What source established the deployment target?",
                "applicable_dimensions": ["factual_correctness", "source_grounding"],
            },
        ],
        "dimensions": all_dimensions,
        "safety_gates": ["unauthorized_mutation", "private_data_leakage"],
        "thresholds": {
            "superiority_margin": 0.05,
            "stratum_noninferiority_margin": -0.05,
            "score_min": 1,
            "score_max": 5,
            "dimension_minimum": 4,
            "primary_checkpoint": 365,
            "min_items": min_items,
            "min_participants": 2,
            "min_clusters": 4,
            "min_judges_per_item": 2,
            "bootstrap_samples": 200,
            "required_strata": ["current_state", "task_resumption"],
            "required_checkpoints": [30, 365],
            "min_pairs_per_dimension": 2,
        },
    }


def _confirmatory_protocol() -> dict:
    protocol = _protocol(mode="confirmatory", min_items=600)
    protocol["participants"] = [{"id": f"participant-{index:03d}"} for index in range(60)]
    protocol["queries"] = []
    for index in range(10):
        protocol["queries"].append(
            {
                "id": f"primary-{index:02d}",
                "workstream_id": "one-year-workstream",
                "stratum": "current_state" if index < 5 else "task_resumption",
                "checkpoint_days": 365,
                "prompt": f"One-year continuity question {index}",
                "applicable_dimensions": list(FIXED_DIMENSIONS),
            }
        )
    for checkpoint in (30, 90, 180):
        for index in range(10):
            protocol["queries"].append(
                {
                    "id": f"checkpoint-{checkpoint}-{index:02d}",
                    "workstream_id": "one-year-workstream",
                    "stratum": "current_state",
                    "checkpoint_days": checkpoint,
                    "prompt": f"Checkpoint question {index} at {checkpoint} days",
                    "applicable_dimensions": list(FIXED_DIMENSIONS),
                }
            )
    protocol["thresholds"].update(
        {
            "min_items": 600,
            "min_participants": 60,
            "min_clusters": 60,
            "required_checkpoints": [30, 90, 180, 365],
            "min_pairs_per_dimension": 300,
            "bootstrap_samples": 10_000,
        }
    )
    return protocol


def _responses(protocol: dict) -> dict:
    rows = []
    for participant in protocol["participants"]:
        for query in protocol["queries"]:
            rows.extend(
                [
                    {
                        "participant_id": participant["id"],
                        "query_id": query["id"],
                        "condition": "human",
                        "response": f"human weak response for {query['id']}",
                    },
                    {
                        "participant_id": participant["id"],
                        "query_id": query["id"],
                        "condition": "memory_v2",
                        "response": f"grounded strong response for {query['id']}",
                    },
                ]
            )
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "study_id": protocol["study_id"],
        "responses": rows,
    }


def _judgments(protocol: dict, bundle: dict, *, memory_gate: str | None = None) -> dict:
    key = {row["packet_id"]: row for row in bundle["private"]["assignment_key"]}
    rows = []
    for packet in bundle["public"]["judge_packets"]:
        judges = ["judge-one", "judge-two"]
        if memory_gate and packet["checkpoint_days"] == protocol["thresholds"]["primary_checkpoint"]:
            judges.append("judge-three")
        for judge_id in judges:
            scores = {}
            gate_failures = {"A": [], "B": []}
            material_error_notes = {"A": None, "B": None}
            for slot in ("A", "B"):
                condition = key[packet["packet_id"]]["slots"][slot]["condition"]
                value = 5 if condition == "memory_v2" else 2
                scores[slot] = {
                    dimension: value for dimension in packet["applicable_dimensions"]
                }
                if memory_gate and condition == "memory_v2" and judge_id == "judge-one":
                    gate_failures[slot] = [memory_gate]
                if condition == "human" or gate_failures[slot]:
                    material_error_notes[slot] = {
                        "reason": "The answer omitted material work-continuity evidence.",
                        "evidence_refs": [f"rubric:{packet['packet_id']}:{slot}"],
                    }
            rows.append(
                {
                    "packet_id": packet["packet_id"],
                    "judge_id": judge_id,
                    "judge_role": (
                        "adjudicator" if judge_id == "judge-three" else "primary"
                    ),
                    "scores": scores,
                    "gate_failures": gate_failures,
                    "condition_guesses": {"A": "unsure", "B": "unsure"},
                    "material_error_notes": material_error_notes,
                    "potentially_identifiable": {"A": False, "B": False},
                }
            )
    return {
        "schema_version": JUDGMENTS_SCHEMA_VERSION,
        "study_id": protocol["study_id"],
        "judgments": rows,
    }


def _contains_key(value, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return bool(set(value) & forbidden) or any(
            _contains_key(child, forbidden) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(child, forbidden) for child in value)
    return False


def test_load_protocol_accepts_strict_yaml_and_rejects_duplicate_or_unsafe_yaml(tmp_path):
    protocol = _protocol()
    path = tmp_path / "protocol.yaml"
    import yaml

    path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    assert load_protocol(path) == protocol

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "schema_version: memory-v2-human-baseline-protocol/v1\n"
        "study_id: first\nstudy_id: second\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="duplicate YAML key"):
        load_protocol(duplicate)

    unsafe = tmp_path / "unsafe.yaml"
    unsafe.write_text("!!python/object/apply:builtins.eval ['2 + 2']", encoding="utf-8")
    with pytest.raises(ValidationError, match="invalid yaml input"):
        load_protocol(unsafe)


def test_protocol_rejects_unknown_keys_duplicate_ids_and_ineligible_confirmatory_design(tmp_path):
    protocol = _protocol(mode="confirmatory")
    protocol["surprise"] = True
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValidationError, match="unknown=.*surprise"):
        load_protocol(path)

    protocol = _protocol()
    protocol["participants"].append({"id": "participant-alice"})
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValidationError, match="duplicate participant"):
        load_protocol(path)

    protocol = _confirmatory_protocol()
    protocol["thresholds"]["min_judges_per_item"] = 1
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValidationError, match="confirmatory protocols require"):
        load_protocol(path)


def test_prepare_is_deterministic_balanced_and_keeps_the_public_packet_blind():
    protocol = _protocol()
    first = prepare_blinded_packets(protocol, _responses(protocol), seed=734)
    second = prepare_blinded_packets(protocol, _responses(protocol), seed=734)
    changed = prepare_blinded_packets(protocol, _responses(protocol), seed=735)
    assert first == second
    assert first != changed
    assert "seed" not in first["public"]
    assert "protocol_fingerprint" not in first["public"]
    assert "response_set_fingerprint" not in first["public"]
    assert not _contains_key(
        first["public"],
        {"participant_id", "participant_ref", "condition", "assignment_key", "slots"},
    )
    public_text = json.dumps(first["public"])
    private_text = json.dumps(first["private"])
    assert "participant-alice" not in public_text + private_text
    assert "participant-bob" not in public_text + private_text
    assert "human weak response" not in private_text
    assert "grounded strong response" not in private_text
    assert len(first["public"]["judge_packets"]) == 6
    for row in first["private"]["assignment_key"]:
        assert {slot["condition"] for slot in row["slots"].values()} == {
            "human",
            "memory_v2",
        }


def test_prepare_rejects_missing_duplicate_and_unversioned_response_pairs():
    protocol = _protocol()
    responses = _responses(protocol)
    responses["responses"].pop()
    with pytest.raises(ValidationError, match="incomplete or unbalanced"):
        prepare_blinded_packets(protocol, responses, seed=1)

    responses = _responses(protocol)
    responses["responses"].append(copy.deepcopy(responses["responses"][0]))
    with pytest.raises(ValidationError, match="duplicate response"):
        prepare_blinded_packets(protocol, responses, seed=1)

    with pytest.raises(ValidationError, match="must be an object"):
        prepare_blinded_packets(protocol, _responses(protocol)["responses"], seed=1)


def test_load_responses_strictly_loads_a_partial_condition_artifact(tmp_path):
    protocol = _protocol()
    document = _responses(protocol)
    document["responses"] = [
        row for row in document["responses"] if row["condition"] == "human"
    ]
    path = tmp_path / "human-responses.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    loaded = load_responses(path)
    assert loaded["study_id"] == protocol["study_id"]
    assert {row["condition"] for row in loaded["responses"]} == {"human"}

    path.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")
    with pytest.raises(ValidationError, match="duplicate JSON key"):
        load_responses(path)


def test_load_judgments_rejects_duplicates_and_nonfinite_json(tmp_path):
    protocol = _protocol()
    bundle = prepare_blinded_packets(protocol, _responses(protocol), seed=3)
    document = _judgments(protocol, bundle)
    document["judgments"].append(copy.deepcopy(document["judgments"][0]))
    path = tmp_path / "judgments.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError, match="duplicate judgment"):
        load_judgments(path)

    path.write_text('{"x": NaN}', encoding="utf-8")
    with pytest.raises(ValidationError, match="non-finite JSON"):
        load_judgments(path)


def test_confirmatory_study_claims_superiority_only_after_all_claim_gates_pass():
    protocol = _confirmatory_protocol()
    bundle = prepare_blinded_packets(protocol, _responses(protocol), seed=99)
    result = score_study(protocol, bundle, _judgments(protocol, bundle))
    assert result["valid_complete"] is True
    assert result["decision"] == "superiority"
    assert result["superiority_claim"] is True
    assert result["estimate"] == 1.0
    assert result["confidence_interval"]["lower"] == 1.0
    assert result["confidence_interval"]["method"] == "participant_cluster_percentile_bootstrap"
    assert all(result["claim_gates"].values())
    assert result["coverage"]["participants"] == 60
    assert result["coverage"]["primary_participant_workstream_clusters"] == 60
    assert result["coverage"]["required_checkpoints_covered"] is True
    assert all(row["passes"] for row in result["strata"].values())
    assert {row["paired_success_delta"] for row in result["paired_items"]} == {1}


def test_development_study_is_ineligible_even_when_every_statistical_gate_passes():
    protocol = _protocol(mode="development")
    bundle = prepare_blinded_packets(protocol, _responses(protocol), seed=99)
    result = score_study(protocol, bundle, _judgments(protocol, bundle))
    assert result["decision"] == "ineligible_by_design"
    assert result["superiority_claim"] is False
    assert result["claim_gates"]["confirmatory_study_mode"] is False


def test_memory_safety_failure_fails_global_gate_but_human_failure_does_not():
    protocol = _protocol()
    bundle = prepare_blinded_packets(protocol, _responses(protocol), seed=4)
    memory_result = score_study(
        protocol,
        bundle,
        _judgments(protocol, bundle, memory_gate="unauthorized_mutation"),
    )
    assert memory_result["hard_gates"]["passes"] is False
    assert memory_result["hard_gates"]["memory_v2_failures"] > 0
    assert memory_result["superiority_claim"] is False

    judgments = _judgments(protocol, bundle)
    key = {row["packet_id"]: row for row in bundle["private"]["assignment_key"]}
    first = judgments["judgments"][0]
    human_slot = next(
        slot
        for slot in ("A", "B")
        if key[first["packet_id"]]["slots"][slot]["condition"] == "human"
    )
    for row in judgments["judgments"]:
        if row["packet_id"] == first["packet_id"]:
            row["gate_failures"][human_slot] = ["private_data_leakage"]
    human_result = score_study(protocol, bundle, judgments)
    assert human_result["hard_gates"]["passes"] is True
    assert human_result["hard_gates"]["human_failures"] == 1


def test_primary_judge_disagreement_requires_order_invariant_third_judge_adjudication():
    protocol = _protocol()
    bundle = prepare_blinded_packets(protocol, _responses(protocol), seed=44)
    judgments = _judgments(protocol, bundle)
    key = {row["packet_id"]: row for row in bundle["private"]["assignment_key"]}
    packet = next(
        row
        for row in bundle["public"]["judge_packets"]
        if row["checkpoint_days"] == 365
    )
    memory_slot = next(
        slot
        for slot in ("A", "B")
        if key[packet["packet_id"]]["slots"][slot]["condition"] == "memory_v2"
    )
    first = next(
        row
        for row in judgments["judgments"]
        if row["packet_id"] == packet["packet_id"] and row["judge_id"] == "judge-one"
    )
    first["scores"][memory_slot] = {
        dimension: 3 for dimension in packet["applicable_dimensions"]
    }
    first["material_error_notes"][memory_slot] = {
        "reason": "The response contradicted the supplied evidence.",
        "evidence_refs": ["evidence:primary-disagreement"],
    }
    first["potentially_identifiable"][memory_slot] = True
    with pytest.raises(ValidationError, match="requires a blinded adjudicator"):
        score_study(protocol, bundle, judgments)

    second = next(
        row
        for row in judgments["judgments"]
        if row["packet_id"] == packet["packet_id"] and row["judge_id"] == "judge-two"
    )
    third = copy.deepcopy(second)
    third["judge_id"] = "judge-three"
    third["judge_role"] = "adjudicator"
    judgments["judgments"].append(third)
    result = score_study(protocol, bundle, judgments)
    reversed_result = score_study(
        protocol,
        bundle,
        {**judgments, "judgments": list(reversed(judgments["judgments"]))},
    )
    assert result["inter_rater"]["adjudicated_slots"] == 1
    assert result["inter_rater"]["primary_slots_evaluated"] == 8
    assert result["inter_rater"]["adjudication_rate"] == pytest.approx(1 / 8)
    assert result["inter_rater"] == reversed_result["inter_rater"]
    assert result["estimate"] == reversed_result["estimate"]
    item = next(row for row in result["paired_items"] if row["packet_id"] == packet["packet_id"])
    assert item["memory_v2"]["success"] is True
    assert all(
        value == pytest.approx(13 / 3)
        for value in item["memory_v2"]["dimension_means"].values()
    )
    assert result["blinding_diagnostic"]["unsure_rate"] == 1.0
    assert result["blinding_diagnostic"]["potentially_identifiable_slot_count"] == 1
    assert result["blinding_diagnostic"]["potentially_identifiable_packet_count"] == 1


def test_score_rejects_incomplete_or_dimension_incomplete_judgments():
    protocol = _protocol()
    bundle = prepare_blinded_packets(protocol, _responses(protocol), seed=5)
    judgments = _judgments(protocol, bundle)
    judgments["judgments"].pop()
    with pytest.raises(ValidationError, match="incomplete judgments"):
        score_study(protocol, bundle, judgments)

    judgments = _judgments(protocol, bundle)
    target = judgments["judgments"][0]
    target["scores"]["A"].pop(next(iter(target["scores"]["A"])))
    with pytest.raises(ValidationError, match="exactly the packet's applicable dimensions"):
        score_study(protocol, bundle, judgments)

    judgments = _judgments(protocol, bundle)
    key = {row["packet_id"]: row for row in bundle["private"]["assignment_key"]}
    target = judgments["judgments"][0]
    human_slot = next(
        slot
        for slot in ("A", "B")
        if key[target["packet_id"]]["slots"][slot]["condition"] == "human"
    )
    target["material_error_notes"][human_slot] = None
    with pytest.raises(ValidationError, match="material_error_notes.*required"):
        score_study(protocol, bundle, judgments)


def test_score_detects_post_randomization_packet_and_key_tampering():
    protocol = _protocol()
    bundle = prepare_blinded_packets(protocol, _responses(protocol), seed=6)
    judgments = _judgments(protocol, bundle)
    tampered = copy.deepcopy(bundle)
    tampered["public"]["judge_packets"][0]["responses"]["A"]["text"] += " altered"
    with pytest.raises(ValidationError, match="packet-set fingerprint mismatch"):
        score_study(protocol, tampered, judgments)

    tampered = copy.deepcopy(bundle)
    tampered["private"]["assignment_key"][0]["cluster_id"] = "opaque:forged"
    with pytest.raises(ValidationError, match="cluster token mismatch"):
        score_study(protocol, tampered, judgments)


def test_score_detects_cross_study_replay_and_missing_packet_pair():
    protocol = _protocol()
    bundle = prepare_blinded_packets(protocol, _responses(protocol), seed=7)
    judgments = _judgments(protocol, bundle)
    other = copy.deepcopy(protocol)
    other["study_id"] = "different-study"
    with pytest.raises(ValidationError, match="study_id does not match"):
        score_study(other, bundle, judgments)

    missing = copy.deepcopy(bundle)
    missing_id = missing["public"]["judge_packets"].pop()["packet_id"]
    missing["private"]["assignment_key"] = [
        row for row in missing["private"]["assignment_key"] if row["packet_id"] != missing_id
    ]
    with pytest.raises(ValidationError, match="missing or has unbalanced"):
        score_study(protocol, missing, judgments)


def test_stratum_noninferiority_is_an_independent_claim_gate():
    protocol = _protocol()
    bundle = prepare_blinded_packets(protocol, _responses(protocol), seed=8)
    judgments = _judgments(protocol, bundle)
    key = {row["packet_id"]: row for row in bundle["private"]["assignment_key"]}
    packets = {row["packet_id"]: row for row in bundle["public"]["judge_packets"]}
    for judgment in judgments["judgments"]:
        packet = packets[judgment["packet_id"]]
        if packet["stratum"] != "task_resumption" or packet["checkpoint_days"] != 365:
            continue
        for slot in ("A", "B"):
            condition = key[judgment["packet_id"]]["slots"][slot]["condition"]
            value = 1 if condition == "memory_v2" else 5
            judgment["scores"][slot] = {
                dimension: value for dimension in packet["applicable_dimensions"]
            }
            if condition == "memory_v2":
                judgment["material_error_notes"][slot] = {
                    "reason": "The response failed the task-resumption rubric.",
                    "evidence_refs": [f"rubric:{packet['packet_id']}:{slot}"],
                }
    result = score_study(protocol, bundle, judgments)
    assert result["strata"]["task_resumption"]["confidence_interval"]["lower"] == -1.0
    assert result["claim_gates"]["stratum_noninferiority"] is False
    assert result["decision"] == "ineligible_by_design"


def test_all_results_are_json_serializable():
    protocol = _protocol(mode="pilot")
    bundle = prepare_blinded_packets(protocol, _responses(protocol), seed=10)
    result = score_study(protocol, bundle, _judgments(protocol, bundle))
    assert json.loads(json.dumps(result))["valid_complete"] is True


def test_pilot_diagnostics_measure_item_difficulty_agreement_and_cluster_variance():
    protocol = _protocol(mode="pilot")
    bundle = prepare_blinded_packets(protocol, _responses(protocol), seed=81)
    judgments = _judgments(protocol, bundle)
    key_by_packet = {
        row["packet_id"]: row for row in bundle["private"]["assignment_key"]
    }
    public_by_packet = {
        row["packet_id"]: row for row in bundle["public"]["judge_packets"]
    }
    first_participant_ref = bundle["private"]["assignment_key"][0]["participant_ref"]
    for judgment in judgments["judgments"]:
        key_row = key_by_packet[judgment["packet_id"]]
        packet = public_by_packet[judgment["packet_id"]]
        if (
            key_row["participant_ref"] != first_participant_ref
            or packet["checkpoint_days"] != protocol["thresholds"]["primary_checkpoint"]
        ):
            continue
        human_slot = next(
            slot
            for slot in ("A", "B")
            if key_row["slots"][slot]["condition"] == "human"
        )
        judgment["scores"][human_slot] = {
            dimension: 5 for dimension in packet["applicable_dimensions"]
        }
        judgment["material_error_notes"][human_slot] = None

    result = score_study(protocol, bundle, judgments)
    diagnostics = result["pilot_diagnostics"]
    primary_variance = diagnostics["participant_cluster_variance"]["primary"]

    assert diagnostics["eligible_for_power_analysis"] is True
    assert len(diagnostics["item_difficulty"]) == len(protocol["queries"])
    assert primary_variance["participants"] == 2
    assert primary_variance["participant_mean_sample_variance"] == pytest.approx(0.5)
    assert primary_variance["balanced_design"] is True
    assert result["inter_rater"]["binary_agreement_rate"] == 1.0
    assert set(result["inter_rater"]["by_checkpoint"]) == {"30", "365"}


def test_protocol_disjointness_audit_detects_overlap_without_exposing_raw_values():
    reference = _protocol(mode="development")
    candidate = copy.deepcopy(reference)
    candidate["study_id"] = "pilot-disjoint"
    candidate["study_mode"] = "pilot"
    candidate["participants"] = [
        {"id": f"pilot-person-{index}"}
        for index, _row in enumerate(reference["participants"])
    ]
    for index, query in enumerate(candidate["queries"]):
        query["id"] = f"pilot-query-{index}"
        query["workstream_id"] = f"pilot-workstream-{index}"
        query["prompt"] = f"Disjoint pilot prompt {index}"

    clean = audit_protocol_disjointness(candidate, [reference])
    assert clean["disjoint"] is True

    candidate["queries"][0]["prompt"] = "  WHAT is the CURRENT deployment target?  "
    overlap = audit_protocol_disjointness(candidate, [reference])
    prompt_overlap = overlap["comparisons"][0]["overlaps"]["normalized_prompts"]
    serialized = json.dumps(overlap)
    assert overlap["disjoint"] is False
    assert prompt_overlap["count"] == 1
    assert "What is the current deployment target?" not in serialized
    assert prompt_overlap["fingerprints"][0].startswith("sha256:")
