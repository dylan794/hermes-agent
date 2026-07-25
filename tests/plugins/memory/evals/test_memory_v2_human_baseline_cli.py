"""Operator CLI tests for the Memory v2 human-baseline workflow."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).parents[4]
FIXTURES = REPO_ROOT / "plugins" / "memory" / "memory_v2" / "evals" / "fixtures"
PROTOCOL = FIXTURES / "human_baseline_protocol_example_v1.yaml"
HUMAN_RESPONSES = FIXTURES / "human_baseline_human_responses_example_v1.json"
MEMORY_RESPONSES = FIXTURES / "human_baseline_memory_responses_example_v1.json"
JUDGMENTS = FIXTURES / "human_baseline_judgments_example_v1.json"
PILOT_TEMPLATE = FIXTURES / "human_baseline_pilot_protocol_template_v1.yaml.example"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "scripts/memory_v2_human_baseline.py", *args],
        check=False,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )


def _prepare(tmp_path: Path, *, suffix: str = "") -> tuple[dict[str, Any], dict[str, Any]]:
    packets = tmp_path / f"packets{suffix}.json"
    key = tmp_path / f"key{suffix}.json"
    completed = _run(
        "prepare",
        "--protocol",
        str(PROTOCOL),
        "--human-responses",
        str(HUMAN_RESPONSES),
        "--memory-responses",
        str(MEMORY_RESPONSES),
        "--packet-output",
        str(packets),
        "--key-output",
        str(key),
        "--seed",
        "1701",
    )
    assert completed.returncode == 0, completed.stderr
    return (
        json.loads(packets.read_text(encoding="utf-8")),
        json.loads(key.read_text(encoding="utf-8")),
    )


def _walk(value: Any):
    yield value
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _materialize_pilot_template(tmp_path: Path) -> Path:
    text = PILOT_TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "REPLACE_WITH_UNIQUE_PILOT_STUDY_ID": "pilot_operational_rehearsal_v1",
        **{
            f"REPLACE_WITH_OPAQUE_PILOT_PARTICIPANT_{index:02d}": (
                f"pilot_opaque_registry_ref_{index:02d}"
            )
            for index in range(1, 7)
        },
    }
    for marker, value in replacements.items():
        text = text.replace(marker, value)
    text = text.replace("REPLACE:", "Pilot item:")
    protocol = tmp_path / "pilot.yaml"
    protocol.write_text(text, encoding="utf-8")
    return protocol


def _write_judgments(
    path: Path, public_packets: dict[str, Any], *, complete: bool = True
) -> None:
    packets = public_packets["judge_packets"]
    if not complete:
        packets = packets[:-1]
    payload = {
        "schema_version": "memory-v2-human-baseline-judgments/v1",
        "study_id": public_packets["study_id"],
        "judgments": [
            {
                "packet_id": packet["packet_id"],
                "judge_id": "synthetic_blinded_judge_01",
                "judge_role": "primary",
                "scores": {
                    slot: {dimension: 4 for dimension in packet["applicable_dimensions"]}
                    for slot in ("A", "B")
                },
                "gate_failures": {"A": [], "B": []},
                "material_error_notes": {"A": None, "B": None},
                "potentially_identifiable": {"A": False, "B": False},
                "condition_guesses": {"A": "unsure", "B": "unsure"},
            }
            for packet in packets
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_human_baseline_cli_direct_script_bootstraps_repo_imports():
    completed = _run("--help")

    assert completed.returncode == 0, completed.stderr
    assert "blinded Memory v2 human-baseline study" in completed.stdout


def test_validate_accepts_explicitly_non_claim_synthetic_protocol():
    completed = _run("validate", "--protocol", str(PROTOCOL))

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["valid"] is True
    assert payload["study_id"] == "human_baseline_synthetic_mechanics_only_not_claim_evidence_v1"
    assert payload["study_mode"] == "development"
    assert payload["participant_count"] == 1
    assert payload["query_count"] == 6


def test_validate_fails_closed_without_traceback(tmp_path):
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("study_id: incomplete\n", encoding="utf-8")

    completed = _run("validate", "--protocol", str(invalid))

    assert completed.returncode == 2
    assert "error:" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_prepare_is_deterministic_and_keeps_public_packets_blind(tmp_path):
    public_a, key_a = _prepare(tmp_path, suffix="_a")
    public_b, key_b = _prepare(tmp_path, suffix="_b")

    assert public_a == public_b
    assert key_a == key_b
    assert len(public_a["judge_packets"]) == 6
    assert "seed" not in public_a
    assert "assignment_key" not in public_a
    assert key_a["seed"] == 1701
    public_scalars = set(item for item in _walk(public_a) if isinstance(item, str))
    assert "human" not in public_scalars
    assert "memory_v2" not in public_scalars
    assert "synthetic_operator_01" not in public_scalars
    assert "participant_id" not in public_scalars
    assert "query_id" not in public_scalars
    assert "workstream_id" not in public_scalars

    serialized_key = json.dumps(key_a, sort_keys=True)
    assert "synthetic_operator_01" not in serialized_key
    assert "current_state_30d" not in serialized_key
    assert "synthetic_project_atlas" not in serialized_key


def test_prepare_rejects_condition_mislabeled_response_file(tmp_path):
    payload = json.loads(HUMAN_RESPONSES.read_text(encoding="utf-8"))
    payload["responses"][0]["condition"] = "memory_v2"
    mislabeled = tmp_path / "mislabeled.json"
    mislabeled.write_text(json.dumps(payload), encoding="utf-8")

    completed = _run(
        "prepare",
        "--protocol",
        str(PROTOCOL),
        "--human-responses",
        str(mislabeled),
        "--memory-responses",
        str(MEMORY_RESPONSES),
        "--packet-output",
        str(tmp_path / "packets.json"),
        "--key-output",
        str(tmp_path / "key.json"),
        "--seed",
        "1701",
    )

    assert completed.returncode == 2
    assert "condition must be 'human'" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_prepare_rejects_participant_identity_embedded_in_judge_text(tmp_path):
    payload = json.loads(HUMAN_RESPONSES.read_text(encoding="utf-8"))
    payload["responses"][0]["response"] += " Participant: synthetic_operator_01."
    identifying = tmp_path / "identifying.json"
    identifying.write_text(json.dumps(payload), encoding="utf-8")

    completed = _run(
        "prepare",
        "--protocol",
        str(PROTOCOL),
        "--human-responses",
        str(identifying),
        "--memory-responses",
        str(MEMORY_RESPONSES),
        "--packet-output",
        str(tmp_path / "packets.json"),
        "--key-output",
        str(tmp_path / "key.json"),
        "--seed",
        "1701",
    )

    assert completed.returncode == 2
    assert "participant identity" in completed.stderr
    assert not (tmp_path / "packets.json").exists()
    assert not (tmp_path / "key.json").exists()


def test_prepare_refuses_to_mix_public_packets_and_private_key(tmp_path):
    shared_output = tmp_path / "shared.json"
    completed = _run(
        "prepare",
        "--protocol",
        str(PROTOCOL),
        "--human-responses",
        str(HUMAN_RESPONSES),
        "--memory-responses",
        str(MEMORY_RESPONSES),
        "--packet-output",
        str(shared_output),
        "--key-output",
        str(shared_output),
        "--seed",
        "1701",
    )

    assert completed.returncode == 2
    assert "must be different files" in completed.stderr
    assert not shared_output.exists()


def test_score_reports_development_example_ineligible_without_treating_it_as_error(
    tmp_path,
):
    public_packets, _ = _prepare(tmp_path, suffix="_score")
    report = tmp_path / "report.json"

    completed = _run(
        "score",
        "--protocol",
        str(PROTOCOL),
        "--packets",
        str(tmp_path / "packets_score.json"),
        "--key",
        str(tmp_path / "key_score.json"),
        "--judgments",
        str(JUDGMENTS),
        "--output",
        str(report),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["valid_complete"] is True
    assert payload["decision"] == "ineligible_by_design"
    assert payload["superiority_claim"] is False
    assert payload["claim_gates"]["confirmatory_study_mode"] is False
    assert payload["coverage"]["required_checkpoints_covered"] is True
    assert payload["coverage"]["required_strata_covered"] is True
    assert payload["blinding_diagnostic"]["unsure_rate"] == 1.0

    required = _run(
        "score",
        "--protocol",
        str(PROTOCOL),
        "--packets",
        str(tmp_path / "packets_score.json"),
        "--key",
        str(tmp_path / "key_score.json"),
        "--judgments",
        str(JUDGMENTS),
        "--require-superiority",
    )
    assert required.returncode == 1
    assert json.loads(required.stdout)["decision"] == "ineligible_by_design"


def test_score_rejects_incomplete_judgments_with_distinct_exit_code(tmp_path):
    public_packets, _ = _prepare(tmp_path, suffix="_incomplete")
    judgments = tmp_path / "incomplete.json"
    _write_judgments(judgments, public_packets, complete=False)

    completed = _run(
        "score",
        "--protocol",
        str(PROTOCOL),
        "--packets",
        str(tmp_path / "packets_incomplete.json"),
        "--key",
        str(tmp_path / "key_incomplete.json"),
        "--judgments",
        str(judgments),
    )

    assert completed.returncode == 2
    assert "incomplete judgments" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_example_protocol_covers_preregistered_dimensions_and_timepoints():
    payload = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    covered_dimensions = {
        dimension
        for query in payload["queries"]
        for dimension in query["applicable_dimensions"]
    }
    primary_strata = {
        query["stratum"]
        for query in payload["queries"]
        if query["checkpoint_days"] == payload["thresholds"]["primary_checkpoint"]
    }

    assert payload["study_mode"] == "development"
    assert covered_dimensions == set(payload["dimensions"])
    assert set(payload["thresholds"]["required_checkpoints"]) == {30, 90, 180, 365}
    assert set(payload["thresholds"]["required_strata"]) <= primary_strata


def test_audit_disjoint_cli_accepts_clean_pilot_and_fails_on_overlap(tmp_path):
    reference = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    candidate = json.loads(json.dumps(reference))
    candidate["study_id"] = "pilot_pool_disjointness_test"
    candidate["study_mode"] = "pilot"
    candidate["participants"] = [{"id": "pilot_operator_disjoint_01"}]
    for index, query in enumerate(candidate["queries"]):
        query["id"] = f"pilot_disjoint_query_{index}"
        query["workstream_id"] = f"pilot_disjoint_workstream_{index}"
        query["prompt"] = f"Pilot-only continuity prompt {index}"
    candidate_path = tmp_path / "candidate.yaml"
    candidate_path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
    report_path = tmp_path / "disjoint.json"

    clean = _run(
        "audit-disjoint",
        "--candidate",
        str(candidate_path),
        "--against",
        str(PROTOCOL),
        "--output",
        str(report_path),
    )
    assert clean.returncode == 0, clean.stderr
    assert json.loads(report_path.read_text(encoding="utf-8"))["disjoint"] is True

    candidate["queries"][0]["workstream_id"] = reference["queries"][0]["workstream_id"]
    candidate_path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
    overlapping = _run(
        "audit-disjoint",
        "--candidate",
        str(candidate_path),
        "--against",
        str(PROTOCOL),
    )
    assert overlapping.returncode == 1
    payload = json.loads(overlapping.stdout)
    assert payload["disjoint"] is False
    assert payload["comparisons"][0]["overlaps"]["workstream_ids"]["count"] == 1


def test_pilot_template_becomes_a_valid_small_pilot_only_after_markers_are_replaced(
    tmp_path,
):
    text = PILOT_TEMPLATE.read_text(encoding="utf-8")
    assert "REPLACE" in text
    protocol = _materialize_pilot_template(tmp_path)

    completed = _run("validate", "--protocol", str(protocol))
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["study_mode"] == "pilot"
    assert payload["participant_count"] == 6
    assert payload["query_count"] == 8


def test_small_pilot_operational_rehearsal_runs_end_to_end(tmp_path):
    protocol_path = _materialize_pilot_template(tmp_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    response_documents = {}
    for condition in ("human", "memory_v2"):
        rows = []
        for participant_index, participant in enumerate(protocol["participants"]):
            for query_index, query in enumerate(protocol["queries"]):
                rows.append(
                    {
                        "participant_id": participant["id"],
                        "query_id": query["id"],
                        "condition": condition,
                        "response": (
                            f"Grounded evidence summary case {participant_index} item {query_index}."
                            if condition == "memory_v2"
                            else f"Open-book work summary case {participant_index} item {query_index}."
                        ),
                    }
                )
        response_documents[condition] = {
            "schema_version": "memory-v2-human-baseline-responses/v1",
            "study_id": protocol["study_id"],
            "responses": rows,
        }

    human_path = tmp_path / "pilot-human.json"
    memory_path = tmp_path / "pilot-memory.json"
    packets_path = tmp_path / "pilot-packets.json"
    key_path = tmp_path / "pilot-key.json"
    human_path.write_text(json.dumps(response_documents["human"]), encoding="utf-8")
    memory_path.write_text(json.dumps(response_documents["memory_v2"]), encoding="utf-8")
    prepared = _run(
        "prepare",
        "--protocol",
        str(protocol_path),
        "--human-responses",
        str(human_path),
        "--memory-responses",
        str(memory_path),
        "--packet-output",
        str(packets_path),
        "--key-output",
        str(key_path),
        "--seed",
        "8128",
    )
    assert prepared.returncode == 0, prepared.stderr

    packets = json.loads(packets_path.read_text(encoding="utf-8"))
    private_key = json.loads(key_path.read_text(encoding="utf-8"))
    key_by_packet = {row["packet_id"]: row for row in private_key["assignment_key"]}
    participant_refs = sorted({row["participant_ref"] for row in private_key["assignment_key"]})
    participant_index = {value: index for index, value in enumerate(participant_refs)}
    judgments = []
    for packet in packets["judge_packets"]:
        key_row = key_by_packet[packet["packet_id"]]
        human_succeeds = participant_index[key_row["participant_ref"]] % 2 == 0
        for judge_id in ("pilot-primary-judge-01", "pilot-primary-judge-02"):
            scores = {}
            notes = {"A": None, "B": None}
            for slot in ("A", "B"):
                condition = key_row["slots"][slot]["condition"]
                value = 4 if condition == "memory_v2" or human_succeeds else 2
                scores[slot] = {
                    dimension: value for dimension in packet["applicable_dimensions"]
                }
                if value < protocol["thresholds"]["dimension_minimum"]:
                    notes[slot] = {
                        "reason": "Synthetic rehearsal material error.",
                        "evidence_refs": [f"rehearsal:{packet['packet_id']}:{slot}"],
                    }
            judgments.append(
                {
                    "packet_id": packet["packet_id"],
                    "judge_id": judge_id,
                    "judge_role": "primary",
                    "scores": scores,
                    "gate_failures": {"A": [], "B": []},
                    "condition_guesses": {"A": "unsure", "B": "unsure"},
                    "material_error_notes": notes,
                    "potentially_identifiable": {"A": False, "B": False},
                }
            )
    judgments_path = tmp_path / "pilot-judgments.json"
    report_path = tmp_path / "pilot-report.json"
    judgments_path.write_text(
        json.dumps(
            {
                "schema_version": "memory-v2-human-baseline-judgments/v1",
                "study_id": protocol["study_id"],
                "judgments": judgments,
            }
        ),
        encoding="utf-8",
    )
    scored = _run(
        "score",
        "--protocol",
        str(protocol_path),
        "--packets",
        str(packets_path),
        "--key",
        str(key_path),
        "--judgments",
        str(judgments_path),
        "--output",
        str(report_path),
    )
    assert scored.returncode == 0, scored.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["decision"] == "ineligible_by_design"
    assert report["pilot_diagnostics"]["eligible_for_power_analysis"] is True
    assert len(report["pilot_diagnostics"]["item_difficulty"]) == 8
    assert report["pilot_diagnostics"]["participant_cluster_variance"]["primary"][
        "participants"
    ] == 6
    assert report["pilot_diagnostics"]["participant_cluster_variance"]["primary"][
        "participant_mean_sample_variance"
    ] > 0
    assert report["inter_rater"]["binary_agreement_rate"] == 1.0
    assert report["blinding_diagnostic"]["unsure_rate"] == 1.0
