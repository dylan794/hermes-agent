"""End-to-end CLI contracts for the private Earn-the-Canary workflow."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import memory_v2_earn_canary as earn_canary_cli
from plugins.memory.memory_v2.evals.earn_canary import (
    DIMENSIONS,
    JUDGMENTS_SCHEMA_VERSION,
    SLOTS,
    ValidationError,
    audit_study_disjointness,
    load_protocol,
)


REPO_ROOT = Path(__file__).parents[4]
FIXTURES = REPO_ROOT / "plugins" / "memory" / "memory_v2" / "evals" / "fixtures"
PROTOCOL = FIXTURES / "earn_canary_protocol_synthetic_v1.yaml"
RESPONSES = FIXTURES / "earn_canary_responses_synthetic_v1.yaml"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "scripts/memory_v2_earn_canary.py", *args],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def _write_judgments(path: Path, packets: dict, key: dict) -> None:
    assignments = {row["packet_id"]: row["slots"] for row in key["assignment_key"]}
    rows = []
    for packet_index, packet in enumerate(packets["judge_packets"], start=1):
        for judge_number in (1, 2):
            scores = {}
            taxonomy = {}
            for slot in SLOTS:
                condition = assignments[packet["packet_id"]][slot]["condition"]
                success = condition in {"memory_v2", "oracle_evidence"}
                score = 5 if success else 2
                scores[slot] = {dimension: score for dimension in DIMENSIONS}
                taxonomy[slot] = (
                    None
                    if success
                    else {
                        "primary": "candidate_recall",
                        "secondary": None,
                        "label_source": "independent_judge",
                        "judgment_ref": "judgment:synthetic-cli",
                    }
                )
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
    path.write_text(
        json.dumps({
            "schema_version": JUDGMENTS_SCHEMA_VERSION,
            "study_id": packets["study_id"],
            "packet_set_fingerprint": packets["packet_set_fingerprint"],
            "judgments": rows,
        }),
        encoding="utf-8",
    )


def _write_disjoint_reference(path: Path) -> None:
    candidate = load_protocol(PROTOCOL)
    reference = json.loads(json.dumps(candidate))
    reference["study_id"] = "earn_canary_cli_disjoint_reference_v1"
    for index, episode in enumerate(reference["episodes"], start=301):
        episode["episode_id"] = f"opaque:{index:016x}"
        episode["participant_ref"] = f"opaque:{index + 100:016x}"
        episode["project_ref"] = f"opaque:{index + 200:016x}"
        episode["workstream_ref"] = f"opaque:{index + 300:016x}"
        episode["query_ref"] = f"opaque:{index + 400:016x}"
        episode["query_hash"] = f"sha256:{index + 500:064x}"
        episode["snapshot"]["corpus_ref"] = f"opaque:{index + 600:016x}"
        episode["snapshot"]["archive_digest"] = f"sha256:{index + 700:064x}"
    assert audit_study_disjointness(candidate, [reference])["disjoint"] is True
    path.write_text(json.dumps(reference), encoding="utf-8")


def test_cli_validates_and_emits_only_a_content_free_receipt(tmp_path):
    output = tmp_path / "validation.json"
    completed = _run(
        "validate",
        "--protocol",
        str(PROTOCOL),
        "--output",
        str(output),
    )

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert receipt == {
        "command": "validate",
        "episode_count": 4,
        "study_mode": "development",
        "valid": True,
    }
    assert report["mutation_authority"] == "none"
    assert "dashboard setting" not in completed.stdout
    assert str(PROTOCOL) not in completed.stdout


def test_cli_prepares_separate_blinded_immutable_artifacts(tmp_path):
    packets_path = tmp_path / "packets.json"
    key_path = tmp_path / "key.json"
    command = (
        "prepare",
        "--protocol",
        str(PROTOCOL),
        "--responses",
        str(RESPONSES),
        "--packet-output",
        str(packets_path),
        "--key-output",
        str(key_path),
        "--seed",
        "17",
    )

    completed = _run(*command)
    repeated = _run(*command)

    assert completed.returncode == 0, completed.stderr
    assert repeated.returncode == 2
    assert "refusing to overwrite" in repeated.stderr
    packets = json.loads(packets_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert len(packets["judge_packets"]) == 4
    assert "assignment_key" not in packets
    assert "judge_packets" not in key
    assert "participant_ref" not in repr(packets)
    assert "condition" not in repr(packets)
    assert "blue synthetic setting" not in completed.stdout


def test_cli_score_requires_frozen_unblind_authority_and_returns_no_go(tmp_path):
    packets_path = tmp_path / "packets.json"
    key_path = tmp_path / "key.json"
    prepared = _run(
        "prepare",
        "--protocol",
        str(PROTOCOL),
        "--responses",
        str(RESPONSES),
        "--packet-output",
        str(packets_path),
        "--key-output",
        str(key_path),
        "--seed",
        "17",
    )
    assert prepared.returncode == 0, prepared.stderr
    packets = json.loads(packets_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    judgments_path = tmp_path / "judgments.json"
    _write_judgments(judgments_path, packets, key)
    reference_path = tmp_path / "prior-protocol.json"
    _write_disjoint_reference(reference_path)
    output = tmp_path / "report.json"
    base = (
        "score",
        "--protocol",
        str(PROTOCOL),
        "--responses",
        str(RESPONSES),
        "--packets",
        str(packets_path),
        "--key",
        str(key_path),
        "--judgments",
        str(judgments_path),
        "--against",
        str(reference_path),
        "--output",
        str(output),
    )

    unauthorized = _run(*base)
    missing_consent = _run(*base, "--authorize-unblind")
    completed = _run(
        *base,
        "--authorize-unblind",
        "--attest-untouched-pool",
        "--attest-consent-active",
        "--require-go",
    )

    assert unauthorized.returncode == 2
    assert "--authorize-unblind" in unauthorized.stderr
    assert missing_consent.returncode == 2
    assert "--attest-consent-active" in missing_consent.stderr
    assert completed.returncode == 1, completed.stderr
    receipt = json.loads(completed.stdout)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["go"] is False
    assert receipt["mutation_authority"] == "none"
    assert report["comparisons"]["memory_v2_vs_raw_fts"]["paired_success_delta"] == 1.0
    assert report["gates"]["pilot_design"] is False
    assert "synthetic resume point" not in completed.stdout


def test_cli_disjointness_uses_distinct_valid_exit_code(tmp_path):
    output = tmp_path / "overlap.json"
    completed = _run(
        "audit-disjoint",
        "--candidate",
        str(PROTOCOL),
        "--against",
        str(PROTOCOL),
        "--output",
        str(output),
    )

    assert completed.returncode == 1, completed.stderr
    assert json.loads(completed.stdout)["disjoint"] is False
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["disjoint"] is False
    assert "opaque:a000000000000001" not in output.read_text(encoding="utf-8")


def test_cli_errors_do_not_echo_private_paths_with_spaces(tmp_path):
    missing = tmp_path / "private data" / "alice secret.yaml"
    completed = _run("validate", "--protocol", str(missing))

    assert completed.returncode == 2
    assert str(missing) not in completed.stderr
    assert "alice secret" not in completed.stderr


def test_pilot_paths_fail_closed_on_native_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(earn_canary_cli.os, "name", "nt")

    with pytest.raises(ValidationError, match="hardened POSIX permissions"):
        earn_canary_cli._enforce_real_study_paths(
            {"study_mode": "pilot"},
            inputs=(tmp_path / "protocol.yaml",),
            outputs=(tmp_path / "report.json",),
        )
