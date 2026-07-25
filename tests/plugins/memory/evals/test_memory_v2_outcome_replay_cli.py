"""CLI tests for the Memory v2 outcome-replay lab."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from plugins.memory.memory_v2.evals.outcome_replay import load_dataset
from plugins.memory.memory_v2.evals.outcome_replay import PRIVATE_INTAKE_SCHEMA_VERSION


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


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "scripts/memory_v2_outcome_replay.py", *args],
        check=False,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_disjoint(dataset: dict) -> dict:
    dataset = json.loads(json.dumps(dataset))
    dataset["lab_id"] = "cli_disjoint_rehearsal_v1"
    for index, episode in enumerate(dataset["episodes"], start=901):
        episode["episode_id"] = f"opaque:{index:016x}"
        episode["participant_ref"] = f"opaque:{index + 100:016x}"
        episode["project_ref"] = f"opaque:{index + 200:016x}"
        episode["workstream_ref"] = f"opaque:{index + 300:016x}"
        episode["query_ref"] = f"opaque:{index + 400:016x}"
        episode["query_hash"] = f"sha256:{index + 500:064x}"
        episode["snapshot"]["archive_digest"] = f"sha256:{index + 600:064x}"
    return dataset


def _private_intake() -> dict:
    dataset = load_dataset(FIXTURE)
    episodes = []
    for index, episode in enumerate(dataset["episodes"], start=1):
        snapshot = episode["snapshot"]
        episodes.append(
            {
                "participant_id": f"cli participant {index}",
                "project_id": f"cli project {index}",
                "workstream_id": f"cli workstream {index}",
                "query_text": f"Resume the CLI project {index}",
                "query_class": episode["query_class"],
                "checkpoint_days": episode["checkpoint_days"],
                "evidence_cutoff": episode["evidence_cutoff"],
                "snapshot": {
                    "corpus_id": f"cli corpus {index}",
                    "archive_digest": f"sha256:{index + 700:064x}",
                    "index_digest": snapshot["index_digest"],
                    "config_digest": snapshot["config_digest"],
                    "code_digest": snapshot["code_digest"],
                    "answerer_digest": snapshot["answerer_digest"],
                },
                "variants": episode["variants"],
            }
        )
    return {
        "schema_version": PRIVATE_INTAKE_SCHEMA_VERSION,
        "lab_id": "cli_private_opt_in_pilot_v1",
        "study_mode": "pilot",
        "created_at": dataset["created_at"],
        "consent": {
            "collection_mode": "opt_in_shadow",
            "consent_granted": True,
            "consent_ref": "policy:cli-private-opt-in-consent",
            "retention_until": dataset["privacy"]["retention_until"],
            "revocation_policy_ref": "policy:cli-private-opt-in-revocation",
            "profile_scope_id": "cli private profile scope",
        },
        "thresholds": dataset["thresholds"],
        "episodes": episodes,
    }


def test_cli_bootstraps_checkout_and_validates_minimized_fixture():
    completed = _run("validate", "--dataset", str(FIXTURE))

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["valid"] is True
    assert payload["episode_count"] == 4
    assert payload["raw_content_stored"] is False
    assert payload["diagnostic_only"] is True


def test_cli_analysis_writes_atomic_json_and_detects_ranking(tmp_path):
    output = tmp_path / "analysis.json"
    completed = _run(
        "analyze",
        "--dataset",
        str(FIXTURE),
        "--output",
        str(output),
        "--require-decisive",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["diagnosis"]["component"] == "ranking"
    assert payload["mutation_authority"] == "none"


def test_cli_require_decisive_uses_distinct_inconclusive_exit(tmp_path):
    dataset = load_dataset(FIXTURE)
    dataset["thresholds"]["minimum_participant_clusters"] = 5
    inconclusive = tmp_path / "inconclusive.json"
    _write_json(inconclusive, dataset)

    completed = _run(
        "analyze",
        "--dataset",
        str(inconclusive),
        "--require-decisive",
    )

    assert completed.returncode == 1, completed.stderr
    assert json.loads(completed.stdout)["diagnosis"]["status"] == "insufficient_evidence"


def test_cli_disjointness_exit_codes_are_operator_friendly(tmp_path):
    dataset = load_dataset(FIXTURE)
    disjoint_path = tmp_path / "disjoint.json"
    _write_json(disjoint_path, _make_disjoint(dataset))

    overlap = _run(
        "audit-disjoint",
        "--candidate",
        str(FIXTURE),
        "--against",
        str(FIXTURE),
    )
    disjoint = _run(
        "audit-disjoint",
        "--candidate",
        str(disjoint_path),
        "--against",
        str(FIXTURE),
    )

    assert overlap.returncode == 1, overlap.stderr
    assert json.loads(overlap.stdout)["disjoint"] is False
    assert disjoint.returncode == 0, disjoint.stderr
    assert json.loads(disjoint.stdout)["disjoint"] is True


def test_cli_invalid_artifact_fails_without_traceback(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"schema_version":"wrong"}', encoding="utf-8")

    completed = _run("validate", "--dataset", str(invalid))

    assert completed.returncode == 2
    assert "error:" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_cli_refuses_to_overwrite_an_input():
    completed = _run(
        "analyze",
        "--dataset",
        str(FIXTURE),
        "--output",
        str(FIXTURE),
    )

    assert completed.returncode == 2
    assert "must use .json" in completed.stderr or "must not replace" in completed.stderr


def test_cli_collects_only_minimized_disjoint_opt_in_episodes(tmp_path):
    intake = tmp_path / "private-intake.json"
    key = tmp_path / "token-key.txt"
    output = tmp_path / "collected.json"
    _write_json(intake, _private_intake())
    key.write_text("ab" * 32, encoding="ascii")
    key.chmod(0o600)

    completed = _run(
        "collect",
        "--intake",
        str(intake),
        "--token-key-file",
        str(key),
        "--against",
        str(FIXTURE),
        "--output",
        str(output),
        "--authorize-opt-in-collection",
    )

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    dataset = load_dataset(output)
    assert receipt["collected"] is True
    assert receipt["disjoint"] is True
    assert receipt["raw_content_stored"] is False
    assert dataset["study_mode"] == "pilot"
    assert "cli participant" not in output.read_text(encoding="utf-8")
    assert "Resume the CLI" not in output.read_text(encoding="utf-8")
    assert all(
        episode["participant_ref"].startswith("opaque:")
        for episode in dataset["episodes"]
    )

    analysis = _run(
        "analyze",
        "--dataset",
        str(output),
        "--require-decisive",
    )
    assert analysis.returncode == 0, analysis.stderr
    assert json.loads(analysis.stdout)["diagnosis"]["component"] == "ranking"

    overlap_output = tmp_path / "overlapping-pool.json"
    overlap = _run(
        "collect",
        "--intake",
        str(intake),
        "--token-key-file",
        str(key),
        "--against",
        str(output),
        "--output",
        str(overlap_output),
        "--authorize-opt-in-collection",
    )
    assert overlap.returncode == 1, overlap.stderr
    assert json.loads(overlap.stdout)["disjoint"] is False
    assert not overlap_output.exists()


def test_cli_collection_requires_operator_attestation_and_external_paths(tmp_path):
    intake = tmp_path / "private-intake.json"
    key = tmp_path / "token-key.txt"
    output = tmp_path / "collected.json"
    _write_json(intake, _private_intake())
    key.write_text("cd" * 32, encoding="ascii")
    key.chmod(0o600)

    missing_attestation = _run(
        "collect",
        "--intake",
        str(intake),
        "--token-key-file",
        str(key),
        "--against",
        str(FIXTURE),
        "--output",
        str(output),
    )
    repository_output = _run(
        "collect",
        "--intake",
        str(intake),
        "--token-key-file",
        str(key),
        "--against",
        str(FIXTURE),
        "--output",
        str(REPO_ROOT / "must-not-be-created.json"),
        "--authorize-opt-in-collection",
    )

    assert missing_attestation.returncode == 2
    assert "--authorize-opt-in-collection" in missing_attestation.stderr
    assert repository_output.returncode == 2
    assert "outside the repository" in repository_output.stderr
    assert not output.exists()
    assert not (REPO_ROOT / "must-not-be-created.json").exists()
