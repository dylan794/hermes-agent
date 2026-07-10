"""CLI tests for Memory v2 deterministic eval runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_memory_v2_eval_cli_writes_json_report(tmp_path):
    output_path = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/memory_v2_eval.py",
            "--dataset",
            "tests/plugins/memory/evals/fixtures/local_memory_eval_v1.yaml",
            "--baseline",
            "no_memory",
            "--baseline",
            "raw_fts",
            "--baseline",
            "memory_v2",
            "--workdir",
            str(tmp_path / "work"),
            "--output",
            str(output_path),
        ],
        check=False,
        cwd=Path(__file__).parents[4],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "local_memory_eval_v1"
    assert set(payload["summary"]) == {"no_memory", "raw_fts", "memory_v2"}
    assert payload["summary"]["memory_v2"]["query_count"] == 3


def test_memory_v2_eval_cli_supports_archive_and_semantic_baselines(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/memory_v2_eval.py",
            "--dataset",
            "plugins/memory/memory_v2/evals/fixtures/local_memory_eval_v1.yaml",
            "--baseline",
            "archive_only",
            "--baseline",
            "semantic_only",
            "--workdir",
            str(tmp_path / "work"),
            "--no-fail-on-acceptance",
        ],
        check=False,
        cwd=Path(__file__).parents[4],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert set(payload["summary"]) == {"archive_only", "semantic_only"}


def test_memory_v2_eval_cli_missing_dataset_is_user_friendly(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/memory_v2_eval.py",
            "--dataset",
            str(tmp_path / "missing.yaml"),
            "--baseline",
            "no_memory",
        ],
        check=False,
        cwd=Path(__file__).parents[4],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "dataset not found" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_memory_v2_eval_cli_fails_nonzero_on_acceptance_failure_by_default(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/memory_v2_eval.py",
            "--dataset",
            "tests/plugins/memory/evals/fixtures/local_memory_eval_v1.yaml",
            "--baseline",
            "no_memory",
            "--workdir",
            str(tmp_path / "work"),
        ],
        check=False,
        cwd=Path(__file__).parents[4],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert "acceptance failed" in completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["acceptance"]["passed"] is False


def test_memory_v2_eval_cli_can_opt_out_of_acceptance_exit_for_exploration(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/memory_v2_eval.py",
            "--dataset",
            "tests/plugins/memory/evals/fixtures/local_memory_eval_v1.yaml",
            "--baseline",
            "no_memory",
            "--workdir",
            str(tmp_path / "work"),
            "--no-fail-on-acceptance",
        ],
        check=False,
        cwd=Path(__file__).parents[4],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["acceptance"]["passed"] is False


def test_memory_v2_eval_cli_hard_gate_fails_closed_when_raw_fts_baseline_missing(tmp_path):
    output_path = tmp_path / "missing_raw.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/memory_v2_eval.py",
            "--dataset",
            "plugins/memory/memory_v2/evals/fixtures/hard_longitudinal_memory_v2_v1.yaml",
            "--baseline",
            "memory_v2",
            "--workdir",
            str(tmp_path / "work"),
            "--fail-on-acceptance",
            "--output",
            str(output_path),
        ],
        check=False,
        cwd=Path(__file__).parents[4],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert "acceptance failed" in completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    missing_check = next(check for check in payload["acceptance"]["checks"] if check["name"] == "missing_required_baselines")
    assert missing_check["passed"] is False
    assert missing_check["details"]["missing_baselines"] == ["raw_fts"]


def test_memory_v2_eval_cli_hard_gate_fails_when_memory_v2_only_ties_raw_fts(tmp_path):
    hard_tie = tmp_path / "hard_tie.yaml"
    hard_tie.write_text(
        """
version: 1
name: hard_tie_requires_win
metadata:
  requires_memory_v2_beats_raw_fts: true
events:
  - id: event_pref_blue
    session_id: s
    role: user
    text: "Remember that Alex prefers blue dashboards."
queries:
  - id: q_pref_blue
    route: preference_recall
    text: "What dashboard color does Alex prefer?"
    expected_answer_contains: ["blue dashboards"]
    expected_source_refs: ["event_pref_blue"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/memory_v2_eval.py",
            "--dataset",
            str(hard_tie),
            "--baseline",
            "raw_fts",
            "--baseline",
            "memory_v2",
            "--workdir",
            str(tmp_path / "work"),
            "--fail-on-acceptance",
        ],
        check=False,
        cwd=Path(__file__).parents[4],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert "acceptance failed" in completed.stderr
    payload = json.loads(completed.stdout)
    hard_check = next(check for check in payload["acceptance"]["checks"] if check["name"] == "memory_v2_beats_raw_fts_source_recall")
    assert hard_check["passed"] is False


def test_memory_v2_eval_cli_hard_benchmark_passes_when_memory_v2_beats_raw_fts(tmp_path):
    output_path = tmp_path / "hard_report.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/memory_v2_eval.py",
            "--dataset",
            "plugins/memory/memory_v2/evals/fixtures/hard_longitudinal_memory_v2_v1.yaml",
            "--baseline",
            "raw_fts",
            "--baseline",
            "memory_v2",
            "--workdir",
            str(tmp_path / "work"),
            "--fail-on-acceptance",
            "--output",
            str(output_path),
        ],
        check=False,
        cwd=Path(__file__).parents[4],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["acceptance"]["passed"] is True
    hard_check = next(check for check in payload["acceptance"]["checks"] if check["name"] == "memory_v2_beats_raw_fts_source_recall")
    assert hard_check["passed"] is True
