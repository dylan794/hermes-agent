from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from hermes_state import SessionDB
from plugins.memory.memory_v2 import MemoryV2Provider
from plugins.memory.memory_v2.session_backfill import SESSION_BACKFILL_CONFIRM


ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts" / "memory_v2_archive_ops.py"


def _run_cli(tmp_path: Path, *args: str, check: bool = True) -> tuple[subprocess.CompletedProcess[str], dict]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--hermes-home", str(tmp_path), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(completed.stdout) if completed.stdout.strip().startswith("{") else {}
    if check:
        assert completed.returncode == 0, completed.stderr or completed.stdout
    return completed, payload


def _provider(home: Path) -> MemoryV2Provider:
    provider = MemoryV2Provider()
    provider.initialize("cli-test-session", hermes_home=str(home), platform="cli")
    return provider


def _seed_archive(home: Path) -> dict:
    provider = _provider(home)
    return provider.store.append_raw_event(
        {
            "id": "evt_cli_safe_1",
            "type": "test_event",
            "session_id": "sess-cli",
            "provider_session_id": "sess-cli",
            "created_at": "2026-06-30T12:00:00+00:00",
            "observed_at": "2026-06-30T12:00:00+00:00",
            "user_content": "Project Atlas safe CLI note; never dump this full sentence in status.",
            "assistant_content": "Acknowledged the safe archive fixture.",
            "trust_level": "untrusted",
            "can_instruct": False,
            "privacy_level": "standard",
            "archive_status": "active",
        }
    )


def _seed_state_db(home: Path) -> None:
    db = SessionDB(home / "state.db")
    db.create_session("sess-discord", "discord", model="test-model")
    db.append_message("sess-discord", "user", "Remember this synthetic CLI backfill fact.")
    db.append_message("sess-discord", "assistant", "Synthetic acknowledgement.")
    db.close()


def test_archive_ops_status_verify_rebuild_search_and_show_are_safe_json(tmp_path: Path):
    event = _seed_archive(tmp_path)

    _, status = _run_cli(tmp_path, "archive", "status")
    assert status["success"] is True
    assert status["command"] == "archive status"
    assert status["counts"]["raw_events"] == 1
    assert "Project Atlas safe CLI note" not in json.dumps(status)
    assert str(tmp_path) not in json.dumps(status)

    _, verify = _run_cli(tmp_path, "archive", "verify")
    assert verify["success"] is True
    assert verify["command"] == "archive verify"
    assert verify["archive"]["status"] == "ok"
    assert verify["archive"]["event_count"] == 1
    assert "Project Atlas safe CLI note" not in json.dumps(verify)

    _, rebuilt = _run_cli(tmp_path, "archive", "rebuild-index")
    assert rebuilt["success"] is True
    assert rebuilt["command"] == "archive rebuild-index"
    assert rebuilt["mutated"] == "derived_index_only"
    assert rebuilt["result"]["raw_events"] == 1

    _, search = _run_cli(tmp_path, "archive", "search", "--query", "Project Atlas", "--limit", "5")
    assert search["success"] is True
    assert search["command"] == "archive search"
    assert search["count"] == 1
    result = search["results"][0]
    assert result["event_id"] == event["id"]
    assert result["can_instruct"] is False
    assert result["labels_trusted"] is False
    assert result["chain"]["record_sha256"] == event["record_sha256"]

    _, shown = _run_cli(
        tmp_path,
        "archive",
        "show",
        event["id"],
        "--expect-record-sha256",
        event["record_sha256"],
    )
    assert shown["success"] is True
    assert shown["command"] == "archive show"
    assert shown["event"]["integrity"]["hash_pin_match"] is True
    assert shown["event"]["event_id"] == event["id"]


def test_session_backfill_cli_defaults_to_dry_run_and_requires_exact_confirmation(tmp_path: Path):
    _seed_state_db(tmp_path)

    _, dry_run = _run_cli(tmp_path, "session-backfill", "dry-run", "--source", "discord", "--limit", "5000")
    assert dry_run["success"] is True
    assert dry_run["command"] == "session-backfill dry-run"
    assert dry_run["dry_run"] is True
    assert dry_run["considered"] == 2
    assert dry_run["imported"] == 2
    assert not (tmp_path / "memory_v2" / "inbox" / "raw_events.jsonl").read_text(encoding="utf-8").strip()

    bad, bad_payload = _run_cli(
        tmp_path,
        "session-backfill",
        "run",
        "--source",
        "discord",
        "--confirm",
        "WRONG",
        check=False,
    )
    assert bad.returncode == 2
    assert bad_payload["success"] is False
    assert SESSION_BACKFILL_CONFIRM in bad_payload["error"]
    assert not (tmp_path / "memory_v2" / "backfill" / "sessiondb.yaml").exists()

    _, imported = _run_cli(
        tmp_path,
        "session-backfill",
        "run",
        "--source",
        "discord",
        "--resume",
        "--confirm",
        SESSION_BACKFILL_CONFIRM,
    )
    assert imported["success"] is True
    assert imported["command"] == "session-backfill run"
    assert imported["dry_run"] is False
    assert imported["resume"] is True
    assert imported["imported"] == 2
    assert "Synthetic CLI backfill fact" not in json.dumps(imported)


def test_archive_ops_privacy_scan_and_eval_wrappers_emit_json(tmp_path: Path):
    _, scan = _run_cli(tmp_path, "privacy-scan", "--paths", "docs/memory-v2-evals.md", "--format", "json")
    assert scan["success"] is True
    assert scan["command"] == "privacy-scan"
    assert "finding_count" in scan

    _, report = _run_cli(
        tmp_path,
        "eval",
        "--dataset",
        "tests/plugins/memory/evals/fixtures/local_memory_eval_v1.yaml",
        "--baseline",
        "memory_v2",
        "--no-fail-on-acceptance",
    )
    assert report["success"] is True
    assert report["command"] == "eval"
    assert report["dataset"] == "local_memory_eval_v1"
    assert set(report["summary"]) == {"memory_v2"}

    _, expanded = _run_cli(
        tmp_path,
        "eval",
        "--dataset",
        "plugins/memory/memory_v2/evals/fixtures/local_memory_eval_v1.yaml",
        "--baseline",
        "archive_only",
        "--baseline",
        "semantic_only",
        "--no-fail-on-acceptance",
    )
    assert expanded["success"] is True
    assert set(expanded["summary"]) == {"archive_only", "semantic_only"}

    failed_proc, failed = _run_cli(
        tmp_path,
        "eval",
        "--dataset",
        "plugins/memory/memory_v2/evals/fixtures/local_memory_eval_v1.yaml",
        "--baseline",
        "no_memory",
        check=False,
    )
    assert failed_proc.returncode == 1
    assert failed["success"] is False
    assert failed["acceptance"]["passed"] is False
