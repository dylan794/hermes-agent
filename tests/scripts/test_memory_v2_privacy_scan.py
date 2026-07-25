from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "memory_v2_privacy_scan.py"
SPEC = importlib.util.spec_from_file_location("memory_v2_privacy_scan", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
scanner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scanner
SPEC.loader.exec_module(scanner)


def kinds_for(line: str) -> list[str]:
    return [finding.kind for finding in scanner.scan_line("fixture.txt", 1, line)]


def test_scan_line_detects_private_paths_and_redacts_usernames():
    unix_path = "/ho" + "me/dylan/private/memory/report.yaml"
    mac_path = "/Us" + "ers/dylan/Library/Application Support/Hermes/log.txt"
    wsl_path = "/m" + "nt/c/Users/dylan/Documents/hermes/report.json"  # privacy-scan: allow test fixture
    win_path = "C:" + "\\Users\\dylan\\Documents\\hermes\\report.json"

    findings = scanner.scan_line("artifact.yaml", 3, f"paths: {unix_path} {mac_path} {wsl_path} {win_path}")

    assert [finding.kind for finding in findings] == [
        "private_path",
        "private_path",
        "private_path",
        "private_path",
    ]
    assert all("dylan" not in finding.snippet for finding in findings)
    assert all("<user>" in finding.snippet for finding in findings)


def test_scan_line_detects_secret_forms_without_leaking_values():
    api_key = "OPENAI_" + "API_" + "KEY=sk-live-real-secret"
    bearer = "Authorization: " + "Bearer live-token-value"
    pem = "-----BEGIN " + "RSA PRIVATE KEY-----"

    findings = scanner.scan_line("report.env", 5, f"{api_key} {bearer} {pem}")

    assert [finding.kind for finding in findings] == [
        "secret_assignment",
        "bearer_token",
        "private_key_pem",
    ]
    rendered = scanner.render_text(findings)
    assert "sk-live-real-secret" not in rendered
    assert "live-token-value" not in rendered
    assert "[REDACTED]" in rendered or "***" in rendered


def test_scan_line_detects_snowflake_like_ids_and_redacts_middle():
    line = "discord_channel_id=" + "1474927302512087112"  # privacy-scan: allow test fixture

    findings = scanner.scan_line("artifact.json", 8, line)

    assert [finding.kind for finding in findings] == ["snowflake_id"]
    assert findings[0].snippet == "1474…7112"


def test_scan_line_allows_obvious_examples_and_tmp_paths():
    lines = [
        "example fixture id " + "1474927302512087112",
        "path=/ho" + "me/alice/example/report.yaml",
        "path=/tmp/hermes/memory_v2/report.yaml",
        "DUMMY_" + "TOKEN=dummy",
        "Authorization: " + "Bearer <token>",
        "SECRET_" + "KEY=REDACTED",
    ]

    for line in lines:
        assert scanner.scan_line("examples.md", 1, line) == []


def test_scan_line_allows_ordinary_code_key_variables_but_catches_real_secret_shapes():
    clean_lines = [
        "key = normalize_key(value)",
        "candidate_key = candidate.id",
        "include_secret = False",
        "subject_key = make_key(subject)",
        "predicate_key = f'{subject}:{predicate}'",
    ]
    for line in clean_lines:
        assert scanner.scan_line("module.py", 1, line) == []

    findings = scanner.scan_line("module.py", 2, "OPENAI_API_KEY='sk-proj-1234567890abcdef'")
    assert [finding.kind for finding in findings] == ["secret_assignment"]


def test_path_scan_json_cli_reports_findings_with_nonzero_exit(tmp_path):
    artifact = tmp_path / "memory_report.txt"
    artifact.write_text(
        "source=/ho" + "me/dylan/.hermes/memory_v2/report.yaml\n"
        "token=HERMES_" + "API_" + "KEY=super-secret-value\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--paths", str(artifact), "--format", "json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert [finding["kind"] for finding in payload["findings"]] == [
        "private_path",
        "secret_assignment",
    ]
    assert "dylan" not in proc.stdout
    assert "super-secret-value" not in proc.stdout


def test_clean_path_scan_exits_zero(tmp_path):
    artifact = tmp_path / "public_report.md"
    artifact.write_text(
        "Public Memory v2 report\n"
        "Source paths are repo-relative, examples use example.com, and tmp files use /tmp.\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--paths", str(artifact)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0
    assert "no findings" in proc.stdout


def test_scan_line_detects_lowercase_json_token_forms_and_aws_access_key_id():
    key_part = "to" + "ken"
    sample_one = f'"{key_part}": "live-json-token"'
    sample_two = "api_" + "k" + "ey" + "='lowercase-secret'"
    sample_three = "AWS_" + "ACCESS_" + "KEY_" + "ID=AKIAIO...MPLE"

    findings = scanner.scan_line("artifact.json", 9, f"{sample_one} {sample_two} {sample_three}")

    assert [finding.kind for finding in findings] == [
        "secret_assignment",
        "secret_assignment",
        "secret_assignment",
    ]
    rendered = scanner.render_text(findings)
    assert "live-json-token" not in rendered
    assert "lowercase-secret" not in rendered
    assert "AKIAIO...MPLE" not in rendered
    assert "to" + "ken=" + "[REDACTED]" in rendered
    assert "api_" + "key=" + "[REDACTED]" in rendered
    assert "AWS_" + "ACCESS_" + "KEY_" + "ID=" + "[REDACTED]" in rendered


def test_release_profile_does_not_let_fake_context_hide_strong_bait():
    line = "fake fixture OPENAI_API_KEY=sk-proj-" + ("a" * 28)  # privacy-scan: synthetic-bait-ok

    default_findings = scanner.scan_line("fixture.md", 1, line)
    release_findings = scanner.scan_line(
        "fixture.md",
        1,
        line,
        profile=scanner.profile_for_mode("memory-v2-release-artifacts"),
    )

    assert [finding.kind for finding in default_findings] == ["secret_assignment"]
    assert [finding.kind for finding in release_findings] == ["secret_assignment"]


def test_adversarial_profile_requires_specific_synthetic_bait_marker_for_fake_snowflakes():
    bait_id = "1474927302512087112"  # privacy-scan: synthetic-bait-ok
    unmarked = "example fixture discord id " + bait_id
    marked = unmarked + "  # privacy-scan: synthetic-bait-ok"
    profile = scanner.profile_for_mode("intentional-adversarial-fixtures")

    assert scanner.scan_line("fixture.py", 1, unmarked) == []
    assert scanner.scan_line("fixture.py", 1, unmarked, profile=profile) != []
    assert scanner.scan_line("fixture.py", 1, marked, profile=profile) == []


def test_release_mode_cli_scans_deterministic_scope_and_reports_mode():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--mode", "memory-v2-release-artifacts", "--format", "json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    payload = json.loads(proc.stdout)
    assert proc.returncode == 0
    assert payload["mode"] == "memory-v2-release-artifacts"
    assert payload["scan_scope"]["source"] == "mode-default-paths"
    assert payload["finding_count"] == 0
    assert "scripts/memory_v2_earn_canary.py" in payload["scan_scope"]["paths"]
    assert "scripts/memory_v2_outcome_replay.py" in payload["scan_scope"]["paths"]
    assert "docs/memory-v2-earn-canary.md" in payload["scan_scope"]["paths"]
    assert "docs/memory-v2-outcome-replay-lab.md" in payload["scan_scope"]["paths"]


def test_release_mode_cli_fails_on_unmarked_temp_release_artifact(tmp_path):
    artifact = tmp_path / "release-notes.md"
    artifact.write_text(
        "synthetic fixture token OPENAI_API_KEY=sk-proj-" + ("c" * 28) + "\n",  # privacy-scan: synthetic-bait-ok
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--mode",
            "memory-v2-release-artifacts",
            "--paths",
            str(artifact),
            "--format",
            "json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    payload = json.loads(proc.stdout)
    assert proc.returncode == 1
    assert payload["mode"] == "memory-v2-release-artifacts"
    assert payload["finding_count"] == 1
    assert payload["findings"][0]["kind"] == "secret_assignment"
    assert "sk-proj" not in proc.stdout


def test_intentional_adversarial_mode_cli_passes_only_when_bait_is_marked(tmp_path):
    unmarked = tmp_path / "unmarked_fixture.py"
    marked = tmp_path / "marked_fixture.py"
    bait_id = "1474927302512087112"  # privacy-scan: synthetic-bait-ok
    unmarked.write_text("FAKE_CHANNEL_ID = '" + bait_id + "'\n", encoding="utf-8")
    marked.write_text(
        "FAKE_CHANNEL_ID = '" + bait_id + "'  # privacy-scan: synthetic-bait-ok\n",
        encoding="utf-8",
    )

    base_cmd = [sys.executable, str(SCRIPT_PATH), "--mode", "intentional-adversarial-fixtures", "--format", "json", "--paths"]
    unmarked_proc = subprocess.run(
        [*base_cmd, str(unmarked)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    marked_proc = subprocess.run(
        [*base_cmd, str(marked)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    unmarked_payload = json.loads(unmarked_proc.stdout)
    marked_payload = json.loads(marked_proc.stdout)
    assert unmarked_proc.returncode == 1
    assert unmarked_payload["mode"] == "intentional-adversarial-fixtures"
    assert [finding["kind"] for finding in unmarked_payload["findings"]] == ["snowflake_id"]
    assert marked_proc.returncode == 0
    assert marked_payload["mode"] == "intentional-adversarial-fixtures"
    assert marked_payload["finding_count"] == 0


def test_full_repo_public_hygiene_mode_cli_has_distinct_mode_and_exit_codes(tmp_path):
    clean = tmp_path / "clean.md"
    dirty = tmp_path / "dirty.md"
    clean.write_text("public docs only\n", encoding="utf-8")
    dirty.write_text("path=/ho" + "me/dylan/private/report.txt\n", encoding="utf-8")

    clean_proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--mode", "full-repo-public-hygiene", "--paths", str(clean), "--format", "json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    dirty_proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--mode", "full-repo-public-hygiene", "--paths", str(dirty), "--format", "json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    clean_payload = json.loads(clean_proc.stdout)
    dirty_payload = json.loads(dirty_proc.stdout)
    assert clean_proc.returncode == 0
    assert clean_payload["mode"] == "full-repo-public-hygiene"
    assert clean_payload["finding_count"] == 0
    assert dirty_proc.returncode == 1
    assert dirty_payload["mode"] == "full-repo-public-hygiene"
    assert [finding["kind"] for finding in dirty_payload["findings"]] == ["private_path"]
    assert "dylan" not in dirty_proc.stdout
