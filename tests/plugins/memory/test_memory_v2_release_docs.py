"""Doc-contract tests for the Memory v2 archive release checklist.

These tests intentionally validate release/runbook text instead of runtime behavior so
Phase 10 rollout gates do not drift silently.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKLIST = REPO_ROOT / "docs" / "memory-v2-archive-release-checklist.md"
README = REPO_ROOT / "plugins" / "memory" / "memory_v2" / "README.md"


REQUIRED_STAGES = [
    "1. Synthetic fixtures only",
    "2. Isolated dogfood profile",
    "3. Limited personal-profile dry-run",
    "4. Small confirmed import with health checks",
    "5. Read-only archive search/show enabled",
    "6. Candidate extraction enabled",
    "7. Semantic promotion with review gate",
    "8. Limited automatic prefetch",
    "9. Broader release",
]


REQUIRED_FLAGS = {
    "memory_v2.archive.enabled": "true",
    "memory_v2.archive.backfill_enabled": "false",
    "memory_v2.archive.search_tools_enabled": "true",
    "memory_v2.archive.show_tools_enabled": "true",
    "memory_v2.archive.include_tool_outputs": "false",
    "memory_v2.extraction.enabled": "false",
    "memory_v2.extraction.candidate_creation_enabled": "false",
    "memory_v2.consolidation.enabled": "false",
    "memory_v2.prefetch.enabled": "false",
    "memory_v2.auto_promote.enabled": "false",
}


REQUIRED_GATE_COMMANDS = [
    "./scripts/run_tests.sh tests/plugins/memory/test_memory_v2_*.py tests/plugins/memory/evals tests/agent/test_memory_provider.py",
    "python -m pytest tests/plugins/memory/test_memory_v2_raw_archive_perf.py -q",
    "python -m pytest tests/plugins/memory/test_memory_v2_adversarial_archive.py -q",
    "python -m pytest tests/plugins/memory/test_memory_v2_archive_readiness.py -q",
    "python -m pytest tests/plugins/memory/test_memory_v2_extraction_rollout.py -q",
    "python scripts/memory_v2_privacy_scan.py --mode memory-v2-release-artifacts --format json",
    "python scripts/memory_v2_privacy_scan.py --mode intentional-adversarial-fixtures --format json",
    "python scripts/memory_v2_eval.py --dataset plugins/memory/memory_v2/evals/fixtures/local_memory_eval_v1.yaml --baseline no_memory --baseline raw_fts --baseline memory_v2",
    "python -m py_compile plugins/memory/memory_v2/*.py scripts/memory_v2_eval.py scripts/memory_v2_privacy_scan.py",
    "./scripts/run_tests.sh",
]


REQUIRED_ACCEPTANCE_PHRASES = [
    "dogfood report proves safety and quality",
    "privacy scan passes",
    "eval thresholds pass",
    "archive health is ok",
    "docs are current and synthetic-only",
    "branch and release artifacts contain no private data",
]


def _checklist_text() -> str:
    assert CHECKLIST.exists(), "Phase 10 release checklist doc must exist"
    return CHECKLIST.read_text(encoding="utf-8")


def test_release_checklist_documents_all_rollout_stages() -> None:
    text = _checklist_text()

    for stage in REQUIRED_STAGES:
        assert stage in text


def test_release_checklist_documents_default_rollout_flags() -> None:
    text = _checklist_text()

    for flag, expected in REQUIRED_FLAGS.items():
        pattern = rf"`?{re.escape(flag)}`?\s*\|\s*`?{expected}`?\b"
        assert re.search(pattern, text), f"missing default for {flag}={expected}"


def test_release_checklist_documents_release_gate_commands() -> None:
    text = _checklist_text()

    for command in REQUIRED_GATE_COMMANDS:
        assert command in text
    assert "ACP caveat" in text


def test_release_checklist_documents_acceptance_and_privacy_boundaries() -> None:
    text = _checklist_text().lower()

    for phrase in REQUIRED_ACCEPTANCE_PHRASES:
        assert phrase in text
    assert "synthetic-only" in text
    assert "do not copy real private conversations" in text


def test_memory_v2_readme_links_phase_10_runbook() -> None:
    text = README.read_text(encoding="utf-8")

    assert "docs/memory-v2-archive-release-checklist.md" in text
    assert "Phase 10" in text
    assert "rollout gate" in text.lower()
