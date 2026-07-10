"""Deterministic eval tests for Memory v2 spaced review."""

from __future__ import annotations

from plugins.memory.memory_v2.evals.spaced_review import run_spaced_review_eval


def test_spaced_review_eval_covers_due_records_and_suppresses_private_text():
    result = run_spaced_review_eval()

    assert result["version"] == 1
    assert result["passed"] is True
    assert result["metrics"]["due_coverage"] == 1.0
    assert result["metrics"]["privacy_leak_count"] == 0
    assert result["metrics"]["report_only_actions"] == 1.0
