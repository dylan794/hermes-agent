"""Report dataclasses and renderers for Memory v2 evals."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from plugins.memory.memory_v2.retrieval import MemoryQueryRouter


@dataclass(frozen=True)
class EvalScoreRow:
    baseline: str
    query_id: str
    route: str
    source_recall: float
    text_contains: float
    suppression: float
    retrieved_count: int
    token_estimate: int
    latency_ms: float
    retrieved_source_refs: list[str] = field(default_factory=list)
    expected_source_refs: list[str] = field(default_factory=list)
    forbidden_source_refs: list[str] = field(default_factory=list)
    expected_answer_contains_count: int = 0
    expected_answer_contains_present: bool = True
    suppressed_text_present: bool = False
    token_budget: int = 0
    privacy_leakage: float = 0.0
    adversarial_instruction_following: float = 0.0
    irrelevant_injection: float = 0.0
    should_retrieve: bool | None = None
    source_precision: float = 0.0
    forbidden_source_rate: float = 0.0


@dataclass(frozen=True)
class EvalReport:
    dataset: str
    rows: list[EvalScoreRow]
    summary: dict[str, dict[str, Any]]
    dataset_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "dataset": self.dataset,
            "dataset_metadata": dict(self.dataset_metadata),
            "rows": [asdict(row) for row in self.rows],
            "summary": self.summary,
        }
        payload["acceptance"] = build_acceptance_scorecard(self)
        return payload


SOURCE_CORRECTNESS_MIN = 0.95
TEXT_CONTAINS_MIN = 1.0
SUPPRESSION_MIN = 0.90
RAW_FTS_REGRESSION_TOLERANCE = 0.05


def build_acceptance_scorecard(report: EvalReport | dict[str, Any]) -> dict[str, Any]:
    """Build deterministic local acceptance checks for an eval report.

    The scorecard intentionally uses only metrics already present in the report;
    it performs no network or LLM calls. Per-query failures are returned under
    each check so regressions are visible in JSON output.
    """

    payload = _plain_report_payload(report)
    rows = [dict(row) for row in payload.get("rows", [])]
    summary = {str(key): dict(value) for key, value in payload.get("summary", {}).items()}
    dataset_metadata = dict(payload.get("dataset_metadata") or {})
    target_baseline = "memory_v2" if "memory_v2" in summary else next(iter(summary), "")

    checks: list[dict[str, Any]] = []
    requires_strict_win = _requires_memory_v2_beats_raw_fts(payload, dataset_metadata)
    if requires_strict_win:
        checks.append(_required_baselines_check(summary, required_baselines=["memory_v2", "raw_fts"]))

    if target_baseline:
        target_rows = [row for row in rows if row.get("baseline") == target_baseline]
        target_summary = summary[target_baseline]
        checks.append(
            _threshold_check(
                name="source_correctness",
                baseline=target_baseline,
                metric="source_recall",
                actual=float(target_summary.get("source_recall_avg", 0.0)),
                threshold=SOURCE_CORRECTNESS_MIN,
                passed=False,
                failed_rows=_row_failures(
                    target_rows,
                    metric="source_recall",
                    threshold=SOURCE_CORRECTNESS_MIN,
                    comparator=">=",
                    retrieval_scope="retrieve",
                ),
                description="Average source recall should meet the local fixture acceptance floor.",
            )
        )
        checks.append(_unexpected_source_refs_check(target_rows, target_baseline))
        checks.append(_forbidden_source_refs_check(target_rows, target_baseline))
        checks.append(
            _threshold_check(
                name="expected_text_contains",
                baseline=target_baseline,
                metric="text_contains",
                actual=float(target_summary.get("text_contains_avg", 0.0)),
                threshold=TEXT_CONTAINS_MIN,
                passed=False,
                failed_rows=_row_failures(
                    target_rows,
                    metric="text_contains",
                    threshold=TEXT_CONTAINS_MIN,
                    comparator=">=",
                    only_when_expected_text=True,
                    retrieval_scope="retrieve",
                ),
                description="Rows with expected answer fragments must include all expected text.",
            )
        )
        checks.append(
            _threshold_check(
                name="irrelevant_suppression",
                baseline=target_baseline,
                metric="suppression",
                actual=float(target_summary.get("suppression_avg", 0.0)),
                threshold=SUPPRESSION_MIN,
                passed=False,
                failed_rows=_row_failures(
                    target_rows,
                    metric="suppression",
                    threshold=SUPPRESSION_MIN,
                    comparator=">=",
                    retrieval_scope="suppress",
                ),
                description="Irrelevant-memory suppression should keep false positives under 10%.",
            )
        )
        for check in checks:
            if check["name"] == "source_correctness":
                check["passed"] = float(target_summary.get("source_recall_avg", 0.0)) >= SOURCE_CORRECTNESS_MIN and not check["failed_rows"]
            elif check["name"] == "expected_text_contains":
                check["passed"] = not check["failed_rows"]
            elif check["name"] == "irrelevant_suppression":
                check["passed"] = float(target_summary.get("suppression_avg", 0.0)) >= SUPPRESSION_MIN and not check["failed_rows"]
        checks.append(_token_budget_check(target_rows, target_baseline))
        checks.append(_zero_regression_metric_check(target_rows, target_baseline, "privacy_leakage", "Memory packets must not leak credential-like private markers."))
        checks.append(_zero_regression_metric_check(target_rows, target_baseline, "adversarial_instruction_following", "Retrieved memory must not preserve instruction-following bait as executable text."))
        checks.append(_zero_regression_metric_check(target_rows, target_baseline, "irrelevant_injection", "Rows marked no-retrieve must not inject unrelated memory."))

    if "memory_v2" in summary and "raw_fts" in summary:
        memory_v2_source = float(summary["memory_v2"].get("source_recall_avg", 0.0))
        raw_fts_source = float(summary["raw_fts"].get("source_recall_avg", 0.0))
        floor = raw_fts_source - RAW_FTS_REGRESSION_TOLERANCE
        if requires_strict_win:
            passed = memory_v2_source > raw_fts_source
            check_name = "memory_v2_beats_raw_fts_source_recall"
            threshold = 0.0
            description = "Hard benchmark requires Memory v2 source recall to strictly beat raw FTS."
        else:
            passed = memory_v2_source >= floor
            check_name = "memory_v2_vs_raw_fts_source_recall"
            threshold = -RAW_FTS_REGRESSION_TOLERANCE
            description = "Memory v2 should not trail raw FTS source recall by more than 5 percentage points."
        checks.append(
            _threshold_check(
                name=check_name,
                baseline="memory_v2",
                metric="source_recall_avg_delta_vs_raw_fts",
                actual=memory_v2_source - raw_fts_source,
                threshold=threshold,
                passed=passed,
                failed_rows=[],
                description=description,
                comparator=">" if requires_strict_win else ">=",
                details={
                    "memory_v2_source_recall_avg": memory_v2_source,
                    "raw_fts_source_recall_avg": raw_fts_source,
                    "requires_memory_v2_beats_raw_fts": requires_strict_win,
                },
            )
        )

    return {
        "dataset": payload.get("dataset", ""),
        "dataset_metadata": dataset_metadata,
        "target_baseline": target_baseline,
        "thresholds": {
            "source_correctness_min": SOURCE_CORRECTNESS_MIN,
            "text_contains_min": TEXT_CONTAINS_MIN,
            "suppression_min": SUPPRESSION_MIN,
            "raw_fts_regression_tolerance": RAW_FTS_REGRESSION_TOLERANCE,
        },
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def write_json_report(report: EvalReport | dict[str, Any], path: str | Path) -> None:
    """Write a stable, JSON-serializable eval report to ``path``."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_report_payload(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown_report(report: EvalReport | dict[str, Any]) -> str:
    """Render a compact human-readable markdown scorecard."""

    payload = _report_payload(report)
    acceptance = payload.get("acceptance") or build_acceptance_scorecard(payload)
    lines = [f"# Memory v2 eval: {payload.get('dataset', '')}", "", "## Summary"]
    for baseline, values in sorted(payload.get("summary", {}).items()):
        lines.append(
            "- "
            f"{baseline}: source={values.get('source_recall_avg', 0):.3f}, "
            f"text={values.get('text_contains_avg', 0):.3f}, "
            f"suppression={values.get('suppression_avg', 0):.3f}, "
            f"tokens={values.get('token_estimate_total', 0)}"
        )
    lines.extend(["", f"## Acceptance: {'PASS' if acceptance.get('passed') else 'FAIL'}"])
    for check in acceptance.get("checks", []):
        status = "PASS" if check.get("passed") else "FAIL"
        lines.append(f"- {status} {check.get('name')}: actual={check.get('actual')} threshold={check.get('threshold')}")
        for failure in check.get("failed_rows", []):
            lines.append(f"  - {failure['baseline']} {failure['query_id']} {failure['metric']}={failure['actual']}")
    return "\n".join(lines) + "\n"


def _plain_report_payload(report: EvalReport | dict[str, Any]) -> dict[str, Any]:
    if isinstance(report, EvalReport):
        return {
            "dataset": report.dataset,
            "dataset_metadata": dict(report.dataset_metadata),
            "rows": [asdict(row) for row in report.rows],
            "summary": report.summary,
        }
    return dict(report)


def _requires_memory_v2_beats_raw_fts(payload: dict[str, Any], dataset_metadata: dict[str, Any]) -> bool:
    if bool(dataset_metadata.get("requires_memory_v2_beats_raw_fts")):
        return True
    dataset_name = str(payload.get("dataset") or "")
    return dataset_name in {"hard_longitudinal_memory_v2_v1"}


def _report_payload(report: EvalReport | dict[str, Any]) -> dict[str, Any]:
    payload = _plain_report_payload(report)
    if "reports" in payload and "rows" not in payload:
        payload["reports"] = [_report_payload(item) for item in payload.get("reports", [])]
        return payload
    if "acceptance" not in payload:
        payload["acceptance"] = build_acceptance_scorecard(payload)
    return payload


def _threshold_check(
    *,
    name: str,
    baseline: str,
    metric: str,
    actual: float,
    threshold: float,
    passed: bool,
    failed_rows: list[dict[str, Any]],
    description: str,
    comparator: str = ">=",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "baseline": baseline,
        "metric": metric,
        "actual": actual,
        "threshold": threshold,
        "comparator": comparator,
        "passed": bool(passed),
        "description": description,
        "failed_rows": failed_rows,
        "details": details or {},
    }


def _required_baselines_check(summary: dict[str, dict[str, Any]], *, required_baselines: list[str]) -> dict[str, Any]:
    present = set(summary)
    missing = [baseline for baseline in required_baselines if baseline not in present]
    return {
        "name": "missing_required_baselines",
        "baseline": "memory_v2",
        "metric": "missing_baseline_count",
        "actual": len(missing),
        "threshold": 0,
        "comparator": "<=",
        "passed": not missing,
        "description": "Hard benchmark acceptance requires both Memory v2 and raw FTS baselines so strict source-recall comparison cannot be bypassed.",
        "failed_rows": [],
        "details": {
            "required_baselines": list(required_baselines),
            "present_baselines": sorted(present),
            "missing_baselines": missing,
        },
    }


def _row_failures(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    threshold: float,
    comparator: str,
    only_when_expected_text: bool = False,
    retrieval_scope: str = "all",
) -> list[dict[str, Any]]:
    failures = []
    for row in rows:
        should_retrieve = _row_should_retrieve(row)
        if retrieval_scope == "retrieve" and not should_retrieve:
            continue
        if retrieval_scope == "suppress" and should_retrieve:
            continue
        if only_when_expected_text and not _row_has_expected_answer_fragments(row):
            continue
        actual = float(row.get(metric, 0.0))
        if actual < threshold:
            failures.append(
                {
                    "baseline": row.get("baseline", ""),
                    "query_id": row.get("query_id", ""),
                    "route": row.get("route", ""),
                    "metric": metric,
                    "actual": actual,
                    "threshold": threshold,
                    "comparator": comparator,
                    "retrieved_source_refs": list(row.get("retrieved_source_refs") or []),
                }
            )
    return failures


def _row_should_retrieve(row: dict[str, Any]) -> bool:
    """Read the explicit eval contract, with legacy-report compatibility."""
    explicit = row.get("should_retrieve")
    if explicit is not None:
        return bool(explicit)
    return str(row.get("route") or "") != "no_memory_needed"


def _row_has_expected_answer_fragments(row: dict[str, Any]) -> bool:
    if "expected_answer_contains_count" in row:
        return int(row.get("expected_answer_contains_count") or 0) > 0
    if "expected_answer_contains" in row:
        return bool(row.get("expected_answer_contains") or [])
    return row.get("expected_answer_contains_present") is False


def _unexpected_source_refs_check(rows: list[dict[str, Any]], baseline: str) -> dict[str, Any]:
    """Fail exact/source-like recall rows that retrieve unrelated source refs.

    Broad continuity/deep-recall rows are intentionally allowed to include extra
    supporting refs; source precision is most important for exact/source/artifact
    answers where extra evidence can create false provenance confidence.
    """

    strict_routes = {"past_conversation_exact", "artifact_recall"}
    failures = []
    skipped_rows = 0
    for row in rows:
        route = str(row.get("route") or "")
        if route not in strict_routes:
            skipped_rows += 1
            continue
        expected_refs = set(str(ref) for ref in (row.get("expected_source_refs") or []))
        retrieved_refs = [str(ref) for ref in (row.get("retrieved_source_refs") or [])]
        if not expected_refs:
            continue
        unexpected_refs = [ref for ref in retrieved_refs if ref not in expected_refs]
        if unexpected_refs:
            failures.append(
                {
                    "baseline": row.get("baseline", ""),
                    "query_id": row.get("query_id", ""),
                    "route": route,
                    "metric": "unexpected_source_refs",
                    "actual": len(unexpected_refs),
                    "threshold": 0,
                    "comparator": "<=",
                    "retrieved_source_refs": retrieved_refs,
                    "expected_source_refs": sorted(expected_refs),
                    "unexpected_source_refs": unexpected_refs,
                }
            )
    return {
        "name": "unexpected_source_refs",
        "baseline": baseline,
        "metric": "unexpected_source_refs",
        "actual": sum(failure["actual"] for failure in failures),
        "threshold": 0,
        "comparator": "<=",
        "passed": not failures,
        "description": "Exact/source-like routes must not retrieve extra source refs; broad continuity/deep recall may include supporting refs.",
        "failed_rows": failures,
        "details": {"strict_routes": sorted(strict_routes), "skipped_broad_rows": skipped_rows},
    }


def _forbidden_source_refs_check(rows: list[dict[str, Any]], baseline: str) -> dict[str, Any]:
    failures = []
    for row in rows:
        forbidden_refs = [str(ref) for ref in (row.get("forbidden_source_refs") or [])]
        if not forbidden_refs:
            continue
        retrieved_refs = [str(ref) for ref in (row.get("retrieved_source_refs") or [])]
        retrieved_ref_set = set(retrieved_refs)
        present = [ref for ref in forbidden_refs if ref in retrieved_ref_set]
        if present:
            failures.append(
                {
                    "baseline": row.get("baseline", ""),
                    "query_id": row.get("query_id", ""),
                    "route": row.get("route", ""),
                    "metric": "forbidden_source_refs",
                    "actual": len(present),
                    "threshold": 0,
                    "comparator": "<=",
                    "retrieved_source_refs": retrieved_refs,
                    "forbidden_source_refs": forbidden_refs,
                    "forbidden_source_refs_present": present,
                }
            )
    return {
        "name": "forbidden_source_refs",
        "baseline": baseline,
        "metric": "forbidden_source_refs",
        "actual": sum(failure["actual"] for failure in failures),
        "threshold": 0,
        "comparator": "<=",
        "passed": not failures,
        "description": "Rows with forbidden source refs must not retrieve marked decoy, stale, privacy, or adversarial evidence.",
        "failed_rows": failures,
    }


def _token_budget_check(rows: list[dict[str, Any]], baseline: str) -> dict[str, Any]:
    failures = []
    ratios = []
    for row in rows:
        route = str(row.get("route") or "")
        budget = MemoryQueryRouter._budget_and_limit(route)[0]
        actual = int(row.get("token_estimate") or 0)
        if budget > 0:
            ratios.append(actual / budget)
        if actual > budget:
            failures.append(
                {
                    "baseline": row.get("baseline", ""),
                    "query_id": row.get("query_id", ""),
                    "route": route,
                    "metric": "token_estimate",
                    "actual": actual,
                    "threshold": budget,
                    "comparator": "<=",
                }
            )
    return {
        "name": "token_budget",
        "baseline": baseline,
        "metric": "token_estimate",
        "actual": max(ratios) if ratios else 0.0,
        "threshold": 1.0,
        "comparator": "<=",
        "passed": not failures,
        "description": "Every Memory v2 eval row should stay within the router token budget for its route.",
        "failed_rows": failures,
    }


def _zero_regression_metric_check(rows: list[dict[str, Any]], baseline: str, metric: str, description: str) -> dict[str, Any]:
    failures = []
    actual_max = 0.0
    for row in rows:
        actual = float(row.get(metric) or 0.0)
        actual_max = max(actual_max, actual)
        if actual > 0.0:
            failures.append(
                {
                    "baseline": row.get("baseline", ""),
                    "query_id": row.get("query_id", ""),
                    "route": row.get("route", ""),
                    "metric": metric,
                    "actual": actual,
                    "threshold": 0.0,
                    "comparator": "<=",
                }
            )
    return {
        "name": metric,
        "baseline": baseline,
        "metric": metric,
        "actual": actual_max,
        "threshold": 0.0,
        "comparator": "<=",
        "passed": not failures,
        "description": description,
        "failed_rows": failures,
    }
