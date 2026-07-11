"""Benchmark runners for Memory v2 evals."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from .baselines import MemoryEvalBaseline
from .datasets import EvalDataset
from .metrics import (
    score_adversarial_instruction_following,
    score_irrelevant_injection,
    score_irrelevant_suppression,
    score_privacy_leakage,
    score_source_recall,
    score_text_contains,
)
from .reports import EvalReport, EvalScoreRow
from plugins.memory.memory_v2.retrieval import MemoryQueryRouter


def run_eval(dataset: EvalDataset, *, baselines: list[MemoryEvalBaseline]) -> EvalReport:
    rows: list[EvalScoreRow] = []
    for baseline in baselines:
        ingest_dataset = getattr(baseline, "ingest_dataset", None)
        if callable(ingest_dataset):
            ingest_dataset(dataset)
        else:
            baseline.ingest(dataset.events)
        consolidate = getattr(baseline, "consolidate", None)
        if callable(consolidate):
            consolidate()
        for query in dataset.queries:
            result = baseline.retrieve(query)
            answer_text = result.answer or result.memory_packet
            suppression = score_irrelevant_suppression(
                should_retrieve=query.should_retrieve,
                retrieved_count=result.retrieved_count,
            )
            if not query.should_retrieve and result.retrieved_count == 0:
                source_recall = 1.0
                text_contains = 1.0
            else:
                source_recall = score_source_recall(result.retrieved_source_refs, query.expected_source_refs)
                text_contains = score_text_contains(answer_text, query.expected_answer_contains)
            expected_present = score_text_contains(answer_text, query.expected_answer_contains) >= 1.0
            suppressed_present = bool(query.suppressed_text and query.suppressed_text.lower() in answer_text.lower())
            token_budget = MemoryQueryRouter._budget_and_limit(result.route or query.route)[0]
            rows.append(
                EvalScoreRow(
                    baseline=result.baseline,
                    query_id=query.id,
                    route=result.route or query.route,
                    source_recall=source_recall,
                    text_contains=text_contains,
                    suppression=suppression,
                    retrieved_count=result.retrieved_count,
                    token_estimate=result.token_estimate,
                    latency_ms=result.latency_ms,
                    retrieved_source_refs=list(result.retrieved_source_refs),
                    expected_source_refs=list(query.expected_source_refs),
                    forbidden_source_refs=list(query.forbidden_source_refs),
                    expected_answer_contains_count=len(query.expected_answer_contains),
                    expected_answer_contains_present=expected_present,
                    suppressed_text_present=suppressed_present,
                    token_budget=token_budget,
                    privacy_leakage=score_privacy_leakage(answer_text),
                    adversarial_instruction_following=score_adversarial_instruction_following(answer_text),
                    irrelevant_injection=score_irrelevant_injection(
                        should_retrieve=query.should_retrieve,
                        retrieved_count=result.retrieved_count,
                        answer=answer_text,
                    ),
                )
            )
    return EvalReport(dataset=dataset.name, rows=rows, summary=_summarize(rows), dataset_metadata=dict(dataset.metadata))


def _summarize(rows: list[EvalScoreRow]) -> dict[str, dict]:
    grouped: dict[str, list[EvalScoreRow]] = defaultdict(list)
    for row in rows:
        grouped[row.baseline].append(row)
    summary: dict[str, dict] = {}
    for baseline, baseline_rows in grouped.items():
        summary[baseline] = {
            "query_count": len(baseline_rows),
            "source_recall_avg": mean(row.source_recall for row in baseline_rows),
            "text_contains_avg": mean(row.text_contains for row in baseline_rows),
            "suppression_avg": mean(row.suppression for row in baseline_rows),
            "token_estimate_total": sum(row.token_estimate for row in baseline_rows),
            "latency_ms_avg": mean(row.latency_ms for row in baseline_rows),
        }
    return summary
