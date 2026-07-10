"""Active recall / spaced-review tests for Memory v2."""

from __future__ import annotations

import json

from plugins.memory.memory_v2.schemas import CandidateMemory, MemoryItem, ProjectCard
from plugins.memory.memory_v2.spaced_review import (
    build_active_recall_review,
    recommend_next_review,
    score_recall_outcome,
)

NOW = "2026-06-27T12:00:00Z"
PRIVATE = "PRIVATE_SENTINEL_CERULEAN_RECALL_LEAK"


def _item(**overrides):
    data = {
        "id": "mem_pref_old",
        "type": "preference",
        "subject": f"Alex {PRIVATE}",
        "value": f"prefers concise updates {PRIVATE}",
        "status": "active",
        "confidence": 0.86,
        "importance": 0.95,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-15T00:00:00Z",
        "source_refs": ["src_1"],
    }
    data.update(overrides)
    return MemoryItem(**data)


def test_stale_high_importance_active_record_becomes_due():
    report = build_active_recall_review(items=[_item()], now=NOW)

    assert report["policy"] == "report_only_no_mutation"
    assert report["untrusted_text"] is True
    assert report["summary"]["due_now"] == 1
    card = report["due_now"][0]
    assert card["id"].startswith("record:")
    assert card["record_ref"].startswith("record:")
    assert card["record_id_sha256"]
    assert card["mutation"] == "none"
    assert "stale" in card["review_reasons"]
    assert card["subject_key"].startswith("subject:")
    assert card["answer_fingerprint"].startswith("sha256:")


def test_project_cards_and_open_loops_prioritized_over_generic_memory():
    generic = _item(id="mem_generic", importance=0.8, updated_at="2026-01-01T00:00:00Z")
    project = ProjectCard(
        id="project:secret-recall",
        name=f"Secret Recall {PRIVATE}",
        current_state=f"private current state {PRIVATE}",
        importance=0.6,
        updated_at="2026-06-01T00:00:00Z",
        source_refs=["src_project"],
    )
    loop = {"id": "loop_1", "text": f"follow up on {PRIVATE}", "status": "open", "updated_at": "2026-06-02T00:00:00Z", "source_refs": ["src_loop"]}

    report = build_active_recall_review(items=[generic], project_cards=[project], open_loops=[loop], now=NOW, max_cards=3)
    due_kinds = [card["kind"] for card in report["due_now"]]

    assert due_kinds[:2] == ["open_loop", "project_card"]
    assert all(card["kind"] in {"open_loop", "project_card", "memory_item"} for card in report["due_now"])


def test_inactive_superseded_archived_rejected_records_excluded():
    items = [
        _item(id="active", status="active"),
        _item(id="uncertain", status="uncertain"),
        _item(id="sup", status="superseded", superseded_by="active"),
        _item(id="arch", status="archived"),
        _item(id="rej", status="rejected"),
    ]
    candidates = [
        CandidateMemory(id="cand_pending", type="preference", claim="safe pending", gate_decision="pending"),
        CandidateMemory(id="cand_rejected", type="preference", claim="safe rejected", gate_decision="rejected", decision_reason="no"),
    ]

    report = build_active_recall_review(items=items, candidates=candidates, now=NOW)
    hashes = {card["record_id_sha256"] for section in ("due_now", "upcoming", "overdue") for card in report[section]}
    import hashlib

    assert {hashlib.sha256(value.encode("utf-8")).hexdigest() for value in {"active", "uncertain", "cand_pending"}}.issubset(hashes)
    assert {hashlib.sha256(value.encode("utf-8")).hexdigest() for value in {"sup", "arch", "rej", "cand_rejected"}}.isdisjoint(hashes)


def test_report_bounded_and_privacy_safe_with_sentinel_absent_from_full_json():
    items = [_item(id=f"mem_{i}", subject=f"subject {i} {PRIVATE}", value=f"value {i} {PRIVATE}") for i in range(30)]

    report = build_active_recall_review(items=items, now=NOW, max_cards=5)
    payload = json.dumps(report, sort_keys=True)

    assert len(report["due_now"]) <= 5
    assert len(report["upcoming"]) <= 5
    assert len(report["overdue"]) <= 5
    assert PRIVATE not in payload
    forbidden_keys = {"claim", "value", "body", "current_state", "name", "quote", "user_content"}
    assert forbidden_keys.isdisjoint(payload.split('"'))


def test_report_hashes_raw_record_ids_when_ids_contain_private_text():
    report = build_active_recall_review(
        open_loops=[{"id": PRIVATE, "text": "safe", "status": "open", "updated_at": "2026-01-01T00:00:00Z"}],
        now=NOW,
    )
    payload = json.dumps(report, sort_keys=True)

    assert PRIVATE not in payload
    assert report["due_now"][0]["id"].startswith("record:")
    assert report["retrieval_probes"][0]["target"].startswith("record:")
    assert report["suggested_actions"][0]["target"].startswith("record:")


def test_suggested_actions_are_report_only_mutation_none():
    report = build_active_recall_review(items=[_item()], now=NOW)

    assert report["suggested_actions"]
    assert {action["mutation"] for action in report["suggested_actions"]} == {"none"}
    assert all(action["policy"] == "report_only_no_mutation" for action in report["suggested_actions"])


def test_retrieval_review_logs_can_lengthen_or_shorten_interval_without_mutation():
    base = recommend_next_review(confidence=0.8, importance=0.8, age_days=100, prior_reviews=0)
    easy = recommend_next_review(confidence=0.8, importance=0.8, age_days=100, prior_reviews=2, last_score=1.0)
    hard = recommend_next_review(confidence=0.8, importance=0.8, age_days=100, prior_reviews=2, last_score=0.0)
    report = build_active_recall_review(
        items=[_item(id="reviewed")],
        retrieval_logs=[{"record_id": "reviewed", "reviewed_at": "2026-06-20T00:00:00Z", "score": 1.0}],
        now=NOW,
    )

    assert easy["interval_days"] > base["interval_days"]
    assert hard["interval_days"] < easy["interval_days"]
    assert report["due_now"][0]["prior_reviews"] == 1
    assert report["due_now"][0]["mutation"] == "none"


def test_score_recall_outcome_keyword_mode_and_fingerprint_only_mode():
    keyword_card = {"expected_keywords": ["source", "grounded"], "answer_fingerprint": "sha256:abc"}
    fingerprint_only = {"answer_fingerprint": "sha256:abc"}

    scored = score_recall_outcome(keyword_card, "The answer should stay source grounded.")
    unscorable = score_recall_outcome(fingerprint_only, "anything")

    assert scored["score"] == 1.0
    assert scored["matched_keywords"] == ["source", "grounded"]
    assert scored["mutation"] == "none"
    assert unscorable["score"] is None
    assert unscorable["reason"] == "fingerprint_only_cannot_score_without_keywords"
