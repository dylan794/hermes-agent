"""Report-safe serializer tests for Memory v2 outward-facing payloads."""

from __future__ import annotations

import json

from plugins.memory.memory_v2 import MemoryV2Provider
from plugins.memory.memory_v2.belief_dashboard import build_belief_update_dashboard
from plugins.memory.memory_v2.dream import run_memory_dream_cycle
from plugins.memory.memory_v2.report_safety import report_safe_serialize
from plugins.memory.memory_v2.schemas import CandidateMemory, MemoryItem

SECRET_SENTINEL = "sk-report-safety-unique-secret-token-0001"
HOME_PATH = "/home/example/.hermes/private.txt"
WINDOWS_PATH = r"C:\Users\Example\Documents\private.txt"
WSL_WINDOWS_PATH = "/mnt/c/Users/Example/Documents/private.txt"
RAW_CANDIDATE_TEXT = "raw candidate text unique report safety marker"
RAW_OPEN_LOOP_TEXT = "raw open-loop text unique report safety marker"
RAW_CONTRADICTION_TEXT = "raw contradiction value unique report safety marker"
RAW_SOURCE_REF = "raw-source-ref-unique-report-safety-marker"

FORBIDDEN = [
    SECRET_SENTINEL,
    HOME_PATH,
    WINDOWS_PATH,
    WSL_WINDOWS_PATH,
    RAW_CANDIDATE_TEXT,
    RAW_OPEN_LOOP_TEXT,
    RAW_CONTRADICTION_TEXT,
    RAW_SOURCE_REF,
]


def _json(payload) -> str:
    return json.dumps(payload, sort_keys=True)


def _assert_forbidden_absent(payload) -> None:
    serialized = _json(payload)
    for forbidden in FORBIDDEN:
        assert forbidden not in serialized


def _provider(tmp_path) -> MemoryV2Provider:
    provider = MemoryV2Provider()
    provider.initialize("session-report-safety", hermes_home=str(tmp_path), platform="cli")
    return provider


def test_report_safe_serialize_redacts_nested_adversarial_strings() -> None:
    payload = {
        "success": True,
        "candidate": CandidateMemory(
            id="candidate_report_safe",
            type="fact",
            claim=f"{RAW_CANDIDATE_TEXT} {SECRET_SENTINEL}",
            source_refs=[RAW_SOURCE_REF],
            proposed_destination=HOME_PATH,
        ),
        "open_loop": {
            "id": "loop_report_safe",
            "text": f"{RAW_OPEN_LOOP_TEXT} {WINDOWS_PATH}",
            "source_refs": [RAW_SOURCE_REF],
        },
        "contradiction": {
            "value": f"{RAW_CONTRADICTION_TEXT} {WSL_WINDOWS_PATH}",
            "source_ref": RAW_SOURCE_REF,
        },
        "tuple_value": (None, 7, True),
    }

    safe = report_safe_serialize(payload)

    _assert_forbidden_absent(safe)
    assert safe["success"] is True
    assert safe["tuple_value"] == [None, 7, True]
    assert safe["candidate"]["claim"]["sha256"]
    assert safe["candidate"]["source_refs"]["source_ref_count"] == 1


def test_review_candidates_dream_and_dashboard_outputs_are_report_safe(tmp_path) -> None:
    provider = _provider(tmp_path)
    event = provider.store.append_raw_event(
        {
            "type": "turn",
            "session_id": "session-report-safety",
            "user_content": f"{RAW_CANDIDATE_TEXT} {SECRET_SENTINEL} {HOME_PATH}",
            "created_at": "2026-06-01T00:00:00Z",
        }
    )
    source_id = event["id"]
    provider.store.write_memory_item(
        MemoryItem(
            id="mem_report_safety_old",
            type="preference",
            subject="report safety subject",
            predicate="prefers_output",
            value=RAW_CONTRADICTION_TEXT,
            source_refs=[source_id, RAW_SOURCE_REF],
            confidence=0.9,
            status="active",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
    )
    provider.store.write_memory_item(
        MemoryItem(
            id="mem_report_safety_new",
            type="preference",
            subject="report safety subject",
            predicate="prefers_output",
            value=f"updated value {SECRET_SENTINEL}",
            source_refs=[source_id],
            confidence=0.95,
            status="active",
            created_at="2026-06-01T00:00:00Z",
            updated_at="2026-06-01T00:00:00Z",
        )
    )
    provider.store.append_candidate(
        CandidateMemory(
            id="cand_report_safety",
            type="preference",
            claim=f"report safety subject prefers {RAW_CANDIDATE_TEXT} {WINDOWS_PATH}",
            proposed_destination=WSL_WINDOWS_PATH,
            source_refs=[source_id, RAW_SOURCE_REF],
            confidence=0.91,
        )
    )
    provider.store.write_open_loops(
        [
            {
                "id": "loop_report_safety",
                "text": f"{RAW_OPEN_LOOP_TEXT} {HOME_PATH}",
                "status": "open",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "source_refs": [RAW_SOURCE_REF],
            }
        ]
    )

    review_payload = json.loads(provider.handle_tool_call("memory_v2_review_queue", {"now": "2026-06-20T00:00:00Z"}))
    candidates_payload = json.loads(provider.handle_tool_call("memory_v2_candidates", {}))
    contradictions_payload = json.loads(provider.handle_tool_call("memory_v2_contradictions", {}))
    dashboard_payload = build_belief_update_dashboard(
        items=provider.store.list_memory_items(),
        candidates=provider.store.list_candidates(),
        sources=provider.store.list_source_refs(),
        now="2026-06-20T00:00:00Z",
    )
    dream_payload = run_memory_dream_cycle(provider.store, provider.index, date="2026-06-20", auto_apply="off")

    for payload in [review_payload, candidates_payload, contradictions_payload, dashboard_payload, dream_payload]:
        _assert_forbidden_absent(payload)

    assert review_payload["items"][0]["source_refs"]["source_ref_count"] == 2
    assert candidates_payload["candidates"][0]["claim"]["sha256"]
    assert contradictions_payload["conflicts"][0]["memory_a"]["value"]["sha256"]
    assert dream_payload["open_loops"][0]["text_sha256"]
