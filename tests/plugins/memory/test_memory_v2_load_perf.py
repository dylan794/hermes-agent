"""Load/perf gates for bounded Memory v2 large-store operations.

These are intentionally generous complexity gates, not microbenchmarks.  They
build moderately large fake stores so CI can catch accidental O(n^2) behavior or
unbounded payload dumps without touching any real profile data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from plugins.memory.memory_v2 import MemoryV2Provider
from plugins.memory.memory_v2.artifacts import (
    compose_artifact_memory_packets,
    search_artifact_segments,
)
from plugins.memory.memory_v2.schemas import ArtifactRecord, ArtifactSegment, CandidateMemory
from plugins.memory.memory_v2.store import MemoryV2Store


# Deliberately loose: these tests should fail only on complexity regressions or
# unexpectedly unbounded output, not ordinary CI noise.
STORE_OP_SECONDS = 2.0
TOOL_OP_SECONDS = 3.0
ARTIFACT_OP_SECONDS = 4.0
DAILY_REPORT_SECONDS = 10.0


def _provider(tmp_path: Path) -> MemoryV2Provider:
    provider = MemoryV2Provider()
    provider.initialize("session-load-perf", hermes_home=str(tmp_path), platform="cli")
    return provider


def _store(tmp_path: Path) -> MemoryV2Store:
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    return store


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _seed_raw_events(store: MemoryV2Store, count: int) -> None:
    _write_jsonl(
        store.raw_events_path,
        [
            {
                "id": f"event_{index:05d}",
                "type": "turn",
                "session_id": f"session_{index % 17}",
                "created_at": f"2026-06-{(index % 28) + 1:02d}T00:00:00Z",
                "user_content": f"Fake raw event {index} with bounded payload.",
                "assistant_content": "ack",
            }
            for index in range(count)
        ],
    )


def _seed_candidates(store: MemoryV2Store, count: int, *, source_count: int = 20) -> None:
    _write_jsonl(
        store.candidates_path,
        [
            CandidateMemory(
                id=f"cand_{index:05d}",
                type="preference" if index % 3 else "fact",
                claim=f"Synthetic bounded candidate claim {index}.",
                proposed_destination="core/user",
                confidence=0.8,
                source_refs=[f"event_{index % source_count:05d}"],
                created_at=f"2026-06-{(index % 28) + 1:02d}T00:00:00Z",
            ).to_dict()
            for index in range(count)
        ],
    )


def _timed(callable_):
    start = time.perf_counter()
    result = callable_()
    elapsed = time.perf_counter() - start
    return result, elapsed


def test_raw_event_tail_reads_are_bounded_and_fast_on_large_jsonl(tmp_path: Path):
    store = _store(tmp_path)
    _seed_raw_events(store, 5_000)

    events, elapsed = _timed(lambda: store.read_raw_events(limit=25))

    assert elapsed < STORE_OP_SECONDS
    assert len(events) == 25
    assert events[0]["id"] == "event_04975"
    assert events[-1]["id"] == "event_04999"
    assert len(json.dumps(events)) < 10_000


def test_candidate_listing_tool_caps_large_store_output_and_stays_fast(tmp_path: Path):
    provider = _provider(tmp_path)
    _seed_candidates(provider.store, 2_000)

    payload, elapsed = _timed(
        lambda: json.loads(
            provider.handle_tool_call("memory_v2_candidates", {"limit": 50})
        )
    )

    assert elapsed < TOOL_OP_SECONDS
    assert payload["success"] is True
    assert payload["count"] == 50
    assert len(payload["candidates"]) == 50
    assert payload["candidates"][0]["id"] == "cand_00000"
    assert "cand_01999" not in json.dumps(payload)
    assert len(json.dumps(payload)) < 40_000


def test_review_queue_caps_items_groups_and_recommendations_on_large_candidate_store(tmp_path: Path):
    provider = _provider(tmp_path)
    _seed_raw_events(provider.store, 5_000)
    _seed_candidates(provider.store, 1_200, source_count=5_000)

    payload, elapsed = _timed(
        lambda: json.loads(
            provider.handle_tool_call(
                "memory_v2_review_queue",
                {"limit": 40, "now": "2026-07-01T00:00:00Z"},
            )
        )
    )

    assert elapsed < TOOL_OP_SECONDS
    assert payload["success"] is True
    assert payload["review_summary"]["pending"] == 1_200
    assert payload["review_summary"]["reviewed"] == 40
    assert payload["review_summary"]["truncated"] is True
    assert len(payload["items"]) == 40
    assert max(len(ids) for ids in payload["groups"]["by_type"].values()) <= 40
    assert all(len(ids) <= 40 for ids in payload["recommended_actions"].values())
    assert "cand_01199" not in json.dumps(payload)
    assert len(json.dumps(payload)) < 120_000


def test_dream_cycle_honors_review_and_action_caps_on_large_candidate_store(tmp_path: Path):
    provider = _provider(tmp_path)
    _seed_raw_events(provider.store, 5_000)
    _seed_candidates(provider.store, 1_000, source_count=5_000)

    report, elapsed = _timed(
        lambda: json.loads(
            provider.handle_tool_call(
                "memory_v2_dream_cycle",
                {
                    "date": "2026-06-28",
                    "max_review_items": 35,
                    "max_actions": 12,
                },
            )
        )
    )

    assert elapsed < TOOL_OP_SECONDS
    assert report["success"] is True
    assert report["review_queue_summary"]["pending"] == 1_000
    assert report["review_queue_summary"]["reviewed"] == 35
    assert report["review_queue_summary"]["truncated"] is True
    plan_summary = report["review_plan"]["summary"]
    assert plan_summary["proposed_promotions"] + plan_summary["proposed_rejections"] <= 12
    assert "cand_00999" not in json.dumps(report)
    assert len(json.dumps(report)) < 180_000


def test_daily_report_honors_recent_raw_limit_on_large_raw_and_candidate_stores(tmp_path: Path):
    provider = _provider(tmp_path)
    _seed_raw_events(provider.store, 5_000)
    _seed_candidates(provider.store, 1_000, source_count=5_000)

    report, elapsed = _timed(
        lambda: json.loads(
            provider.handle_tool_call(
                "memory_v2_daily_report",
                {"date": "2026-06-28"},
            )
        )
    )

    report_json = json.dumps(report, sort_keys=True)
    assert elapsed < DAILY_REPORT_SECONDS
    assert report["success"] is True
    assert report["before_counts"]["raw_events"] == 5_000
    assert report["before_counts"]["candidates"] == 1_000
    assert len(report["recent_raw_event_ids"]) <= 50
    assert "event_00000" not in report_json
    assert "cand_00999" not in report_json
    assert len(report_json) < 160_000


def _seed_artifacts(store: MemoryV2Store, *, artifacts: int, segments_per_artifact: int) -> None:
    for artifact_index in range(artifacts):
        artifact_id = f"art_perf_{artifact_index:03d}"
        record = ArtifactRecord(
            id=artifact_id,
            content_hash=f"sha256:{artifact_index:064x}"[-71:],
            modality="text",
            source_type="local_file",
            source_uri=f"file:///tmp/fake-artifact-{artifact_index}.txt",
            metadata={"raw_ref": f"artifacts/raw/{artifact_index:02x}/fake-{artifact_index}.txt"},
        )
        store.write_artifact_record(record)
        segment_dir = store.derived_artifacts_dir / artifact_id / "segments"
        segment_dir.mkdir(parents=True, exist_ok=True)
        for segment_index in range(segments_per_artifact):
            global_index = artifact_index * segments_per_artifact + segment_index
            segment = ArtifactSegment(
                id=f"seg_{artifact_index:03d}_{segment_index:03d}",
                artifact_id=artifact_id,
                segment_type="text_chunk",
                text=(
                    "needle " if global_index % 7 == 0 else "filler "
                )
                + f"synthetic artifact segment {global_index} "
                + ("x" * 300),
                summary=f"summary {global_index}",
                location={"char_start": global_index * 100, "char_end": global_index * 100 + 100},
                source_ref={
                    "artifact_id": artifact_id,
                    "content_hash": record.content_hash,
                    "path": record.metadata["raw_ref"],
                },
            )
            (segment_dir / f"{segment.id}.yaml").write_text(
                yaml.safe_dump(segment.to_dict(), sort_keys=False),
                encoding="utf-8",
            )


def test_artifact_retrieval_and_packets_are_limited_budgeted_and_fast(tmp_path: Path):
    store = _store(tmp_path)
    _seed_artifacts(store, artifacts=80, segments_per_artifact=12)

    segments, search_elapsed = _timed(
        lambda: search_artifact_segments(store, "needle", limit=7)
    )
    packets, packet_elapsed = _timed(
        lambda: compose_artifact_memory_packets(
            store, "needle", limit=5, token_budget=180
        )
    )

    packet_json = json.dumps(packets, sort_keys=True)
    assert search_elapsed < ARTIFACT_OP_SECONDS
    assert packet_elapsed < ARTIFACT_OP_SECONDS
    assert len(segments) == 7
    assert len(packets) <= 5
    assert len(packet_json) <= 180 * 4
    assert "synthetic artifact segment 959" not in packet_json
    assert all(packet["type"] == "artifact_memory_packet" for packet in packets)
