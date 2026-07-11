from __future__ import annotations

import json

import yaml

from plugins.memory.memory_v2 import MemoryV2Provider


def _provider(tmp_path, memory_v2: dict | None = None) -> MemoryV2Provider:
    config = {
        "archive": {
            "enabled": True,
            "backfill_enabled": False,
            "search_tools_enabled": True,
            "show_tools_enabled": True,
            "include_tool_outputs": False,
        },
        "extraction": {"enabled": False, "candidate_creation_enabled": False},
        "consolidation": {"enabled": False},
        "prefetch": {"enabled": False},
        "review_apply": {"enabled": False},
        "auto_promote": {"enabled": False},
    }
    if memory_v2:
        for key, value in memory_v2.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"memory_v2": config}, sort_keys=False),
        encoding="utf-8",
    )
    provider = MemoryV2Provider()
    provider.initialize("archive-readiness", hermes_home=str(tmp_path), platform="cli")
    return provider


def _tool(provider: MemoryV2Provider, name: str, args: dict | None = None) -> dict:
    payload = json.loads(provider.handle_tool_call(name, args or {}))
    assert isinstance(payload, dict)
    return payload


def _seed_event(provider: MemoryV2Provider) -> str:
    event = provider.store.append_raw_event(
        {
            "type": "turn",
            "session_id": "archive-readiness",
            "created_at": "2026-06-20T00:00:00Z",
            "user_content": "Synthetic alpha proof for read only archive readiness.",
            "assistant_content": "Synthetic acknowledgement only.",
            "content": "Synthetic imported evidence row.",
            "privacy_level": "standard",
            "archive_status": "raw_evidence",
        }
    )
    return str(event["id"])


def _mutation_counts(provider: MemoryV2Provider) -> dict[str, int | bool]:
    return {
        "pending_candidates": provider.store.count_pending_candidates(),
        "candidates": provider.store.count_candidates(),
        "memory_items": len(provider.store.list_memory_items()),
        "project_cards": len(provider.store.list_project_cards()),
        "open_loops": len(provider.store.list_open_loops()),
        "operation_records": len(provider.store.list_operation_records()),
        "operations_file_exists": provider.store.operations_path.exists(),
    }


def test_archive_readiness_gate_proves_step5_contract_without_mutations(tmp_path) -> None:
    provider = _provider(tmp_path)
    event_id = _seed_event(provider)
    before = _mutation_counts(provider)

    payload = _tool(provider, "memory_v2_archive_readiness", {})

    assert payload["success"] is True
    assert payload["ready"] is True
    assert payload["rollout_step"] == 5
    assert payload["mutations_allowed"] is False
    assert payload["mutations"] == {"allowed": False, "created_candidates": 0, "created_memories": 0, "created_project_cards": 0, "created_open_loops": 0}
    assert payload["blockers"] == []
    assert payload["checks"]["feature_flags"]["ok"] is True
    assert payload["checks"]["archive_integrity"]["ok"] is True
    assert payload["checks"]["raw_index"]["ok"] is True
    assert payload["checks"]["source_refs"]["ok"] is True
    assert payload["checks"]["packet_contract"]["ok"] is True
    assert payload["proof"]["sample_event_ids"] == [event_id]

    for packet in [payload["proof"]["archive_search"]["results"][0], payload["proof"]["archive_show"]["event"]]:
        assert packet["event_id"] == event_id
        assert packet["untrusted_text"] is True
        assert packet["can_instruct"] is False
        assert packet["labels_trusted"] is False
        assert packet["evidence_boundary"] == "UNTRUSTED ARCHIVE EVIDENCE"
        assert packet["source_ref"]["id"] == event_id
        assert packet["source_ref"]["uri"] == f"raw_event:{event_id}"
        assert packet["chain"]["record_sha256"].startswith("sha256:")
        assert packet["integrity"]["status"] == "ok"
        assert packet["excerpts"]["user_content"]["sha256"].startswith("sha256:")
        assert packet["excerpts"]["user_content"]["chars"] <= payload["limits"]["excerpt_chars"] + 3

    assert _mutation_counts(provider) == before


def test_archive_readiness_gate_fails_closed_when_search_or_show_flags_disabled(tmp_path) -> None:
    provider = _provider(tmp_path, {"archive": {"search_tools_enabled": False}})
    _seed_event(provider)

    payload = _tool(provider, "memory_v2_archive_readiness", {})

    assert payload["success"] is False
    assert "disabled" in payload["error"].lower()


def test_archive_readiness_gate_fails_closed_on_tampered_raw_event_integrity(tmp_path) -> None:
    provider = _provider(tmp_path)
    _seed_event(provider)
    raw_path = provider.store.raw_events_path
    raw_path.write_text(
        raw_path.read_text(encoding="utf-8").replace("alpha", "bravo", 1),
        encoding="utf-8",
    )

    payload = _tool(provider, "memory_v2_archive_readiness", {})

    assert payload["success"] is False
    assert payload["ready"] is False
    assert payload["mutations_allowed"] is False
    assert payload["checks"]["archive_integrity"]["ok"] is False
    assert any("raw archive integrity" in blocker for blocker in payload["blockers"])


def test_archive_readiness_gate_fails_closed_on_missing_source_ref(tmp_path) -> None:
    provider = _provider(tmp_path)
    event_id = _seed_event(provider)
    provider.store._source_ref_path(event_id).unlink()

    payload = _tool(provider, "memory_v2_archive_readiness", {})

    assert payload["success"] is False
    assert payload["ready"] is False
    assert payload["checks"]["source_refs"]["ok"] is False
    assert any("source ref" in blocker for blocker in payload["blockers"])
