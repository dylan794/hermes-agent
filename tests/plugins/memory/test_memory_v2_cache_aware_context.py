from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from plugins.memory.memory_v2.context_packets import (
    DYNAMIC_MEMORY_BOUNDARY_BEGIN,
    DYNAMIC_MEMORY_BOUNDARY_END,
    CacheAwareContextPacket,
    estimate_tokens,
    render_dynamic_memory_packet,
    render_stable_prompt,
    stable_prompt_blocks,
)
from plugins.memory.memory_v2 import MemoryV2Provider
from plugins.memory.memory_v2.index import MemoryV2Index
from plugins.memory.memory_v2.retrieval import MemoryPacketComposer
from plugins.memory.memory_v2.schemas import (
    CoreMemoryRecord,
    MemoryItem,
    MemoryPacket,
    WorkingMemory,
)


@pytest.fixture()
def seeded_memory_index(tmp_path: Path) -> MemoryV2Index:
    provider = MemoryV2Provider()
    provider.initialize("cache-aware-context", hermes_home=str(tmp_path), platform="pytest")
    index = provider.index
    index.index_memory_item(
        MemoryItem(
            id="memory_v2_project_card",
            type="project_state",
            subject="Memory v2",
            predicate="current_state",
            value="Memory v2 is adding cache-aware context packet budgets.",
            summary="Memory v2 project continuity should use compact source-grounded packets.",
            source_refs=["source://session/cache-aware-test"],
            tags=["memory_v2", "project_continuity"],
            importance=0.9,
        )
    )
    return index


def _enabled_provider(tmp_path: Path, *, session_id: str) -> MemoryV2Provider:
    (tmp_path / "config.yaml").write_text(
        """
memory_v2:
  prefetch:
    enabled: true
  working_memory:
    enabled: true
""".lstrip(),
        encoding="utf-8",
    )
    provider = MemoryV2Provider()
    provider.initialize(session_id, hermes_home=str(tmp_path), platform="pytest")
    return provider


def test_stable_prompt_blocks_are_cache_safe_and_do_not_include_dynamic_memory():
    blocks = stable_prompt_blocks()

    assert [block["id"] for block in blocks] == [
        "memory_v2_contract",
        "memory_v2_cache_policy",
        "memory_v2_safety_policy",
    ]
    rendered = yaml.safe_dump(blocks, sort_keys=False)
    assert DYNAMIC_MEMORY_BOUNDARY_BEGIN not in rendered
    assert "session_id" not in rendered.lower()
    assert "created_at" not in rendered.lower()
    assert "retrieved memory item" not in rendered.lower()


def test_stable_prompt_only_includes_source_grounded_cache_safe_core_records():
    rendered = render_stable_prompt(
        [
            CoreMemoryRecord(
                id="core_safe",
                category="user",
                statement="Fixture user prefers concise, evidence-backed answers.",
                source_refs=["source_fixture_001"],
            ),
            CoreMemoryRecord(
                id="core_without_source",
                category="user",
                statement="Unsourced fixture claim must stay out.",
            ),
            CoreMemoryRecord(
                id="core_session_local",
                category="user",
                statement=(
                    "session_id fixture_session_001 uses channel_id "
                    "fixture_channel_001 and /home/fixture_user/private.txt"
                ),
                source_refs=["source_fixture_002"],
            ),
            CoreMemoryRecord(
                id="core_private_value",
                category="user",
                statement="Fixture authorization is Bearer fake-token-123.",
                source_refs=["source_fixture_003"],
            ),
        ]
    )

    assert "Fixture user prefers concise" in rendered
    assert "source_refs=verified" in rendered
    assert "source_fixture_001" not in rendered
    assert "Unsourced fixture claim" not in rendered
    assert "fixture_session_001" not in rendered
    assert "fixture_channel_001" not in rendered
    assert "/home/fixture_user" not in rendered
    assert "fake-token-123" not in rendered


def test_dynamic_memory_packet_has_explicit_boundary_and_untrusted_label():
    packet = MemoryPacket(
        route="project_continuity",
        confidence="high",
        token_budget=400,
        items=[
            {
                "id": "project_memory_v2",
                "type": "project_state",
                "status": "active",
                "summary": "Memory v2 needs source-grounded project cards.",
                "source_refs": ["source://session/example"],
            }
        ],
        retrieval_plan={"route": "project_continuity"},
    )

    rendered = render_dynamic_memory_packet(packet)

    assert rendered.startswith(DYNAMIC_MEMORY_BOUNDARY_BEGIN)
    assert rendered.rstrip().endswith(DYNAMIC_MEMORY_BOUNDARY_END)
    assert "untrusted data" in rendered
    assert "project_memory_v2" in rendered


def test_cache_aware_context_packet_keeps_stable_and_dynamic_sections_separate():
    packet = MemoryPacket(
        route="preference_recall",
        confidence="medium",
        token_budget=300,
        items=[{"id": "pref_concise", "summary": "User prefers concise responses."}],
        retrieval_plan={"route": "preference_recall"},
    )
    context = CacheAwareContextPacket(dynamic_packet=packet)

    rendered = context.render()
    stable_part, dynamic_part = rendered.split(DYNAMIC_MEMORY_BOUNDARY_BEGIN, 1)

    assert "memory_v2_contract" in stable_part
    assert "pref_concise" not in stable_part
    assert "pref_concise" in dynamic_part
    assert DYNAMIC_MEMORY_BOUNDARY_END in dynamic_part


def test_memory_packet_renderer_respects_budget_after_dynamic_boundary_overhead():
    item = {
        "id": "large_memory",
        "type": "project_state",
        "status": "active",
        "summary": "Memory v2 " + ("source grounded continuity " * 200),
        "source_refs": ["source://session/one"],
    }
    packet = MemoryPacket(
        route="project_continuity",
        confidence="high",
        token_budget=160,
        items=[item],
        retrieval_plan={"route": "project_continuity"},
    )

    rendered = render_dynamic_memory_packet(packet, include_boundary=True)

    assert estimate_tokens(rendered) <= 160
    assert "large_memory" in rendered
    assert DYNAMIC_MEMORY_BOUNDARY_BEGIN in rendered


def test_memory_packet_composer_budget_rows_fit_dynamic_boundary(seeded_memory_index):
    composer = MemoryPacketComposer(seeded_memory_index)
    packet = composer.compose("where did we leave Memory v2 project continuity?")

    rendered = render_dynamic_memory_packet(packet, include_boundary=True)

    assert estimate_tokens(rendered) <= packet.token_budget
    assert DYNAMIC_MEMORY_BOUNDARY_BEGIN in rendered
    assert DYNAMIC_MEMORY_BOUNDARY_END in rendered


def test_provider_keeps_working_state_dynamic_without_changing_stable_prompt(
    tmp_path: Path,
):
    provider = _enabled_provider(tmp_path, session_id="session_fixture_001")
    provider.store.write_core_memory_record(
        CoreMemoryRecord(
            id="core_fixture_preference",
            category="user",
            statement="Fixture user prefers compact project updates.",
            source_refs=["source_fixture_core_001"],
        )
    )
    stable_before = provider.system_prompt_block()
    provider.store.write_current_working_memory(
        WorkingMemory(
            session_id="session_fixture_001",
            focus={"current_user_message": "synthetic current-turn recall needle"},
            scratchpad={"open_item": "synthetic session-local state"},
        )
    )

    dynamic = provider.prefetch(
        "what current work is pending?", session_id="session_fixture_001"
    )
    stable_after = provider.system_prompt_block()

    assert stable_after == stable_before
    assert "Fixture user prefers compact project updates." in stable_before
    assert "session_fixture_001" not in stable_before
    assert "synthetic current-turn recall needle" not in stable_before
    assert dynamic.startswith(DYNAMIC_MEMORY_BOUNDARY_BEGIN)
    assert dynamic.rstrip().endswith(DYNAMIC_MEMORY_BOUNDARY_END)
    assert "untrusted data" in dynamic.lower()
    assert "working_memory" in dynamic
    assert "synthetic current-turn recall needle" in dynamic
    assert estimate_tokens(dynamic) <= 800
