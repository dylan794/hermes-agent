from __future__ import annotations

import time

import pytest

from plugins.memory.memory_v2.scoped_retrieval import ScopedIndexIntegrityError
from plugins.memory.memory_v2.shadow_pipeline import (
    ShadowPipelineConfig,
    ShadowPipelineError,
    ShadowRetrievalPipeline,
)
from plugins.memory.memory_v2.shadow_reranker import ShadowRerankerConfig


ADAPTER_DIGEST = "sha256:" + ("b" * 64)


def _enabled_config(**overrides) -> ShadowPipelineConfig:
    values = {
        "enabled": True,
        "reranker": ShadowRerankerConfig(enabled=True, minimum_utility=0.2),
    }
    values.update(overrides)
    return ShadowPipelineConfig(
        **values,
    )


def _events() -> list[dict]:
    events = [
        {
            "id": "event-decision",
            "type": "turn",
            "user_content": (
                "Decision: use SQLite FTS for the alpha migration. "
                "Next action: verify the migration."
            ),
            "observed_at": "2026-05-01T10:00:00Z",
            "provider_session_id": "root-alpha",
            "project_id": "alpha",
            "workstream_id": "migration",
        },
        {
            "id": "event-claim",
            "type": "turn",
            "assistant_content": "Implemented the alpha migration.",
            "observed_at": "2026-05-01T11:00:00Z",
            "provider_session_id": "root-alpha",
            "project_id": "alpha",
            "workstream_id": "migration",
        },
        {
            "id": "event-tool",
            "type": "tool",
            "result": "All tests passed for the alpha SQLite migration.",
            "observed_at": "2026-05-01T12:00:00Z",
            "provider_session_id": "root-alpha",
            "project_id": "alpha",
            "workstream_id": "migration",
        },
        {
            "id": "event-other",
            "type": "turn",
            "user_content": "Decision: ship the unrelated beta dashboard.",
            "observed_at": "2026-05-01T12:30:00Z",
            "project_id": "beta",
            "workstream_id": "dashboard",
        },
    ]
    for event in events:
        event["profile_id"] = "profile-a"
        event["tenant_id"] = "tenant-a"
    return events


def test_disabled_default_does_not_build_index_or_call_adapters(tmp_path):
    calls = []
    pipeline = ShadowRetrievalPipeline(
        tmp_path / "derived.sqlite",
        structured_extractor=lambda payload: calls.append(payload),
        reranker_adapter=lambda payload: calls.append(payload),
        scratch_root=tmp_path,
    )

    result = pipeline.run(
        query="What did we decide?",
        raw_events=_events(),
        profile_id="profile-a",
        tenant_id="tenant-a",
        evidence_cutoff="2026-06-01T00:00:00Z",
        context={},
    )

    assert result["enabled"] is False
    assert result["mutation_authority"] == "none"
    assert result["retrieval"]["candidate_count"] == 0
    assert calls == []
    assert not (tmp_path / "derived.sqlite").exists()


def test_current_context_gate_skips_derivation_and_index(tmp_path):
    pipeline = ShadowRetrievalPipeline(
        tmp_path / "derived.sqlite",
        config=_enabled_config(),
        scratch_root=tmp_path,
    )

    result = pipeline.run(
        query="Can you explain what you just said?",
        raw_events=_events(),
        profile_id="profile-a",
        tenant_id="tenant-a",
        evidence_cutoff="2026-06-01T00:00:00Z",
        context={"has_current_context": True, "gap_days": 0},
    )

    assert result["enabled"] is True
    assert result["intent"]["decision"] == "none"
    assert result["intent"]["reason"] == "current_context_sufficient"
    assert not (tmp_path / "derived.sqlite").exists()


def test_derivation_budget_is_enforced_during_extraction(tmp_path):
    result = ShadowRetrievalPipeline(
        tmp_path / "derived.sqlite",
        config=_enabled_config(max_derived_nodes=1),
        scratch_root=tmp_path,
    ).run(
        query="What did we previously decide for alpha?",
        raw_events=_events(),
        profile_id="profile-a",
        tenant_id="tenant-a",
        evidence_cutoff="2026-06-01T00:00:00Z",
        context={"gap_days": 30, "project_id": "alpha"},
    )

    assert result["corpus"]["derived_node_count"] == 1
    assert "total_node_budget_exhausted" in result["corpus"]["overflow_rejections"]
    assert result["corpus"]["rejected_node_count"] >= 1
    assert result["index"]["indexed_units"] == 1
    assert not (tmp_path / "derived.sqlite").exists()


def test_end_to_end_returns_scoped_exact_citations_and_verified_neighbor(tmp_path):
    pipeline = ShadowRetrievalPipeline(
        tmp_path / "derived.sqlite",
        config=_enabled_config(),
        scratch_root=tmp_path,
    )

    result = pipeline.run(
        query="What did we decide and verify for the alpha migration?",
        raw_events=_events(),
        profile_id="profile-a",
        tenant_id="tenant-a",
        evidence_cutoff="2026-06-01T00:00:00Z",
        context={
            "has_current_context": False,
            "gap_days": 30,
            "project_id": "alpha",
            "workstream_id": "migration",
        },
    )

    assert result["enabled"] is True
    assert result["shadow_only"] is True
    assert result["read_only"] is True
    assert result["mutation_authority"] == "none"
    assert result["query_scope"]["project_ids"] == ["project:alpha"]
    assert result["query_scope"]["workstream_ids"] == ["workstream:migration"]
    assert result["corpus"]["raw_event_count"] == 4
    assert result["retrieval"]["candidate_count"] <= 20

    bundle = result["result"]["bundle"]
    assert bundle["untrusted_data"] is True
    assert bundle["bounded"] is True
    assert len(bundle["items"]) <= bundle["max_items"]
    by_type = {item["type"]: item for item in bundle["items"]}
    assert "decision" in by_type
    assert "verified_result" in by_type
    assert by_type["verified_result"]["verified"] is True
    citation = by_type["decision"]["citations"][0]
    source_text = _events()[0]["user_content"]
    assert source_text[citation["start"] : citation["end"]] == citation["text"]
    assert citation["source_id"] == "event-decision"
    assert citation["evidence_at"] == "2026-05-01T10:00:00Z"
    assert {
        citation["source_id"]
        for item in result["result"]["ranked_candidates"]
        for citation in item["citations"]
    }.isdisjoint({"event-other"})


def test_future_evidence_is_suppressed(tmp_path):
    events = _events()
    events.append(
        {
            "id": "event-future",
            "type": "tool",
            "result": "All tests passed for the future-only alpha migration.",
            "observed_at": "2027-01-01T00:00:00Z",
            "project_id": "alpha",
            "workstream_id": "migration",
            "profile_id": "profile-a",
            "tenant_id": "tenant-a",
        }
    )
    captured = {}

    def extractor(payload):
        captured.update(payload)
        return {
            "schema_version": 1,
            "mutation_authority": "none",
            "proposals": [],
        }
    extractor.artifact_digest = ADAPTER_DIGEST

    pipeline = ShadowRetrievalPipeline(
        tmp_path / "derived.sqlite",
        config=_enabled_config(
            structured_extractor_artifact_digest=ADAPTER_DIGEST
        ),
        structured_extractor=extractor,
        scratch_root=tmp_path,
    )

    result = pipeline.run(
        query="What did we previously verify for alpha migration?",
        raw_events=events,
        profile_id="profile-a",
        tenant_id="tenant-a",
        evidence_cutoff="2026-06-01T00:00:00Z",
        context={"gap_days": 30, "project_id": "alpha", "workstream_id": "migration"},
    )

    citation_sources = {
        citation["source_id"]
        for item in result["result"]["ranked_candidates"]
        for citation in item["citations"]
    }
    assert "event-future" not in citation_sources
    assert "event-future" not in {
        event["source_id"] for event in captured["events"]
    }
    assert result["corpus"]["policy_exclusions"]["future"] == 1
    assert not (tmp_path / "derived.sqlite").exists()


def test_current_suppresses_explicit_superseded_evidence_but_history_keeps_it(
    tmp_path,
):
    events = _events()
    events[0]["memory_status"] = "superseded"

    current = ShadowRetrievalPipeline(
        tmp_path / "current.sqlite",
        config=_enabled_config(),
        scratch_root=tmp_path,
    ).run(
        query="What is the current alpha migration decision?",
        raw_events=events,
        profile_id="profile-a",
        tenant_id="tenant-a",
        evidence_cutoff="2026-06-01T00:00:00Z",
        context={"gap_days": 30, "project_id": "alpha", "workstream_id": "migration"},
    )
    history = ShadowRetrievalPipeline(
        tmp_path / "history.sqlite",
        config=_enabled_config(),
        scratch_root=tmp_path,
    ).run(
        query="What did we previously decide for alpha migration?",
        raw_events=events,
        profile_id="profile-a",
        tenant_id="tenant-a",
        evidence_cutoff="2026-06-01T00:00:00Z",
        context={"gap_days": 30, "project_id": "alpha", "workstream_id": "migration"},
    )

    current_sources = {
        citation["source_id"]
        for item in current["result"]["ranked_candidates"]
        for citation in item["citations"]
    }
    history_sources = {
        citation["source_id"]
        for item in history["result"]["ranked_candidates"]
        for citation in item["citations"]
    }
    assert "event-decision" not in current_sources
    assert "event-decision" in history_sources


def test_structured_extractor_is_only_called_when_enabled(tmp_path):
    calls = []

    def extractor(payload):
        calls.append(payload)
        return {
            "schema_version": 1,
            "mutation_authority": "none",
            "proposals": [],
        }
    extractor.artifact_digest = ADAPTER_DIGEST

    pipeline = ShadowRetrievalPipeline(
        tmp_path / "derived.sqlite",
        config=_enabled_config(
            structured_extractor_artifact_digest=ADAPTER_DIGEST
        ),
        structured_extractor=extractor,
        scratch_root=tmp_path,
    )
    pipeline.run(
        query="What did we previously decide for alpha?",
        raw_events=_events(),
        profile_id="profile-a",
        tenant_id="tenant-a",
        evidence_cutoff="2026-06-01T00:00:00Z",
        context={"gap_days": 30, "project_id": "alpha"},
    )

    assert len(calls) == 1
    assert calls[0]["mutation_authority"] == "none"


@pytest.mark.parametrize(
    ("events", "context", "message"),
    [
        (
            [
                {
                    "id": "duplicate",
                    "type": "turn",
                    "user_content": "Decision: choose alpha.",
                    "observed_at": "2026-05-01T00:00:00Z",
                },
                {
                    "id": "duplicate",
                    "type": "turn",
                    "user_content": "Decision: choose beta.",
                    "observed_at": "2026-05-02T00:00:00Z",
                },
            ],
            {"gap_days": 30},
            "unique",
        ),
        (
            _events(),
            {"gap_days": 30, "mutation_authority": "promote"},
            "unknown fields",
        ),
    ],
)
def test_ambiguous_identity_and_context_fail_closed(
    tmp_path, events, context, message
):
    pipeline = ShadowRetrievalPipeline(
        tmp_path / "derived.sqlite",
        config=_enabled_config(),
        scratch_root=tmp_path,
    )

    with pytest.raises(ShadowPipelineError, match=message):
        pipeline.run(
            query="What did we previously decide?",
            raw_events=events,
            profile_id="profile-a",
            tenant_id="tenant-a",
            evidence_cutoff="2026-06-01T00:00:00Z",
            context=context,
        )

    assert not (tmp_path / "derived.sqlite").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"profile_id": "other-profile"}, "profile"),
        ({"tenant_id": "other-tenant"}, "tenant"),
        ({"profile_id": ""}, "explicit profile_id"),
    ],
)
def test_source_scope_cannot_be_relabelled(tmp_path, mutation, message):
    events = _events()
    events[0].update(mutation)
    pipeline = ShadowRetrievalPipeline(
        tmp_path / "derived.sqlite",
        config=_enabled_config(),
        scratch_root=tmp_path,
    )

    with pytest.raises(ShadowPipelineError, match=message):
        pipeline.run(
            query="What did we previously decide?",
            raw_events=events,
            profile_id="profile-a",
            tenant_id="tenant-a",
            evidence_cutoff="2026-06-01T00:00:00Z",
            context={"gap_days": 30},
        )


def test_policy_disabled_and_tombstoned_events_never_reach_extractor(tmp_path):
    events = _events()
    events[0]["privacy_level"] = "secret"
    events[1]["retrieval_disabled"] = True
    events[2]["tombstone"] = True
    captured = {}

    def extractor(payload):
        captured.update(payload)
        return {
            "schema_version": 1,
            "mutation_authority": "none",
            "proposals": [],
        }
    extractor.artifact_digest = ADAPTER_DIGEST

    result = ShadowRetrievalPipeline(
        tmp_path / "derived.sqlite",
        config=_enabled_config(
            structured_extractor_artifact_digest=ADAPTER_DIGEST
        ),
        structured_extractor=extractor,
        scratch_root=tmp_path,
    ).run(
        query="What did we previously decide about beta?",
        raw_events=events,
        profile_id="profile-a",
        tenant_id="tenant-a",
        evidence_cutoff="2026-06-01T00:00:00Z",
        context={"gap_days": 30},
    )

    adapter_sources = {event["source_id"] for event in captured["events"]}
    assert adapter_sources == {"event-other"}
    assert result["corpus"]["policy_exclusions"]["privacy"] == 1
    assert result["corpus"]["policy_exclusions"]["retrieval_disabled"] == 1
    assert result["corpus"]["policy_exclusions"]["tombstone"] == 1


def test_existing_or_out_of_capability_index_target_is_refused(tmp_path):
    existing = tmp_path / "existing.sqlite"
    existing.write_text("canonical data", encoding="utf-8")

    with pytest.raises(ShadowPipelineError, match="overwrite"):
        ShadowRetrievalPipeline(
            existing,
            config=_enabled_config(),
            scratch_root=tmp_path,
        ).run(
            query="What did we previously decide?",
            raw_events=_events(),
            profile_id="profile-a",
            tenant_id="tenant-a",
            evidence_cutoff="2026-06-01T00:00:00Z",
            context={"gap_days": 30},
        )
    assert existing.read_text(encoding="utf-8") == "canonical data"

    outside = tmp_path.parent / "outside.sqlite"
    with pytest.raises(ShadowPipelineError, match="scratch_root"):
        ShadowRetrievalPipeline(
            outside,
            config=_enabled_config(),
            scratch_root=tmp_path,
        ).run(
            query="What did we previously decide?",
            raw_events=_events(),
            profile_id="profile-a",
            tenant_id="tenant-a",
            evidence_cutoff="2026-06-01T00:00:00Z",
            context={"gap_days": 30},
        )


def test_adapter_digest_and_event_byte_bounds_fail_closed(tmp_path):
    def extractor(_payload):
        return {
            "schema_version": 1,
            "mutation_authority": "none",
            "proposals": [],
        }

    extractor.artifact_digest = "sha256:" + ("c" * 64)
    mismatched = ShadowRetrievalPipeline(
        tmp_path / "mismatch.sqlite",
        config=_enabled_config(
            structured_extractor_artifact_digest=ADAPTER_DIGEST
        ),
        structured_extractor=extractor,
        scratch_root=tmp_path,
    )
    with pytest.raises(ShadowPipelineError, match="pinned artifact digest"):
        mismatched.run(
            query="What did we previously decide?",
            raw_events=_events(),
            profile_id="profile-a",
            tenant_id="tenant-a",
            evidence_cutoff="2026-06-01T00:00:00Z",
            context={"gap_days": 30},
        )

    oversized = _events()
    oversized[0]["user_content"] = "x" * 256_001
    bounded = ShadowRetrievalPipeline(
        tmp_path / "bounded.sqlite",
        config=_enabled_config(),
        scratch_root=tmp_path,
    )
    with pytest.raises(ShadowPipelineError, match="byte bound"):
        bounded.run(
            query="What did we previously decide?",
            raw_events=oversized,
            profile_id="profile-a",
            tenant_id="tenant-a",
            evidence_cutoff="2026-06-01T00:00:00Z",
            context={"gap_days": 30},
        )

    assert not (tmp_path / "mismatch.sqlite").exists()
    assert not (tmp_path / "bounded.sqlite").exists()


@pytest.mark.parametrize(
    "field_value",
    [
        "private",
        b"private",
        {"private": True},
        ["private", 7],
        [],
        [f"scope-{index}" for index in range(17)],
        ["x" * 81],
    ],
)
def test_policy_scope_config_requires_bounded_non_string_sequences(field_value):
    with pytest.raises(ShadowPipelineError, match="scope|privacy"):
        _enabled_config(allowed_privacy=field_value)


def test_policy_scope_config_is_normalized_and_defensively_copied(tmp_path):
    privacy_scopes = [" private ", "private"]
    visibility_scopes = [" private ", "private"]
    config = _enabled_config(
        allowed_privacy=privacy_scopes,
        allowed_visibility=visibility_scopes,
        structured_extractor_artifact_digest=ADAPTER_DIGEST,
    )
    privacy_scopes.append("secret")
    visibility_scopes.append("shared")
    captured = {}

    def extractor(payload):
        captured.update(payload)
        return {
            "schema_version": 1,
            "mutation_authority": "none",
            "proposals": [],
        }

    extractor.artifact_digest = ADAPTER_DIGEST
    events = _events()
    events[0]["privacy_level"] = "secret"
    events[0]["visibility"] = "shared"
    result = ShadowRetrievalPipeline(
        tmp_path / "derived.sqlite",
        config=config,
        structured_extractor=extractor,
        scratch_root=tmp_path,
    ).run(
        query="What did we previously decide about beta?",
        raw_events=events,
        profile_id="profile-a",
        tenant_id="tenant-a",
        evidence_cutoff="2026-06-01T00:00:00Z",
        context={"gap_days": 30},
    )

    assert config.allowed_privacy == ("private",)
    assert config.allowed_visibility == ("private",)
    assert "event-decision" not in {
        event["source_id"] for event in captured["events"]
    }
    assert result["corpus"]["policy_exclusions"]["privacy"] == 1


@pytest.mark.parametrize(
    "gap_days",
    [float("nan"), float("inf"), 36_501, 10**400],
)
def test_invalid_gap_days_fail_before_any_adapter_sees_data(tmp_path, gap_days):
    calls = []

    def extractor(payload):
        calls.append(("extractor", payload))
        return {
            "schema_version": 1,
            "mutation_authority": "none",
            "proposals": [],
        }

    def reranker(payload):
        calls.append(("reranker", payload))
        return {}

    extractor.artifact_digest = ADAPTER_DIGEST
    reranker.artifact_digest = ADAPTER_DIGEST
    pipeline = ShadowRetrievalPipeline(
        tmp_path / "derived.sqlite",
        config=_enabled_config(
            structured_extractor_artifact_digest=ADAPTER_DIGEST,
            reranker=ShadowRerankerConfig(
                enabled=True,
                minimum_utility=0.2,
                model_artifact_digest=ADAPTER_DIGEST,
            ),
        ),
        structured_extractor=extractor,
        reranker_adapter=reranker,
        scratch_root=tmp_path,
    )

    with pytest.raises(ShadowPipelineError, match="gap_days"):
        pipeline.run(
            query="What did we previously decide?",
            raw_events=_events(),
            profile_id="profile-a",
            tenant_id="tenant-a",
            evidence_cutoff="2026-06-01T00:00:00Z",
            context={"gap_days": gap_days},
        )

    assert calls == []
    assert not (tmp_path / "derived.sqlite").exists()


def test_embedding_timeout_circuit_persists_across_pipeline_runs(tmp_path):
    class SlowEmbedding:
        model = "slow-local"
        version = "1"
        dimension = 2
        artifact_digest = ADAPTER_DIGEST

        def __init__(self):
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            time.sleep(0.05)
            return [[0.0, 1.0] for _ in texts]

    adapter = SlowEmbedding()
    pipeline = ShadowRetrievalPipeline(
        tmp_path / "derived.sqlite",
        config=_enabled_config(embedding_timeout_ms=5),
        embedding_adapter=adapter,
        scratch_root=tmp_path,
    )
    request = {
        "query": "What did we previously decide for alpha?",
        "raw_events": _events(),
        "profile_id": "profile-a",
        "tenant_id": "tenant-a",
        "evidence_cutoff": "2026-06-01T00:00:00Z",
        "context": {"gap_days": 30, "project_id": "alpha"},
    }

    with pytest.raises(ScopedIndexIntegrityError, match="embedding adapter failed"):
        pipeline.run(**request)
    with pytest.raises(ScopedIndexIntegrityError, match="embedding adapter failed"):
        pipeline.run(**request)

    assert adapter.calls == 1
    assert not (tmp_path / "derived.sqlite").exists()


def test_reranker_timeout_circuit_persists_across_pipeline_runs(tmp_path):
    calls = 0

    def slow_reranker(_payload):
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        return {}

    slow_reranker.artifact_digest = ADAPTER_DIGEST
    pipeline = ShadowRetrievalPipeline(
        tmp_path / "derived.sqlite",
        config=_enabled_config(
            reranker=ShadowRerankerConfig(
                enabled=True,
                minimum_utility=0.2,
                model_artifact_digest=ADAPTER_DIGEST,
                adapter_timeout_ms=5,
                adapter_failure_mode="none",
            )
        ),
        reranker_adapter=slow_reranker,
        scratch_root=tmp_path,
    )
    request = {
        "query": "What did we previously decide for alpha?",
        "raw_events": _events(),
        "profile_id": "profile-a",
        "tenant_id": "tenant-a",
        "evidence_cutoff": "2026-06-01T00:00:00Z",
        "context": {"gap_days": 30, "project_id": "alpha"},
    }

    timed_out = pipeline.run(**request)
    circuit_open = pipeline.run(**request)

    assert timed_out["result"]["adapter"]["status"] == "timeout"
    assert circuit_open["result"]["adapter"]["status"] == "circuit_open"
    assert calls == 1
    assert not (tmp_path / "derived.sqlite").exists()
