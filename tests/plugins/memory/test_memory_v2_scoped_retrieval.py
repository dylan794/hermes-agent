from __future__ import annotations

import math
import sqlite3

import pytest

from plugins.memory.memory_v2.scoped_retrieval import (
    ScopedEvidenceIndex,
    ScopedIndexIntegrityError,
)


def _unit(
    unit_id: str,
    text: str,
    *,
    profile: str = "profile-a",
    tenant: str = "tenant-a",
    workstreams: list[str] | None = None,
    evidence_at: str = "2026-05-01T12:00:00Z",
    status: str = "current",
    privacy: str = "private",
    visibility: str = "private",
    entities: list[str] | None = None,
    artifacts: list[str] | None = None,
    tombstone: bool = False,
) -> dict:
    return {
        "id": unit_id,
        "type": "verified_result",
        "title": f"Evidence {unit_id}",
        "text": text,
        "profile_id": profile,
        "tenant_id": tenant,
        "workstreams": workstreams or [],
        "evidence_at": evidence_at,
        "status": status,
        "privacy_level": privacy,
        "visibility": visibility,
        "entity_aliases": entities or [],
        "artifact_aliases": artifacts or [],
        "source_refs": [
            {
                "id": f"source-{unit_id}",
                "uri": f"session://{unit_id}",
                "observed_at": evidence_at,
            }
        ],
        "tombstone": tombstone,
    }


class _EmbeddingAdapter:
    model = "tiny-local"
    version = "1.2"
    dimension = 3
    artifact_digest = "sha256:" + ("a" * 64)

    def embed(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    float("alpha" in lower),
                    float("banana" in lower),
                    float("migration" in lower),
                ]
            )
        return vectors


def _search(index: ScopedEvidenceIndex, query: str, **kwargs):
    return index.search(
        query,
        profile_id="profile-a",
        tenant_id="tenant-a",
        evidence_cutoff="2026-06-01T00:00:00Z",
        workstreams=["alpha"],
        **kwargs,
    )


def test_hard_scope_and_temporal_filters_apply_before_ranking(tmp_path):
    index = ScopedEvidenceIndex(tmp_path / "derived.sqlite")
    index.rebuild(
        [
            _unit("good", "alpha migration verified", workstreams=["alpha"]),
            _unit("wrong-profile", "alpha migration exact", profile="profile-b", workstreams=["alpha"]),
            _unit("wrong-tenant", "alpha migration exact", tenant="tenant-b", workstreams=["alpha"]),
            _unit("wrong-project", "alpha migration exact", workstreams=["beta"]),
            _unit(
                "future",
                "alpha migration exact",
                workstreams=["alpha"],
                evidence_at="2026-07-01T00:00:00Z",
            ),
            _unit(
                "fraction-future",
                "alpha migration exact",
                workstreams=["alpha"],
                evidence_at="2026-06-01T00:00:00.500000Z",
            ),
            _unit("superseded", "alpha migration exact", workstreams=["alpha"], status="superseded"),
            _unit("stale", "alpha migration exact", workstreams=["alpha"], status="stale"),
            _unit("resolved", "alpha migration exact", workstreams=["alpha"], status="resolved"),
            _unit("rejected", "alpha migration exact", workstreams=["alpha"], status="rejected"),
            _unit("secret", "alpha migration exact", workstreams=["alpha"], privacy="restricted"),
            _unit("shared", "alpha migration exact", workstreams=["alpha"], visibility="shared"),
        ]
    )

    results = _search(
        index,
        "alpha migration",
        allowed_privacy=["private"],
        allowed_visibility=["private"],
    )

    assert [result["id"] for result in results] == ["good"]
    assert results[0]["evidence_at"] == "2026-05-01T12:00:00Z"
    assert results[0]["source_refs"] == [
        {
            "id": "source-good",
            "uri": "session://good",
            "observed_at": "2026-05-01T12:00:00Z",
        }
    ]

    history = _search(
        index,
        "alpha migration",
        history=True,
        allowed_privacy=["private"],
        allowed_visibility=["private"],
    )
    assert {result["id"] for result in history} == {
        "good",
        "superseded",
        "stale",
        "resolved",
    }


def test_unknown_and_multi_workstream_fallback_is_bounded(tmp_path):
    index = ScopedEvidenceIndex(tmp_path / "derived.sqlite")
    index.rebuild(
        [
            _unit("exact", "banana release decision", workstreams=["alpha"]),
            _unit("multi", "banana release rationale", workstreams=["alpha", "gamma"]),
            _unit("unknown-a", "banana background note"),
            _unit("unknown-b", "banana secondary note"),
            _unit("other", "banana unrelated", workstreams=["beta"]),
        ]
    )

    results = _search(
        index,
        "banana release",
        unknown_workstream_limit=1,
    )

    assert {"exact", "multi"} <= {row["id"] for row in results}
    assert "other" not in {row["id"] for row in results}
    assert len([row for row in results if not row["workstreams"]]) <= 1
    assert all(row["scope_diagnostics"]["workstream_match"] in {"exact", "multi", "unknown"} for row in results)

    exact_only = _search(
        index,
        "banana release",
        unknown_workstream_limit=0,
        multi_workstream_limit=0,
    )
    assert [row["id"] for row in exact_only] == ["exact"]


def test_fallback_caps_apply_after_relevance_ranking(tmp_path):
    index = ScopedEvidenceIndex(tmp_path / "derived.sqlite")
    index.rebuild(
        [
            _unit(
                "unknown-new-noise",
                "migration filler",
                evidence_at="2026-05-03T00:00:00Z",
            ),
            _unit(
                "unknown-old-relevant",
                "rare migration decision verification",
                evidence_at="2026-05-01T00:00:00Z",
            ),
            _unit(
                "multi-new-noise",
                "migration filler",
                workstreams=["alpha", "gamma"],
                evidence_at="2026-05-03T00:00:00Z",
            ),
            _unit(
                "multi-old-relevant",
                "rare migration decision verification",
                workstreams=["alpha", "gamma"],
                evidence_at="2026-05-01T00:00:00Z",
            ),
        ]
    )

    results = _search(
        index,
        "rare migration decision verification",
        unknown_workstream_limit=1,
        multi_workstream_limit=1,
    )

    assert {row["id"] for row in results} == {
        "unknown-old-relevant",
        "multi-old-relevant",
    }


def test_fielded_fts_aliases_and_capped_typed_neighbors_are_fused(tmp_path):
    index = ScopedEvidenceIndex(tmp_path / "derived.sqlite")
    index.rebuild(
        [
            _unit(
                "decision",
                "Selected the blue deployment.",
                workstreams=["alpha"],
                artifacts=["launch-manifest"],
            ),
            _unit("verification", "The deployment passed all checks.", workstreams=["alpha"]),
            _unit("unrelated-neighbor", "A different check.", workstreams=["alpha"]),
        ],
        aliases={
            "decision": {
                "artifact": ["ship sheet"],
                "entity": ["Project Starling"],
            }
        },
        edges=[
            {"source_id": "decision", "target_id": "verification", "type": "verified_by"},
            {"source_id": "decision", "target_id": "unrelated-neighbor", "type": "mentions"},
        ],
    )

    results = _search(
        index,
        "Project Starling ship sheet",
        neighbor_types=["verified_by"],
        neighbor_limit=1,
    )

    by_id = {row["id"]: row for row in results}
    assert "alias" in by_id["decision"]["retrieval_pools"]
    assert by_id["decision"]["score_diagnostics"]["rrf_score"] > 0
    assert "verification" in by_id
    assert by_id["verification"]["neighbor_diagnostics"] == [
        {"from_id": "decision", "relation_type": "verified_by"}
    ]
    assert "unrelated-neighbor" not in by_id


def test_embedding_pool_is_versioned_finite_and_tombstones_disappear_on_rebuild(tmp_path):
    path = tmp_path / "derived.sqlite"
    adapter = _EmbeddingAdapter()
    index = ScopedEvidenceIndex(path, embedding_adapter=adapter)
    units = [
        _unit("semantic", "banana plan", workstreams=["alpha"]),
        _unit("lexical", "alpha exact words", workstreams=["alpha"]),
    ]
    first = index.rebuild(units)
    assert first["embedding"]["model"] == "tiny-local"
    assert first["embedding"]["version"] == "1.2"
    assert len(first["embedding"]["digest"]) == 64

    results = _search(index, "banana")
    assert results[0]["id"] == "semantic"
    assert "dense" in results[0]["retrieval_pools"]
    assert math.isfinite(results[0]["score_diagnostics"]["cosine_score"])

    index.rebuild(units, tombstones=["semantic"])
    assert "semantic" not in {row["id"] for row in _search(index, "banana")}


def test_malformed_embedding_and_index_metadata_fail_closed(tmp_path):
    class BadAdapter(_EmbeddingAdapter):
        def embed(self, texts):
            return [[float("nan"), 0.0, 1.0] for _ in texts]

    with pytest.raises(ScopedIndexIntegrityError, match="finite"):
        ScopedEvidenceIndex(tmp_path / "bad.sqlite", embedding_adapter=BadAdapter()).rebuild(
            [_unit("bad", "alpha", workstreams=["alpha"])]
        )

    path = tmp_path / "tampered.sqlite"
    index = ScopedEvidenceIndex(path)
    index.rebuild([_unit("good", "alpha", workstreams=["alpha"])])
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE metadata SET value = '999' WHERE key = 'schema_version'")

    with pytest.raises(ScopedIndexIntegrityError, match="metadata"):
        _search(index, "alpha")


def test_embedding_adapter_identity_and_dimension_are_pinned_to_rebuilt_index(tmp_path):
    path = tmp_path / "identity.sqlite"
    index = ScopedEvidenceIndex(path, embedding_adapter=_EmbeddingAdapter())
    index.rebuild([_unit("good", "alpha banana", workstreams=["alpha"])])

    class DifferentModel(_EmbeddingAdapter):
        model = "different-local"

    class DifferentVersion(_EmbeddingAdapter):
        version = "9.9"

    class DifferentDimension(_EmbeddingAdapter):
        dimension = 4

        def embed(self, texts):
            return [[1.0, 1.0, 0.0, 0.0] for _ in texts]

    for adapter in (DifferentModel(), DifferentVersion(), DifferentDimension()):
        mismatched = ScopedEvidenceIndex(path, embedding_adapter=adapter)
        with pytest.raises(ScopedIndexIntegrityError, match="identity"):
            _search(mismatched, "alpha")

    class ExcessiveDimension(_EmbeddingAdapter):
        dimension = 8_193

    with pytest.raises(ScopedIndexIntegrityError, match="between 1 and 8192"):
        ScopedEvidenceIndex(
            tmp_path / "excessive.sqlite",
            embedding_adapter=ExcessiveDimension(),
        ).rebuild([_unit("good", "alpha", workstreams=["alpha"])])


def test_content_fts_edges_and_vectors_are_covered_by_integrity_manifest(tmp_path):
    def built_path(name: str, *, embedding: bool = False):
        path = tmp_path / f"{name}.sqlite"
        adapter = _EmbeddingAdapter() if embedding else None
        index = ScopedEvidenceIndex(path, embedding_adapter=adapter)
        index.rebuild(
            [
                _unit("a", "alpha migration", workstreams=["alpha"]),
                _unit("b", "alpha verification", workstreams=["alpha"]),
            ],
            edges=[{"source_id": "a", "target_id": "b", "type": "verified_by"}],
        )
        return path, index

    evidence_path, evidence_index = built_path("evidence")
    with sqlite3.connect(evidence_path) as conn:
        conn.execute("UPDATE evidence SET text = 'tampered' WHERE id = 'a'")
        conn.execute("UPDATE evidence_fts SET text = 'tampered' WHERE id = 'a'")
    with pytest.raises(ScopedIndexIntegrityError, match="integrity manifest"):
        _search(evidence_index, "tampered")

    fts_path, fts_index = built_path("fts")
    with sqlite3.connect(fts_path) as conn:
        conn.execute("UPDATE evidence_fts SET text = 'tampered' WHERE id = 'a'")
    with pytest.raises(ScopedIndexIntegrityError, match="FTS content"):
        _search(fts_index, "tampered")

    duplicate_path, duplicate_index = built_path("fts-duplicate")
    with sqlite3.connect(duplicate_path) as conn:
        row = conn.execute(
            "SELECT id, title, text, entity_aliases, artifact_aliases, workstreams "
            "FROM evidence_fts WHERE id = 'a'"
        ).fetchone()
        conn.execute(
            "INSERT INTO evidence_fts VALUES (?, ?, ?, ?, ?, ?)",
            row,
        )
    with pytest.raises(ScopedIndexIntegrityError, match="duplicated"):
        _search(duplicate_index, "alpha")

    edge_path, edge_index = built_path("edge")
    with sqlite3.connect(edge_path) as conn:
        conn.execute("DELETE FROM edges")
    with pytest.raises(ScopedIndexIntegrityError, match="integrity manifest"):
        _search(edge_index, "alpha")

    vector_path, vector_index = built_path("vector", embedding=True)
    with sqlite3.connect(vector_path) as conn:
        conn.execute(
            "UPDATE embeddings SET vector = zeroblob(length(vector)) WHERE unit_id = 'a'"
        )
    with pytest.raises(ScopedIndexIntegrityError, match="integrity manifest"):
        _search(vector_index, "alpha")


def test_repeated_atomic_rebuild_closes_windows_handle_and_leaves_no_temp(tmp_path):
    path = tmp_path / "derived.sqlite"
    index = ScopedEvidenceIndex(path)

    index.rebuild([_unit("first", "alpha first", workstreams=["alpha"])])
    index.rebuild([_unit("second", "alpha second", workstreams=["alpha"])])

    assert [row["id"] for row in _search(index, "alpha")] == ["second"]
    assert list(tmp_path.glob(".derived.sqlite.*.tmp")) == []


def test_results_are_bounded_and_rebuild_does_not_mutate_inputs(tmp_path):
    units = [_unit(f"u-{number}", "alpha banana migration", workstreams=["alpha"]) for number in range(40)]
    before = repr(units)
    index = ScopedEvidenceIndex(tmp_path / "derived.sqlite")
    summary = index.rebuild(units)

    results = _search(index, "alpha banana migration", limit=100)

    assert len(results) == 30
    assert repr(units) == before
    assert summary["indexed_units"] == 40


def test_unscoped_fts_query_does_not_require_a_dense_full_corpus_scan(tmp_path):
    units = [
        _unit(
            f"u-{number:04d}",
            "ordinary background evidence",
            workstreams=[f"stream-{number % 10}"],
        )
        for number in range(1_001)
    ]
    units[777]["text"] = "globally unique continuity needle"
    index = ScopedEvidenceIndex(tmp_path / "derived.sqlite")
    index.rebuild(units)

    results = index.search(
        "globally unique continuity needle",
        profile_id="profile-a",
        tenant_id="tenant-a",
        evidence_cutoff="2026-06-01T00:00:00Z",
        workstreams=[],
        limit=5,
    )

    assert results[0]["id"] == "u-0777"


def test_evidence_node_mapping_can_receive_explicit_rebuild_scope(tmp_path):
    node_mapping = {
        "id": "evidence:node",
        "kind": "decision",
        "text": "Use the alpha migration plan.",
        "observed_at": "2026-05-01T12:00:00Z",
        "source_refs": ["raw-event-1"],
        "project_ids": ["project:alpha"],
        "workstream_ids": ["workstream:alpha-release"],
    }
    index = ScopedEvidenceIndex(tmp_path / "derived.sqlite")

    index.rebuild(
        [node_mapping],
        default_profile_id="profile-a",
        default_tenant_id="tenant-a",
    )
    results = index.search(
        "alpha migration",
        profile_id="profile-a",
        tenant_id="tenant-a",
        evidence_cutoff="2026-06-01T00:00:00Z",
        workstreams=["workstream:alpha-release"],
    )

    assert results[0]["id"] == "evidence:node"
    assert results[0]["workstreams"] == [
        "project:alpha",
        "workstream:alpha-release",
    ]
    assert results[0]["source_refs"] == [
        {"id": "raw-event-1", "observed_at": "2026-05-01T12:00:00Z"}
    ]
    assert results[0]["scope_diagnostics"]["workstream_match"] == "exact"
