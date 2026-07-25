"""Behavior contracts for the portable SkillRegistry interface."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from plugins.memory.memory_v2 import skill_registry as registry_module
from plugins.memory.memory_v2.skill_registry import (
    MAX_SKILL_SEARCH_RESULTS,
    SkillAuthorityDecision,
    SkillAuthorizationError,
    SkillBundle,
    SkillDescriptor,
    SkillExecutionOutcome,
    SkillExecutionReceipt,
    SkillLifecycle,
    SkillMutationAction,
    SkillMutationIntent,
    SkillProvenance,
    SkillRef,
    SkillRegistry,
    SkillRegistryError,
    SkillSearchQuery,
    SkillTrust,
    SourceSkillCandidate,
    compute_skill_bundle_digest,
    normalize_skill_bundle_files,
    require_skill_mutation_authority,
)


NOW = "2026-07-20T12:00:00Z"
LATER = "2026-07-20T13:00:00Z"
OTHER_DIGEST = "sha256:" + ("f" * 64)


def _files(body: bytes = b"# Review incidents\n") -> dict[str, bytes]:
    return {
        "SKILL.md": body,
        "references/checklist.md": b"Use the approved checklist.\n",
    }


def _ref(*, version: str = "1.0.0", files: dict[str, bytes] | None = None) -> SkillRef:
    content = files or _files()
    return SkillRef(
        skill_id="memory-v2/review-incidents",
        version=version,
        bundle_digest=compute_skill_bundle_digest(content),
    )


def _provenance() -> SkillProvenance:
    return SkillProvenance(
        source_type="git",
        source_id="example/repo@0123456",
        source_uri="https://example.invalid/example/repo/tree/0123456/review-incidents",
        observed_at="2026-07-19T10:00:00Z",
        acquired_at=NOW,
        evidence_refs=("raw:source-event",),
        signer_refs=("signer:operator-a",),
    )


def _staged_descriptor() -> SkillDescriptor:
    return SkillDescriptor(
        ref=_ref(),
        name="review-incidents",
        description="Review operational incidents from source evidence.",
        provenance=_provenance(),
        created_at=NOW,
        required_capabilities=("read:workspace",),
    )


def test_bundle_digest_is_order_independent_and_content_bound():
    files = _files()
    reversed_files = dict(reversed(tuple(files.items())))

    assert compute_skill_bundle_digest(files) == compute_skill_bundle_digest(reversed_files)
    assert compute_skill_bundle_digest(files) != compute_skill_bundle_digest(
        _files(b"# Changed procedure\n")
    )

    ref = _ref(files=files)
    bundle = SkillBundle(ref=ref, files=files)
    assert bundle.ref.bundle_digest == compute_skill_bundle_digest(bundle.files)
    with pytest.raises(TypeError):
        bundle.files["new.txt"] = b"not mutable"  # type: ignore[index]


@pytest.mark.parametrize(
    "path",
    (
        "../escape",
        "references/../../escape",
        "/absolute",
        "C:/windows",
        "a\\..\\escape",
        "references/CON.txt",
        "references/trailing. ",
    ),
)
def test_bundle_rejects_unsafe_paths(path: str):
    with pytest.raises(SkillRegistryError, match="unsafe bundle path"):
        normalize_skill_bundle_files({"SKILL.md": b"ok", path: b"bad"})


def test_bundle_rejects_missing_manifest_duplicate_canonical_paths_and_bounds(monkeypatch):
    with pytest.raises(SkillRegistryError, match="SKILL.md"):
        normalize_skill_bundle_files({"references/a.md": b"a"})
    with pytest.raises(SkillRegistryError, match="duplicate canonical"):
        normalize_skill_bundle_files(
            {"SKILL.md": b"ok", "references//a.md": b"a", "references/a.md": b"b"}
        )
    with pytest.raises(SkillRegistryError, match="duplicate canonical"):
        normalize_skill_bundle_files(
            {"SKILL.md": b"ok", "references/A.md": b"a", "references/a.md": b"b"}
        )

    monkeypatch.setattr(registry_module, "MAX_SKILL_FILE_BYTES", 2)
    with pytest.raises(SkillRegistryError, match="size limit"):
        normalize_skill_bundle_files({"SKILL.md": b"too large"})


def test_bundle_constructor_fails_closed_on_tampering():
    ref = _ref()
    with pytest.raises(SkillRegistryError, match="does not match"):
        SkillBundle(ref=ref, files=_files(b"# Tampered\n"))


def test_refs_are_fully_pinned_and_reject_latest_shortcuts():
    with pytest.raises(SkillRegistryError, match="version"):
        SkillRef(skill_id="review-incidents", version="", bundle_digest=OTHER_DIGEST)
    with pytest.raises(SkillRegistryError, match="immutable"):
        SkillRef(skill_id="review-incidents", version="latest", bundle_digest=OTHER_DIGEST)
    with pytest.raises(SkillRegistryError, match="bundle_digest"):
        SkillRef(skill_id="review-incidents", version="1", bundle_digest="latest")
    with pytest.raises(SkillRegistryError, match="skill_id"):
        SkillRef(skill_id="../review-incidents", version="1", bundle_digest=OTHER_DIGEST)


def test_descriptor_lifecycle_requires_review_and_complete_history_links():
    staged = _staged_descriptor()
    assert staged.lifecycle == SkillLifecycle.STAGED
    assert staged.trust == SkillTrust.UNVERIFIED
    assert staged.metadata_is_untrusted is True

    with pytest.raises(SkillRegistryError, match="active skills require"):
        SkillDescriptor(
            ref=staged.ref,
            name=staged.name,
            description=staged.description,
            provenance=staged.provenance,
            created_at=NOW,
            lifecycle="active",
        )

    active = SkillDescriptor(
        ref=staged.ref,
        name=staged.name,
        description=staged.description,
        provenance=staged.provenance,
        created_at=NOW,
        lifecycle="active",
        trust="reviewed",
        reviewed_at=NOW,
        activated_at=LATER,
    )
    assert active.lifecycle == SkillLifecycle.ACTIVE
    assert active.record_fingerprint().startswith("sha256:")

    with pytest.raises(SkillRegistryError, match="superseded skills require"):
        SkillDescriptor(
            ref=staged.ref,
            name=staged.name,
            description=staged.description,
            provenance=staged.provenance,
            created_at=NOW,
            lifecycle="superseded",
        )
    with pytest.raises(SkillRegistryError, match="revoked skills require"):
        SkillDescriptor(
            ref=staged.ref,
            name=staged.name,
            description=staged.description,
            provenance=staged.provenance,
            created_at=NOW,
            lifecycle="revoked",
        )


def test_descriptor_fingerprint_covers_lifecycle_and_provenance():
    staged = _staged_descriptor()
    changed = SkillDescriptor(
        ref=staged.ref,
        name=staged.name,
        description=staged.description,
        provenance=SkillProvenance(
            source_type=staged.provenance.source_type,
            source_id=staged.provenance.source_id,
            source_uri=staged.provenance.source_uri,
            observed_at=staged.provenance.observed_at,
            acquired_at="2026-07-20T12:00:01Z",
            evidence_refs=staged.provenance.evidence_refs,
            signer_refs=staged.provenance.signer_refs,
        ),
        created_at=staged.created_at,
    )
    assert staged.record_fingerprint() != changed.record_fingerprint()


def test_search_contract_is_bounded_and_current_by_default():
    query = SkillSearchQuery(required_capabilities=("read:workspace", "read:workspace"))
    assert query.lifecycle == (SkillLifecycle.ACTIVE,)
    assert query.required_capabilities == ("read:workspace",)
    with pytest.raises(SkillRegistryError, match="limit"):
        SkillSearchQuery(limit=MAX_SKILL_SEARCH_RESULTS + 1)


def test_source_candidate_fingerprint_binds_displayed_provenance_and_digest():
    first = SourceSkillCandidate(
        ref=_ref(),
        name="review-incidents",
        description="Review operational incidents from source evidence.",
        provenance=_provenance(),
    )
    second = SourceSkillCandidate(
        ref=_ref(version="1.0.1"),
        name=first.name,
        description=first.description,
        provenance=first.provenance,
    )
    assert first.candidate_fingerprint() != second.candidate_fingerprint()


class _Verifier:
    def __init__(self, decision_factory):
        self.decision_factory = decision_factory

    def verify(self, authority: object, intent: SkillMutationIntent) -> SkillAuthorityDecision:
        assert authority is AUTHORITY
        return self.decision_factory(intent)


class _BrokenVerifier:
    def verify(self, authority: object, intent: SkillMutationIntent) -> SkillAuthorityDecision:
        raise RuntimeError("verifier unavailable")


AUTHORITY = object()


def _stage_intent(*, confirm: bool = True) -> SkillMutationIntent:
    ref = _ref()
    return SkillMutationIntent(
        action="stage",
        subject_ref=ref,
        candidate_fingerprint=ref.bundle_digest,
        confirm=confirm,
        reason="Operator reviewed the exact quarantined bundle.",
        requested_by="workflow:skill-review",
        requested_at=NOW,
    )


def _allow(intent: SkillMutationIntent, **overrides) -> SkillAuthorityDecision:
    values = {
        "authorized": True,
        "principal_ref": "operator:alice",
        "authority_class": "operator",
        "scopes": (SkillMutationAction.STAGE,),
        "request_fingerprint": intent.request_fingerprint(),
        "issued_at": NOW,
        "expires_at": LATER,
    }
    values.update(overrides)
    return SkillAuthorityDecision(**values)


def test_mutation_authority_accepts_only_exact_host_verified_intent():
    intent = _stage_intent()
    decision = require_skill_mutation_authority(
        intent,
        AUTHORITY,
        _Verifier(_allow),
        now=datetime(2026, 7, 20, 12, 30, tzinfo=timezone.utc),
    )
    assert decision.principal_ref == "operator:alice"


@pytest.mark.parametrize(
    ("intent", "authority", "verifier", "message"),
    (
        (_stage_intent(confirm=False), AUTHORITY, _Verifier(_allow), "confirm"),
        (_stage_intent(), None, _Verifier(_allow), "authority"),
        (
            _stage_intent(),
            AUTHORITY,
            _Verifier(lambda intent: _allow(intent, request_fingerprint=OTHER_DIGEST)),
            "fingerprint",
        ),
        (
            _stage_intent(),
            AUTHORITY,
            _Verifier(lambda intent: _allow(intent, scopes=(SkillMutationAction.REVOKE,))),
            "scope",
        ),
        (_stage_intent(), AUTHORITY, _BrokenVerifier(), "failed closed"),
    ),
)
def test_mutation_authority_fails_closed(intent, authority, verifier, message):
    with pytest.raises(SkillAuthorizationError, match=message):
        require_skill_mutation_authority(
            intent,
            authority,
            verifier,
            now=datetime(2026, 7, 20, 12, 30, tzinfo=timezone.utc),
        )


def test_mutation_authority_rejects_expired_grant():
    intent = _stage_intent()
    with pytest.raises(SkillAuthorizationError, match="currently valid"):
        require_skill_mutation_authority(
            intent,
            AUTHORITY,
            _Verifier(_allow),
            now=datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc),
        )


def test_canonical_mutations_require_fresh_record_fingerprints():
    ref = _ref()
    with pytest.raises(SkillRegistryError, match="expected_record_fingerprint"):
        SkillMutationIntent(
            action="activate",
            subject_ref=ref,
            candidate_fingerprint=ref.bundle_digest,
            confirm=True,
            reason="Activate reviewed version.",
            requested_by="operator:alice",
            requested_at=NOW,
        )
    with pytest.raises(SkillRegistryError, match="reviewed record fingerprint"):
        SkillMutationIntent(
            action="revoke",
            subject_ref=ref,
            candidate_fingerprint=ref.bundle_digest,
            expected_record_fingerprint=OTHER_DIGEST,
            confirm=True,
            reason="Revoke compromised version.",
            requested_by="operator:alice",
            requested_at=NOW,
        )


def test_execution_receipt_is_exact_and_carries_no_execution_authority():
    receipt = SkillExecutionReceipt(
        execution_id="execution:1",
        ref=_ref(),
        host_id="host:test-sandbox",
        outcome=SkillExecutionOutcome.SUCCEEDED,
        started_at=NOW,
        completed_at=LATER,
        granted_capabilities=("read:workspace",),
        evidence_refs=("raw:execution-event",),
        output_digest=OTHER_DIGEST,
    )
    assert receipt.ref.version == "1.0.0"
    assert receipt.granted_capabilities == ("read:workspace",)
    assert not hasattr(receipt, "authority")

    with pytest.raises(SkillRegistryError, match="cannot precede"):
        SkillExecutionReceipt(
            execution_id="execution:2",
            ref=_ref(),
            host_id="host:test-sandbox",
            outcome="failed",
            started_at=LATER,
            completed_at=NOW,
        )


def test_registry_contract_has_no_execution_or_mutation_capability():
    assert not hasattr(SkillRegistry, "execute")
    assert not hasattr(SkillRegistry, "activate")
    assert not hasattr(SkillRegistry, "install")
