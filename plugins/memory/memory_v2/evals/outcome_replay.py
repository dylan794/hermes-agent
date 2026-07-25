"""Offline outcome-replay and bottleneck diagnostics for Memory v2.

The lab consumes minimized, frozen shadow-episode artifacts.  It never invokes
tools, replays external side effects, mutates memory, or authorizes skill
lifecycle changes.  Oracle variants are paired diagnostic observations used to
locate performance regret across the memory pipeline.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import random
import statistics
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATASET_SCHEMA_VERSION = "memory-v2-outcome-replay-dataset/v1"
PRIVATE_INTAKE_SCHEMA_VERSION = "memory-v2-outcome-replay-private-intake/v1"
RESULT_SCHEMA_VERSION = "memory-v2-outcome-replay-result/v1"
DISJOINTNESS_SCHEMA_VERSION = "memory-v2-outcome-replay-disjointness/v1"
MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_EPISODES = 10_000

QUERY_CLASSES = (
    "current_state",
    "historical_state",
    "source_rationale",
    "project_resumption",
    "open_loop",
    "procedure",
    "correction_contradiction",
    "suppression_abstention",
)

ABLATION_COMPONENTS = {
    "current": "none",
    "no_memory": "memory_contribution",
    "oracle_archive": "archive_extraction",
    "oracle_candidate": "candidate_recall",
    "oracle_route": "routing",
    "oracle_rerank": "ranking",
    "oracle_temporal": "temporal_resolution",
    "oracle_packet": "packet_composition",
    "oracle_synthesis": "answer_synthesis",
}

COMPONENT_ACTIONS = {
    "archive_extraction": "Improve capture and source-grounded extraction coverage.",
    "candidate_recall": "Improve index coverage and candidate generation before ranking.",
    "routing": "Improve the query planner and route classification.",
    "ranking": "Calibrate a shadow-only reranker under hard temporal filters.",
    "temporal_resolution": "Improve current/history conflict and supersession resolution.",
    "packet_composition": "Improve bounded evidence selection and packet structure.",
    "answer_synthesis": "Improve grounded answer use, citation checks, and abstention.",
}

LABEL_SOURCES = {"independent_judge", "trusted_operator", "deterministic_harness"}
_HEX = frozenset("0123456789abcdef")


class ValidationError(ValueError):
    """Raised when a replay artifact violates its strict schema."""


def load_dataset(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate a JSON or YAML replay dataset."""

    return validate_dataset(_load_document(path))


def collect_private_intake(
    raw: Mapping[str, Any],
    *,
    token_material: bytes,
) -> dict[str, Any]:
    """Minimize a private opt-in intake into the public replay schema.

    The caller owns the private intake and token key. Neither is returned or
    persisted here. Stable, domain-separated HMACs preserve exact overlap
    auditing without placing low-entropy identities or query text in the
    replay dataset.
    """

    if not isinstance(token_material, bytes) or len(token_material) != 32:
        raise ValidationError("token material must contain exactly 32 bytes")
    value = _mapping(raw, "private_intake")
    _exact_keys(
        value,
        {
            "schema_version",
            "lab_id",
            "study_mode",
            "created_at",
            "consent",
            "thresholds",
            "episodes",
        },
        "private_intake",
    )
    if value["schema_version"] != PRIVATE_INTAKE_SCHEMA_VERSION:
        raise ValidationError(
            "private_intake.schema_version must equal "
            f"{PRIVATE_INTAKE_SCHEMA_VERSION!r}"
        )
    consent = _mapping(value["consent"], "private_intake.consent")
    _exact_keys(
        consent,
        {
            "collection_mode",
            "consent_granted",
            "consent_ref",
            "retention_until",
            "revocation_policy_ref",
            "profile_scope_id",
        },
        "private_intake.consent",
    )
    if consent["collection_mode"] != "opt_in_shadow":
        raise ValidationError(
            "private_intake.consent.collection_mode must equal 'opt_in_shadow'"
        )
    if consent["consent_granted"] is not True:
        raise ValidationError("private_intake.consent.consent_granted must be true")

    episodes_raw = _list(value["episodes"], "private_intake.episodes")
    transformed_episodes: list[dict[str, Any]] = []
    for index, raw_episode in enumerate(episodes_raw):
        where = f"private_intake.episodes[{index}]"
        episode = _mapping(raw_episode, where)
        _exact_keys(
            episode,
            {
                "participant_id",
                "project_id",
                "workstream_id",
                "query_text",
                "query_class",
                "checkpoint_days",
                "evidence_cutoff",
                "snapshot",
                "variants",
            },
            where,
        )
        participant_id = _private_identifier(
            episode["participant_id"], f"{where}.participant_id"
        )
        project_id = _private_identifier(episode["project_id"], f"{where}.project_id")
        workstream_id = _private_identifier(
            episode["workstream_id"], f"{where}.workstream_id"
        )
        query_text = _private_query(episode["query_text"], f"{where}.query_text")
        evidence_cutoff = _timestamp(
            episode["evidence_cutoff"], f"{where}.evidence_cutoff"
        )
        snapshot = _mapping(episode["snapshot"], f"{where}.snapshot")
        _exact_keys(
            snapshot,
            {
                "corpus_id",
                "archive_digest",
                "index_digest",
                "config_digest",
                "code_digest",
                "answerer_digest",
            },
            f"{where}.snapshot",
        )
        corpus_id = _private_identifier(snapshot["corpus_id"], f"{where}.snapshot.corpus_id")
        episode_identity = json.dumps(
            {
                "participant_id": participant_id,
                "project_id": project_id,
                "workstream_id": workstream_id,
                "query_text": query_text,
                "evidence_cutoff": evidence_cutoff,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        transformed_episodes.append(
            {
                "episode_id": _keyed_opaque(token_material, "episode", episode_identity),
                "participant_ref": _keyed_opaque(
                    token_material, "participant", participant_id
                ),
                "project_ref": _keyed_opaque(token_material, "project", project_id),
                "workstream_ref": _keyed_opaque(
                    token_material, "workstream", workstream_id
                ),
                "query_ref": _keyed_opaque(token_material, "query-ref", query_text),
                "query_hash": _keyed_digest(token_material, "query-hash", query_text),
                "query_class": episode["query_class"],
                "checkpoint_days": episode["checkpoint_days"],
                "evidence_cutoff": evidence_cutoff,
                "snapshot": {
                    "corpus_ref": _keyed_opaque(token_material, "corpus", corpus_id),
                    **{
                        field_name: snapshot[field_name]
                        for field_name in (
                            "archive_digest",
                            "index_digest",
                            "config_digest",
                            "code_digest",
                            "answerer_digest",
                        )
                    },
                },
                "variants": episode["variants"],
            }
        )

    minimized = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "lab_id": value["lab_id"],
        "study_mode": value["study_mode"],
        "created_at": value["created_at"],
        "privacy": {
            "collection_mode": "opt_in_shadow",
            "consent_ref": consent["consent_ref"],
            "retention_until": consent["retention_until"],
            "revocation_policy_ref": consent["revocation_policy_ref"],
            "profile_scope": _keyed_opaque(
                token_material,
                "profile-scope",
                _private_identifier(
                    consent["profile_scope_id"],
                    "private_intake.consent.profile_scope_id",
                ),
            ),
            "raw_query_stored": False,
            "raw_answer_stored": False,
            "raw_tool_output_stored": False,
        },
        "thresholds": value["thresholds"],
        "episodes": transformed_episodes,
    }
    return validate_dataset(minimized)


def load_private_intake(path: str | Path, *, token_material: bytes) -> dict[str, Any]:
    """Load a private intake and return only its minimized replay dataset."""

    return collect_private_intake(
        _load_document(path),
        token_material=token_material,
    )


def validate_dataset(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized, JSON-serializable replay dataset."""

    _mapping(raw, "dataset")
    _exact_keys(
        raw,
        {
            "schema_version",
            "lab_id",
            "study_mode",
            "created_at",
            "privacy",
            "thresholds",
            "episodes",
        },
        "dataset",
    )
    if raw["schema_version"] != DATASET_SCHEMA_VERSION:
        raise ValidationError(
            f"dataset.schema_version must equal {DATASET_SCHEMA_VERSION!r}"
        )
    lab_id = _identifier(raw["lab_id"], "dataset.lab_id")
    study_mode = _choice(raw["study_mode"], {"development", "pilot"}, "dataset.study_mode")
    created_at = _timestamp(raw["created_at"], "dataset.created_at")
    privacy = _validate_privacy(raw["privacy"], created_at=created_at)
    thresholds = _validate_thresholds(raw["thresholds"])
    episodes_raw = _list(raw["episodes"], "dataset.episodes")
    if not episodes_raw:
        raise ValidationError("dataset.episodes must be nonempty")
    if len(episodes_raw) > MAX_EPISODES:
        raise ValidationError(f"dataset.episodes exceeds {MAX_EPISODES} rows")
    episodes = [
        _validate_episode(row, where=f"dataset.episodes[{index}]", created_at=created_at)
        for index, row in enumerate(episodes_raw)
    ]
    _require_unique((row["episode_id"] for row in episodes), "episode_id")
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "lab_id": lab_id,
        "study_mode": study_mode,
        "created_at": created_at,
        "privacy": privacy,
        "thresholds": thresholds,
        "episodes": episodes,
    }


def audit_dataset_disjointness(
    candidate: Mapping[str, Any],
    references: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Check exact opaque identities and frozen-corpus hashes for overlap."""

    validated_candidate = validate_dataset(candidate)
    if not references:
        raise ValidationError("disjointness audit requires at least one reference dataset")
    validated_references = [validate_dataset(reference) for reference in references]

    def values(dataset: Mapping[str, Any]) -> dict[str, set[str]]:
        episodes = dataset["episodes"]
        return {
            "participant_refs": {row["participant_ref"] for row in episodes},
            "project_refs": {row["project_ref"] for row in episodes},
            "workstream_refs": {row["workstream_ref"] for row in episodes},
            "query_hashes": {row["query_hash"] for row in episodes},
            "corpus_digests": {
                row["snapshot"]["archive_digest"] for row in episodes
            },
        }

    candidate_values = values(validated_candidate)
    comparisons: list[dict[str, Any]] = []
    all_disjoint = True
    for reference in validated_references:
        overlaps: dict[str, dict[str, Any]] = {}
        reference_values = values(reference)
        for category, candidate_category in candidate_values.items():
            shared = sorted(candidate_category & reference_values[category])
            overlaps[category] = {
                "count": len(shared),
                "fingerprints": [
                    _fingerprint({"category": category, "value": value})
                    for value in shared
                ],
            }
        same_lab_id = validated_candidate["lab_id"] == reference["lab_id"]
        disjoint = not same_lab_id and all(row["count"] == 0 for row in overlaps.values())
        all_disjoint = all_disjoint and disjoint
        comparisons.append(
            {
                "reference_lab_id": reference["lab_id"],
                "reference_dataset_fingerprint": _fingerprint(reference),
                "same_lab_id": same_lab_id,
                "overlaps": overlaps,
                "disjoint": disjoint,
            }
        )
    return {
        "schema_version": DISJOINTNESS_SCHEMA_VERSION,
        "candidate_lab_id": validated_candidate["lab_id"],
        "candidate_dataset_fingerprint": _fingerprint(validated_candidate),
        "reference_count": len(comparisons),
        "comparisons": comparisons,
        "disjoint": all_disjoint,
        "limitations": (
            "Exact opaque references and corpus/query hashes are checked. Related people, "
            "project lineage, paraphrases, and shared upstream evidence require operator audit."
        ),
    }


def analyze_dataset(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Measure paired oracle gaps with participant-cluster uncertainty."""

    dataset = validate_dataset(raw)
    episodes = dataset["episodes"]
    thresholds = dataset["thresholds"]
    diagnostics: list[dict[str, Any]] = []
    for ablation, component in ABLATION_COMPONENTS.items():
        if ablation == "current":
            continue
        pairs: list[dict[str, Any]] = []
        for episode in episodes:
            by_ablation = {row["ablation"]: row for row in episode["variants"]}
            if ablation not in by_ablation:
                continue
            current = by_ablation["current"]["outcome"]
            variant = by_ablation[ablation]["outcome"]
            pairs.append(
                {
                    "episode_id": episode["episode_id"],
                    "participant_ref": episode["participant_ref"],
                    "project_ref": episode["project_ref"],
                    "query_class": episode["query_class"],
                    "current_success": bool(current["work_continuity_success"]),
                    "variant_success": bool(variant["work_continuity_success"]),
                    "delta": float(variant["work_continuity_success"])
                    - float(current["work_continuity_success"]),
                    "new_safety_failures": sorted(
                        set(variant["safety_failures"])
                        - set(current["safety_failures"])
                    ),
                    "latency_delta_ms": variant["latency_ms"] - current["latency_ms"],
                    "token_delta": variant["total_tokens"] - current["total_tokens"],
                    "rework_delta_seconds": variant["rework_seconds"]
                    - current["rework_seconds"],
                }
            )
        diagnostics.append(
            _ablation_diagnostic(
                ablation,
                component,
                pairs,
                thresholds=thresholds,
            )
        )

    bottleneck_candidates = [
        row
        for row in diagnostics
        if row["ablation"] not in {"no_memory"} and row["decisive_positive_gap"]
    ]
    bottleneck_candidates.sort(
        key=lambda row: (
            float(row["paired_success_delta"] or 0.0),
            float(row["rescue_rate"] or 0.0),
            row["component"],
        ),
        reverse=True,
    )
    oracle_diagnostics = [row for row in diagnostics if row["ablation"] != "no_memory"]
    oracle_panel_complete_and_comparable = bool(
        oracle_diagnostics
        and all(
            row["coverage_eligible"] and row["paired_episodes"] == len(episodes)
            for row in oracle_diagnostics
        )
        and len(
            {
                row["episode_set_fingerprint"]
                for row in oracle_diagnostics
                if row["episode_set_fingerprint"] is not None
            }
        )
        == 1
    )
    if bottleneck_candidates and oracle_panel_complete_and_comparable:
        top = bottleneck_candidates[0]
        diagnosis = {
            "status": "decisive_bottleneck",
            "component": top["component"],
            "ablation": top["ablation"],
            "recommended_next_action": COMPONENT_ACTIONS[top["component"]],
            "basis": (
                "The participant-cluster confidence interval is above zero, coverage floors "
                "pass, and the oracle variant introduces no observed safety regression."
            ),
        }
    elif bottleneck_candidates:
        top = bottleneck_candidates[0]
        diagnosis = {
            "status": "partial_bottleneck_signal",
            "component": top["component"],
            "ablation": top["ablation"],
            "recommended_next_action": (
                "Complete a common, coverage-eligible oracle panel before selecting the "
                "largest component for optimization."
            ),
            "basis": (
                "At least one component has a positive safe gap, but missing or different "
                "episode panels make cross-component ranking invalid."
            ),
        }
    else:
        diagnosis = {
            "status": "insufficient_evidence",
            "component": None,
            "ablation": None,
            "recommended_next_action": (
                "Collect more disjoint opt-in shadow episodes or add missing single-change "
                "oracle variants before changing a production component."
            ),
            "basis": (
                "No oracle component has a positive participant-cluster lower confidence "
                "bound while meeting coverage and safety requirements."
            ),
        }

    current_outcomes = [
        next(row for row in episode["variants"] if row["ablation"] == "current")["outcome"]
        for episode in episodes
    ]
    missing_by_ablation = {
        ablation: sum(
            1
            for episode in episodes
            if ablation not in {row["ablation"] for row in episode["variants"]}
        )
        for ablation in ABLATION_COMPONENTS
        if ablation != "current"
    }
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "lab_id": dataset["lab_id"],
        "study_mode": dataset["study_mode"],
        "dataset_fingerprint": _fingerprint(dataset),
        "diagnostic_only": True,
        "eligible_for_human_superiority_claim": False,
        "mutation_authority": "none",
        "episode_count": len(episodes),
        "participant_clusters": len({row["participant_ref"] for row in episodes}),
        "project_clusters": len({row["project_ref"] for row in episodes}),
        "current_success_rate": statistics.mean(
            float(row["work_continuity_success"]) for row in current_outcomes
        ),
        "current_safety_failure_count": sum(
            len(row["safety_failures"]) for row in current_outcomes
        ),
        "missing_episode_counts_by_ablation": missing_by_ablation,
        "oracle_panel_complete_and_comparable": oracle_panel_complete_and_comparable,
        "ablation_diagnostics": diagnostics,
        "diagnosis": diagnosis,
        "limitations": (
            "Oracle labels are causal only when the replay operator truly changed one component. "
            "The schema records that assertion but cannot independently prove it."
        ),
    }


def _ablation_diagnostic(
    ablation: str,
    component: str,
    pairs: Sequence[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    paired_count = len(pairs)
    clusters = len({row["participant_ref"] for row in pairs})
    eligible = (
        paired_count >= thresholds["minimum_paired_episodes"]
        and clusters >= thresholds["minimum_participant_clusters"]
    )
    if not pairs:
        return {
            "ablation": ablation,
            "component": component,
            "paired_episodes": 0,
            "participant_clusters": 0,
            "project_clusters": 0,
            "episode_set_fingerprint": None,
            "current_success_rate": None,
            "variant_success_rate": None,
            "paired_success_delta": None,
            "confidence_interval": None,
            "rescue_count": 0,
            "rescue_rate": None,
            "regression_count": 0,
            "regression_rate": None,
            "safety_regression_count": 0,
            "mean_latency_delta_ms": None,
            "mean_token_delta": None,
            "mean_rework_delta_seconds": None,
            "coverage_eligible": False,
            "decisive_positive_gap": False,
        }

    current_failures = [row for row in pairs if not row["current_success"]]
    current_successes = [row for row in pairs if row["current_success"]]
    rescue_count = sum(
        1 for row in current_failures if row["variant_success"]
    )
    regression_count = sum(
        1 for row in current_successes if not row["variant_success"]
    )
    safety_regressions = sum(
        1
        for row in pairs
        if row["new_safety_failures"]
    )
    point = statistics.mean(row["delta"] for row in pairs)
    confidence_interval = _cluster_bootstrap_interval(
        pairs,
        samples=thresholds["bootstrap_samples"],
        seed=_derived_seed(thresholds["bootstrap_seed"], ablation),
        confidence_level=thresholds["confidence_level"],
    )
    decisive = bool(
        eligible
        and safety_regressions == 0
        and confidence_interval["lower"] > 0.0
    )
    return {
        "ablation": ablation,
        "component": component,
        "paired_episodes": paired_count,
        "participant_clusters": clusters,
        "project_clusters": len({row["project_ref"] for row in pairs}),
        "episode_set_fingerprint": _fingerprint(
            sorted(row["episode_id"] for row in pairs)
        ),
        "current_success_rate": statistics.mean(float(row["current_success"]) for row in pairs),
        "variant_success_rate": statistics.mean(float(row["variant_success"]) for row in pairs),
        "paired_success_delta": point,
        "confidence_interval": confidence_interval,
        "rescue_count": rescue_count,
        "rescue_rate": rescue_count / len(current_failures) if current_failures else None,
        "regression_count": regression_count,
        "regression_rate": regression_count / len(current_successes) if current_successes else None,
        "safety_regression_count": safety_regressions,
        "mean_latency_delta_ms": statistics.mean(row["latency_delta_ms"] for row in pairs),
        "mean_token_delta": statistics.mean(row["token_delta"] for row in pairs),
        "mean_rework_delta_seconds": statistics.mean(
            row["rework_delta_seconds"] for row in pairs
        ),
        "coverage_eligible": eligible,
        "decisive_positive_gap": decisive,
    }


def _cluster_bootstrap_interval(
    pairs: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float]:
    by_participant: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        by_participant[row["participant_ref"]].append(row)
    participant_refs = sorted(by_participant)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled_rows: list[Mapping[str, Any]] = []
        for _cluster in participant_refs:
            sampled_ref = rng.choice(participant_refs)
            sampled_rows.extend(by_participant[sampled_ref])
        estimates.append(statistics.mean(row["delta"] for row in sampled_rows))
    estimates.sort()
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "confidence_level": confidence_level,
        "lower": _percentile(estimates, alpha),
        "upper": _percentile(estimates, 1.0 - alpha),
        "method": "participant_cluster_percentile_bootstrap",
        "samples": samples,
    }


def _validate_privacy(raw: Any, *, created_at: str) -> dict[str, Any]:
    where = "dataset.privacy"
    value = _mapping(raw, where)
    _exact_keys(
        value,
        {
            "collection_mode",
            "consent_ref",
            "retention_until",
            "revocation_policy_ref",
            "profile_scope",
            "raw_query_stored",
            "raw_answer_stored",
            "raw_tool_output_stored",
        },
        where,
    )
    if value["collection_mode"] != "opt_in_shadow":
        raise ValidationError(f"{where}.collection_mode must equal 'opt_in_shadow'")
    retention_until = _timestamp(value["retention_until"], f"{where}.retention_until")
    if _time_value(retention_until) < _time_value(created_at):
        raise ValidationError(f"{where}.retention_until cannot precede dataset.created_at")
    for field_name in ("raw_query_stored", "raw_answer_stored", "raw_tool_output_stored"):
        if value[field_name] is not False:
            raise ValidationError(f"{where}.{field_name} must be false")
    return {
        "collection_mode": "opt_in_shadow",
        "consent_ref": _ref(value["consent_ref"], f"{where}.consent_ref"),
        "retention_until": retention_until,
        "revocation_policy_ref": _ref(
            value["revocation_policy_ref"], f"{where}.revocation_policy_ref"
        ),
        "profile_scope": _opaque(value["profile_scope"], f"{where}.profile_scope"),
        "raw_query_stored": False,
        "raw_answer_stored": False,
        "raw_tool_output_stored": False,
    }


def _validate_thresholds(raw: Any) -> dict[str, Any]:
    where = "dataset.thresholds"
    value = _mapping(raw, where)
    _exact_keys(
        value,
        {
            "bootstrap_samples",
            "bootstrap_seed",
            "confidence_level",
            "minimum_paired_episodes",
            "minimum_participant_clusters",
        },
        where,
    )
    return {
        "bootstrap_samples": _integer(
            value["bootstrap_samples"], f"{where}.bootstrap_samples", minimum=100, maximum=100_000
        ),
        "bootstrap_seed": _integer(
            value["bootstrap_seed"], f"{where}.bootstrap_seed", minimum=0
        ),
        "confidence_level": _number(
            value["confidence_level"], f"{where}.confidence_level", minimum=0.8, maximum=0.999
        ),
        "minimum_paired_episodes": _integer(
            value["minimum_paired_episodes"], f"{where}.minimum_paired_episodes", minimum=2
        ),
        "minimum_participant_clusters": _integer(
            value["minimum_participant_clusters"],
            f"{where}.minimum_participant_clusters",
            minimum=2,
        ),
    }


def _validate_episode(raw: Any, *, where: str, created_at: str) -> dict[str, Any]:
    value = _mapping(raw, where)
    _exact_keys(
        value,
        {
            "episode_id",
            "participant_ref",
            "project_ref",
            "workstream_ref",
            "query_ref",
            "query_hash",
            "query_class",
            "checkpoint_days",
            "evidence_cutoff",
            "snapshot",
            "variants",
        },
        where,
    )
    evidence_cutoff = _timestamp(value["evidence_cutoff"], f"{where}.evidence_cutoff")
    if _time_value(evidence_cutoff) > _time_value(created_at):
        raise ValidationError(f"{where}.evidence_cutoff cannot follow dataset.created_at")
    variants_raw = _list(value["variants"], f"{where}.variants")
    variants = [
        _validate_variant(
            row,
            where=f"{where}.variants[{index}]",
            evidence_cutoff=evidence_cutoff,
            created_at=created_at,
        )
        for index, row in enumerate(variants_raw)
    ]
    ablations = [row["ablation"] for row in variants]
    _require_unique(ablations, f"{where}.variants.ablation")
    _require_unique(
        (row["replay_artifact_digest"] for row in variants),
        f"{where}.variants.replay_artifact_digest",
    )
    if "current" not in ablations:
        raise ValidationError(f"{where}.variants must include current")
    if len(variants) < 2:
        raise ValidationError(f"{where}.variants must include current and at least one ablation")
    return {
        "episode_id": _opaque(value["episode_id"], f"{where}.episode_id"),
        "participant_ref": _opaque(value["participant_ref"], f"{where}.participant_ref"),
        "project_ref": _opaque(value["project_ref"], f"{where}.project_ref"),
        "workstream_ref": _opaque(value["workstream_ref"], f"{where}.workstream_ref"),
        "query_ref": _opaque(value["query_ref"], f"{where}.query_ref"),
        "query_hash": _digest(value["query_hash"], f"{where}.query_hash"),
        "query_class": _choice(value["query_class"], set(QUERY_CLASSES), f"{where}.query_class"),
        "checkpoint_days": _integer(
            value["checkpoint_days"], f"{where}.checkpoint_days", minimum=0, maximum=10_000
        ),
        "evidence_cutoff": evidence_cutoff,
        "snapshot": _validate_snapshot(value["snapshot"], where=f"{where}.snapshot"),
        "variants": variants,
    }


def _validate_snapshot(raw: Any, *, where: str) -> dict[str, Any]:
    value = _mapping(raw, where)
    _exact_keys(
        value,
        {
            "corpus_ref",
            "archive_digest",
            "index_digest",
            "config_digest",
            "code_digest",
            "answerer_digest",
        },
        where,
    )
    return {
        "corpus_ref": _opaque(value["corpus_ref"], f"{where}.corpus_ref"),
        **{
            field_name: _digest(value[field_name], f"{where}.{field_name}")
            for field_name in (
                "archive_digest",
                "index_digest",
                "config_digest",
                "code_digest",
                "answerer_digest",
            )
        },
    }


def _validate_variant(
    raw: Any,
    *,
    where: str,
    evidence_cutoff: str,
    created_at: str,
) -> dict[str, Any]:
    value = _mapping(raw, where)
    _exact_keys(
        value,
        {
            "ablation",
            "changed_component",
            "replay_artifact_digest",
            "offline_side_effects_disabled",
            "trace",
            "outcome",
        },
        where,
    )
    ablation = _choice(value["ablation"], set(ABLATION_COMPONENTS), f"{where}.ablation")
    expected_component = ABLATION_COMPONENTS[ablation]
    if value["changed_component"] != expected_component:
        raise ValidationError(
            f"{where}.changed_component must equal {expected_component!r} for {ablation}"
        )
    if value["offline_side_effects_disabled"] is not True:
        raise ValidationError(f"{where}.offline_side_effects_disabled must be true")
    trace = _validate_trace(
        value["trace"],
        where=f"{where}.trace",
        evidence_cutoff=evidence_cutoff,
    )
    outcome = _validate_outcome(
        value["outcome"],
        where=f"{where}.outcome",
        evidence_cutoff=evidence_cutoff,
        created_at=created_at,
    )
    if outcome["total_tokens"] < trace["packet_tokens"]:
        raise ValidationError(f"{where}.outcome.total_tokens cannot be below packet_tokens")
    if not set(trace["cited_ref_fingerprints"]).issubset(trace["ranked_ref_fingerprints"]):
        raise ValidationError(
            f"{where}.trace.cited_ref_fingerprints must be a subset of ranked references"
        )
    return {
        "ablation": ablation,
        "changed_component": expected_component,
        "replay_artifact_digest": _digest(
            value["replay_artifact_digest"], f"{where}.replay_artifact_digest"
        ),
        "offline_side_effects_disabled": True,
        "trace": trace,
        "outcome": outcome,
    }


def _validate_trace(raw: Any, *, where: str, evidence_cutoff: str) -> dict[str, Any]:
    value = _mapping(raw, where)
    _exact_keys(
        value,
        {
            "route",
            "candidate_set_digest",
            "ranked_ref_fingerprints",
            "evidence_timestamps",
            "packet_digest",
            "packet_tokens",
            "retrieved_count",
            "cited_ref_fingerprints",
            "safety_filters_digest",
        },
        where,
    )
    ranked = _digest_list(value["ranked_ref_fingerprints"], f"{where}.ranked_ref_fingerprints")
    retrieved_count = _integer(
        value["retrieved_count"], f"{where}.retrieved_count", minimum=0, maximum=10_000
    )
    if len(ranked) != retrieved_count:
        raise ValidationError(f"{where}.retrieved_count must equal ranked_ref_fingerprints length")
    evidence_timestamps = [
        _timestamp(item, f"{where}.evidence_timestamps[{index}]")
        for index, item in enumerate(_list(value["evidence_timestamps"], f"{where}.evidence_timestamps"))
    ]
    if len(evidence_timestamps) != len(ranked):
        raise ValidationError(f"{where}.evidence_timestamps must align with ranked references")
    if any(
        _time_value(observed_at) > _time_value(evidence_cutoff)
        for observed_at in evidence_timestamps
    ):
        raise ValidationError(f"{where}.evidence_timestamps cannot follow evidence_cutoff")
    return {
        "route": _ref(value["route"], f"{where}.route"),
        "candidate_set_digest": _digest(
            value["candidate_set_digest"], f"{where}.candidate_set_digest"
        ),
        "ranked_ref_fingerprints": ranked,
        "evidence_timestamps": evidence_timestamps,
        "packet_digest": _digest(value["packet_digest"], f"{where}.packet_digest"),
        "packet_tokens": _integer(
            value["packet_tokens"], f"{where}.packet_tokens", minimum=0, maximum=1_000_000
        ),
        "retrieved_count": retrieved_count,
        "cited_ref_fingerprints": _digest_list(
            value["cited_ref_fingerprints"], f"{where}.cited_ref_fingerprints"
        ),
        "safety_filters_digest": _digest(
            value["safety_filters_digest"], f"{where}.safety_filters_digest"
        ),
    }


def _validate_outcome(
    raw: Any,
    *,
    where: str,
    evidence_cutoff: str,
    created_at: str,
) -> dict[str, Any]:
    value = _mapping(raw, where)
    _exact_keys(
        value,
        {
            "work_continuity_success",
            "label_source",
            "judgment_ref",
            "verified_at",
            "corrections_count",
            "rework_seconds",
            "latency_ms",
            "total_tokens",
            "safety_failures",
            "citation_correct",
            "stale_conflict_error",
        },
        where,
    )
    success = _boolean(value["work_continuity_success"], f"{where}.work_continuity_success")
    verified_at = _timestamp(value["verified_at"], f"{where}.verified_at")
    if _time_value(verified_at) < _time_value(evidence_cutoff):
        raise ValidationError(f"{where}.verified_at cannot precede evidence_cutoff")
    if _time_value(verified_at) > _time_value(created_at):
        raise ValidationError(f"{where}.verified_at cannot follow dataset.created_at")
    failures = _string_list(value["safety_failures"], f"{where}.safety_failures")
    stale_error = _boolean(value["stale_conflict_error"], f"{where}.stale_conflict_error")
    citation_raw = value["citation_correct"]
    if citation_raw is not None and not isinstance(citation_raw, bool):
        raise ValidationError(f"{where}.citation_correct must be boolean or null")
    if success and (failures or stale_error or citation_raw is not True):
        raise ValidationError(
            f"{where} cannot be successful without verified citation correctness or with "
            "a safety or stale-conflict failure"
        )
    return {
        "work_continuity_success": success,
        "label_source": _choice(value["label_source"], LABEL_SOURCES, f"{where}.label_source"),
        "judgment_ref": _ref(value["judgment_ref"], f"{where}.judgment_ref"),
        "verified_at": verified_at,
        "corrections_count": _integer(
            value["corrections_count"], f"{where}.corrections_count", minimum=0
        ),
        "rework_seconds": _integer(value["rework_seconds"], f"{where}.rework_seconds", minimum=0),
        "latency_ms": _integer(value["latency_ms"], f"{where}.latency_ms", minimum=0),
        "total_tokens": _integer(value["total_tokens"], f"{where}.total_tokens", minimum=0),
        "safety_failures": failures,
        "citation_correct": citation_raw,
        "stale_conflict_error": stale_error,
    }


def _load_document(path: str | Path) -> Any:
    input_path = Path(path)
    if input_path.suffix.lower() not in {".json", ".yaml", ".yml"}:
        raise ValidationError("input file must use .json, .yaml, or .yml")
    try:
        if input_path.stat().st_size > MAX_INPUT_BYTES:
            raise ValidationError(f"input file exceeds {MAX_INPUT_BYTES} bytes")
        text = input_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"cannot read UTF-8 input file: {exc}") from exc
    try:
        if input_path.suffix.lower() == ".json":
            return json.loads(
                text,
                object_pairs_hook=_json_object,
                parse_constant=_reject_json_constant,
            )
        return _safe_yaml_load(text)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"invalid structured input: {exc}") from exc


def _safe_yaml_load(text: str) -> Any:
    import yaml

    class StrictSafeLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        loader.flatten_mapping(node)
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValidationError("YAML mapping keys must be strings")
            if key in result:
                raise ValidationError(f"duplicate YAML key: {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    StrictSafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    return yaml.load(text, Loader=StrictSafeLoader)


def _json_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON constant is not allowed: {value}")


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{where} must be an object")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{where} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValidationError(f"{where} keys mismatch; missing={missing}, extra={extra}")


def _identifier(value: Any, where: str) -> str:
    text = _ref(value, where)
    if not all(char in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in text):
        raise ValidationError(f"{where} must use lowercase letters, digits, hyphens, or underscores")
    return text


def _ref(value: Any, where: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 1024 or any(ord(char) < 32 for char in text):
        raise ValidationError(f"{where} must be a bounded nonblank reference")
    return text


def _opaque(value: Any, where: str) -> str:
    text = _ref(value, where)
    if not text.startswith("opaque:"):
        raise ValidationError(f"{where} must be an opaque: reference")
    suffix = text[7:]
    if not 16 <= len(suffix) <= 64 or any(char not in _HEX for char in suffix):
        raise ValidationError(f"{where} must contain 16-64 lowercase hexadecimal characters")
    return text


def _private_identifier(value: Any, where: str) -> str:
    text = _normalized_private_text(value, where, maximum=1024, allow_newlines=False)
    return text


def _private_query(value: Any, where: str) -> str:
    return _normalized_private_text(value, where, maximum=65_536, allow_newlines=True)


def _normalized_private_text(
    value: Any,
    where: str,
    *,
    maximum: int,
    allow_newlines: bool,
) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{where} must be text")
    text = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text or len(text) > maximum:
        raise ValidationError(f"{where} must contain 1-{maximum} characters")
    if any(
        ord(char) < 32 and not (allow_newlines and char in {"\n", "\t"})
        for char in text
    ):
        raise ValidationError(f"{where} contains unsupported control characters")
    return text


def _keyed_digest(token_material: bytes, namespace: str, value: str) -> str:
    message = (
        "memory-v2-outcome-replay-private-intake/v1\0"
        f"{namespace}\0{value}"
    ).encode("utf-8")
    return "sha256:" + hmac.new(token_material, message, hashlib.sha256).hexdigest()


def _keyed_opaque(token_material: bytes, namespace: str, value: str) -> str:
    return "opaque:" + _keyed_digest(token_material, namespace, value)[7:]


def _digest(value: Any, where: str) -> str:
    text = str(value or "").strip().lower()
    if not text.startswith("sha256:") or len(text) != 71 or any(
        char not in _HEX for char in text[7:]
    ):
        raise ValidationError(f"{where} must be a sha256: digest")
    return text


def _digest_list(value: Any, where: str) -> list[str]:
    items = _list(value, where)
    normalized = [_digest(item, f"{where}[{index}]") for index, item in enumerate(items)]
    _require_unique(normalized, where)
    return normalized


def _string_list(value: Any, where: str) -> list[str]:
    items = _list(value, where)
    normalized = [_ref(item, f"{where}[{index}]") for index, item in enumerate(items)]
    _require_unique(normalized, where)
    return normalized


def _choice(value: Any, allowed: set[str], where: str) -> str:
    text = str(value or "").strip()
    if text not in allowed:
        raise ValidationError(f"{where} must be one of: {sorted(allowed)}")
    return text


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{where} must be boolean")
    return value


def _integer(
    value: Any,
    where: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{where} must be an integer")
    if minimum is not None and value < minimum:
        raise ValidationError(f"{where} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValidationError(f"{where} must be at most {maximum}")
    return value


def _number(
    value: Any,
    where: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValidationError(f"{where} must be between {minimum} and {maximum}")
    return result


def _timestamp(value: Any, where: str) -> str:
    text = _ref(value, where)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{where} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{where} must include a timezone offset")
    return text


def _time_value(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _require_unique(values: Iterable[str], where: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValidationError(f"{where} values must be unique")
        seen.add(value)


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _derived_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}|outcome-replay|{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


__all__ = [
    "ABLATION_COMPONENTS",
    "DATASET_SCHEMA_VERSION",
    "DISJOINTNESS_SCHEMA_VERSION",
    "PRIVATE_INTAKE_SCHEMA_VERSION",
    "QUERY_CLASSES",
    "RESULT_SCHEMA_VERSION",
    "ValidationError",
    "analyze_dataset",
    "audit_dataset_disjointness",
    "collect_private_intake",
    "load_dataset",
    "load_private_intake",
    "validate_dataset",
]
