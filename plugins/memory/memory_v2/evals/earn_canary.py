"""Blinded four-arm longitudinal canary study for Memory v2.

The study is deliberately offline and diagnostic.  It prepares condition-
blinded judge packets, keeps assignments in a separate sealed key, scores
paired outcomes with participant/project clustered uncertainty, and evaluates
conjunctive safety and quality gates.  It cannot mutate memory or authorize a
human-superiority claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import stat
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..shadow_reranker import ShadowRerankerError, compute_shadow_metrics


PROTOCOL_SCHEMA_VERSION = "memory-v2-earn-canary-protocol/v1"
RESPONSE_SCHEMA_VERSION = "memory-v2-earn-canary-responses/v1"
PUBLIC_PACKETS_SCHEMA_VERSION = "memory-v2-earn-canary-public-packets/v1"
PRIVATE_KEY_SCHEMA_VERSION = "memory-v2-earn-canary-private-key/v1"
JUDGMENTS_SCHEMA_VERSION = "memory-v2-earn-canary-judgments/v1"
SHADOW_METRICS_SCHEMA_VERSION = "memory-v2-earn-canary-shadow-metrics/v1"
RESULT_SCHEMA_VERSION = "memory-v2-earn-canary-result/v1"
DISJOINTNESS_SCHEMA_VERSION = "memory-v2-earn-canary-disjointness/v1"

CONDITIONS = ("no_memory", "raw_fts", "memory_v2", "oracle_evidence")
SLOTS = ("A", "B", "C", "D")
DIMENSIONS = (
    "factual_correctness",
    "temporal_correctness",
    "completeness",
    "source_grounding",
    "actionability",
    "calibration_restraint",
)
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
PRIMARY_COMPARISONS = (
    "memory_v2_vs_no_memory",
    "memory_v2_vs_raw_fts",
    "raw_fts_vs_no_memory",
    "oracle_evidence_vs_memory_v2",
)
GATE_FAILURES = (
    "irrelevant_injection",
    "stale_as_current",
    "invalid_citation",
    "unattested_citation",
    "security_boundary",
    "privacy_leak",
    "post_cutoff_evidence",
    "artifact_integrity",
)
FAILURE_TAXONOMY = (
    "capture_extraction",
    "candidate_recall",
    "routing",
    "ranking",
    "temporal_resolution",
    "packet_composition",
    "multi_event_synthesis",
    "answer_synthesis",
    "unnecessary_memory",
    "oracle_ceiling_failure",
    "inconclusive",
)
LABEL_SOURCES = ("independent_judge", "trusted_operator", "deterministic_harness")
MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_EPISODES = 10_000
MAX_DOCUMENT_NODES = 250_000
MAX_CLOCK_SKEW_SECONDS = 300
_HEX = frozenset("0123456789abcdef")


class ValidationError(ValueError):
    """Raised when an Earn-the-Canary artifact violates its strict contract."""


def load_protocol(path: str | Path) -> dict[str, Any]:
    return validate_protocol(_load_document(path))


def load_responses(path: str | Path) -> dict[str, Any]:
    return validate_responses(_load_document(path))


def load_judgments(path: str | Path) -> dict[str, Any]:
    return validate_judgments(_load_document(path))


def validate_protocol(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = _mapping(raw, "protocol")
    _exact(
        value,
        {
            "schema_version",
            "study_id",
            "study_mode",
            "created_at",
            "privacy",
            "design",
            "thresholds",
            "episodes",
        },
        "protocol",
    )
    if value["schema_version"] != PROTOCOL_SCHEMA_VERSION:
        raise ValidationError("protocol schema version is invalid")
    study_id = _identifier(value["study_id"], "protocol.study_id")
    study_mode = _choice(
        value["study_mode"], {"development", "pilot"}, "protocol.study_mode"
    )
    created_at = _timestamp(value["created_at"], "protocol.created_at")
    privacy = _validate_privacy(value["privacy"], created_at=created_at)
    design = _validate_design(value["design"], study_mode=study_mode)
    thresholds = _validate_thresholds(value["thresholds"], study_mode=study_mode)
    raw_episodes = _list(value["episodes"], "protocol.episodes")
    if not raw_episodes or len(raw_episodes) > MAX_EPISODES:
        raise ValidationError("protocol episodes must be nonempty and bounded")
    episodes = [
        _validate_episode(row, f"protocol.episodes[{index}]", created_at=created_at)
        for index, row in enumerate(raw_episodes)
    ]
    _unique((row["episode_id"] for row in episodes), "episode_id")
    _unique((row["query_hash"] for row in episodes), "query_hash")
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "study_id": study_id,
        "study_mode": study_mode,
        "created_at": created_at,
        "privacy": privacy,
        "design": design,
        "thresholds": thresholds,
        "episodes": episodes,
    }


def validate_responses(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = _mapping(raw, "responses")
    _exact(
        value,
        {"schema_version", "study_id", "protocol_fingerprint", "responses"},
        "responses",
    )
    if value["schema_version"] != RESPONSE_SCHEMA_VERSION:
        raise ValidationError("response schema version is invalid")
    rows_raw = _list(value["responses"], "responses.responses")
    if not rows_raw or len(rows_raw) > MAX_EPISODES * len(CONDITIONS):
        raise ValidationError("response rows must be nonempty and bounded")
    rows = [
        _validate_response(row, f"responses.responses[{index}]")
        for index, row in enumerate(rows_raw)
    ]
    _unique((row["response_id"] for row in rows), "response_id")
    _unique(
        (f"{row['episode_id']}\0{row['condition']}" for row in rows),
        "episode-condition response",
    )
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "study_id": _identifier(value["study_id"], "responses.study_id"),
        "protocol_fingerprint": _digest(
            value["protocol_fingerprint"], "responses.protocol_fingerprint"
        ),
        "responses": rows,
    }


def validate_judgments(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = _mapping(raw, "judgments")
    _exact(
        value,
        {
            "schema_version",
            "study_id",
            "packet_set_fingerprint",
            "judgments",
        },
        "judgments",
    )
    if value["schema_version"] != JUDGMENTS_SCHEMA_VERSION:
        raise ValidationError("judgment schema version is invalid")
    rows_raw = _list(value["judgments"], "judgments.judgments")
    if not rows_raw or len(rows_raw) > MAX_EPISODES * 3:
        raise ValidationError("judgment rows must be nonempty and bounded")
    rows = [
        _validate_judgment(row, f"judgments.judgments[{index}]")
        for index, row in enumerate(rows_raw)
    ]
    _unique(
        (f"{row['packet_id']}\0{row['judge_id']}" for row in rows),
        "packet-judge judgment",
    )
    return {
        "schema_version": JUDGMENTS_SCHEMA_VERSION,
        "study_id": _identifier(value["study_id"], "judgments.study_id"),
        "packet_set_fingerprint": _digest(
            value["packet_set_fingerprint"], "judgments.packet_set_fingerprint"
        ),
        "judgments": rows,
    }


def prepare_blinded_packets(
    protocol: Mapping[str, Any],
    responses: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Create balanced A-D packets and a separate sealed assignment key."""

    study = validate_protocol(protocol)
    response_set = validate_responses(responses)
    seed = _integer(
        seed,
        "blinding seed",
        minimum=-(2**63),
        maximum=2**63 - 1,
    )
    if _fingerprint({"seed": seed}) != study["design"]["blinding_seed_commitment"]:
        raise ValidationError("blinding seed does not match protocol commitment")
    protocol_fingerprint = _fingerprint(study)
    if (
        response_set["study_id"] != study["study_id"]
        or response_set["protocol_fingerprint"] != protocol_fingerprint
    ):
        raise ValidationError("responses do not belong to the frozen protocol")

    episodes = {row["episode_id"]: row for row in study["episodes"]}
    by_episode: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in response_set["responses"]:
        episode = episodes.get(row["episode_id"])
        if episode is None:
            raise ValidationError("response refers to an unknown episode")
        if row["answerer_digest"] != episode["snapshot"]["answerer_digest"]:
            raise ValidationError("response answerer digest differs from protocol")
        if row["config_digest"] != episode["snapshot"]["config_digest"]:
            raise ValidationError("response config digest differs from protocol")
        if row["code_digest"] != episode["snapshot"]["code_digest"]:
            raise ValidationError("response code digest differs from protocol")
        if (
            row["input_packet_digest"]
            != episode["condition_input_digests"][row["condition"]]
        ):
            raise ValidationError("response input digest differs from protocol")
        if _time(row["evidence_max_at"]) > _time(episode["evidence_cutoff"]):
            raise ValidationError("response uses evidence after the frozen cutoff")
        if _time(row["generated_at"]) < _time(study["created_at"]):
            raise ValidationError("response predates protocol preregistration")
        by_episode[row["episode_id"]][row["condition"]] = row
    if set(by_episode) != set(episodes):
        raise ValidationError("every protocol episode requires responses")
    for episode_id, conditions in by_episode.items():
        if set(conditions) != set(CONDITIONS):
            raise ValidationError(
                "every episode requires exactly four condition responses"
            )

    ordered_episodes = sorted(study["episodes"], key=lambda row: row["episode_id"])
    condition_orders = _condition_orders(ordered_episodes, seed)
    public_packets: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for episode in ordered_episodes:
        condition_order = condition_orders[episode["episode_id"]]
        packet_id = (
            "packet:"
            + _fingerprint({
                "study_id": study["study_id"],
                "episode_id": episode["episode_id"],
            }).removeprefix("sha256:")[:24]
        )
        public_responses: dict[str, dict[str, str]] = {}
        assignments: dict[str, dict[str, Any]] = {}
        for slot, condition in zip(SLOTS, condition_order, strict=True):
            row = by_episode[episode["episode_id"]][condition]
            blinded_response_id = (
                "response:"
                + _fingerprint({
                    "study_id": study["study_id"],
                    "packet_id": packet_id,
                    "slot": slot,
                }).removeprefix("sha256:")[:24]
            )
            public_responses[slot] = {
                "response_id": blinded_response_id,
                "text": row["text"],
            }
            assignments[slot] = {
                "condition": condition,
                "response_id": blinded_response_id,
                "response_artifact_digest": row["response_artifact_digest"],
                "trace_digest": row["trace_digest"],
                "latency_ms": row["latency_ms"],
                "total_tokens": row["total_tokens"],
            }
        public_packets.append({
            "packet_id": packet_id,
            "prompt": episode["prompt"],
            "evidence_pack_ref": episode["evidence_pack_ref"],
            "query_class": episode["query_class"],
            "checkpoint_days": episode["checkpoint_days"],
            "dimensions": list(DIMENSIONS),
            "responses": public_responses,
        })
        private_rows.append({
            "packet_id": packet_id,
            "episode_id": episode["episode_id"],
            "participant_ref": episode["participant_ref"],
            "project_ref": episode["project_ref"],
            "workstream_ref": episode["workstream_ref"],
            "query_ref": episode["query_ref"],
            "slots": assignments,
        })
    random.Random(_derived_seed(seed, "public-packet-order")).shuffle(public_packets)
    response_set_fingerprint = _fingerprint(response_set)
    packet_set_fingerprint = _fingerprint({
        "protocol_fingerprint": protocol_fingerprint,
        "response_set_fingerprint": response_set_fingerprint,
        "judge_packets": public_packets,
    })
    return {
        "public": {
            "schema_version": PUBLIC_PACKETS_SCHEMA_VERSION,
            "study_id": study["study_id"],
            "protocol_fingerprint": protocol_fingerprint,
            "response_set_fingerprint": response_set_fingerprint,
            "packet_set_fingerprint": packet_set_fingerprint,
            "judge_packets": public_packets,
        },
        "private": {
            "schema_version": PRIVATE_KEY_SCHEMA_VERSION,
            "study_id": study["study_id"],
            "seed": seed,
            "protocol_fingerprint": protocol_fingerprint,
            "response_set_fingerprint": response_set_fingerprint,
            "packet_set_fingerprint": packet_set_fingerprint,
            "assignment_key": private_rows,
        },
    }


def _condition_orders(
    episodes: Sequence[Mapping[str, Any]], seed: int
) -> dict[str, list[str]]:
    """Return independently randomized balanced Latin-square blocks."""

    rng = random.Random(_derived_seed(seed, "condition-orders"))
    result: dict[str, list[str]] = {}
    for start in range(0, len(episodes), len(CONDITIONS)):
        block = list(episodes[start : start + len(CONDITIONS)])
        base = list(CONDITIONS)
        rng.shuffle(base)
        rotations = [base[offset:] + base[:offset] for offset in range(len(CONDITIONS))]
        rng.shuffle(rotations)
        for episode, order in zip(block, rotations, strict=False):
            result[str(episode["episode_id"])] = order
    return result


def audit_study_disjointness(
    candidate: Mapping[str, Any],
    references: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit exact protected identities without returning the identities."""

    study = validate_protocol(candidate)
    if not references:
        raise ValidationError("disjointness audit requires reference protocols")
    reference_studies = [validate_protocol(row) for row in references]

    def values(protocol: Mapping[str, Any]) -> dict[str, set[str]]:
        episodes = protocol["episodes"]
        return {
            "participants": {row["participant_ref"] for row in episodes},
            "projects": {row["project_ref"] for row in episodes},
            "workstreams": {row["workstream_ref"] for row in episodes},
            "queries": {row["query_hash"] for row in episodes},
            "corpora": {row["snapshot"]["corpus_ref"] for row in episodes},
            "archives": {row["snapshot"]["archive_digest"] for row in episodes},
        }

    candidate_values = values(study)
    comparisons: list[dict[str, Any]] = []
    all_disjoint = True
    for reference in reference_studies:
        overlap: dict[str, dict[str, Any]] = {}
        reference_values = values(reference)
        for category, candidates in candidate_values.items():
            shared = sorted(candidates & reference_values[category])
            overlap[category] = {
                "count": len(shared),
                "fingerprints": [
                    _fingerprint({"category": category, "value": item})
                    for item in shared
                ],
            }
        same_study = study["study_id"] == reference["study_id"]
        namespace_match = (
            study["privacy"]["identity_namespace_digest"]
            == reference["privacy"]["identity_namespace_digest"]
        )
        disjoint = (
            namespace_match
            and not same_study
            and all(item["count"] == 0 for item in overlap.values())
        )
        all_disjoint = all_disjoint and disjoint
        comparisons.append({
            "reference_study_id": reference["study_id"],
            "reference_protocol_fingerprint": _fingerprint(reference),
            "same_study_id": same_study,
            "identity_namespace_match": namespace_match,
            "overlaps": overlap,
            "disjoint": disjoint,
        })
    return {
        "schema_version": DISJOINTNESS_SCHEMA_VERSION,
        "candidate_study_id": study["study_id"],
        "candidate_protocol_fingerprint": _fingerprint(study),
        "reference_count": len(comparisons),
        "comparisons": comparisons,
        "disjoint": all_disjoint,
        "limitations": (
            "Exact protected identities are checked. Related people, project lineage, "
            "paraphrases, and shared upstream evidence require operator audit."
        ),
    }


def _validate_disjointness_report(
    raw: Mapping[str, Any],
    study: Mapping[str, Any],
) -> dict[str, Any]:
    value = _mapping(raw, "disjointness report")
    _exact(
        value,
        {
            "schema_version",
            "candidate_study_id",
            "candidate_protocol_fingerprint",
            "reference_count",
            "comparisons",
            "disjoint",
            "limitations",
        },
        "disjointness report",
    )
    if value["schema_version"] != DISJOINTNESS_SCHEMA_VERSION:
        raise ValidationError("disjointness report schema version is invalid")
    if value["candidate_study_id"] != study["study_id"] or value[
        "candidate_protocol_fingerprint"
    ] != _fingerprint(study):
        raise ValidationError("disjointness report belongs to a different protocol")
    comparisons_raw = _bounded_list(
        value["comparisons"],
        "disjointness report.comparisons",
        maximum=MAX_EPISODES,
    )
    comparisons: list[dict[str, Any]] = []
    categories = {
        "participants",
        "projects",
        "workstreams",
        "queries",
        "corpora",
        "archives",
    }
    for index, raw_comparison in enumerate(comparisons_raw):
        where = f"disjointness report.comparisons[{index}]"
        comparison = _mapping(raw_comparison, where)
        _exact(
            comparison,
            {
                "reference_study_id",
                "reference_protocol_fingerprint",
                "same_study_id",
                "identity_namespace_match",
                "overlaps",
                "disjoint",
            },
            where,
        )
        overlaps_raw = _mapping(comparison["overlaps"], f"{where}.overlaps")
        _exact(overlaps_raw, categories, f"{where}.overlaps")
        overlaps: dict[str, Any] = {}
        for category in sorted(categories):
            overlap = _mapping(
                overlaps_raw[category],
                f"{where}.overlaps.{category}",
            )
            _exact(overlap, {"count", "fingerprints"}, f"{where}.overlaps.{category}")
            fingerprints = [
                _digest(item, f"{where}.overlaps.{category}.fingerprints")
                for item in _bounded_list(
                    overlap["fingerprints"],
                    f"{where}.overlaps.{category}.fingerprints",
                    maximum=MAX_EPISODES,
                )
            ]
            count = _integer(
                overlap["count"],
                f"{where}.overlaps.{category}.count",
                minimum=0,
                maximum=MAX_EPISODES,
            )
            if count != len(fingerprints):
                raise ValidationError("disjointness overlap count is invalid")
            overlaps[category] = {
                "count": count,
                "fingerprints": fingerprints,
            }
        namespace_match = _boolean(
            comparison["identity_namespace_match"],
            f"{where}.identity_namespace_match",
        )
        same_study = _boolean(comparison["same_study_id"], f"{where}.same_study_id")
        disjoint = _boolean(comparison["disjoint"], f"{where}.disjoint")
        expected_disjoint = (
            namespace_match
            and not same_study
            and all(row["count"] == 0 for row in overlaps.values())
        )
        if disjoint != expected_disjoint:
            raise ValidationError("disjointness comparison conclusion is invalid")
        comparisons.append({
            "reference_study_id": _identifier(
                comparison["reference_study_id"],
                f"{where}.reference_study_id",
            ),
            "reference_protocol_fingerprint": _digest(
                comparison["reference_protocol_fingerprint"],
                f"{where}.reference_protocol_fingerprint",
            ),
            "same_study_id": same_study,
            "identity_namespace_match": namespace_match,
            "overlaps": overlaps,
            "disjoint": disjoint,
        })
    reference_count = _integer(
        value["reference_count"],
        "disjointness report.reference_count",
        minimum=1,
        maximum=MAX_EPISODES,
    )
    if reference_count != len(comparisons):
        raise ValidationError("disjointness reference count is invalid")
    disjoint = _boolean(value["disjoint"], "disjointness report.disjoint")
    if disjoint != all(row["disjoint"] for row in comparisons):
        raise ValidationError("disjointness report conclusion is invalid")
    return {
        "schema_version": DISJOINTNESS_SCHEMA_VERSION,
        "candidate_study_id": study["study_id"],
        "candidate_protocol_fingerprint": _fingerprint(study),
        "reference_count": reference_count,
        "comparisons": comparisons,
        "disjoint": disjoint,
        "limitations": _text(
            value["limitations"],
            "disjointness report.limitations",
            maximum=1_000,
        ),
    }


def _validate_shadow_metric_bundle(
    raw: Mapping[str, Any],
    *,
    study: Mapping[str, Any],
    response_set: Mapping[str, Any],
) -> dict[str, Any]:
    value = _mapping(raw, "shadow metric bundle")
    _exact(
        value,
        {
            "schema_version",
            "study_id",
            "protocol_fingerprint",
            "response_set_fingerprint",
            "records",
        },
        "shadow metric bundle",
    )
    if value["schema_version"] != SHADOW_METRICS_SCHEMA_VERSION:
        raise ValidationError("shadow metric bundle schema version is invalid")
    if (
        value["study_id"] != study["study_id"]
        or value["protocol_fingerprint"] != _fingerprint(study)
        or value["response_set_fingerprint"] != _fingerprint(response_set)
    ):
        raise ValidationError("shadow metric bundle belongs to different artifacts")
    episodes = {row["episode_id"]: row for row in study["episodes"]}
    rows_raw = _bounded_list(
        value["records"],
        "shadow metric bundle.records",
        maximum=MAX_EPISODES,
    )
    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows_raw):
        where = f"shadow metric bundle.records[{index}]"
        row = _mapping(raw_row, where)
        _exact(row, {"episode_id", "query_hash", "metric_record"}, where)
        episode_id = _opaque(row["episode_id"], f"{where}.episode_id")
        episode = episodes.get(episode_id)
        if episode is None:
            raise ValidationError("shadow metric row refers to an unknown episode")
        query_hash = _digest(row["query_hash"], f"{where}.query_hash")
        if query_hash != episode["query_hash"]:
            raise ValidationError("shadow metric query hash differs from protocol")
        rows.append({
            "episode_id": episode_id,
            "query_hash": query_hash,
            "metric_record": dict(
                _mapping(row["metric_record"], f"{where}.metric_record")
            ),
        })
    _unique((row["episode_id"] for row in rows), "shadow metric episode_id")
    if len(rows) != len(episodes) or {row["episode_id"] for row in rows} != set(
        episodes
    ):
        raise ValidationError("shadow metric bundle must cover the exact episode panel")
    return {
        "schema_version": SHADOW_METRICS_SCHEMA_VERSION,
        "study_id": study["study_id"],
        "protocol_fingerprint": _fingerprint(study),
        "response_set_fingerprint": _fingerprint(response_set),
        "records": rows,
    }


def score_study(
    protocol: Mapping[str, Any],
    bundle: Mapping[str, Any],
    judgments: Mapping[str, Any],
    *,
    responses: Mapping[str, Any],
    reference_protocols: Sequence[Mapping[str, Any]],
    untouched_pool_attested: bool,
    consent_active_attested: bool,
    shadow_metric_bundle: Mapping[str, Any] | None = None,
    outcome_replay_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a frozen study and return a diagnostic canary decision."""

    study = validate_protocol(protocol)
    if consent_active_attested is not True:
        raise ValidationError("active consent attestation is required before scoring")
    scored_at = datetime.now(timezone.utc)
    retention_active = _time(study["privacy"]["retention_until"]) > scored_at
    if not retention_active:
        raise ValidationError("study retention expired before scoring")
    response_set = validate_responses(responses)
    _validate_operation_time(study, response_set, scored_at)
    public, private = _validate_bundle(bundle, study, response_set)
    disjointness = audit_study_disjointness(study, reference_protocols)
    if not isinstance(untouched_pool_attested, bool):
        raise ValidationError("operator attestations must be boolean")
    judged = validate_judgments(judgments)
    if judged["study_id"] != study["study_id"]:
        raise ValidationError("judgments belong to a different study")
    if judged["packet_set_fingerprint"] != public["packet_set_fingerprint"]:
        raise ValidationError("judgments belong to a different packet set")

    packets = {row["packet_id"]: row for row in public["judge_packets"]}
    keys = {row["packet_id"]: row for row in private["assignment_key"]}
    judgments_by_packet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judged["judgments"]:
        if row["packet_id"] not in packets:
            raise ValidationError("judgment refers to an unknown packet")
        judgments_by_packet[row["packet_id"]].append(row)
    if set(judgments_by_packet) != set(packets):
        raise ValidationError("every packet requires frozen judgments")

    dimension_minimum = study["design"]["minimum_dimension_score"]
    episode_results: list[dict[str, Any]] = []
    agreement_pairs: list[tuple[bool, bool]] = []
    agreement_packets: list[list[tuple[bool, bool]]] = []
    guess_packets: list[tuple[int, int]] = []
    guess_correct = 0
    guess_total = 0
    identifiable = 0
    adjudicated_slots = 0
    taxonomy_counts: Counter[str] = Counter()
    taxonomy_complete = True

    for packet_id in sorted(packets):
        packet_agreement_pairs: list[tuple[bool, bool]] = []
        packet_guess_correct = 0
        packet_guess_total = 0
        rows = judgments_by_packet[packet_id]
        primaries = sorted(
            (row for row in rows if row["judge_role"] == "primary"),
            key=lambda row: row["judge_id"],
        )
        adjudicators = [row for row in rows if row["judge_role"] == "adjudicator"]
        if len(primaries) != 2:
            raise ValidationError("every packet requires exactly two primary judges")
        disagreements = {
            slot: (
                _judge_success(primaries[0], slot, dimension_minimum)
                != _judge_success(primaries[1], slot, dimension_minimum)
            )
            for slot in SLOTS
        }
        if any(disagreements.values()) and len(adjudicators) != 1:
            raise ValidationError(
                "primary success disagreement requires exactly one adjudicator"
            )
        if not any(disagreements.values()) and adjudicators:
            raise ValidationError(
                "adjudicator is forbidden without a primary disagreement"
            )
        arm_results: dict[str, dict[str, Any]] = {}
        for slot in SLOTS:
            primary_success = [
                _judge_success(row, slot, dimension_minimum) for row in primaries
            ]
            disagreement = primary_success[0] != primary_success[1]
            agreement_pairs.append((primary_success[0], primary_success[1]))
            packet_agreement_pairs.append((primary_success[0], primary_success[1]))
            if disagreement:
                final_judge = adjudicators[0]
                final_success = _judge_success(final_judge, slot, dimension_minimum)
                adjudicated_slots += 1
            else:
                final_judge = primaries[0]
                final_success = primary_success[0]
            assignment = keys[packet_id]["slots"][slot]
            condition = assignment["condition"]
            relevant_judges = [
                *primaries,
                *(adjudicators if disagreement else []),
            ]
            observed_failures = sorted({
                failure
                for row in relevant_judges
                for failure in row["gate_failures"][slot]
            })
            taxonomy = _resolved_taxonomy(
                primaries,
                adjudicators,
                slot=slot,
                success=final_success,
                adjudicated=disagreement,
            )
            if taxonomy is not None:
                taxonomy_counts[taxonomy["primary"]] += 1
            arm_results[condition] = {
                "success": final_success,
                "gate_failures": observed_failures,
                "taxonomy": taxonomy,
                "dimension_scores": {
                    dimension: statistics.mean(
                        row["scores"][slot][dimension] for row in primaries
                    )
                    for dimension in DIMENSIONS
                },
                "primary_rubric_disagreement": statistics.mean(
                    abs(
                        primaries[0]["scores"][slot][dimension]
                        - primaries[1]["scores"][slot][dimension]
                    )
                    for dimension in DIMENSIONS
                ),
                "adjudicator_dimension_scores": (
                    dict(final_judge["scores"][slot]) if disagreement else None
                ),
                "primary_success_disagreement": disagreement,
                "latency_ms": assignment["latency_ms"],
                "total_tokens": assignment["total_tokens"],
            }
            for row in primaries:
                guess_total += 1
                correct = int(row["condition_guesses"][slot] == condition)
                guess_correct += correct
                packet_guess_correct += correct
                packet_guess_total += 1
                identifiable += int(row["potentially_identifiable"][slot])

        if set(arm_results) != set(CONDITIONS):
            raise ValidationError("sealed packet assignments are incomplete")
        if (
            not arm_results["memory_v2"]["success"]
            and arm_results["oracle_evidence"]["success"]
            and (
                arm_results["memory_v2"]["taxonomy"] is None
                or arm_results["memory_v2"]["taxonomy"]["primary"] == "inconclusive"
            )
        ):
            taxonomy_complete = False
        episode_results.append({
            "episode_id": keys[packet_id]["episode_id"],
            "participant_ref": keys[packet_id]["participant_ref"],
            "project_ref": keys[packet_id]["project_ref"],
            "query_class": packets[packet_id]["query_class"],
            "checkpoint_days": packets[packet_id]["checkpoint_days"],
            "arms": arm_results,
        })
        agreement_packets.append(packet_agreement_pairs)
        guess_packets.append((packet_guess_correct, packet_guess_total))

    thresholds = study["thresholds"]
    comparisons = {
        name: _comparison(
            episode_results,
            left=left,
            right=right,
            name=name,
            thresholds=thresholds,
        )
        for name, left, right in (
            ("memory_v2_vs_no_memory", "memory_v2", "no_memory"),
            ("memory_v2_vs_raw_fts", "memory_v2", "raw_fts"),
            ("raw_fts_vs_no_memory", "raw_fts", "no_memory"),
            (
                "oracle_evidence_vs_memory_v2",
                "oracle_evidence",
                "memory_v2",
            ),
        )
    }
    strata = _stratum_comparisons(episode_results, thresholds)
    arms = {
        condition: _arm_summary(episode_results, condition) for condition in CONDITIONS
    }
    agreement = _agreement(agreement_pairs)
    agreement_interval = _packet_cluster_agreement_interval(
        agreement_packets,
        samples=thresholds["bootstrap_samples"],
        seed=_derived_seed(
            thresholds["bootstrap_seed"],
            "judge-agreement:packet",
        ),
        confidence=thresholds["confidence_level"],
    )
    blinding = {
        "forced_guess_count": guess_total,
        "correct_guess_count": guess_correct,
        "accuracy": guess_correct / guess_total if guess_total else None,
        "chance_accuracy": 1.0 / len(CONDITIONS),
        "potentially_identifiable_count": identifiable,
        "potentially_identifiable_rate": (
            identifiable / guess_total if guess_total else None
        ),
        "slot_wilson_interval": _wilson_interval(guess_correct, guess_total),
        "accuracy_interval": _packet_cluster_guess_interval(
            guess_packets,
            samples=thresholds["bootstrap_samples"],
            seed=_derived_seed(
                thresholds["bootstrap_seed"],
                "condition-guess:packet",
            ),
            confidence=thresholds["confidence_level"],
        ),
    }
    shadow_metrics = None
    shadow_panel = None
    if shadow_metric_bundle is not None:
        shadow_panel = _validate_shadow_metric_bundle(
            shadow_metric_bundle,
            study=study,
            response_set=response_set,
        )
        try:
            shadow_metrics = compute_shadow_metrics([
                row["metric_record"] for row in shadow_panel["records"]
            ])
        except ShadowRerankerError as exc:
            raise ValidationError("shadow metric records are invalid") from exc
    replay_summary = _outcome_replay_summary(outcome_replay_result)

    coverage = _coverage(study, episode_results)
    required_shadow = _shadow_gate_status(shadow_metrics, thresholds)
    gates = {
        "pilot_design": study["study_mode"] == "pilot",
        "exact_cross_pool_disjointness": disjointness["disjoint"],
        "untouched_pool_operator_attestation": untouched_pool_attested,
        "active_consent_operator_attestation": consent_active_attested,
        "retention_active_at_score": retention_active,
        "episode_coverage": coverage["episode_coverage"],
        "participant_coverage": coverage["participant_coverage"],
        "project_coverage": coverage["project_coverage"],
        "participant_effective_coverage": coverage["participant_effective_coverage"],
        "project_effective_coverage": coverage["project_effective_coverage"],
        "participant_dominance": coverage["participant_dominance"],
        "project_dominance": coverage["project_dominance"],
        "checkpoint_coverage": coverage["checkpoint_coverage"],
        "query_class_coverage": coverage["query_class_coverage"],
        "required_strata_noninferiority": bool(
            strata
            and all(
                row["conservative_interval"]["lower"]
                >= thresholds["stratum_noninferiority_margin"]
                for row in strata
            )
        ),
        "memory_uplift_vs_no_memory": (
            comparisons["memory_v2_vs_no_memory"]["conservative_interval"]["lower"]
            > thresholds["uplift_margin_vs_no_memory"]
        ),
        "memory_uplift_vs_raw_fts": (
            comparisons["memory_v2_vs_raw_fts"]["conservative_interval"]["lower"]
            > thresholds["uplift_margin_vs_raw_fts"]
        ),
        "oracle_gap_bounded": (
            comparisons["oracle_evidence_vs_memory_v2"]["paired_success_delta"] >= 0.0
            and comparisons["oracle_evidence_vs_memory_v2"]["regression_rate"]
            in {None, 0.0}
            and arms["oracle_evidence"]["hard_safety_failure_count"] == 0
            and comparisons["oracle_evidence_vs_memory_v2"]["conservative_interval"][
                "upper"
            ]
            <= thresholds["max_oracle_gap"]
        ),
        "judge_agreement": (
            agreement["raw_agreement"] is not None
            and agreement_interval["raw_agreement"]["lower"]
            >= thresholds["min_judge_agreement"]
            and agreement_interval["cohen_kappa"]["lower"]
            >= thresholds["min_judge_kappa"]
        ),
        "structural_blinding": (
            blinding["potentially_identifiable_rate"] is not None
            and blinding["potentially_identifiable_rate"]
            <= thresholds["max_identifiable_rate"]
            and blinding["accuracy_interval"]["upper"]
            <= thresholds["max_condition_guess_accuracy"]
        ),
        "irrelevant_injection": (
            arms["memory_v2"]["irrelevant_injection_rate"]
            <= thresholds["max_irrelevant_injection_rate"]
        ),
        "stale_as_current": (
            arms["memory_v2"]["stale_as_current_rate"]
            <= thresholds["max_stale_as_current_rate"]
        ),
        "citation_validity": (
            arms["memory_v2"]["citation_validity_rate"]
            >= thresholds["min_citation_validity_rate"]
        ),
        "hard_safety": arms["memory_v2"]["hard_safety_failure_count"] == 0,
        "latency": (
            arms["memory_v2"]["p95_latency_ms"] <= thresholds["max_p95_latency_ms"]
        ),
        "token_budget": (
            arms["memory_v2"]["mean_total_tokens"] <= thresholds["max_mean_tokens"]
        ),
        "oracle_rescue_taxonomy": taxonomy_complete,
        **required_shadow,
    }
    go = all(gates.values())
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "study_id": study["study_id"],
        "study_mode": study["study_mode"],
        "scored_at": scored_at.isoformat().replace("+00:00", "Z"),
        "protocol_fingerprint": _fingerprint(study),
        "response_set_fingerprint": _fingerprint(response_set),
        "packet_set_fingerprint": public["packet_set_fingerprint"],
        "judgment_set_fingerprint": _fingerprint(judged),
        "disjointness_report": disjointness,
        "disjointness_report_fingerprint": _fingerprint(disjointness),
        "diagnostic_only": True,
        "eligible_for_human_superiority_claim": False,
        "shadow_only": True,
        "read_only": True,
        "mutation_authority": "none",
        "episode_count": len(episode_results),
        "coverage": coverage,
        "comparisons": comparisons,
        "strata": strata,
        "arms": arms,
        "judge_agreement": {
            **agreement,
            "packet_cluster_interval": agreement_interval,
            "adjudicated_slot_count": adjudicated_slots,
        },
        "blinding": blinding,
        "item_difficulty": _item_difficulty(episode_results),
        "failure_taxonomy": {
            "counts": dict(sorted(taxonomy_counts.items())),
            "oracle_rescue_labels_complete": taxonomy_complete,
            "causal_limitation": (
                "Four-arm oracle headroom cannot identify a pipeline stage. "
                "Use common-panel single-change Outcome Replay variants."
            ),
        },
        "shadow_metrics": shadow_metrics,
        "shadow_metric_bundle_fingerprint": (
            _fingerprint(shadow_panel) if shadow_panel is not None else None
        ),
        "outcome_replay": replay_summary,
        "gates": gates,
        "go": go,
        "decision": (
            "ready_for_limited_answer_injection_canary" if go else "not_ready"
        ),
        "limitations": (
            "This pilot decision is not a production rollout, promotion authority, "
            "or evidence of superiority to experienced humans."
        ),
    }


def _validate_privacy(raw: Any, *, created_at: str) -> dict[str, Any]:
    value = _mapping(raw, "protocol.privacy")
    _exact(
        value,
        {
            "collection_mode",
            "consent_ref",
            "identity_namespace_digest",
            "retention_until",
            "revocation_policy_ref",
            "raw_work_evidence_stored",
        },
        "protocol.privacy",
    )
    if value["collection_mode"] != "opt_in_shadow":
        raise ValidationError("privacy collection mode must be opt_in_shadow")
    if value["raw_work_evidence_stored"] is not False:
        raise ValidationError("protocol cannot store raw work evidence")
    retention = _timestamp(value["retention_until"], "protocol.privacy.retention_until")
    if _time(retention) <= _time(created_at):
        raise ValidationError("privacy retention must end after protocol creation")
    return {
        "collection_mode": "opt_in_shadow",
        "consent_ref": _ref(value["consent_ref"], "protocol.privacy.consent_ref"),
        "identity_namespace_digest": _digest(
            value["identity_namespace_digest"],
            "protocol.privacy.identity_namespace_digest",
        ),
        "retention_until": retention,
        "revocation_policy_ref": _ref(
            value["revocation_policy_ref"],
            "protocol.privacy.revocation_policy_ref",
        ),
        "raw_work_evidence_stored": False,
    }


def _validate_operation_time(
    study: Mapping[str, Any],
    response_set: Mapping[str, Any],
    scored_at: datetime,
) -> None:
    latest = scored_at.timestamp() + MAX_CLOCK_SKEW_SECONDS
    timestamps = [study["created_at"]]
    for episode in study["episodes"]:
        timestamps.extend((
            episode["query_as_of"],
            episode["target_evidence_at"],
            episode["evidence_cutoff"],
        ))
    for response in response_set["responses"]:
        timestamps.extend((response["generated_at"], response["evidence_max_at"]))
    if any(_time(value).timestamp() > latest for value in timestamps):
        raise ValidationError("study artifact timestamp is in the future")


def _validate_design(raw: Any, *, study_mode: str) -> dict[str, Any]:
    value = _mapping(raw, "protocol.design")
    _exact(
        value,
        {
            "conditions",
            "slots",
            "dimensions",
            "minimum_dimension_score",
            "primary_comparisons",
            "blinding_seed_commitment",
            "rubric_digest",
            "judge_instructions_digest",
            "judge_calibration_digest",
            "adjudication_policy_digest",
            "packet_renderer_digest",
        },
        "protocol.design",
    )
    if value["conditions"] != list(CONDITIONS):
        raise ValidationError("design conditions must use the fixed four-arm order")
    if value["slots"] != list(SLOTS):
        raise ValidationError("design slots must equal A-D")
    if value["dimensions"] != list(DIMENSIONS):
        raise ValidationError("design dimensions must use the fixed rubric")
    if value["primary_comparisons"] != list(PRIMARY_COMPARISONS):
        raise ValidationError("design primary comparisons are invalid")
    minimum = _integer(
        value["minimum_dimension_score"],
        "protocol.design.minimum_dimension_score",
        minimum=1,
        maximum=5,
    )
    if study_mode == "pilot" and minimum != 3:
        raise ValidationError(
            "pilot minimum_dimension_score must equal the fixed value 3"
        )
    return {
        "conditions": list(CONDITIONS),
        "slots": list(SLOTS),
        "dimensions": list(DIMENSIONS),
        "minimum_dimension_score": minimum,
        "primary_comparisons": list(PRIMARY_COMPARISONS),
        "blinding_seed_commitment": _digest(
            value["blinding_seed_commitment"],
            "protocol.design.blinding_seed_commitment",
        ),
        **{
            field: _digest(value[field], f"protocol.design.{field}")
            for field in (
                "rubric_digest",
                "judge_instructions_digest",
                "judge_calibration_digest",
                "adjudication_policy_digest",
                "packet_renderer_digest",
            )
        },
    }


def _validate_thresholds(raw: Any, *, study_mode: str) -> dict[str, Any]:
    where = "protocol.thresholds"
    value = _mapping(raw, where)
    expected = {
        "minimum_episodes",
        "minimum_participant_clusters",
        "minimum_project_clusters",
        "minimum_effective_participant_clusters",
        "minimum_effective_project_clusters",
        "max_participant_episode_share",
        "max_project_episode_share",
        "minimum_episodes_per_checkpoint",
        "minimum_episodes_per_query_class",
        "minimum_participant_clusters_per_stratum",
        "minimum_project_clusters_per_stratum",
        "minimum_shadow_eligible_records",
        "required_checkpoints",
        "required_query_classes",
        "bootstrap_samples",
        "bootstrap_seed",
        "confidence_level",
        "uplift_margin_vs_no_memory",
        "uplift_margin_vs_raw_fts",
        "stratum_noninferiority_margin",
        "max_oracle_gap",
        "max_irrelevant_injection_rate",
        "max_stale_as_current_rate",
        "min_citation_validity_rate",
        "min_judge_agreement",
        "min_judge_kappa",
        "max_identifiable_rate",
        "max_condition_guess_accuracy",
        "max_p95_latency_ms",
        "max_mean_tokens",
        "min_top1_useful",
        "min_evidence_completeness",
        "min_none_precision",
        "min_none_recall",
    }
    _exact(value, expected, where)
    minimum_episodes = _integer(
        value["minimum_episodes"],
        f"{where}.minimum_episodes",
        minimum=1,
        maximum=MAX_EPISODES,
    )
    if study_mode == "pilot" and minimum_episodes < 500:
        raise ValidationError("pilot minimum_episodes must be at least 500")
    checkpoints = _integer_list(
        value["required_checkpoints"],
        f"{where}.required_checkpoints",
        minimum=0,
        maximum=10_000,
    )
    if study_mode == "pilot" and not {30, 90, 365} <= set(checkpoints):
        raise ValidationError("pilot checkpoints must include 30, 90, and 365")
    query_classes = _choice_list(
        value["required_query_classes"],
        set(QUERY_CLASSES),
        f"{where}.required_query_classes",
    )
    if study_mode == "pilot" and set(query_classes) != set(QUERY_CLASSES):
        raise ValidationError("pilot must register every fixed query class")
    participant_minimum = _integer(
        value["minimum_participant_clusters"],
        f"{where}.minimum_participant_clusters",
        minimum=1,
        maximum=10_000,
    )
    project_minimum = _integer(
        value["minimum_project_clusters"],
        f"{where}.minimum_project_clusters",
        minimum=1,
        maximum=10_000,
    )
    effective_participant_minimum = _number(
        value["minimum_effective_participant_clusters"],
        f"{where}.minimum_effective_participant_clusters",
        minimum=1.0,
        maximum=10_000.0,
    )
    effective_project_minimum = _number(
        value["minimum_effective_project_clusters"],
        f"{where}.minimum_effective_project_clusters",
        minimum=1.0,
        maximum=10_000.0,
    )
    bootstrap_samples = _integer(
        value["bootstrap_samples"],
        f"{where}.bootstrap_samples",
        minimum=100,
        maximum=100_000,
    )
    confidence_level = _number(
        value["confidence_level"],
        f"{where}.confidence_level",
        minimum=0.8,
        maximum=0.999,
    )
    if study_mode == "pilot":
        if participant_minimum < 30 or project_minimum < 20:
            raise ValidationError(
                "pilot requires at least 30 participant and 20 project clusters"
            )
        if effective_participant_minimum < 20 or effective_project_minimum < 15:
            raise ValidationError(
                "pilot effective cluster floors must be at least 20 and 15"
            )
        if bootstrap_samples < 10_000 or confidence_level < 0.95:
            raise ValidationError(
                "pilot requires 10000 bootstrap samples and 95% confidence"
            )
    result = {
        "minimum_episodes": minimum_episodes,
        "minimum_participant_clusters": participant_minimum,
        "minimum_project_clusters": project_minimum,
        "minimum_effective_participant_clusters": effective_participant_minimum,
        "minimum_effective_project_clusters": effective_project_minimum,
        "max_participant_episode_share": _number(
            value["max_participant_episode_share"],
            f"{where}.max_participant_episode_share",
            minimum=0.0,
            maximum=1.0,
        ),
        "max_project_episode_share": _number(
            value["max_project_episode_share"],
            f"{where}.max_project_episode_share",
            minimum=0.0,
            maximum=1.0,
        ),
        "minimum_episodes_per_checkpoint": _integer(
            value["minimum_episodes_per_checkpoint"],
            f"{where}.minimum_episodes_per_checkpoint",
            minimum=1,
            maximum=MAX_EPISODES,
        ),
        "minimum_episodes_per_query_class": _integer(
            value["minimum_episodes_per_query_class"],
            f"{where}.minimum_episodes_per_query_class",
            minimum=1,
            maximum=MAX_EPISODES,
        ),
        "minimum_participant_clusters_per_stratum": _integer(
            value["minimum_participant_clusters_per_stratum"],
            f"{where}.minimum_participant_clusters_per_stratum",
            minimum=1,
            maximum=10_000,
        ),
        "minimum_project_clusters_per_stratum": _integer(
            value["minimum_project_clusters_per_stratum"],
            f"{where}.minimum_project_clusters_per_stratum",
            minimum=1,
            maximum=10_000,
        ),
        "minimum_shadow_eligible_records": _integer(
            value["minimum_shadow_eligible_records"],
            f"{where}.minimum_shadow_eligible_records",
            minimum=1,
            maximum=MAX_EPISODES,
        ),
        "required_checkpoints": checkpoints,
        "required_query_classes": query_classes,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": _integer(
            value["bootstrap_seed"],
            f"{where}.bootstrap_seed",
            minimum=0,
            maximum=2**63 - 1,
        ),
        "confidence_level": confidence_level,
        "max_p95_latency_ms": _number(
            value["max_p95_latency_ms"],
            f"{where}.max_p95_latency_ms",
            minimum=0.0,
            maximum=600_000.0,
        ),
        "max_mean_tokens": _number(
            value["max_mean_tokens"],
            f"{where}.max_mean_tokens",
            minimum=1.0,
            maximum=1_000_000.0,
        ),
    }
    for field in (
        "uplift_margin_vs_no_memory",
        "uplift_margin_vs_raw_fts",
        "max_oracle_gap",
        "max_irrelevant_injection_rate",
        "max_stale_as_current_rate",
        "min_citation_validity_rate",
        "min_judge_agreement",
        "min_judge_kappa",
        "max_identifiable_rate",
        "max_condition_guess_accuracy",
        "min_top1_useful",
        "min_evidence_completeness",
        "min_none_precision",
        "min_none_recall",
    ):
        result[field] = _number(
            value[field], f"{where}.{field}", minimum=0.0, maximum=1.0
        )
    result["stratum_noninferiority_margin"] = _number(
        value["stratum_noninferiority_margin"],
        f"{where}.stratum_noninferiority_margin",
        minimum=-1.0,
        maximum=0.0,
    )
    if study_mode == "pilot":
        if (
            result["minimum_episodes_per_checkpoint"] < 25
            or result["minimum_episodes_per_query_class"] < 25
            or result["minimum_participant_clusters_per_stratum"] < 10
            or result["minimum_project_clusters_per_stratum"] < 8
        ):
            raise ValidationError(
                "pilot strata require 25 episodes, 10 participants, and 8 projects"
            )
        if result["minimum_shadow_eligible_records"] < 50:
            raise ValidationError(
                "pilot shadow metric eligibility floor must be at least 50"
            )
        if (
            result["max_participant_episode_share"] > 0.10
            or result["max_project_episode_share"] > 0.20
        ):
            raise ValidationError(
                "pilot cluster-share ceilings must be at most 0.10 and 0.20"
            )
        if (
            result["min_judge_agreement"] < 0.80
            or result["min_judge_kappa"] < 0.60
            or result["max_condition_guess_accuracy"] > 0.35
        ):
            raise ValidationError(
                "pilot judge and blinding thresholds are too permissive"
            )
        if result["min_citation_validity_rate"] != 1.0:
            raise ValidationError("pilot requires zero invalid citation episodes")
        if result["stratum_noninferiority_margin"] < -0.05:
            raise ValidationError(
                "pilot stratum noninferiority margin cannot be below -0.05"
            )
        if (
            result["max_identifiable_rate"] > 0.05
            or result["max_irrelevant_injection_rate"] > 0.02
            or result["max_stale_as_current_rate"] > 0.01
            or result["max_oracle_gap"] > 0.15
            or result["min_top1_useful"] < 0.70
            or result["min_evidence_completeness"] < 0.80
            or result["min_none_precision"] < 0.90
            or result["min_none_recall"] < 0.90
        ):
            raise ValidationError("pilot readiness thresholds are too permissive")
    return result


def _validate_episode(raw: Any, where: str, *, created_at: str) -> dict[str, Any]:
    value = _mapping(raw, where)
    _exact(
        value,
        {
            "episode_id",
            "participant_ref",
            "project_ref",
            "workstream_ref",
            "query_ref",
            "query_hash",
            "consent_receipt_digest",
            "query_class",
            "checkpoint_days",
            "evidence_cutoff",
            "query_as_of",
            "target_evidence_at",
            "prompt",
            "evidence_pack_ref",
            "condition_input_digests",
            "snapshot",
        },
        where,
    )
    cutoff = _timestamp(value["evidence_cutoff"], f"{where}.evidence_cutoff")
    query_as_of = _timestamp(value["query_as_of"], f"{where}.query_as_of")
    target_evidence_at = _timestamp(
        value["target_evidence_at"],
        f"{where}.target_evidence_at",
    )
    if _time(query_as_of) > _time(created_at):
        raise ValidationError(f"{where}.query_as_of is after protocol creation")
    if not (_time(target_evidence_at) <= _time(cutoff) <= _time(query_as_of)):
        raise ValidationError(
            f"{where} timestamps must satisfy target <= cutoff <= query"
        )
    if _time(cutoff) > _time(created_at):
        raise ValidationError(f"{where}.evidence_cutoff is after protocol creation")
    checkpoint_days = _integer(
        value["checkpoint_days"],
        f"{where}.checkpoint_days",
        minimum=0,
        maximum=10_000,
    )
    derived_gap_days = (
        _time(query_as_of) - _time(target_evidence_at)
    ).total_seconds() / 86_400.0
    if abs(derived_gap_days - checkpoint_days) > 1.0:
        raise ValidationError(
            f"{where}.checkpoint_days differs from its timestamp-derived gap"
        )
    return {
        "episode_id": _opaque(value["episode_id"], f"{where}.episode_id"),
        "participant_ref": _opaque(
            value["participant_ref"], f"{where}.participant_ref"
        ),
        "project_ref": _opaque(value["project_ref"], f"{where}.project_ref"),
        "workstream_ref": _opaque(value["workstream_ref"], f"{where}.workstream_ref"),
        "query_ref": _opaque(value["query_ref"], f"{where}.query_ref"),
        "query_hash": _digest(value["query_hash"], f"{where}.query_hash"),
        "consent_receipt_digest": _digest(
            value["consent_receipt_digest"],
            f"{where}.consent_receipt_digest",
        ),
        "query_class": _choice(
            value["query_class"], set(QUERY_CLASSES), f"{where}.query_class"
        ),
        "checkpoint_days": checkpoint_days,
        "evidence_cutoff": cutoff,
        "query_as_of": query_as_of,
        "target_evidence_at": target_evidence_at,
        "prompt": _text(value["prompt"], f"{where}.prompt", maximum=8_000),
        "evidence_pack_ref": _ref(
            value["evidence_pack_ref"], f"{where}.evidence_pack_ref"
        ),
        "condition_input_digests": _condition_digest_mapping(
            value["condition_input_digests"],
            f"{where}.condition_input_digests",
        ),
        "snapshot": _validate_snapshot(value["snapshot"], f"{where}.snapshot"),
    }


def _validate_snapshot(raw: Any, where: str) -> dict[str, str]:
    value = _mapping(raw, where)
    fields = {
        "corpus_ref",
        "archive_digest",
        "index_digest",
        "config_digest",
        "code_digest",
        "answerer_digest",
        "safety_filters_digest",
    }
    _exact(value, fields, where)
    return {
        "corpus_ref": _opaque(value["corpus_ref"], f"{where}.corpus_ref"),
        **{
            field: _digest(value[field], f"{where}.{field}")
            for field in fields - {"corpus_ref"}
        },
    }


def _validate_response(raw: Any, where: str) -> dict[str, Any]:
    value = _mapping(raw, where)
    _exact(
        value,
        {
            "episode_id",
            "condition",
            "response_id",
            "text",
            "response_artifact_digest",
            "answerer_digest",
            "config_digest",
            "code_digest",
            "input_packet_digest",
            "trace_digest",
            "latency_ms",
            "total_tokens",
            "side_effects_disabled",
            "evidence_max_at",
            "generated_at",
        },
        where,
    )
    if value["side_effects_disabled"] is not True:
        raise ValidationError(f"{where} must disable all side effects")
    evidence_at = _timestamp(value["evidence_max_at"], f"{where}.evidence_max_at")
    generated_at = _timestamp(value["generated_at"], f"{where}.generated_at")
    if _time(generated_at) < _time(evidence_at):
        raise ValidationError(f"{where}.generated_at precedes its evidence")
    return {
        "episode_id": _opaque(value["episode_id"], f"{where}.episode_id"),
        "condition": _choice(value["condition"], set(CONDITIONS), f"{where}.condition"),
        "response_id": _identifier(value["response_id"], f"{where}.response_id"),
        "text": _text(value["text"], f"{where}.text", maximum=32_000),
        **{
            field: _digest(value[field], f"{where}.{field}")
            for field in (
                "response_artifact_digest",
                "answerer_digest",
                "config_digest",
                "code_digest",
                "input_packet_digest",
                "trace_digest",
            )
        },
        "latency_ms": _number(
            value["latency_ms"], f"{where}.latency_ms", minimum=0, maximum=600_000
        ),
        "total_tokens": _integer(
            value["total_tokens"],
            f"{where}.total_tokens",
            minimum=0,
            maximum=1_000_000,
        ),
        "side_effects_disabled": True,
        "evidence_max_at": evidence_at,
        "generated_at": generated_at,
    }


def _validate_judgment(raw: Any, where: str) -> dict[str, Any]:
    value = _mapping(raw, where)
    _exact(
        value,
        {
            "packet_id",
            "judge_id",
            "judge_role",
            "scores",
            "gate_failures",
            "condition_guesses",
            "guess_confidence",
            "material_error_notes",
            "potentially_identifiable",
            "failure_taxonomy",
        },
        where,
    )
    scores_raw = _slot_mapping(value["scores"], f"{where}.scores")
    scores: dict[str, dict[str, int]] = {}
    for slot, raw_dimensions in scores_raw.items():
        dimensions = _mapping(raw_dimensions, f"{where}.scores.{slot}")
        _exact(dimensions, set(DIMENSIONS), f"{where}.scores.{slot}")
        scores[slot] = {
            dimension: _integer(
                dimensions[dimension],
                f"{where}.scores.{slot}.{dimension}",
                minimum=1,
                maximum=5,
            )
            for dimension in DIMENSIONS
        }
    failures_raw = _slot_mapping(value["gate_failures"], f"{where}.gate_failures")
    failures = {
        slot: _choice_list(
            item,
            set(GATE_FAILURES),
            f"{where}.gate_failures.{slot}",
        )
        for slot, item in failures_raw.items()
    }
    guesses_raw = _slot_mapping(
        value["condition_guesses"], f"{where}.condition_guesses"
    )
    guesses = {
        slot: _choice(item, set(CONDITIONS), f"{where}.condition_guesses.{slot}")
        for slot, item in guesses_raw.items()
    }
    if set(guesses.values()) != set(CONDITIONS):
        raise ValidationError(
            f"{where}.condition_guesses must assign every condition once"
        )
    confidence_raw = _slot_mapping(
        value["guess_confidence"], f"{where}.guess_confidence"
    )
    confidence = {
        slot: _choice(
            item,
            {"low", "medium", "high"},
            f"{where}.guess_confidence.{slot}",
        )
        for slot, item in confidence_raw.items()
    }
    notes_raw = _slot_mapping(
        value["material_error_notes"], f"{where}.material_error_notes"
    )
    notes = {
        slot: _validate_error_note(item, f"{where}.material_error_notes.{slot}")
        for slot, item in notes_raw.items()
    }
    identifiable_raw = _slot_mapping(
        value["potentially_identifiable"],
        f"{where}.potentially_identifiable",
    )
    identifiable = {
        slot: _boolean(item, f"{where}.potentially_identifiable.{slot}")
        for slot, item in identifiable_raw.items()
    }
    taxonomy_raw = _slot_mapping(value["failure_taxonomy"], f"{where}.failure_taxonomy")
    taxonomy = {
        slot: _validate_taxonomy(item, f"{where}.failure_taxonomy.{slot}")
        for slot, item in taxonomy_raw.items()
    }
    return {
        "packet_id": _identifier(value["packet_id"], f"{where}.packet_id"),
        "judge_id": _opaque(value["judge_id"], f"{where}.judge_id"),
        "judge_role": _choice(
            value["judge_role"], {"primary", "adjudicator"}, f"{where}.judge_role"
        ),
        "scores": scores,
        "gate_failures": failures,
        "condition_guesses": guesses,
        "guess_confidence": confidence,
        "material_error_notes": notes,
        "potentially_identifiable": identifiable,
        "failure_taxonomy": taxonomy,
    }


def _validate_error_note(raw: Any, where: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    value = _mapping(raw, where)
    _exact(value, {"reason", "evidence_refs"}, where)
    return {
        "reason": _text(value["reason"], f"{where}.reason", maximum=1_000),
        "evidence_refs": [
            _ref(item, f"{where}.evidence_refs")
            for item in _bounded_list(
                value["evidence_refs"], f"{where}.evidence_refs", maximum=20
            )
        ],
    }


def _validate_taxonomy(raw: Any, where: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    value = _mapping(raw, where)
    _exact(
        value,
        {"primary", "secondary", "label_source", "judgment_ref"},
        where,
    )
    primary = _choice(value["primary"], set(FAILURE_TAXONOMY), f"{where}.primary")
    secondary_raw = value["secondary"]
    secondary = (
        None
        if secondary_raw is None
        else _choice(secondary_raw, set(FAILURE_TAXONOMY), f"{where}.secondary")
    )
    if secondary == primary:
        raise ValidationError(f"{where}.secondary must differ from primary")
    return {
        "primary": primary,
        "secondary": secondary,
        "label_source": _choice(
            value["label_source"], set(LABEL_SOURCES), f"{where}.label_source"
        ),
        "judgment_ref": _ref(value["judgment_ref"], f"{where}.judgment_ref"),
    }


def _validate_bundle(
    raw: Mapping[str, Any],
    study: Mapping[str, Any],
    response_set: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = _mapping(raw, "packet bundle")
    _exact(bundle, {"public", "private"}, "packet bundle")
    public_raw = _mapping(bundle["public"], "packet bundle.public")
    private_raw = _mapping(bundle["private"], "packet bundle.private")
    _exact(
        public_raw,
        {
            "schema_version",
            "study_id",
            "protocol_fingerprint",
            "response_set_fingerprint",
            "packet_set_fingerprint",
            "judge_packets",
        },
        "packet bundle.public",
    )
    _exact(
        private_raw,
        {
            "schema_version",
            "study_id",
            "seed",
            "protocol_fingerprint",
            "response_set_fingerprint",
            "packet_set_fingerprint",
            "assignment_key",
        },
        "packet bundle.private",
    )
    if public_raw["schema_version"] != PUBLIC_PACKETS_SCHEMA_VERSION:
        raise ValidationError("public packet schema version is invalid")
    if private_raw["schema_version"] != PRIVATE_KEY_SCHEMA_VERSION:
        raise ValidationError("private key schema version is invalid")
    if (
        public_raw["study_id"] != study["study_id"]
        or private_raw["study_id"] != study["study_id"]
    ):
        raise ValidationError("packet bundle belongs to a different study")
    packets = [
        _validate_public_packet(row, f"packet bundle.public.judge_packets[{index}]")
        for index, row in enumerate(
            _list(public_raw["judge_packets"], "packet bundle.public.judge_packets")
        )
    ]
    keys = [
        _validate_assignment(row, f"packet bundle.private.assignment_key[{index}]")
        for index, row in enumerate(
            _list(private_raw["assignment_key"], "packet bundle.private.assignment_key")
        )
    ]
    _unique((row["packet_id"] for row in packets), "public packet_id")
    _unique((row["packet_id"] for row in keys), "private packet_id")
    _unique((row["episode_id"] for row in keys), "private episode_id")
    _unique(
        (
            response["response_id"]
            for packet in packets
            for response in packet["responses"].values()
        ),
        "public response_id",
    )
    if {row["packet_id"] for row in packets} != {row["packet_id"] for row in keys}:
        raise ValidationError("public packets and private key disagree")
    packet_by_id = {row["packet_id"]: row for row in packets}
    key_by_id = {row["packet_id"]: row for row in keys}
    episode_by_id = {row["episode_id"]: row for row in study["episodes"]}
    if {row["episode_id"] for row in keys} != set(episode_by_id):
        raise ValidationError("private key episodes differ from the protocol")
    if len(keys) != len(episode_by_id) or len(packets) != len(episode_by_id):
        raise ValidationError("packet bundle must contain one row per episode")
    for packet_id, key in key_by_id.items():
        episode = episode_by_id[key["episode_id"]]
        for field in (
            "participant_ref",
            "project_ref",
            "workstream_ref",
            "query_ref",
        ):
            if key[field] != episode[field]:
                raise ValidationError("private key cluster identity is invalid")
        public_packet = packet_by_id[packet_id]
        for field in (
            "prompt",
            "evidence_pack_ref",
            "query_class",
            "checkpoint_days",
        ):
            if public_packet[field] != episode[field]:
                raise ValidationError("public packet metadata differs from protocol")
        for slot in SLOTS:
            if (
                key["slots"][slot]["response_id"]
                != packet_by_id[packet_id]["responses"][slot]["response_id"]
            ):
                raise ValidationError("public/private response assignments disagree")
    public_protocol_fingerprint = _digest(
        public_raw["protocol_fingerprint"],
        "public protocol fingerprint",
    )
    public_response_fingerprint = _digest(
        public_raw["response_set_fingerprint"],
        "public response fingerprint",
    )
    packet_fingerprint = _digest(
        public_raw["packet_set_fingerprint"], "public packet fingerprint"
    )
    if packet_fingerprint != _fingerprint({
        "protocol_fingerprint": public_protocol_fingerprint,
        "response_set_fingerprint": public_response_fingerprint,
        "judge_packets": packets,
    }):
        raise ValidationError("public packet set fingerprint is invalid")
    if private_raw["packet_set_fingerprint"] != packet_fingerprint:
        raise ValidationError("private key packet fingerprint is invalid")
    if private_raw["protocol_fingerprint"] != _fingerprint(study):
        raise ValidationError("private key protocol fingerprint is invalid")
    if (
        public_protocol_fingerprint != private_raw["protocol_fingerprint"]
        or public_response_fingerprint != private_raw["response_set_fingerprint"]
    ):
        raise ValidationError("public/private source fingerprints disagree")
    normalized_public = {
        "schema_version": PUBLIC_PACKETS_SCHEMA_VERSION,
        "study_id": study["study_id"],
        "protocol_fingerprint": public_protocol_fingerprint,
        "response_set_fingerprint": public_response_fingerprint,
        "packet_set_fingerprint": packet_fingerprint,
        "judge_packets": packets,
    }
    seed = _integer(
        private_raw["seed"],
        "private key seed",
        minimum=-(2**63),
        maximum=2**63 - 1,
    )
    normalized_private = {
        "schema_version": PRIVATE_KEY_SCHEMA_VERSION,
        "study_id": study["study_id"],
        "seed": seed,
        "protocol_fingerprint": private_raw["protocol_fingerprint"],
        "response_set_fingerprint": _digest(
            private_raw["response_set_fingerprint"],
            "private response fingerprint",
        ),
        "packet_set_fingerprint": packet_fingerprint,
        "assignment_key": keys,
    }
    expected = prepare_blinded_packets(study, response_set, seed)
    if (
        normalized_public != expected["public"]
        or normalized_private != expected["private"]
    ):
        raise ValidationError(
            "packet bundle differs from the frozen protocol, responses, or seed"
        )
    return normalized_public, normalized_private


def _validate_public_packet(raw: Any, where: str) -> dict[str, Any]:
    value = _mapping(raw, where)
    _exact(
        value,
        {
            "packet_id",
            "prompt",
            "evidence_pack_ref",
            "query_class",
            "checkpoint_days",
            "dimensions",
            "responses",
        },
        where,
    )
    if value["dimensions"] != list(DIMENSIONS):
        raise ValidationError(f"{where}.dimensions are invalid")
    responses = _slot_mapping(value["responses"], f"{where}.responses")
    normalized: dict[str, dict[str, str]] = {}
    for slot, raw_response in responses.items():
        response = _mapping(raw_response, f"{where}.responses.{slot}")
        _exact(response, {"response_id", "text"}, f"{where}.responses.{slot}")
        normalized[slot] = {
            "response_id": _identifier(
                response["response_id"], f"{where}.responses.{slot}.response_id"
            ),
            "text": _text(
                response["text"], f"{where}.responses.{slot}.text", maximum=32_000
            ),
        }
    _unique((row["response_id"] for row in normalized.values()), "packet response_id")
    return {
        "packet_id": _identifier(value["packet_id"], f"{where}.packet_id"),
        "prompt": _text(value["prompt"], f"{where}.prompt", maximum=8_000),
        "evidence_pack_ref": _ref(
            value["evidence_pack_ref"], f"{where}.evidence_pack_ref"
        ),
        "query_class": _choice(
            value["query_class"], set(QUERY_CLASSES), f"{where}.query_class"
        ),
        "checkpoint_days": _integer(
            value["checkpoint_days"],
            f"{where}.checkpoint_days",
            minimum=0,
            maximum=10_000,
        ),
        "dimensions": list(DIMENSIONS),
        "responses": normalized,
    }


def _validate_assignment(raw: Any, where: str) -> dict[str, Any]:
    value = _mapping(raw, where)
    _exact(
        value,
        {
            "packet_id",
            "episode_id",
            "participant_ref",
            "project_ref",
            "workstream_ref",
            "query_ref",
            "slots",
        },
        where,
    )
    slots_raw = _slot_mapping(value["slots"], f"{where}.slots")
    slots: dict[str, dict[str, Any]] = {}
    for slot, raw_assignment in slots_raw.items():
        assignment = _mapping(raw_assignment, f"{where}.slots.{slot}")
        _exact(
            assignment,
            {
                "condition",
                "response_id",
                "response_artifact_digest",
                "trace_digest",
                "latency_ms",
                "total_tokens",
            },
            f"{where}.slots.{slot}",
        )
        slots[slot] = {
            "condition": _choice(
                assignment["condition"],
                set(CONDITIONS),
                f"{where}.slots.{slot}.condition",
            ),
            "response_id": _identifier(
                assignment["response_id"], f"{where}.slots.{slot}.response_id"
            ),
            "response_artifact_digest": _digest(
                assignment["response_artifact_digest"],
                f"{where}.slots.{slot}.response_artifact_digest",
            ),
            "trace_digest": _digest(
                assignment["trace_digest"], f"{where}.slots.{slot}.trace_digest"
            ),
            "latency_ms": _number(
                assignment["latency_ms"],
                f"{where}.slots.{slot}.latency_ms",
                minimum=0,
                maximum=600_000,
            ),
            "total_tokens": _integer(
                assignment["total_tokens"],
                f"{where}.slots.{slot}.total_tokens",
                minimum=0,
                maximum=1_000_000,
            ),
        }
    if {row["condition"] for row in slots.values()} != set(CONDITIONS):
        raise ValidationError(
            f"{where}.slots must contain every condition exactly once"
        )
    _unique((row["response_id"] for row in slots.values()), "assignment response_id")
    return {
        "packet_id": _identifier(value["packet_id"], f"{where}.packet_id"),
        "episode_id": _opaque(value["episode_id"], f"{where}.episode_id"),
        "participant_ref": _opaque(
            value["participant_ref"], f"{where}.participant_ref"
        ),
        "project_ref": _opaque(value["project_ref"], f"{where}.project_ref"),
        "workstream_ref": _opaque(value["workstream_ref"], f"{where}.workstream_ref"),
        "query_ref": _opaque(value["query_ref"], f"{where}.query_ref"),
        "slots": slots,
    }


def _judge_success(judgment: Mapping[str, Any], slot: str, minimum: int) -> bool:
    return (
        all(judgment["scores"][slot][dimension] >= minimum for dimension in DIMENSIONS)
        and not judgment["gate_failures"][slot]
    )


def _resolved_taxonomy(
    primaries: Sequence[Mapping[str, Any]],
    adjudicators: Sequence[Mapping[str, Any]],
    *,
    slot: str,
    success: bool,
    adjudicated: bool,
) -> dict[str, Any] | None:
    if success:
        return None
    if adjudicated and adjudicators:
        return adjudicators[0]["failure_taxonomy"][slot]
    labels = [row["failure_taxonomy"][slot] for row in primaries]
    if labels[0] is not None and labels[0] == labels[1]:
        return labels[0]
    return None


def _comparison(
    episodes: Sequence[Mapping[str, Any]],
    *,
    left: str,
    right: str,
    name: str,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    pairs = [
        {
            "participant_ref": row["participant_ref"],
            "project_ref": row["project_ref"],
            "delta": float(row["arms"][left]["success"])
            - float(row["arms"][right]["success"]),
            "left_success": row["arms"][left]["success"],
            "right_success": row["arms"][right]["success"],
        }
        for row in episodes
    ]
    estimate = statistics.mean(row["delta"] for row in pairs)
    participant_interval = _cluster_interval(
        pairs,
        cluster_key="participant_ref",
        samples=thresholds["bootstrap_samples"],
        seed=_derived_seed(thresholds["bootstrap_seed"], f"{name}:participant"),
        confidence=thresholds["confidence_level"],
    )
    project_interval = _cluster_interval(
        pairs,
        cluster_key="project_ref",
        samples=thresholds["bootstrap_samples"],
        seed=_derived_seed(thresholds["bootstrap_seed"], f"{name}:project"),
        confidence=thresholds["confidence_level"],
    )
    two_way_interval = _two_way_cluster_interval(
        pairs,
        samples=thresholds["bootstrap_samples"],
        seed=_derived_seed(thresholds["bootstrap_seed"], f"{name}:two-way"),
        confidence=thresholds["confidence_level"],
    )
    baseline_failures = [row for row in pairs if not row["right_success"]]
    baseline_successes = [row for row in pairs if row["right_success"]]
    return {
        "left": left,
        "right": right,
        "paired_episodes": len(pairs),
        "paired_success_delta": estimate,
        "rescue_rate": (
            sum(row["left_success"] for row in baseline_failures)
            / len(baseline_failures)
            if baseline_failures
            else None
        ),
        "regression_rate": (
            sum(not row["left_success"] for row in baseline_successes)
            / len(baseline_successes)
            if baseline_successes
            else None
        ),
        "participant_cluster_interval": participant_interval,
        "project_cluster_interval": project_interval,
        "one_way_interval_envelope": {
            "level": thresholds["confidence_level"],
            "lower": min(participant_interval["lower"], project_interval["lower"]),
            "upper": max(participant_interval["upper"], project_interval["upper"]),
            "method": "participant_project_cluster_envelope",
        },
        "two_way_cluster_interval": two_way_interval,
        "conservative_interval": dict(two_way_interval),
        "participant_cluster_variance": _cluster_diagnostics(pairs, "participant_ref"),
        "project_cluster_variance": _cluster_diagnostics(pairs, "project_ref"),
    }


def _stratum_comparisons(
    episodes: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = [
        (
            "checkpoint",
            str(checkpoint),
            [row for row in episodes if row["checkpoint_days"] == checkpoint],
        )
        for checkpoint in thresholds["required_checkpoints"]
    ]
    groups.extend(
        (
            "query_class",
            query_class,
            [row for row in episodes if row["query_class"] == query_class],
        )
        for query_class in thresholds["required_query_classes"]
    )
    for dimension, value, subset in groups:
        if not subset:
            continue
        result = _comparison(
            subset,
            left="memory_v2",
            right="raw_fts",
            name=f"stratum:{dimension}:{value}",
            thresholds=thresholds,
        )
        rows.append({"dimension": dimension, "value": value, **result})
    return rows


def _cluster_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    cluster_key: str,
    samples: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[cluster_key])].append(row)
    clusters = sorted(grouped)
    if not clusters:
        raise ValidationError("cluster bootstrap requires observations")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        selected = [rng.choice(clusters) for _ in clusters]
        deltas = [
            float(row["delta"]) for cluster in selected for row in grouped[cluster]
        ]
        estimates.append(statistics.mean(deltas))
    estimates.sort()
    alpha = 1.0 - confidence
    return {
        "level": confidence,
        "lower": _percentile(estimates, alpha / 2.0),
        "upper": _percentile(estimates, 1.0 - alpha / 2.0),
        "method": f"{cluster_key}_cluster_percentile_bootstrap",
        "samples": samples,
        "clusters": len(clusters),
    }


def _two_way_cluster_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    """Pigeonhole bootstrap for crossed participant/project dependence."""

    participants = sorted({str(row["participant_ref"]) for row in rows})
    projects = sorted({str(row["project_ref"]) for row in rows})
    if not participants or not projects:
        raise ValidationError("two-way cluster bootstrap requires observations")
    rng = random.Random(seed)
    estimates: list[float] = []
    attempts = 0
    maximum_attempts = max(samples * 20, samples + 100)
    while len(estimates) < samples and attempts < maximum_attempts:
        attempts += 1
        participant_counts = Counter(rng.choice(participants) for _ in participants)
        project_counts = Counter(rng.choice(projects) for _ in projects)
        weighted_total = 0.0
        total_weight = 0
        for row in rows:
            weight = (
                participant_counts[str(row["participant_ref"])]
                * project_counts[str(row["project_ref"])]
            )
            weighted_total += weight * float(row["delta"])
            total_weight += weight
        if total_weight:
            estimates.append(weighted_total / total_weight)
    if len(estimates) != samples:
        raise ValidationError(
            "two-way cluster bootstrap could not produce enough observations"
        )
    estimates.sort()
    alpha = 1.0 - confidence
    return {
        "level": confidence,
        "lower": _percentile(estimates, alpha / 2.0),
        "upper": _percentile(estimates, 1.0 - alpha / 2.0),
        "method": "participant_project_pigeonhole_bootstrap",
        "samples": samples,
        "participant_clusters": len(participants),
        "project_clusters": len(projects),
        "discarded_zero_weight_draws": attempts - samples,
    }


def _cluster_diagnostics(
    rows: Sequence[Mapping[str, Any]], cluster_key: str
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[cluster_key])].append(float(row["delta"]))
    sizes = [len(values) for values in grouped.values()]
    means = [statistics.mean(values) for values in grouped.values()]
    total = sum(sizes)
    effective = (total * total) / sum(size * size for size in sizes) if sizes else 0.0
    cluster_count = len(grouped)
    grand_mean = (
        statistics.mean(value for values in grouped.values() for value in values)
        if total
        else 0.0
    )
    ss_between = sum(
        len(values) * (statistics.mean(values) - grand_mean) ** 2
        for values in grouped.values()
    )
    ss_within = sum(
        sum((value - statistics.mean(values)) ** 2 for value in values)
        for values in grouped.values()
    )
    ms_between = ss_between / (cluster_count - 1) if cluster_count > 1 else None
    ms_within = ss_within / (total - cluster_count) if total > cluster_count else None
    n0 = (
        (total - (sum(size * size for size in sizes) / total)) / (cluster_count - 1)
        if total and cluster_count > 1
        else None
    )
    tau_squared = (
        max(0.0, (ms_between - ms_within) / n0)
        if ms_between is not None
        and ms_within is not None
        and n0 is not None
        and n0 > 0
        else None
    )
    intraclass_correlation = (
        tau_squared / (tau_squared + ms_within)
        if tau_squared is not None
        and ms_within is not None
        and tau_squared + ms_within > 0
        else None
    )
    return {
        "clusters": len(grouped),
        "effective_clusters": effective,
        "max_cluster_share": max(sizes, default=0) / total if total else None,
        "between_cluster_variance": (
            statistics.variance(means) if len(means) > 1 else None
        ),
        "ms_between": ms_between,
        "ms_within": ms_within,
        "unbalanced_n0": n0,
        "between_cluster_variance_component": tau_squared,
        "intraclass_correlation": intraclass_correlation,
    }


def _arm_summary(
    episodes: Sequence[Mapping[str, Any]], condition: str
) -> dict[str, Any]:
    rows = [row["arms"][condition] for row in episodes]
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    tokens = [int(row["total_tokens"]) for row in rows]
    irrelevant = sum("irrelevant_injection" in row["gate_failures"] for row in rows)
    stale = sum("stale_as_current" in row["gate_failures"] for row in rows)
    invalid_citation = sum(
        bool({"invalid_citation", "unattested_citation"} & set(row["gate_failures"]))
        for row in rows
    )
    hard_failures = {
        "invalid_citation",
        "unattested_citation",
        "security_boundary",
        "privacy_leak",
        "post_cutoff_evidence",
        "artifact_integrity",
    }
    return {
        "episodes": len(rows),
        "success_rate": statistics.mean(float(row["success"]) for row in rows),
        "irrelevant_injection_rate": irrelevant / len(rows),
        "stale_as_current_rate": stale / len(rows),
        "citation_validity_rate": 1.0 - (invalid_citation / len(rows)),
        "citation_failure_free_episode_rate": 1.0 - (invalid_citation / len(rows)),
        "hard_safety_failure_count": sum(
            bool(hard_failures & set(row["gate_failures"])) for row in rows
        ),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "mean_total_tokens": statistics.mean(tokens),
    }


def _agreement(pairs: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    if not pairs:
        return {"comparisons": 0, "raw_agreement": None, "cohen_kappa": None}
    agreements = sum(left == right for left, right in pairs)
    observed = agreements / len(pairs)
    first_positive = sum(left for left, _right in pairs) / len(pairs)
    second_positive = sum(right for _left, right in pairs) / len(pairs)
    expected = first_positive * second_positive + (1.0 - first_positive) * (
        1.0 - second_positive
    )
    kappa = None if expected == 1.0 else (observed - expected) / (1.0 - expected)
    return {
        "comparisons": len(pairs),
        "agreements": agreements,
        "raw_agreement": observed,
        "cohen_kappa": kappa,
    }


def _wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> dict[str, Any]:
    if total <= 0:
        return {"level": 0.95, "lower": None, "upper": None, "method": "wilson"}
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    center = (proportion + (z * z / (2.0 * total))) / denominator
    margin = (
        z
        * math.sqrt(
            (proportion * (1.0 - proportion) / total) + (z * z / (4.0 * total * total))
        )
        / denominator
    )
    return {
        "level": 0.95,
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
        "method": "wilson",
    }


def _packet_cluster_guess_interval(
    packets: Sequence[tuple[int, int]],
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    if not packets or any(total <= 0 for _correct, total in packets):
        raise ValidationError("guess interval requires complete packets")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        selected = [rng.choice(packets) for _packet in packets]
        estimates.append(
            sum(correct for correct, _total in selected)
            / sum(total for _correct, total in selected)
        )
    estimates.sort()
    alpha = 1.0 - confidence
    return {
        "level": confidence,
        "lower": _percentile(estimates, alpha / 2.0),
        "upper": _percentile(estimates, 1.0 - alpha / 2.0),
        "method": "packet_cluster_percentile_bootstrap",
        "samples": samples,
        "packets": len(packets),
    }


def _packet_cluster_agreement_interval(
    packets: Sequence[Sequence[tuple[bool, bool]]],
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    if not packets or any(not packet for packet in packets):
        raise ValidationError("agreement interval requires complete packets")
    rng = random.Random(seed)
    raw_estimates: list[float] = []
    kappa_estimates: list[float] = []
    degenerate = 0
    for _ in range(samples):
        pairs = [pair for _packet in packets for pair in rng.choice(packets)]
        estimate = _agreement(pairs)
        raw_estimates.append(float(estimate["raw_agreement"]))
        if estimate["cohen_kappa"] is None:
            degenerate += 1
            kappa_estimates.append(-1.0)
        else:
            kappa_estimates.append(float(estimate["cohen_kappa"]))
    raw_estimates.sort()
    kappa_estimates.sort()
    alpha = 1.0 - confidence

    def interval(values: Sequence[float]) -> dict[str, float]:
        return {
            "lower": _percentile(values, alpha / 2.0),
            "upper": _percentile(values, 1.0 - alpha / 2.0),
        }

    return {
        "level": confidence,
        "raw_agreement": interval(raw_estimates),
        "cohen_kappa": interval(kappa_estimates),
        "method": "packet_cluster_percentile_bootstrap",
        "samples": samples,
        "packets": len(packets),
        "degenerate_kappa_draws": degenerate,
    }


def _item_difficulty(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = []
    for row in episodes:
        oracle_scores = row["arms"]["oracle_evidence"]["dimension_scores"]
        oracle_quality = statistics.mean(oracle_scores.values())
        difficulty = 1.0 - ((oracle_quality - 1.0) / 4.0)
        items.append({
            "item_fingerprint": _fingerprint({"episode_id": row["episode_id"]}),
            "checkpoint_days": row["checkpoint_days"],
            "query_class": row["query_class"],
            "oracle_rubric_difficulty": difficulty,
            "oracle_success": row["arms"]["oracle_evidence"]["success"],
            "primary_disagreement_arms": sum(
                row["arms"][condition]["primary_success_disagreement"]
                for condition in CONDITIONS
            ),
            "mean_primary_rubric_disagreement": statistics.mean(
                row["arms"][condition]["primary_rubric_disagreement"]
                for condition in CONDITIONS
            ),
            "rubric_means": {
                condition: statistics.mean(
                    row["arms"][condition]["dimension_scores"].values()
                )
                for condition in CONDITIONS
            },
        })
    difficulty = [row["oracle_rubric_difficulty"] for row in items]
    return {
        "definition": (
            "One minus normalized oracle-evidence rubric quality; treatment "
            "failures are reported separately and do not define item difficulty."
        ),
        "mean": statistics.mean(difficulty),
        "p50": _percentile(sorted(difficulty), 0.5),
        "p90": _percentile(sorted(difficulty), 0.9),
        "hard_items": sum(value >= 0.75 for value in difficulty),
        "medium_items": sum(0.25 <= value < 0.75 for value in difficulty),
        "easy_items": sum(value < 0.25 for value in difficulty),
        "items": items,
    }


def _coverage(
    study: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    thresholds = study["thresholds"]
    participant_diagnostics = _cluster_size_diagnostics(episodes, "participant_ref")
    project_diagnostics = _cluster_size_diagnostics(episodes, "project_ref")
    participants = {row["participant_ref"] for row in episodes}
    projects = {row["project_ref"] for row in episodes}
    checkpoints = {row["checkpoint_days"] for row in episodes}
    query_classes = {row["query_class"] for row in episodes}
    checkpoint_strata = {
        str(checkpoint): _stratum_coverage(
            [row for row in episodes if row["checkpoint_days"] == checkpoint],
            minimum_episodes=thresholds["minimum_episodes_per_checkpoint"],
            minimum_participants=thresholds["minimum_participant_clusters_per_stratum"],
            minimum_projects=thresholds["minimum_project_clusters_per_stratum"],
        )
        for checkpoint in thresholds["required_checkpoints"]
    }
    query_strata = {
        query_class: _stratum_coverage(
            [row for row in episodes if row["query_class"] == query_class],
            minimum_episodes=thresholds["minimum_episodes_per_query_class"],
            minimum_participants=thresholds["minimum_participant_clusters_per_stratum"],
            minimum_projects=thresholds["minimum_project_clusters_per_stratum"],
        )
        for query_class in thresholds["required_query_classes"]
    }
    return {
        "episodes": len(episodes),
        "participant_clusters": len(participants),
        "project_clusters": len(projects),
        "participant_diagnostics": participant_diagnostics,
        "project_diagnostics": project_diagnostics,
        "checkpoints": sorted(checkpoints),
        "query_classes": sorted(query_classes),
        "episode_coverage": len(episodes) >= thresholds["minimum_episodes"],
        "participant_coverage": (
            len(participants) >= thresholds["minimum_participant_clusters"]
        ),
        "project_coverage": (len(projects) >= thresholds["minimum_project_clusters"]),
        "participant_effective_coverage": (
            participant_diagnostics["effective_clusters"]
            >= thresholds["minimum_effective_participant_clusters"]
        ),
        "project_effective_coverage": (
            project_diagnostics["effective_clusters"]
            >= thresholds["minimum_effective_project_clusters"]
        ),
        "participant_dominance": (
            participant_diagnostics["max_cluster_share"]
            <= thresholds["max_participant_episode_share"]
        ),
        "project_dominance": (
            project_diagnostics["max_cluster_share"]
            <= thresholds["max_project_episode_share"]
        ),
        "checkpoint_strata": checkpoint_strata,
        "query_class_strata": query_strata,
        "checkpoint_coverage": bool(checkpoint_strata)
        and all(row["eligible"] for row in checkpoint_strata.values()),
        "query_class_coverage": bool(query_strata)
        and all(row["eligible"] for row in query_strata.values()),
    }


def _cluster_size_diagnostics(
    rows: Sequence[Mapping[str, Any]], cluster_key: str
) -> dict[str, Any]:
    counts = Counter(str(row[cluster_key]) for row in rows)
    sizes = list(counts.values())
    total = sum(sizes)
    return {
        "clusters": len(counts),
        "effective_clusters": (
            (total * total) / sum(size * size for size in sizes) if sizes else 0.0
        ),
        "max_cluster_share": (max(sizes, default=0) / total if total else 1.0),
    }


def _stratum_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_episodes: int,
    minimum_participants: int,
    minimum_projects: int,
) -> dict[str, Any]:
    participant = _cluster_size_diagnostics(rows, "participant_ref")
    project = _cluster_size_diagnostics(rows, "project_ref")
    return {
        "episodes": len(rows),
        "participant_clusters": participant["clusters"],
        "effective_participant_clusters": participant["effective_clusters"],
        "project_clusters": project["clusters"],
        "effective_project_clusters": project["effective_clusters"],
        "eligible": (
            len(rows) >= minimum_episodes
            and participant["clusters"] >= minimum_participants
            and participant["effective_clusters"] >= 0.8 * minimum_participants
            and project["clusters"] >= minimum_projects
            and project["effective_clusters"] >= 0.8 * minimum_projects
        ),
    }


def _shadow_gate_status(
    metrics: Mapping[str, Any] | None, thresholds: Mapping[str, Any]
) -> dict[str, bool]:
    names = {
        "shadow_top1": (
            "top1_useful",
            "min_top1_useful",
            False,
            "memory_needed",
        ),
        "shadow_evidence_completeness": (
            "evidence_set_completeness",
            "min_evidence_completeness",
            False,
            "required_evidence_sets",
        ),
        "shadow_none_precision": (
            "none_precision",
            "min_none_precision",
            False,
            "predicted_none",
        ),
        "shadow_none_recall": (
            "none_recall",
            "min_none_recall",
            False,
            "expected_none",
        ),
        "shadow_irrelevant_injection": (
            "irrelevant_injection",
            "max_irrelevant_injection_rate",
            True,
            "expected_none",
        ),
        "shadow_stale_as_current": (
            "stale_as_current",
            "max_stale_as_current_rate",
            True,
            "current_selected_items",
        ),
        "shadow_citation_validity": (
            "citation_validity",
            "min_citation_validity_rate",
            False,
            "citations",
        ),
    }
    if metrics is None:
        return {name: False for name in names}
    result: dict[str, bool] = {}
    eligibility = metrics.get("eligible_gates", {})
    coverage = metrics.get("coverage", {})
    for gate, (metric, threshold, maximum, denominator) in names.items():
        value = metrics.get(metric)
        result[gate] = bool(
            isinstance(eligibility, Mapping)
            and eligibility.get(metric) is True
            and isinstance(coverage, Mapping)
            and isinstance(coverage.get(denominator), int)
            and coverage[denominator] >= thresholds["minimum_shadow_eligible_records"]
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and (
                float(value) <= thresholds[threshold]
                if maximum
                else float(value) >= thresholds[threshold]
            )
        )
    return result


def _outcome_replay_summary(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValidationError("outcome replay result must be a mapping")
    if (
        value.get("schema_version") != "memory-v2-outcome-replay-result/v1"
        or value.get("diagnostic_only") is not True
        or value.get("mutation_authority") != "none"
    ):
        raise ValidationError("outcome replay result is not a safe diagnostic")
    diagnosis = value.get("diagnosis")
    if not isinstance(diagnosis, Mapping):
        raise ValidationError("outcome replay diagnosis is missing")
    return {
        "schema_version": value["schema_version"],
        "dataset_fingerprint": _digest(
            value.get("dataset_fingerprint"), "outcome replay dataset fingerprint"
        ),
        "diagnosis": {
            "status": _text(
                diagnosis.get("status"),
                "outcome replay diagnosis status",
                maximum=100,
            ),
            "component": (
                None
                if diagnosis.get("component") is None
                else _text(
                    diagnosis.get("component"),
                    "outcome replay diagnosis component",
                    maximum=100,
                )
            ),
            "ablation": (
                None
                if diagnosis.get("ablation") is None
                else _text(
                    diagnosis.get("ablation"),
                    "outcome replay diagnosis ablation",
                    maximum=100,
                )
            ),
        },
        "mutation_authority": "none",
    }


def _load_document(path: str | Path) -> Any:
    source = Path(path).expanduser()
    try:
        before_path = source.stat()
        if not stat.S_ISREG(before_path.st_mode):
            raise ValidationError("study artifact must be a regular file")
        with source.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValidationError("study artifact must be a regular file")
            if (
                before.st_dev != before_path.st_dev
                or before.st_ino != before_path.st_ino
            ):
                raise ValidationError("study artifact changed before it was read")
            if before.st_size > MAX_INPUT_BYTES:
                raise ValidationError("study artifact exceeds its byte bound")
            encoded = handle.read(MAX_INPUT_BYTES + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ValidationError("cannot read study artifact") from exc
    if len(encoded) > MAX_INPUT_BYTES:
        raise ValidationError("study artifact exceeds its byte bound")
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(encoded) != after.st_size
    ):
        raise ValidationError("study artifact changed while it was read")
    try:
        text = encoded.decode("utf-8")
    except UnicodeError as exc:
        raise ValidationError("study artifact must be UTF-8") from exc
    suffix = source.suffix.lower()
    try:
        if suffix == ".json":
            value = json.loads(
                text,
                object_pairs_hook=_json_object,
                parse_constant=_reject_json_constant,
                parse_float=_finite_float,
            )
        elif suffix in {".yaml", ".yml"}:
            value = _safe_yaml_load(text)
        else:
            raise ValidationError("study artifacts must use JSON or YAML")
    except RecursionError as exc:
        raise ValidationError("study artifact exceeds its nesting bound") from exc
    _bounded_depth(value)
    return value


def _safe_yaml_load(text: str) -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is in Hermes
        raise ValidationError("YAML support is unavailable") from exc

    class StrictSafeLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        loader.flatten_mapping(node)
        pairs = loader.construct_pairs(node, deep=deep)
        result: dict[Any, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError("duplicate YAML key is forbidden")
            result[key] = value
        return result

    StrictSafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    try:
        if any(
            isinstance(event, yaml.events.AliasEvent)
            for event in yaml.parse(text, Loader=StrictSafeLoader)
        ):
            raise ValidationError("YAML aliases are forbidden")
        value = yaml.load(text, Loader=StrictSafeLoader)
    except yaml.YAMLError as exc:
        raise ValidationError("study YAML is invalid") from exc
    return value


def _json_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("duplicate JSON key is forbidden")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON constant is forbidden: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValidationError("non-finite JSON number is forbidden")
    return parsed


def _bounded_depth(value: Any, maximum: int = 64) -> None:
    pending = [(value, 1)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        current, depth = pending.pop()
        if isinstance(current, Mapping):
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            nodes += 1
            if nodes > MAX_DOCUMENT_NODES:
                raise ValidationError("study artifact exceeds its node bound")
            if depth > maximum:
                raise ValidationError("study artifact exceeds its nesting bound")
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            nodes += 1
            if nodes > MAX_DOCUMENT_NODES:
                raise ValidationError("study artifact exceeds its node bound")
            if depth > maximum:
                raise ValidationError("study artifact exceeds its nesting bound")
            pending.extend((item, depth + 1) for item in current)
        else:
            nodes += 1
            if nodes > MAX_DOCUMENT_NODES:
                raise ValidationError("study artifact exceeds its node bound")


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{where} must be an object")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{where} must be a list")
    return value


def _bounded_list(value: Any, where: str, *, maximum: int) -> list[Any]:
    rows = _list(value, where)
    if len(rows) > maximum:
        raise ValidationError(f"{where} exceeds its safety bound")
    return rows


def _slot_mapping(value: Any, where: str) -> dict[str, Any]:
    rows = _mapping(value, where)
    _exact(rows, set(SLOTS), where)
    return {slot: rows[slot] for slot in SLOTS}


def _condition_digest_mapping(value: Any, where: str) -> dict[str, str]:
    rows = _mapping(value, where)
    _exact(rows, set(CONDITIONS), where)
    return {
        condition: _digest(rows[condition], f"{where}.{condition}")
        for condition in CONDITIONS
    }


def _exact(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValidationError(
            f"{where} fields are invalid; missing={missing}, extra={extra}"
        )


def _text(value: Any, where: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{where} must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > maximum:
        raise ValidationError(f"{where} must be nonempty and bounded")
    return normalized


def _identifier(value: Any, where: str) -> str:
    text = _text(value, where, maximum=256)
    if any(ord(char) < 32 for char in text):
        raise ValidationError(f"{where} contains control characters")
    return text


def _ref(value: Any, where: str) -> str:
    text = _identifier(value, where)
    if ":" not in text:
        raise ValidationError(f"{where} must be a namespaced reference")
    return text


def _opaque(value: Any, where: str) -> str:
    text = _identifier(value, where)
    if not text.startswith("opaque:") or len(text) < 15:
        raise ValidationError(f"{where} must be a protected opaque reference")
    return text


def _digest(value: Any, where: str) -> str:
    text = _text(value, where, maximum=71)
    if (
        not text.startswith("sha256:")
        or len(text) != 71
        or any(char not in _HEX for char in text.removeprefix("sha256:"))
    ):
        raise ValidationError(f"{where} must be a lowercase sha256 digest")
    return text


def _choice(value: Any, allowed: set[str], where: str) -> str:
    text = _identifier(value, where)
    if text not in allowed:
        raise ValidationError(f"{where} has an unsupported value")
    return text


def _choice_list(value: Any, allowed: set[str], where: str) -> list[str]:
    rows = _bounded_list(value, where, maximum=len(allowed))
    result = [_choice(item, allowed, where) for item in rows]
    _unique(result, where)
    return result


def _integer_list(value: Any, where: str, *, minimum: int, maximum: int) -> list[int]:
    rows = _bounded_list(value, where, maximum=32)
    result = [_integer(item, where, minimum=minimum, maximum=maximum) for item in rows]
    _unique((str(item) for item in result), where)
    return result


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{where} must be boolean")
    return value


def _integer(value: Any, where: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{where} must be an integer")
    if not minimum <= value <= maximum:
        raise ValidationError(f"{where} is outside its bound")
    return value


def _number(value: Any, where: str, *, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise ValidationError(f"{where} must be a finite bounded number")
    return float(value)


def _timestamp(value: Any, where: str) -> str:
    text = _text(value, where, maximum=128)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{where} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{where} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _unique(values: Iterable[str], where: str) -> None:
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
    digest = hashlib.sha256(f"{seed}|{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValidationError("percentile requires observations")
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    fraction = position - lower
    return float(values[lower] * (1.0 - fraction) + values[upper] * fraction)


__all__ = [
    "CONDITIONS",
    "DIMENSIONS",
    "DISJOINTNESS_SCHEMA_VERSION",
    "FAILURE_TAXONOMY",
    "JUDGMENTS_SCHEMA_VERSION",
    "PRIVATE_KEY_SCHEMA_VERSION",
    "PROTOCOL_SCHEMA_VERSION",
    "PUBLIC_PACKETS_SCHEMA_VERSION",
    "RESPONSE_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "SLOTS",
    "ValidationError",
    "audit_study_disjointness",
    "load_judgments",
    "load_protocol",
    "load_responses",
    "prepare_blinded_packets",
    "score_study",
    "validate_judgments",
    "validate_protocol",
    "validate_responses",
]
