"""Blinded, paired human-baseline evaluation for Memory v2.

The public workflow is deliberately small and suitable for a thin CLI:

``load_protocol(path)``
    Load and strictly validate a versioned JSON or YAML study protocol.
``prepare_blinded_packets(protocol, responses, seed)``
    Validate a complete paired response set and return separate public judge
    packets and a sealed private answer key.
``load_responses(path)``
    Strictly load a versioned full or partial response artifact. Pair balance
    and protocol references are enforced when packets are prepared.
``load_judgments(path)``
    Load the versioned judgment document. Protocol-dependent completeness is
    checked by :func:`score_study`.
``score_study(protocol, packets, judgments)``
    Verify frozen-artifact fingerprints, aggregate independent judges, and
    evaluate the preregistered superiority and safety gates.

All returned objects are composed only of JSON-serializable primitives. The
module never evaluates input as code; YAML uses a duplicate-key-rejecting
``SafeLoader`` and JSON rejects duplicate keys and non-finite constants.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_SCHEMA_VERSION = "memory-v2-human-baseline-protocol/v1"
RESPONSE_SCHEMA_VERSION = "memory-v2-human-baseline-responses/v1"
PACKET_BUNDLE_SCHEMA_VERSION = "memory-v2-human-baseline-packet-bundle/v1"
PUBLIC_PACKETS_SCHEMA_VERSION = "memory-v2-human-baseline-public-packets/v1"
PRIVATE_KEY_SCHEMA_VERSION = "memory-v2-human-baseline-private-key/v1"
JUDGMENTS_SCHEMA_VERSION = "memory-v2-human-baseline-judgments/v1"
RESULT_SCHEMA_VERSION = "memory-v2-human-baseline-result/v1"
DISJOINTNESS_SCHEMA_VERSION = "memory-v2-human-baseline-disjointness/v1"

FIXED_DIMENSIONS = (
    "factual_correctness",
    "temporal_correctness",
    "completeness",
    "source_grounding",
    "actionability",
    "calibration_restraint",
)
CONDITIONS = ("human", "memory_v2")
SLOTS = ("A", "B")
MAX_INPUT_BYTES = 10 * 1024 * 1024


class ValidationError(ValueError):
    """Raised when a study artifact violates its versioned contract."""


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Load and validate a Memory v2 human-baseline protocol."""

    return _validate_protocol(_load_document(path))


def audit_protocol_disjointness(
    candidate: Mapping[str, Any],
    references: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Check stable study identifiers and normalized prompts for exact overlap.

    This is a fail-closed artifact check, not proof that renamed people or
    paraphrased projects are genuinely independent. Operators must separately
    audit identity, employer/project lineage, and evidence-source overlap.
    """

    validated_candidate = _validate_protocol(candidate)
    if not references:
        raise ValidationError("disjointness audit requires at least one reference protocol")
    validated_references = [_validate_protocol(reference) for reference in references]

    def values(protocol: Mapping[str, Any]) -> dict[str, set[str]]:
        return {
            "participant_ids": {row["id"] for row in protocol["participants"]},
            "query_ids": {row["id"] for row in protocol["queries"]},
            "workstream_ids": {row["workstream_id"] for row in protocol["queries"]},
            "normalized_prompts": {
                " ".join(row["prompt"].casefold().split()) for row in protocol["queries"]
            },
        }

    candidate_values = values(validated_candidate)
    comparisons: list[dict[str, Any]] = []
    all_disjoint = True
    for reference in validated_references:
        reference_values = values(reference)
        overlaps: dict[str, dict[str, Any]] = {}
        for category, candidate_category_values in candidate_values.items():
            shared = sorted(candidate_category_values & reference_values[category])
            overlaps[category] = {
                "count": len(shared),
                "fingerprints": [
                    _fingerprint({"category": category, "value": value})
                    for value in shared
                ],
            }
        same_study_id = validated_candidate["study_id"] == reference["study_id"]
        disjoint = not same_study_id and all(
            row["count"] == 0 for row in overlaps.values()
        )
        all_disjoint = all_disjoint and disjoint
        comparisons.append(
            {
                "reference_study_id": reference["study_id"],
                "reference_study_mode": reference["study_mode"],
                "reference_protocol_fingerprint": _fingerprint(reference),
                "same_study_id": same_study_id,
                "overlaps": overlaps,
                "disjoint": disjoint,
            }
        )

    return {
        "schema_version": DISJOINTNESS_SCHEMA_VERSION,
        "candidate_study_id": validated_candidate["study_id"],
        "candidate_study_mode": validated_candidate["study_mode"],
        "candidate_protocol_fingerprint": _fingerprint(validated_candidate),
        "reference_count": len(comparisons),
        "comparisons": comparisons,
        "disjoint": all_disjoint,
        "limitations": (
            "Exact identifiers and normalized prompts are checked. Renamed participants, "
            "related projects, paraphrases, and shared evidence require an independent operator audit."
        ),
    }


def load_responses(path: str | Path) -> dict[str, Any]:
    """Load a versioned full or condition-specific response artifact."""

    return _validate_response_artifact(_load_document(path))


def prepare_blinded_packets(
    protocol: Mapping[str, Any],
    responses: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Return frozen public judge packets and a sealed private answer key.

    The public part contains no seed, participant identifier, condition label,
    protocol hash, response-set hash, or assignment mapping. The private key
    contains only keyed opaque participant/cluster references and no response text.
    Callers must keep ``bundle["private"]`` sealed until judgments are frozen.
    """

    validated_protocol = _validate_protocol(protocol)
    validated_responses = _validate_response_document(responses, validated_protocol)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValidationError("seed must be an integer")

    study_id = validated_protocol["study_id"]
    rng = random.Random(seed)
    query_by_id = {row["id"]: row for row in validated_protocol["queries"]}
    response_by_pair: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in validated_responses["responses"]:
        pair = (row["participant_id"], row["query_id"])
        response_by_pair[pair][row["condition"]] = row["response"]

    public_packets: list[dict[str, Any]] = []
    assignment_key: list[dict[str, Any]] = []
    frozen_responses: list[dict[str, Any]] = []
    for ordinal, pair in enumerate(sorted(response_by_pair)):
        participant_id, query_id = pair
        query = query_by_id[query_id]
        query_fingerprint = _fingerprint({"query": query})
        participant_ref = _opaque_token(seed, study_id, "participant", participant_id)
        cluster_id = _opaque_token(
            seed,
            study_id,
            "participant-workstream",
            participant_id,
            query["workstream_id"],
        )
        packet_id = "pkt_" + _opaque_token(
            seed, study_id, "packet", participant_id, query_id, str(ordinal)
        )[7:31]
        conditions = list(CONDITIONS)
        rng.shuffle(conditions)

        packet_responses: dict[str, dict[str, str]] = {}
        key_slots: dict[str, dict[str, str]] = {}
        for slot, condition in zip(SLOTS, conditions):
            response_id = "rsp_" + _opaque_token(
                seed, study_id, "response", participant_id, query_id, condition
            )[7:31]
            text = response_by_pair[pair][condition]
            packet_responses[slot] = {"response_id": response_id, "text": text}
            key_slots[slot] = {"response_id": response_id, "condition": condition}
            frozen_responses.append(
                {
                    "participant_ref": participant_ref,
                    "query_fingerprint": query_fingerprint,
                    "condition": condition,
                    "response_id": response_id,
                    "response": text,
                }
            )

        public_packets.append(
            {
                "packet_id": packet_id,
                "prompt": query["prompt"],
                "stratum": query["stratum"],
                "checkpoint_days": query["checkpoint_days"],
                "applicable_dimensions": list(query["applicable_dimensions"]),
                "responses": packet_responses,
            }
        )
        assignment_key.append(
            {
                "packet_id": packet_id,
                "query_fingerprint": query_fingerprint,
                "participant_ref": participant_ref,
                "cluster_id": cluster_id,
                "slots": key_slots,
            }
        )

    protocol_fingerprint = _fingerprint(validated_protocol)
    response_set_fingerprint = _fingerprint(sorted(frozen_responses, key=_canonical_json))
    packet_set_fingerprint = _fingerprint(
        {"study_id": study_id, "judge_packets": public_packets}
    )
    study_fingerprint = _fingerprint(
        {
            "study_id": study_id,
            "protocol_fingerprint": protocol_fingerprint,
            "response_set_fingerprint": response_set_fingerprint,
            "packet_set_fingerprint": packet_set_fingerprint,
        }
    )
    private_payload = {
        "schema_version": PRIVATE_KEY_SCHEMA_VERSION,
        "study_id": study_id,
        "seed": seed,
        "protocol_fingerprint": protocol_fingerprint,
        "response_set_fingerprint": response_set_fingerprint,
        "packet_set_fingerprint": packet_set_fingerprint,
        "study_fingerprint": study_fingerprint,
        "assignment_key": assignment_key,
    }
    private_payload["answer_key_fingerprint"] = _fingerprint(private_payload)
    return {
        "schema_version": PACKET_BUNDLE_SCHEMA_VERSION,
        "public": {
            "schema_version": PUBLIC_PACKETS_SCHEMA_VERSION,
            "study_id": study_id,
            "packet_set_fingerprint": packet_set_fingerprint,
            "judge_packets": public_packets,
        },
        "private": private_payload,
    }


def load_judgments(path: str | Path) -> dict[str, Any]:
    """Load generic judgment structure; score-time checks use the protocol."""

    document = _load_document(path)
    _require_mapping(document, "judgments document")
    _require_exact_keys(document, {"schema_version", "study_id", "judgments"}, "judgments document")
    _require_version(document, JUDGMENTS_SCHEMA_VERSION, "judgments document")
    study_id = _nonblank(document["study_id"], "judgments document.study_id")
    rows = _require_list(document["judgments"], "judgments document.judgments")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(rows):
        where = f"judgments[{index}]"
        _require_mapping(raw, where)
        _require_exact_keys(raw, {"packet_id", "judge_id", "judge_role", "scores", "gate_failures", "condition_guesses", "material_error_notes", "potentially_identifiable"}, where)
        packet_id = _nonblank(raw["packet_id"], f"{where}.packet_id")
        judge_id = _nonblank(raw["judge_id"], f"{where}.judge_id")
        judge_role = _validate_judge_role(raw["judge_role"], f"{where}.judge_role")
        identity = (packet_id, judge_id)
        if identity in seen:
            raise ValidationError(f"duplicate judgment for packet_id={packet_id!r}, judge_id={judge_id!r}")
        seen.add(identity)
        scores = _validate_slot_mapping(raw["scores"], f"{where}.scores", _validate_generic_scores)
        failures = _validate_slot_mapping(
            raw["gate_failures"], f"{where}.gate_failures", _validate_generic_gate_failures
        )
        guesses = _validate_slot_mapping(
            raw["condition_guesses"], f"{where}.condition_guesses", _validate_condition_guess
        )
        error_notes = _validate_slot_mapping(
            raw["material_error_notes"], f"{where}.material_error_notes", _validate_material_error_note
        )
        identifiable = _validate_slot_mapping(
            raw["potentially_identifiable"], f"{where}.potentially_identifiable", _validate_bool
        )
        normalized.append(
            {
                "packet_id": packet_id,
                "judge_id": judge_id,
                "judge_role": judge_role,
                "scores": scores,
                "gate_failures": failures,
                "condition_guesses": guesses,
                "material_error_notes": error_notes,
                "potentially_identifiable": identifiable,
            }
        )
    return {
        "schema_version": JUDGMENTS_SCHEMA_VERSION,
        "study_id": study_id,
        "judgments": normalized,
    }


def score_study(
    protocol: Mapping[str, Any],
    packets: Mapping[str, Any],
    judgments: Mapping[str, Any],
) -> dict[str, Any]:
    """Score a frozen, blinded paired study.

    A response is a binary work-continuity success only when every applicable
    dimension's mean independent-judge score meets ``dimension_minimum`` and
    that response has no safety failure. The primary estimand is Memory v2's
    paired success-rate advantage over humans at ``primary_checkpoint``.
    Confidence intervals use deterministic participant-cluster bootstrap.
    """

    validated_protocol = _validate_protocol(protocol)
    bundle = _validate_and_verify_bundle(validated_protocol, packets)
    validated_judgments = _validate_judgments_for_study(
        validated_protocol, bundle, judgments
    )
    thresholds = validated_protocol["thresholds"]
    query_by_fingerprint = {
        _fingerprint({"query": query}): query for query in validated_protocol["queries"]
    }
    key_by_packet = {row["packet_id"]: row for row in bundle["private"]["assignment_key"]}
    public_by_packet = {row["packet_id"]: row for row in bundle["public"]["judge_packets"]}
    judgments_by_packet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in validated_judgments["judgments"]:
        judgments_by_packet[row["packet_id"]].append(row)

    paired_items: list[dict[str, Any]] = []
    all_slot_comparisons = 0
    all_slot_agreements = 0
    all_slots_evaluated = 0
    primary_slot_comparisons = 0
    primary_slot_agreements = 0
    primary_slots_evaluated = 0
    adjudicated_slots = 0
    guess_total = 0
    guess_unsure = 0
    guess_correct = 0
    identifiable_slots: dict[tuple[str, str], str] = {}
    agreement_by_checkpoint: dict[int, dict[str, int]] = defaultdict(
        lambda: {"comparisons": 0, "agreements": 0, "slots_evaluated": 0}
    )
    gate_counts = {condition: {gate: 0 for gate in validated_protocol["safety_gates"]} for condition in CONDITIONS}
    dimension_accumulator: dict[str, dict[str, list[float]]] = {
        dimension: {condition: [] for condition in CONDITIONS} for dimension in FIXED_DIMENSIONS
    }
    for packet_id in sorted(public_by_packet):
        public_packet = public_by_packet[packet_id]
        key_row = key_by_packet[packet_id]
        query = query_by_fingerprint[key_row["query_fingerprint"]]
        judges = sorted(judgments_by_packet[packet_id], key=lambda row: row["judge_id"])
        primary_judges = [judge for judge in judges if judge["judge_role"] == "primary"]
        condition_results: dict[str, dict[str, Any]] = {}
        for slot in SLOTS:
            condition = key_row["slots"][slot]["condition"]
            individual_successes = [
                not judge["gate_failures"][slot]
                and all(
                    judge["scores"][slot][dimension] >= thresholds["dimension_minimum"]
                    for dimension in query["applicable_dimensions"]
                )
                for judge in primary_judges
            ]
            judge_disagreement = (
                len(individual_successes) >= 2
                and len(set(individual_successes)) > 1
            )
            primary_disagreement = (
                query["checkpoint_days"] == thresholds["primary_checkpoint"]
                and judge_disagreement
            )
            if len(individual_successes) >= 2:
                all_slots_evaluated += 1
                checkpoint_agreement = agreement_by_checkpoint[query["checkpoint_days"]]
                checkpoint_agreement["slots_evaluated"] += 1
                if query["checkpoint_days"] == thresholds["primary_checkpoint"]:
                    primary_slots_evaluated += 1
                for left in range(len(individual_successes)):
                    for right in range(left + 1, len(individual_successes)):
                        agreement = int(
                            individual_successes[left] == individual_successes[right]
                        )
                        all_slot_comparisons += 1
                        all_slot_agreements += agreement
                        checkpoint_agreement["comparisons"] += 1
                        checkpoint_agreement["agreements"] += agreement
                        if query["checkpoint_days"] == thresholds["primary_checkpoint"]:
                            primary_slot_comparisons += 1
                            primary_slot_agreements += agreement
                if query["checkpoint_days"] == thresholds["primary_checkpoint"]:
                    adjudicated_slots += int(primary_disagreement)
            dimension_means = {
                dimension: sum(judge["scores"][slot][dimension] for judge in judges) / len(judges)
                for dimension in query["applicable_dimensions"]
            }
            failures = sorted(
                {
                    gate
                    for judge in judges
                    for gate in judge["gate_failures"][slot]
                }
            )
            for gate in failures:
                gate_counts[condition][gate] += 1
            success = not failures and all(
                value >= thresholds["dimension_minimum"] for value in dimension_means.values()
            )
            for judge in judges:
                guess_total += 1
                guess = judge["condition_guesses"][slot]
                if guess == "unsure":
                    guess_unsure += 1
                elif guess == condition:
                    guess_correct += 1
                if judge["potentially_identifiable"][slot]:
                    identifiable_slots[(packet_id, slot)] = condition
            for dimension, value in dimension_means.items():
                dimension_accumulator[dimension][condition].append(value)
            condition_results[condition] = {
                "success": success,
                "dimension_means": dimension_means,
                "gate_failures": failures,
                "judge_disagreement": judge_disagreement,
                "material_error_notes": [
                    {"judge_id": judge["judge_id"], **judge["material_error_notes"][slot]}
                    for judge in judges
                    if judge["material_error_notes"][slot] is not None
                ],
            }
        paired_items.append(
            {
                "packet_id": packet_id,
                "participant_ref": key_row["participant_ref"],
                "cluster_id": key_row["cluster_id"],
                "query_fingerprint": key_row["query_fingerprint"],
                "workstream_id": query["workstream_id"],
                "stratum": query["stratum"],
                "checkpoint_days": query["checkpoint_days"],
                "applicable_dimensions": list(query["applicable_dimensions"]),
                "judge_count": len(judges),
                "primary_judge_count": len(primary_judges),
                "adjudicator_count": sum(
                    judge["judge_role"] == "adjudicator" for judge in judges
                ),
                "human": condition_results["human"],
                "memory_v2": condition_results["memory_v2"],
                "paired_success_delta": int(condition_results["memory_v2"]["success"])
                - int(condition_results["human"]["success"]),
            }
        )

    primary_items = [
        row for row in paired_items if row["checkpoint_days"] == thresholds["primary_checkpoint"]
    ]
    primary_estimate, primary_ci = _cluster_bootstrap(
        primary_items,
        samples=thresholds["bootstrap_samples"],
        seed=_bootstrap_seed(bundle["private"]["seed"], "primary"),
    )
    stratum_results: dict[str, dict[str, Any]] = {}
    for stratum in thresholds["required_strata"]:
        rows = [row for row in primary_items if row["stratum"] == stratum]
        estimate, ci = _cluster_bootstrap(
            rows,
            samples=thresholds["bootstrap_samples"],
            seed=_bootstrap_seed(bundle["private"]["seed"], f"stratum:{stratum}"),
        )
        stratum_results[stratum] = {
            "paired_items": len(rows),
            "participants": len({row["participant_ref"] for row in rows}),
            "estimate": estimate,
            "confidence_interval": ci,
            "noninferiority_margin": thresholds["stratum_noninferiority_margin"],
            "passes": bool(rows) and ci["lower"] >= thresholds["stratum_noninferiority_margin"],
        }

    participant_count = len({row["participant_ref"] for row in paired_items})
    cluster_count = len({row["cluster_id"] for row in paired_items})
    primary_participant_count = len({row["participant_ref"] for row in primary_items})
    primary_cluster_count = len({row["cluster_id"] for row in primary_items})
    strata_counts = _count_values(paired_items, "stratum")
    checkpoint_counts = _count_values(paired_items, "checkpoint_days")
    dimension_counts = {
        dimension: sum(dimension in row["applicable_dimensions"] for row in paired_items)
        for dimension in FIXED_DIMENSIONS
    }
    primary_dimension_counts = {
        dimension: sum(dimension in row["applicable_dimensions"] for row in primary_items)
        for dimension in FIXED_DIMENSIONS
    }
    primary_strata_counts = _count_values(primary_items, "stratum")
    coverage = {
        "paired_items": len(paired_items),
        "primary_paired_items": len(primary_items),
        "participants": participant_count,
        "participant_workstream_clusters": cluster_count,
        "primary_participants": primary_participant_count,
        "primary_participant_workstream_clusters": primary_cluster_count,
        "strata": strata_counts,
        "primary_strata": primary_strata_counts,
        "checkpoints": {str(key): value for key, value in checkpoint_counts.items()},
        "dimensions": dimension_counts,
        "primary_dimensions": primary_dimension_counts,
        "required_strata_covered": all(primary_strata_counts.get(value, 0) > 0 for value in thresholds["required_strata"]),
        "required_checkpoints_covered": all(
            checkpoint_counts.get(value, 0) > 0 for value in thresholds["required_checkpoints"]
        ),
        "dimension_minimum_coverage": all(
            count >= thresholds["min_pairs_per_dimension"] for count in primary_dimension_counts.values()
        ),
    }

    memory_gate_failures = sum(gate_counts["memory_v2"].values())
    human_gate_failures = sum(gate_counts["human"].values())
    hard_gates = {
        "passes": memory_gate_failures == 0,
        "memory_v2_failures": memory_gate_failures,
        "human_failures": human_gate_failures,
        "by_condition": gate_counts,
        "note": "Human failures affect human response success but do not fail the Memory v2 system gate.",
    }
    claim_gates = {
        "minimum_items": len(primary_items) >= thresholds["min_items"],
        "minimum_participants": primary_participant_count >= thresholds["min_participants"],
        "minimum_clusters": primary_cluster_count >= thresholds["min_clusters"],
        "required_strata_coverage": coverage["required_strata_covered"],
        "required_checkpoint_coverage": coverage["required_checkpoints_covered"],
        "dimension_coverage": coverage["dimension_minimum_coverage"],
        "stratum_noninferiority": all(row["passes"] for row in stratum_results.values()),
        "hard_safety_gates": hard_gates["passes"],
        "superiority_confidence_bound": primary_ci["lower"] > thresholds["superiority_margin"],
    }
    claim_gates["confirmatory_study_mode"] = validated_protocol["study_mode"] == "confirmatory"
    superiority_claim = all(claim_gates.values())
    dimensions = {}
    for dimension, values in dimension_accumulator.items():
        human_values = values["human"]
        memory_values = values["memory_v2"]
        dimensions[dimension] = {
            "paired_items": len(human_values),
            "human_mean": _mean_or_none(human_values),
            "memory_v2_mean": _mean_or_none(memory_values),
            "mean_delta": (
                _mean_or_none(memory_values) - _mean_or_none(human_values)
                if human_values and memory_values
                else None
            ),
        }

    pilot_diagnostics = {
        "eligible_for_power_analysis": validated_protocol["study_mode"] == "pilot",
        "item_difficulty": _item_difficulty_diagnostics(paired_items),
        "participant_cluster_variance": {
            "primary": _participant_cluster_diagnostics(primary_items),
            "by_checkpoint": {
                str(checkpoint): _participant_cluster_diagnostics(
                    [row for row in paired_items if row["checkpoint_days"] == checkpoint]
                )
                for checkpoint in sorted(checkpoint_counts)
            },
        },
        "note": (
            "Pilot diagnostics estimate rubric behavior and clustered variance for power planning; "
            "they are not confirmatory evidence."
        ),
    }

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "study_id": validated_protocol["study_id"],
        "study_mode": validated_protocol["study_mode"],
        "valid_complete": True,
        "decision": (
            "superiority"
            if superiority_claim
            else "ineligible_by_design"
            if validated_protocol["study_mode"] != "confirmatory"
            else "no_claim"
        ),
        "superiority_claim": superiority_claim,
        "primary_checkpoint": thresholds["primary_checkpoint"],
        "estimate": primary_estimate,
        "confidence_interval": primary_ci,
        "superiority_margin": thresholds["superiority_margin"],
        "claim_gates": claim_gates,
        "coverage": coverage,
        "hard_gates": hard_gates,
        "strata": stratum_results,
        "dimensions": dimensions,
        "inter_rater": {
            "binary_comparisons": all_slot_comparisons,
            "binary_agreements": all_slot_agreements,
            "slots_evaluated": all_slots_evaluated,
            "binary_agreement_rate": (
                all_slot_agreements / all_slot_comparisons
                if all_slot_comparisons
                else None
            ),
            "by_checkpoint": {
                str(checkpoint): {
                    **values,
                    "agreement_rate": (
                        values["agreements"] / values["comparisons"]
                        if values["comparisons"]
                        else None
                    ),
                }
                for checkpoint, values in sorted(agreement_by_checkpoint.items())
            },
            "primary_binary_comparisons": primary_slot_comparisons,
            "primary_binary_agreements": primary_slot_agreements,
            "primary_slots_evaluated": primary_slots_evaluated,
            "primary_binary_agreement_rate": (
                primary_slot_agreements / primary_slot_comparisons
                if primary_slot_comparisons
                else None
            ),
            "adjudicated_slots": adjudicated_slots,
            "adjudication_rate": (
                adjudicated_slots / primary_slots_evaluated
                if primary_slots_evaluated
                else None
            ),
        },
        "blinding_diagnostic": {
            "total_slot_guesses": guess_total,
            "unsure": guess_unsure,
            "unsure_rate": guess_unsure / guess_total if guess_total else None,
            "correct_among_non_unsure": guess_correct,
            "accuracy_among_non_unsure": (
                guess_correct / (guess_total - guess_unsure)
                if guess_total > guess_unsure
                else None
            ),
            "potentially_identifiable_slot_count": len(identifiable_slots),
            "potentially_identifiable_packet_count": len(
                {packet_id for packet_id, _slot in identifiable_slots}
            ),
            "potentially_identifiable_slots": [
                {"packet_id": packet_id, "slot": slot, "condition": identifiable_slots[(packet_id, slot)]}
                for packet_id, slot in sorted(identifiable_slots)
            ],
            "note": "Structural identifiers are stripped, but content and style can still reveal condition identity.",
        },
        "pilot_diagnostics": pilot_diagnostics,
        "paired_items": paired_items,
        "fingerprints": {
            "protocol": bundle["private"]["protocol_fingerprint"],
            "responses": bundle["private"]["response_set_fingerprint"],
            "packets": bundle["private"]["packet_set_fingerprint"],
            "study": bundle["private"]["study_fingerprint"],
            "answer_key": bundle["private"]["answer_key_fingerprint"],
            "judgments": _fingerprint(validated_judgments),
        },
    }


def _validate_protocol(raw: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(raw, "protocol")
    _require_exact_keys(
        raw,
        {"schema_version", "study_id", "study_mode", "participants", "queries", "dimensions", "safety_gates", "thresholds"},
        "protocol",
    )
    _require_version(raw, PROTOCOL_SCHEMA_VERSION, "protocol")
    study_id = _nonblank(raw["study_id"], "protocol.study_id")
    study_mode = _nonblank(raw["study_mode"], "protocol.study_mode")
    if study_mode not in {"development", "pilot", "confirmatory"}:
        raise ValidationError("protocol.study_mode must be development, pilot, or confirmatory")
    dimensions = _string_list(raw["dimensions"], "protocol.dimensions", nonempty=True)
    if dimensions != list(FIXED_DIMENSIONS):
        raise ValidationError(f"protocol.dimensions must exactly equal {list(FIXED_DIMENSIONS)!r}")

    participants_raw = _require_list(raw["participants"], "protocol.participants", nonempty=True)
    participants: list[dict[str, str]] = []
    participant_ids: set[str] = set()
    for index, row in enumerate(participants_raw):
        where = f"protocol.participants[{index}]"
        _require_mapping(row, where)
        _require_exact_keys(row, {"id"}, where)
        participant_id = _nonblank(row["id"], f"{where}.id")
        if participant_id in participant_ids:
            raise ValidationError(f"duplicate participant id: {participant_id!r}")
        participant_ids.add(participant_id)
        participants.append({"id": participant_id})

    queries_raw = _require_list(raw["queries"], "protocol.queries", nonempty=True)
    queries: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    for index, row in enumerate(queries_raw):
        where = f"protocol.queries[{index}]"
        _require_mapping(row, where)
        _require_exact_keys(
            row,
            {"id", "workstream_id", "stratum", "checkpoint_days", "prompt", "applicable_dimensions"},
            where,
        )
        query_id = _nonblank(row["id"], f"{where}.id")
        if query_id in query_ids:
            raise ValidationError(f"duplicate query id: {query_id!r}")
        query_ids.add(query_id)
        checkpoint = _integer(row["checkpoint_days"], f"{where}.checkpoint_days", minimum=1)
        applicable = _string_list(row["applicable_dimensions"], f"{where}.applicable_dimensions", nonempty=True)
        if len(applicable) != len(set(applicable)) or not set(applicable).issubset(FIXED_DIMENSIONS):
            raise ValidationError(f"{where}.applicable_dimensions must be a unique subset of protocol.dimensions")
        queries.append(
            {
                "id": query_id,
                "workstream_id": _nonblank(row["workstream_id"], f"{where}.workstream_id"),
                "stratum": _nonblank(row["stratum"], f"{where}.stratum"),
                "checkpoint_days": checkpoint,
                "prompt": _nonblank(row["prompt"], f"{where}.prompt"),
                "applicable_dimensions": applicable,
            }
        )

    gates = _string_list(raw["safety_gates"], "protocol.safety_gates", nonempty=True)
    if len(gates) != len(set(gates)):
        raise ValidationError("protocol.safety_gates contains duplicate IDs")
    thresholds = _validate_thresholds(raw["thresholds"], queries)
    primary_queries = [
        query for query in queries if query["checkpoint_days"] == thresholds["primary_checkpoint"]
    ]
    participant_count = len(participants)
    if participant_count < thresholds["min_participants"]:
        raise ValidationError("protocol declares fewer participants than min_participants")
    if participant_count * len(primary_queries) < thresholds["min_items"]:
        raise ValidationError("protocol cannot produce min_items at the primary checkpoint")
    primary_strata = {query["stratum"] for query in primary_queries}
    if not set(thresholds["required_strata"]).issubset(primary_strata):
        raise ValidationError("every required stratum must have a primary-checkpoint query")
    for dimension in FIXED_DIMENSIONS:
        capacity = participant_count * sum(
            dimension in query["applicable_dimensions"] for query in primary_queries
        )
        if capacity < thresholds["min_pairs_per_dimension"]:
            raise ValidationError(
                f"protocol cannot produce min_pairs_per_dimension for {dimension!r} at the primary checkpoint"
            )
    primary_workstreams = {query["workstream_id"] for query in primary_queries}
    if participant_count * len(primary_workstreams) < thresholds["min_clusters"]:
        raise ValidationError("protocol cannot produce min_clusters at the primary checkpoint")
    if study_mode == "confirmatory":
        if len(participants) < 60 or thresholds["min_participants"] < 60:
            raise ValidationError("confirmatory protocols require at least 60 participants")
        if thresholds["min_clusters"] < 60:
            raise ValidationError("confirmatory protocols require min_clusters >= 60")
        if thresholds["min_items"] < 600:
            raise ValidationError("confirmatory protocols require min_items >= 600 primary items")
        if thresholds["min_pairs_per_dimension"] < 300:
            raise ValidationError("confirmatory protocols require min_pairs_per_dimension >= 300")
        if thresholds["min_judges_per_item"] < 2:
            raise ValidationError("confirmatory protocols require min_judges_per_item >= 2")
        if not {30, 90, 180, 365}.issubset(thresholds["required_checkpoints"]) or thresholds["primary_checkpoint"] != 365:
            raise ValidationError("confirmatory protocols require checkpoints 30/90/180/365 and primary checkpoint 365")
        if thresholds["superiority_margin"] < 0.05:
            raise ValidationError("confirmatory protocols require superiority_margin >= 0.05")
        if thresholds["stratum_noninferiority_margin"] < -0.05:
            raise ValidationError("confirmatory protocols require stratum_noninferiority_margin >= -0.05")
        if thresholds["bootstrap_samples"] < 10_000:
            raise ValidationError("confirmatory protocols require bootstrap_samples >= 10000")
        if participant_count * len(queries) < 2_400:
            raise ValidationError("confirmatory protocols require at least 2400 total paired items")
        for checkpoint in thresholds["required_checkpoints"]:
            checkpoint_pairs = participant_count * sum(
                query["checkpoint_days"] == checkpoint for query in queries
            )
            if checkpoint_pairs < 600:
                raise ValidationError(
                    f"confirmatory checkpoint {checkpoint} requires at least 600 paired items"
                )
        for stratum in thresholds["required_strata"]:
            stratum_pairs = participant_count * sum(
                query["stratum"] == stratum for query in primary_queries
            )
            if stratum_pairs < 50 or participant_count < 30:
                raise ValidationError(
                    f"confirmatory primary stratum {stratum!r} requires at least 50 pairs and 30 participants"
                )
        for dimension in FIXED_DIMENSIONS:
            dimension_pairs = participant_count * sum(
                dimension in query["applicable_dimensions"] for query in primary_queries
            )
            if dimension_pairs < 300 or participant_count < 45:
                raise ValidationError(
                    f"confirmatory primary dimension {dimension!r} requires at least 300 pairs and 45 participants"
                )
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "study_id": study_id,
        "study_mode": study_mode,
        "participants": participants,
        "queries": queries,
        "dimensions": list(FIXED_DIMENSIONS),
        "safety_gates": gates,
        "thresholds": thresholds,
    }


def _validate_thresholds(raw: Any, queries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    where = "protocol.thresholds"
    _require_mapping(raw, where)
    expected = {
        "superiority_margin",
        "stratum_noninferiority_margin",
        "score_min",
        "score_max",
        "dimension_minimum",
        "primary_checkpoint",
        "min_items",
        "min_participants",
        "min_clusters",
        "min_judges_per_item",
        "bootstrap_samples",
        "required_strata",
        "required_checkpoints",
        "min_pairs_per_dimension",
    }
    _require_exact_keys(raw, expected, where)
    score_min = _number(raw["score_min"], f"{where}.score_min")
    score_max = _number(raw["score_max"], f"{where}.score_max")
    if score_max <= score_min:
        raise ValidationError("protocol.thresholds.score_max must be greater than score_min")
    dimension_minimum = _number(raw["dimension_minimum"], f"{where}.dimension_minimum")
    if not score_min <= dimension_minimum <= score_max:
        raise ValidationError("protocol.thresholds.dimension_minimum must be within the score scale")
    superiority_margin = _number(raw["superiority_margin"], f"{where}.superiority_margin")
    stratum_margin = _number(raw["stratum_noninferiority_margin"], f"{where}.stratum_noninferiority_margin")
    if not -1.0 <= superiority_margin <= 1.0 or not -1.0 <= stratum_margin <= 1.0:
        raise ValidationError("paired success-rate margins must be between -1 and 1")
    required_strata = _string_list(raw["required_strata"], f"{where}.required_strata", nonempty=True)
    required_checkpoints_raw = _require_list(raw["required_checkpoints"], f"{where}.required_checkpoints", nonempty=True)
    required_checkpoints = [
        _integer(value, f"{where}.required_checkpoints[{index}]", minimum=1)
        for index, value in enumerate(required_checkpoints_raw)
    ]
    if len(required_strata) != len(set(required_strata)) or len(required_checkpoints) != len(set(required_checkpoints)):
        raise ValidationError("required strata/checkpoints must be unique")
    available_strata = {row["stratum"] for row in queries}
    available_checkpoints = {row["checkpoint_days"] for row in queries}
    if not set(required_strata).issubset(available_strata):
        raise ValidationError("protocol.thresholds.required_strata contains an unknown stratum")
    if not set(required_checkpoints).issubset(available_checkpoints):
        raise ValidationError("protocol.thresholds.required_checkpoints contains an unknown checkpoint")
    primary_checkpoint = _integer(raw["primary_checkpoint"], f"{where}.primary_checkpoint", minimum=1)
    if primary_checkpoint not in required_checkpoints:
        raise ValidationError("primary_checkpoint must be one of required_checkpoints")
    return {
        "superiority_margin": superiority_margin,
        "stratum_noninferiority_margin": stratum_margin,
        "score_min": score_min,
        "score_max": score_max,
        "dimension_minimum": dimension_minimum,
        "primary_checkpoint": primary_checkpoint,
        "min_items": _integer(raw["min_items"], f"{where}.min_items", minimum=1),
        "min_participants": _integer(raw["min_participants"], f"{where}.min_participants", minimum=1),
        "min_clusters": _integer(raw["min_clusters"], f"{where}.min_clusters", minimum=1),
        "min_judges_per_item": _integer(raw["min_judges_per_item"], f"{where}.min_judges_per_item", minimum=1),
        "bootstrap_samples": _integer(raw["bootstrap_samples"], f"{where}.bootstrap_samples", minimum=100, maximum=100_000),
        "required_strata": required_strata,
        "required_checkpoints": required_checkpoints,
        "min_pairs_per_dimension": _integer(raw["min_pairs_per_dimension"], f"{where}.min_pairs_per_dimension", minimum=1),
    }


def _validate_response_document(raw: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _validate_response_artifact(raw)
    if artifact["study_id"] != protocol["study_id"]:
        raise ValidationError("responses document study_id does not match protocol")
    participant_ids = {row["id"] for row in protocol["participants"]}
    query_ids = {row["id"] for row in protocol["queries"]}
    normalized = artifact["responses"]
    seen = {
        (row["participant_id"], row["query_id"], row["condition"])
        for row in normalized
    }
    for index, row in enumerate(normalized):
        if row["participant_id"] not in participant_ids or row["query_id"] not in query_ids:
            raise ValidationError(f"responses[{index}] references an unknown participant or query")
    expected = {
        (participant_id, query_id, condition)
        for participant_id in participant_ids
        for query_id in query_ids
        for condition in CONDITIONS
    }
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValidationError(f"response pairs are incomplete or unbalanced; missing={missing!r}, extra={extra!r}")
    return artifact


def _validate_response_artifact(raw: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(raw, "responses document")
    _require_exact_keys(raw, {"schema_version", "study_id", "responses"}, "responses document")
    _require_version(raw, RESPONSE_SCHEMA_VERSION, "responses document")
    study_id = _nonblank(raw["study_id"], "responses document.study_id")
    rows = _require_list(raw["responses"], "responses document.responses", nonempty=True)
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        where = f"responses[{index}]"
        _require_mapping(row, where)
        _require_exact_keys(row, {"participant_id", "query_id", "condition", "response"}, where)
        participant_id = _nonblank(row["participant_id"], f"{where}.participant_id")
        query_id = _nonblank(row["query_id"], f"{where}.query_id")
        condition = _nonblank(row["condition"], f"{where}.condition")
        response = _nonblank(row["response"], f"{where}.response")
        if condition not in CONDITIONS:
            raise ValidationError(f"{where}.condition must be one of {CONDITIONS!r}")
        identity = (participant_id, query_id, condition)
        if identity in seen:
            raise ValidationError(f"duplicate response for participant/query/condition: {identity!r}")
        seen.add(identity)
        normalized.append(
            {"participant_id": participant_id, "query_id": query_id, "condition": condition, "response": response}
        )
    return {"schema_version": RESPONSE_SCHEMA_VERSION, "study_id": study_id, "responses": normalized}


def _validate_and_verify_bundle(protocol: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(raw, "packet bundle")
    _require_exact_keys(raw, {"schema_version", "public", "private"}, "packet bundle")
    _require_version(raw, PACKET_BUNDLE_SCHEMA_VERSION, "packet bundle")
    public = raw["public"]
    private = raw["private"]
    _require_mapping(public, "packet bundle.public")
    _require_mapping(private, "packet bundle.private")
    _require_exact_keys(public, {"schema_version", "study_id", "packet_set_fingerprint", "judge_packets"}, "packet bundle.public")
    _require_exact_keys(
        private,
        {
            "schema_version", "study_id", "seed", "protocol_fingerprint", "response_set_fingerprint",
            "packet_set_fingerprint", "study_fingerprint", "answer_key_fingerprint", "assignment_key",
        },
        "packet bundle.private",
    )
    _require_version(public, PUBLIC_PACKETS_SCHEMA_VERSION, "packet bundle.public")
    _require_version(private, PRIVATE_KEY_SCHEMA_VERSION, "packet bundle.private")
    if public["study_id"] != protocol["study_id"] or private["study_id"] != protocol["study_id"]:
        raise ValidationError("packet bundle study_id does not match protocol")
    seed = private["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValidationError("private seed must be an integer")
    expected_protocol_fp = _fingerprint(protocol)
    if private["protocol_fingerprint"] != expected_protocol_fp:
        raise ValidationError("protocol fingerprint mismatch")

    packets = _require_list(public["judge_packets"], "packet bundle.public.judge_packets", nonempty=True)
    keys = _require_list(private["assignment_key"], "packet bundle.private.assignment_key", nonempty=True)
    packet_ids: set[str] = set()
    normalized_packets: list[dict[str, Any]] = []
    for index, packet in enumerate(packets):
        where = f"judge_packets[{index}]"
        _require_mapping(packet, where)
        _require_exact_keys(packet, {"packet_id", "prompt", "stratum", "checkpoint_days", "applicable_dimensions", "responses"}, where)
        packet_id = _nonblank(packet["packet_id"], f"{where}.packet_id")
        if packet_id in packet_ids:
            raise ValidationError(f"duplicate packet id: {packet_id!r}")
        packet_ids.add(packet_id)
        responses = _validate_slot_mapping(packet["responses"], f"{where}.responses", _validate_public_response)
        normalized_packets.append(
            {
                "packet_id": packet_id,
                "prompt": _nonblank(packet["prompt"], f"{where}.prompt"),
                "stratum": _nonblank(packet["stratum"], f"{where}.stratum"),
                "checkpoint_days": _integer(packet["checkpoint_days"], f"{where}.checkpoint_days", minimum=1),
                "applicable_dimensions": _string_list(packet["applicable_dimensions"], f"{where}.applicable_dimensions", nonempty=True),
                "responses": responses,
            }
        )

    query_by_fp = {_fingerprint({"query": query}): query for query in protocol["queries"]}
    participant_refs = {
        _opaque_token(seed, protocol["study_id"], "participant", row["id"]): row["id"]
        for row in protocol["participants"]
    }
    normalized_keys: list[dict[str, Any]] = []
    key_ids: set[str] = set()
    frozen_responses: list[dict[str, str]] = []
    packet_lookup = {row["packet_id"]: row for row in normalized_packets}
    observed_pairs: set[tuple[str, str]] = set()
    for index, key in enumerate(keys):
        where = f"assignment_key[{index}]"
        _require_mapping(key, where)
        _require_exact_keys(key, {"packet_id", "query_fingerprint", "participant_ref", "cluster_id", "slots"}, where)
        packet_id = _nonblank(key["packet_id"], f"{where}.packet_id")
        if packet_id in key_ids:
            raise ValidationError(f"duplicate answer-key packet id: {packet_id!r}")
        key_ids.add(packet_id)
        query_fp = _nonblank(key["query_fingerprint"], f"{where}.query_fingerprint")
        participant_ref = _nonblank(key["participant_ref"], f"{where}.participant_ref")
        cluster_id = _nonblank(key["cluster_id"], f"{where}.cluster_id")
        if query_fp not in query_by_fp or participant_ref not in participant_refs:
            raise ValidationError("answer key references an unknown query or participant reference")
        query = query_by_fp[query_fp]
        participant_id = participant_refs[participant_ref]
        expected_cluster = _opaque_token(seed, protocol["study_id"], "participant-workstream", participant_id, query["workstream_id"])
        if cluster_id != expected_cluster:
            raise ValidationError("answer-key cluster token mismatch")
        pair = (participant_ref, query_fp)
        if pair in observed_pairs:
            raise ValidationError("answer key contains duplicate participant/query pairs")
        observed_pairs.add(pair)
        slots = _validate_slot_mapping(key["slots"], f"{where}.slots", _validate_key_slot)
        if {slots[slot]["condition"] for slot in SLOTS} != set(CONDITIONS):
            raise ValidationError(f"{where}.slots must contain one human and one memory_v2 assignment")
        public_packet = packet_lookup.get(packet_id)
        if public_packet is None:
            raise ValidationError("answer key and public packets have different packet IDs")
        if (
            public_packet["prompt"] != query["prompt"]
            or public_packet["stratum"] != query["stratum"]
            or public_packet["checkpoint_days"] != query["checkpoint_days"]
            or public_packet["applicable_dimensions"] != query["applicable_dimensions"]
        ):
            raise ValidationError("public packet query metadata does not match frozen protocol")
        for slot in SLOTS:
            public_response = public_packet["responses"][slot]
            key_slot = slots[slot]
            if public_response["response_id"] != key_slot["response_id"]:
                raise ValidationError("public response ID does not match answer-key assignment")
            frozen_responses.append(
                {
                    "participant_ref": participant_ref,
                    "query_fingerprint": query_fp,
                    "condition": key_slot["condition"],
                    "response_id": key_slot["response_id"],
                    "response": public_response["text"],
                }
            )
        normalized_keys.append(
            {"packet_id": packet_id, "query_fingerprint": query_fp, "participant_ref": participant_ref, "cluster_id": cluster_id, "slots": slots}
        )
    if packet_ids != key_ids:
        raise ValidationError("answer key and public packets have different packet IDs")
    expected_pairs = {
        (participant_ref, query_fp)
        for participant_ref in participant_refs
        for query_fp in query_by_fp
    }
    if observed_pairs != expected_pairs:
        raise ValidationError("packet bundle is missing or has unbalanced participant/query pairs")

    normalized_public = {
        "schema_version": PUBLIC_PACKETS_SCHEMA_VERSION,
        "study_id": protocol["study_id"],
        "packet_set_fingerprint": public["packet_set_fingerprint"],
        "judge_packets": normalized_packets,
    }
    expected_packet_fp = _fingerprint({"study_id": protocol["study_id"], "judge_packets": normalized_packets})
    if public["packet_set_fingerprint"] != expected_packet_fp or private["packet_set_fingerprint"] != expected_packet_fp:
        raise ValidationError("packet-set fingerprint mismatch")
    expected_response_fp = _fingerprint(sorted(frozen_responses, key=_canonical_json))
    if private["response_set_fingerprint"] != expected_response_fp:
        raise ValidationError("response-set fingerprint mismatch")
    expected_study_fp = _fingerprint(
        {
            "study_id": protocol["study_id"],
            "protocol_fingerprint": expected_protocol_fp,
            "response_set_fingerprint": expected_response_fp,
            "packet_set_fingerprint": expected_packet_fp,
        }
    )
    if private["study_fingerprint"] != expected_study_fp:
        raise ValidationError("study fingerprint mismatch")
    private_without_fp = dict(private)
    answer_key_fp = private_without_fp.pop("answer_key_fingerprint")
    if answer_key_fp != _fingerprint(private_without_fp):
        raise ValidationError("answer-key fingerprint mismatch")
    normalized_private = dict(private)
    normalized_private["assignment_key"] = normalized_keys
    return {"schema_version": PACKET_BUNDLE_SCHEMA_VERSION, "public": normalized_public, "private": normalized_private}


def _validate_judgments_for_study(protocol: Mapping[str, Any], bundle: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    # Accept an already-loaded plain mapping while applying the same generic checks.
    _require_mapping(raw, "judgments document")
    _require_exact_keys(raw, {"schema_version", "study_id", "judgments"}, "judgments document")
    _require_version(raw, JUDGMENTS_SCHEMA_VERSION, "judgments document")
    if raw["study_id"] != protocol["study_id"]:
        raise ValidationError("judgments study_id does not match protocol")
    packets = {row["packet_id"]: row for row in bundle["public"]["judge_packets"]}
    allowed_gates = set(protocol["safety_gates"])
    score_min = protocol["thresholds"]["score_min"]
    score_max = protocol["thresholds"]["score_max"]
    rows = _require_list(raw["judgments"], "judgments document.judgments")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    counts: dict[str, int] = defaultdict(int)
    for index, row in enumerate(rows):
        where = f"judgments[{index}]"
        _require_mapping(row, where)
        _require_exact_keys(row, {"packet_id", "judge_id", "judge_role", "scores", "gate_failures", "condition_guesses", "material_error_notes", "potentially_identifiable"}, where)
        packet_id = _nonblank(row["packet_id"], f"{where}.packet_id")
        judge_id = _nonblank(row["judge_id"], f"{where}.judge_id")
        judge_role = _validate_judge_role(row["judge_role"], f"{where}.judge_role")
        if packet_id not in packets:
            raise ValidationError(f"judgment references unknown packet_id: {packet_id!r}")
        identity = (packet_id, judge_id)
        if identity in seen:
            raise ValidationError(f"duplicate judgment for packet_id={packet_id!r}, judge_id={judge_id!r}")
        seen.add(identity)
        if judge_role == "primary":
            counts[packet_id] += 1
        applicable = set(packets[packet_id]["applicable_dimensions"])
        scores = _validate_slot_mapping(row["scores"], f"{where}.scores", _validate_generic_scores)
        failures = _validate_slot_mapping(row["gate_failures"], f"{where}.gate_failures", _validate_generic_gate_failures)
        guesses = _validate_slot_mapping(row["condition_guesses"], f"{where}.condition_guesses", _validate_condition_guess)
        error_notes = _validate_slot_mapping(
            row["material_error_notes"], f"{where}.material_error_notes", _validate_material_error_note
        )
        identifiable = _validate_slot_mapping(
            row["potentially_identifiable"], f"{where}.potentially_identifiable", _validate_bool
        )
        for slot in SLOTS:
            if set(scores[slot]) != applicable:
                raise ValidationError(f"{where}.scores.{slot} must contain exactly the packet's applicable dimensions")
            for dimension, value in scores[slot].items():
                if not score_min <= value <= score_max:
                    raise ValidationError(f"{where}.scores.{slot}.{dimension} is outside the protocol score scale")
            unknown_gates = set(failures[slot]) - allowed_gates
            if unknown_gates:
                raise ValidationError(f"{where}.gate_failures.{slot} contains unknown gates: {sorted(unknown_gates)!r}")
            is_material_error = bool(failures[slot]) or any(
                value < protocol["thresholds"]["dimension_minimum"]
                for value in scores[slot].values()
            )
            if is_material_error and error_notes[slot] is None:
                raise ValidationError(
                    f"{where}.material_error_notes.{slot} is required for a below-threshold score or safety failure"
                )
        normalized.append({"packet_id": packet_id, "judge_id": judge_id, "judge_role": judge_role, "scores": scores, "gate_failures": failures, "condition_guesses": guesses, "material_error_notes": error_notes, "potentially_identifiable": identifiable})
    minimum = protocol["thresholds"]["min_judges_per_item"]
    missing = sorted(packet_id for packet_id in packets if counts[packet_id] < minimum)
    if missing:
        raise ValidationError(f"incomplete judgments: packets below min_judges_per_item={minimum}: {missing!r}")
    rows_by_packet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        rows_by_packet[row["packet_id"]].append(row)
    primary_checkpoint = protocol["thresholds"]["primary_checkpoint"]
    dimension_minimum = protocol["thresholds"]["dimension_minimum"]
    for packet_id, packet in packets.items():
        packet_rows = sorted(rows_by_packet[packet_id], key=lambda row: row["judge_id"])
        adjudicators = [row for row in packet_rows if row["judge_role"] == "adjudicator"]
        if len(adjudicators) > 1:
            raise ValidationError(
                f"packet {packet_id!r} has more than one adjudicator"
            )
        if packet["checkpoint_days"] != primary_checkpoint:
            if adjudicators:
                raise ValidationError(
                    f"packet {packet_id!r} includes an adjudicator outside the primary checkpoint"
                )
            continue
        primary_rows = [row for row in packet_rows if row["judge_role"] == "primary"]
        if len(primary_rows) < 2:
            continue
        has_disagreement = False
        for slot in SLOTS:
            judge_successes = [
                not judge["gate_failures"][slot]
                and all(
                    judge["scores"][slot][dimension] >= dimension_minimum
                    for dimension in packet["applicable_dimensions"]
                )
                for judge in primary_rows
            ]
            has_disagreement = has_disagreement or len(set(judge_successes)) > 1
        if has_disagreement and not adjudicators:
            raise ValidationError(
                f"incomplete adjudication: packet {packet_id!r} has a primary binary-success disagreement and requires a blinded adjudicator"
            )
        if adjudicators and not has_disagreement:
            raise ValidationError(
                f"packet {packet_id!r} includes an adjudicator without a primary binary-success disagreement"
            )
    return {"schema_version": JUDGMENTS_SCHEMA_VERSION, "study_id": protocol["study_id"], "judgments": normalized}


def _cluster_bootstrap(items: Sequence[Mapping[str, Any]], *, samples: int, seed: int) -> tuple[float, dict[str, float]]:
    if not items:
        return 0.0, {"level": 0.95, "lower": 0.0, "upper": 0.0, "method": "participant_cluster_percentile_bootstrap", "samples": samples}
    by_participant: dict[str, list[float]] = defaultdict(list)
    for item in items:
        by_participant[item["participant_ref"]].append(float(item["paired_success_delta"]))
    participants = sorted(by_participant)
    point = sum(value for values in by_participant.values() for value in values) / len(items)
    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(samples):
        selected = [participants[rng.randrange(len(participants))] for _ in participants]
        values = [value for participant in selected for value in by_participant[participant]]
        replicates.append(sum(values) / len(values))
    replicates.sort()
    return point, {
        "level": 0.95,
        "lower": _percentile(replicates, 0.025),
        "upper": _percentile(replicates, 0.975),
        "method": "participant_cluster_percentile_bootstrap",
        "samples": samples,
    }


def _item_difficulty_diagnostics(
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize item performance without exposing protocol query identifiers."""

    by_query: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        by_query[item["query_fingerprint"]].append(item)

    diagnostics: list[dict[str, Any]] = []
    for query_fingerprint, rows in sorted(by_query.items()):
        human_successes = [float(row["human"]["success"]) for row in rows]
        memory_successes = [float(row["memory_v2"]["success"]) for row in rows]
        human_scores = [
            statistics.mean(row["human"]["dimension_means"].values())
            for row in rows
        ]
        memory_scores = [
            statistics.mean(row["memory_v2"]["dimension_means"].values())
            for row in rows
        ]
        disagreement_slots = sum(
            int(row[condition]["judge_disagreement"])
            for row in rows
            for condition in CONDITIONS
        )
        diagnostics.append(
            {
                "query_fingerprint": query_fingerprint,
                "checkpoint_days": rows[0]["checkpoint_days"],
                "stratum": rows[0]["stratum"],
                "paired_items": len(rows),
                "human_success_rate": statistics.mean(human_successes),
                "memory_v2_success_rate": statistics.mean(memory_successes),
                "paired_success_delta": statistics.mean(
                    row["paired_success_delta"] for row in rows
                ),
                "combined_success_rate": statistics.mean(
                    [*human_successes, *memory_successes]
                ),
                "human_mean_rubric_score": statistics.mean(human_scores),
                "memory_v2_mean_rubric_score": statistics.mean(memory_scores),
                "judge_disagreement_slots": disagreement_slots,
                "judge_disagreement_rate": disagreement_slots / (2 * len(rows)),
                "note": "Lower success rates indicate a more difficult item.",
            }
        )
    return diagnostics


def _participant_cluster_diagnostics(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Estimate participant-level heterogeneity for pilot power planning."""

    by_participant: dict[str, list[float]] = defaultdict(list)
    for item in items:
        by_participant[item["participant_ref"]].append(
            float(item["paired_success_delta"])
        )
    if not by_participant:
        return {
            "participants": 0,
            "items": 0,
            "items_per_participant_min": 0,
            "items_per_participant_max": 0,
            "mean_paired_success_delta": None,
            "participant_mean_sample_variance": None,
            "participant_mean_standard_deviation": None,
            "participant_mean_standard_error": None,
            "item_level_sample_variance": None,
            "intraclass_correlation": None,
            "between_participant_variance": None,
            "within_participant_variance": None,
            "balanced_design": True,
            "method": "participant_mean_and_one_way_random_effects",
        }

    participant_means = [statistics.mean(values) for values in by_participant.values()]
    all_values = [value for values in by_participant.values() for value in values]
    counts = [len(values) for values in by_participant.values()]
    participant_variance = (
        statistics.variance(participant_means) if len(participant_means) > 1 else None
    )
    participant_stddev = (
        math.sqrt(participant_variance) if participant_variance is not None else None
    )
    participant_standard_error = (
        participant_stddev / math.sqrt(len(participant_means))
        if participant_stddev is not None
        else None
    )

    balanced = min(counts) == max(counts)
    intraclass_correlation: float | None = None
    between_participant_variance: float | None = None
    within_participant_variance: float | None = None
    if balanced and len(participant_means) > 1 and counts[0] > 1:
        item_count = counts[0]
        grand_mean = statistics.mean(all_values)
        between_sum_squares = item_count * sum(
            (mean - grand_mean) ** 2 for mean in participant_means
        )
        between_mean_square = between_sum_squares / (len(participant_means) - 1)
        within_sum_squares = sum(
            sum((value - statistics.mean(values)) ** 2 for value in values)
            for values in by_participant.values()
        )
        within_mean_square = within_sum_squares / (
            len(participant_means) * (item_count - 1)
        )
        denominator = between_mean_square + (item_count - 1) * within_mean_square
        intraclass_correlation = (
            (between_mean_square - within_mean_square) / denominator
            if denominator
            else 0.0
        )
        between_participant_variance = max(
            (between_mean_square - within_mean_square) / item_count,
            0.0,
        )
        within_participant_variance = within_mean_square

    return {
        "participants": len(participant_means),
        "items": len(all_values),
        "items_per_participant_min": min(counts),
        "items_per_participant_max": max(counts),
        "mean_paired_success_delta": statistics.mean(all_values),
        "participant_mean_sample_variance": participant_variance,
        "participant_mean_standard_deviation": participant_stddev,
        "participant_mean_standard_error": participant_standard_error,
        "item_level_sample_variance": (
            statistics.variance(all_values) if len(all_values) > 1 else None
        ),
        "intraclass_correlation": intraclass_correlation,
        "between_participant_variance": between_participant_variance,
        "within_participant_variance": within_participant_variance,
        "balanced_design": balanced,
        "method": "participant_mean_and_one_way_random_effects",
    }


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _load_document(path: str | Path) -> Any:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix not in {".json", ".yaml", ".yml"}:
        raise ValidationError("input file must use .json, .yaml, or .yml")
    try:
        size = input_path.stat().st_size
    except OSError as exc:
        raise ValidationError(f"cannot read input file: {exc}") from exc
    if size > MAX_INPUT_BYTES:
        raise ValidationError(f"input file exceeds {MAX_INPUT_BYTES} bytes")
    try:
        text = input_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"cannot read UTF-8 input file: {exc}") from exc
    try:
        if suffix == ".json":
            return json.loads(text, object_pairs_hook=_json_object, parse_constant=_reject_json_constant)
        return _safe_yaml_load(text)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"invalid {suffix.lstrip('.')} input: {exc}") from exc


def _safe_yaml_load(text: str) -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - package dependency in supported installs
        raise ValidationError("YAML input requires PyYAML; use JSON instead") from exc

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
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
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


def _fingerprint(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _opaque_token(seed: int, study_id: str, namespace: str, *parts: str) -> str:
    payload = _canonical_json([study_id, str(seed), namespace, *parts])
    return "opaque:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bootstrap_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}|bootstrap|{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _validate_slot_mapping(raw: Any, where: str, validator: Any) -> dict[str, Any]:
    _require_mapping(raw, where)
    _require_exact_keys(raw, set(SLOTS), where)
    return {slot: validator(raw[slot], f"{where}.{slot}") for slot in SLOTS}


def _validate_generic_scores(raw: Any, where: str) -> dict[str, float]:
    _require_mapping(raw, where)
    if not raw:
        raise ValidationError(f"{where} must not be empty")
    unknown = set(raw) - set(FIXED_DIMENSIONS)
    if unknown:
        raise ValidationError(f"{where} contains unknown dimensions: {sorted(unknown)!r}")
    return {dimension: _number(value, f"{where}.{dimension}") for dimension, value in raw.items()}


def _validate_generic_gate_failures(raw: Any, where: str) -> list[str]:
    values = _string_list(raw, where)
    if len(values) != len(set(values)):
        raise ValidationError(f"{where} contains duplicate gate IDs")
    return values


def _validate_condition_guess(raw: Any, where: str) -> str:
    guess = _nonblank(raw, where)
    if guess not in {"human", "memory_v2", "unsure"}:
        raise ValidationError(f"{where} must be human, memory_v2, or unsure")
    return guess


def _validate_judge_role(raw: Any, where: str) -> str:
    role = _nonblank(raw, where)
    if role not in {"primary", "adjudicator"}:
        raise ValidationError(f"{where} must be primary or adjudicator")
    return role


def _validate_material_error_note(raw: Any, where: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    _require_mapping(raw, where)
    _require_exact_keys(raw, {"reason", "evidence_refs"}, where)
    return {
        "reason": _nonblank(raw["reason"], f"{where}.reason"),
        "evidence_refs": _string_list(
            raw["evidence_refs"], f"{where}.evidence_refs", nonempty=True
        ),
    }


def _validate_bool(raw: Any, where: str) -> bool:
    if not isinstance(raw, bool):
        raise ValidationError(f"{where} must be a boolean")
    return raw


def _validate_public_response(raw: Any, where: str) -> dict[str, str]:
    _require_mapping(raw, where)
    _require_exact_keys(raw, {"response_id", "text"}, where)
    return {"response_id": _nonblank(raw["response_id"], f"{where}.response_id"), "text": _nonblank(raw["text"], f"{where}.text")}


def _validate_key_slot(raw: Any, where: str) -> dict[str, str]:
    _require_mapping(raw, where)
    _require_exact_keys(raw, {"response_id", "condition"}, where)
    condition = _nonblank(raw["condition"], f"{where}.condition")
    if condition not in CONDITIONS:
        raise ValidationError(f"{where}.condition must be one of {CONDITIONS!r}")
    return {"response_id": _nonblank(raw["response_id"], f"{where}.response_id"), "condition": condition}


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{where} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValidationError(f"{where} keys must be strings")
    return value


def _require_list(value: Any, where: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{where} must be an array")
    if nonempty and not value:
        raise ValidationError(f"{where} must not be empty")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValidationError(f"{where} keys must be exactly {sorted(expected)!r}; missing={sorted(expected - actual)!r}, unknown={sorted(actual - expected)!r}")


def _require_version(value: Mapping[str, Any], expected: str, where: str) -> None:
    if value.get("schema_version") != expected:
        raise ValidationError(f"{where}.schema_version must equal {expected!r}")


def _nonblank(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{where} must be a nonblank string")
    return value.strip()


def _string_list(value: Any, where: str, *, nonempty: bool = False) -> list[str]:
    rows = _require_list(value, where, nonempty=nonempty)
    return [_nonblank(row, f"{where}[{index}]") for index, row in enumerate(rows)]


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValidationError(f"{where} must be a finite number")
    return float(value)


def _integer(value: Any, where: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{where} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"between {minimum} and {maximum}" if maximum is not None else f">= {minimum}"
        raise ValidationError(f"{where} must be {bound}")
    return value


def _count_values(rows: Sequence[Mapping[str, Any]], key: str) -> dict[Any, int]:
    counts: dict[Any, int] = defaultdict(int)
    for row in rows:
        counts[row[key]] += 1
    return dict(sorted(counts.items(), key=lambda item: str(item[0])))


def _mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


__all__ = [
    "CONDITIONS",
    "FIXED_DIMENSIONS",
    "JUDGMENTS_SCHEMA_VERSION",
    "PACKET_BUNDLE_SCHEMA_VERSION",
    "PRIVATE_KEY_SCHEMA_VERSION",
    "PROTOCOL_SCHEMA_VERSION",
    "PUBLIC_PACKETS_SCHEMA_VERSION",
    "RESPONSE_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "ValidationError",
    "load_judgments",
    "load_protocol",
    "load_responses",
    "prepare_blinded_packets",
    "score_study",
]
