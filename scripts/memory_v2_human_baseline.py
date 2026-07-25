#!/usr/bin/env python3
"""Run the preregistered Memory v2 human-baseline workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# Direct script execution sets ``sys.path[0]`` to ``scripts/``. Bootstrap the
# checkout so this operator workflow never resolves another Hermes install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plugins.memory.memory_v2.evals.human_baseline import (
    PACKET_BUNDLE_SCHEMA_VERSION,
    ValidationError,
    audit_protocol_disjointness,
    load_judgments,
    load_protocol,
    load_responses,
    prepare_blinded_packets,
    score_study,
)

EXIT_SUCCESS = 0
EXIT_NO_CLAIM = 1
EXIT_INVALID = 2


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _validate(args)
        if args.command == "prepare":
            return _prepare(args)
        if args.command == "score":
            return _score(args)
        if args.command == "audit-disjoint":
            return _audit_disjoint(args)
        parser.error("a subcommand is required")
    except (
        ValidationError,
        ValueError,
        OSError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    return EXIT_INVALID


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and score a blinded Memory v2 human-baseline study."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="Validate a preregistered human-baseline protocol."
    )
    validate.add_argument("--protocol", required=True, help="Protocol JSON or YAML path.")
    validate.add_argument("--output", default="", help="Optional JSON validation report path.")

    prepare = subparsers.add_parser(
        "prepare", help="Create deterministic public judge packets and a separate private key."
    )
    prepare.add_argument("--protocol", required=True, help="Validated protocol JSON or YAML path.")
    prepare.add_argument(
        "--human-responses", required=True, help="Human response JSON or YAML path."
    )
    prepare.add_argument(
        "--memory-responses", required=True, help="Memory v2 response JSON or YAML path."
    )
    prepare.add_argument(
        "--packet-output", required=True, help="Public, condition-blinded judge packet path."
    )
    prepare.add_argument(
        "--key-output", required=True, help="Private condition-assignment key path."
    )
    prepare.add_argument(
        "--seed",
        required=True,
        type=int,
        help="Sealed integer blinding seed; preregister it and reveal only after judgments freeze.",
    )

    score = subparsers.add_parser(
        "score", help="Score completed blinded judgments against the private assignment key."
    )
    score.add_argument("--protocol", required=True, help="Preregistered protocol path.")
    score.add_argument("--packets", required=True, help="Public judge packet path.")
    score.add_argument("--key", required=True, help="Private assignment-key path.")
    score.add_argument("--judgments", required=True, help="Completed judgments JSON or YAML path.")
    score.add_argument("--output", default="", help="Write the JSON result to this path.")
    score.add_argument(
        "--require-superiority",
        action="store_true",
        help="Exit 1 when a valid completed study reaches a no-claim decision.",
    )

    disjoint = subparsers.add_parser(
        "audit-disjoint",
        help="Fail when a candidate protocol overlaps a development or prior-study protocol.",
    )
    disjoint.add_argument("--candidate", required=True, help="Candidate pilot protocol path.")
    disjoint.add_argument(
        "--against",
        required=True,
        action="append",
        help="Reference protocol path; repeat for every development, pilot, or confirmation pool.",
    )
    disjoint.add_argument("--output", default="", help="Optional JSON audit report path.")
    return parser


def _validate(args: argparse.Namespace) -> int:
    if args.output:
        _require_json_path(args.output, label="validation output")
        _ensure_output_does_not_replace_inputs(args.output, args.protocol)
    protocol = load_protocol(args.protocol)
    payload = {
        "valid": True,
        "schema_version": protocol["schema_version"],
        "study_id": protocol["study_id"],
        "study_mode": protocol["study_mode"],
        "participant_count": len(protocol["participants"]),
        "query_count": len(protocol["queries"]),
    }
    _emit_json(payload, args.output)
    return EXIT_SUCCESS


def _prepare(args: argparse.Namespace) -> int:
    packet_output = Path(args.packet_output).expanduser().resolve()
    key_output = Path(args.key_output).expanduser().resolve()
    _require_json_path(packet_output, label="public packet output")
    _require_json_path(key_output, label="private key output")
    if packet_output == key_output:
        raise ValidationError("--packet-output and --key-output must be different files")
    input_paths = (args.protocol, args.human_responses, args.memory_responses)
    _ensure_output_does_not_replace_inputs(packet_output, *input_paths)
    _ensure_output_does_not_replace_inputs(key_output, *input_paths)

    protocol = load_protocol(args.protocol)
    human_responses = _load_response_document(
        args.human_responses,
        study_id=protocol["study_id"],
        expected_condition="human",
    )
    memory_responses = _load_response_document(
        args.memory_responses,
        study_id=protocol["study_id"],
        expected_condition="memory_v2",
    )
    if human_responses["schema_version"] != memory_responses["schema_version"]:
        raise ValidationError("human and Memory v2 response schemas do not match")
    responses = {
        "schema_version": human_responses["schema_version"],
        "study_id": protocol["study_id"],
        "responses": [
            *human_responses["responses"],
            *memory_responses["responses"],
        ],
    }
    bundle = prepare_blinded_packets(
        protocol,
        responses,
        seed=args.seed,
    )
    # The core deliberately separates publishable packet data from the sealed
    # seed, identities, assignments, and preregistration fingerprints.
    public_bundle = bundle["public"]
    private_key = bundle["private"]
    _assert_public_bundle_is_blind(
        public_bundle,
        participant_ids={row["id"] for row in protocol["participants"]},
    )
    _assert_private_key_uses_opaque_clusters(
        private_key,
        participant_ids={row["id"] for row in protocol["participants"]},
    )
    _write_json_atomic(packet_output, public_bundle)
    _write_json_atomic(key_output, private_key, private=True)
    print(
        json.dumps(
            {
                "prepared": True,
                "study_id": public_bundle["study_id"],
                "packet_count": len(public_bundle["judge_packets"]),
                "packet_output": str(packet_output),
                "key_output": str(key_output),
            },
            sort_keys=True,
        )
    )
    return EXIT_SUCCESS


def _score(args: argparse.Namespace) -> int:
    if args.output:
        _require_json_path(args.output, label="score output")
        _ensure_output_does_not_replace_inputs(
            args.output, args.protocol, args.packets, args.key, args.judgments
        )
    protocol = load_protocol(args.protocol)
    public_bundle = _load_mapping(args.packets, label="judge packets")
    private_key = _load_mapping(args.key, label="assignment key")
    _validate_bundle_halves(protocol, public_bundle, private_key)
    _assert_public_bundle_is_blind(
        public_bundle,
        participant_ids={row["id"] for row in protocol["participants"]},
    )
    _assert_private_key_uses_opaque_clusters(
        private_key,
        participant_ids={row["id"] for row in protocol["participants"]},
    )
    bundle = {
        "schema_version": PACKET_BUNDLE_SCHEMA_VERSION,
        "public": public_bundle,
        "private": private_key,
    }
    judgments = load_judgments(args.judgments)
    result = score_study(protocol, bundle, judgments)
    _emit_json(result, args.output)
    if args.require_superiority and not _result_supports_claim(result):
        return EXIT_NO_CLAIM
    return EXIT_SUCCESS


def _audit_disjoint(args: argparse.Namespace) -> int:
    if args.output:
        _require_json_path(args.output, label="disjointness output")
        _ensure_output_does_not_replace_inputs(
            args.output, args.candidate, *args.against
        )
    candidate = load_protocol(args.candidate)
    if candidate["study_mode"] != "pilot":
        raise ValidationError("audit-disjoint candidate must use study_mode=pilot")
    references = [load_protocol(path) for path in args.against]
    report = audit_protocol_disjointness(candidate, references)
    _emit_json(report, args.output)
    return EXIT_SUCCESS if report["disjoint"] else EXIT_NO_CLAIM


def _load_response_document(
    path: str, *, study_id: str, expected_condition: str
) -> dict[str, Any]:
    payload = load_responses(path)
    if payload["study_id"] != study_id:
        raise ValidationError(f"{path}: study_id does not match the protocol")
    rows = payload["responses"]
    if not isinstance(rows, list) or not rows:
        raise ValidationError(f"{path}: expected a non-empty responses list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValidationError(f"{path}: responses[{index}] must be an object")
        if row.get("condition") != expected_condition:
            raise ValidationError(
                f"{path}: responses[{index}].condition must be {expected_condition!r}"
            )
    return payload


def _load_mapping(path: str, *, label: str) -> dict[str, Any]:
    payload = _load_structured(path)
    if not isinstance(payload, dict):
        raise ValidationError(f"{path}: {label} must be an object")
    return payload


def _load_structured(path: str) -> Any:
    source = Path(path).expanduser()
    if not source.is_file():
        raise ValidationError(f"file not found: {source}")
    if source.suffix.lower() != ".json":
        raise ValidationError(f"{source}: generated packet and key artifacts must use .json")
    if source.stat().st_size > 10 * 1024 * 1024:
        raise ValidationError(f"{source}: input file exceeds 10485760 bytes")
    text = source.read_text(encoding="utf-8")
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonfinite_json,
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValidationError(f"non-finite JSON constant is not allowed: {value}")


def _ensure_output_does_not_replace_inputs(output: str | Path, *inputs: str) -> None:
    output_path = Path(output).expanduser().resolve()
    input_paths = {Path(path).expanduser().resolve() for path in inputs}
    if output_path in input_paths:
        raise ValidationError("an output path must not replace an input artifact")


def _require_json_path(path: str | Path, *, label: str) -> None:
    if Path(path).suffix.lower() != ".json":
        raise ValidationError(f"{label} must use a .json filename")


def _validate_bundle_halves(
    protocol: dict[str, Any],
    public_bundle: dict[str, Any],
    private_key: dict[str, Any],
) -> None:
    for field in ("schema_version", "study_id", "packet_set_fingerprint", "judge_packets"):
        if field not in public_bundle:
            raise ValidationError(f"public judge packet file is missing {field}")
    for field in (
        "schema_version",
        "study_id",
        "seed",
        "protocol_fingerprint",
        "response_set_fingerprint",
        "packet_set_fingerprint",
        "study_fingerprint",
        "answer_key_fingerprint",
        "assignment_key",
    ):
        if field not in private_key:
            raise ValidationError(f"private assignment-key file is missing {field}")
    if any(
        field in public_bundle
        for field in (
            "seed",
            "protocol_fingerprint",
            "response_set_fingerprint",
            "study_fingerprint",
            "answer_key_fingerprint",
            "assignment_key",
        )
    ):
        raise ValidationError("public judge packet file contains private blinding data")
    if "judge_packets" in private_key:
        raise ValidationError("private assignment-key file unexpectedly contains judge packets")
    for field in ("study_id", "packet_set_fingerprint"):
        if public_bundle[field] != private_key[field]:
            raise ValidationError(f"public packets and private key disagree on {field}")
    if public_bundle["study_id"] != protocol["study_id"]:
        raise ValidationError("packet study_id does not match the protocol")


def _assert_public_bundle_is_blind(
    payload: dict[str, Any], *, participant_ids: set[str]
) -> None:
    forbidden_keys = {
        "assignment_key",
        "condition",
        "participant_id",
        "query_id",
        "slots",
        "workstream_id",
    }
    participant_fragments = {value.casefold() for value in participant_ids}

    def contains_participant_identity(value: str) -> bool:
        folded = value.casefold()
        return any(
            folded == fragment or (len(fragment) >= 4 and fragment in folded)
            for fragment in participant_fragments
        )

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            leaked_keys = forbidden_keys.intersection(value)
            if leaked_keys:
                raise ValidationError(
                    "public judge packet leaks private fields: "
                    + ", ".join(sorted(leaked_keys))
                )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            if value in {"human", "memory_v2"} or contains_participant_identity(value):
                raise ValidationError(
                    "public judge packet leaks a condition or participant identity"
                )

    visit(payload)


def _assert_private_key_uses_opaque_clusters(
    payload: dict[str, Any], *, participant_ids: set[str]
) -> None:
    participant_fragments = {value.casefold() for value in participant_ids}

    def contains_participant_identity(value: str) -> bool:
        folded = value.casefold()
        return any(
            folded == fragment or (len(fragment) >= 4 and fragment in folded)
            for fragment in participant_fragments
        )

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "participant_id" in value:
                raise ValidationError("private assignment key contains raw participant_id fields")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and contains_participant_identity(value):
            raise ValidationError("private assignment key contains a raw participant identity")

    visit(payload)


def _result_supports_claim(result: dict[str, Any]) -> bool:
    return bool(result.get("superiority_claim")) or result.get("decision") == "superiority"


def _emit_json(payload: dict[str, Any], output: str) -> None:
    if output:
        _write_json_atomic(Path(output).expanduser().resolve(), payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


def _write_json_atomic(path: Path, payload: dict[str, Any], *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        if private:
            os.chmod(temp_path, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        if private:
            os.chmod(path, 0o600)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
