#!/usr/bin/env python3
"""Prepare and score the private, offline Memory v2 Earn-the-Canary study."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plugins.memory.memory_v2.evals.earn_canary import (
    ValidationError,
    audit_study_disjointness,
    load_judgments,
    load_protocol,
    load_responses,
    prepare_blinded_packets,
    score_study,
)


EXIT_SUCCESS = 0
EXIT_NO_GO = 1
EXIT_INVALID = 2
_MAX_JSON_BYTES = 20 * 1024 * 1024
_MAX_JSON_DEPTH = 64


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _validate(args)
        if args.command == "audit-disjoint":
            return _audit_disjoint(args)
        if args.command == "prepare":
            return _prepare(args)
        if args.command == "score":
            return _score(args)
        parser.error("a subcommand is required")
    except (
        ValidationError,
        ValueError,
        OSError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {_safe_error_message(exc)}", file=sys.stderr)
        return EXIT_INVALID
    return EXIT_INVALID


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a private, opt-in, four-arm Memory v2 longitudinal study. "
            "This command is offline and has no mutation authority."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="Validate a preregistered Earn-the-Canary protocol."
    )
    validate.add_argument(
        "--protocol", required=True, help="Protocol JSON or YAML path."
    )
    validate.add_argument(
        "--output", default="", help="Optional immutable JSON report."
    )

    disjoint = subparsers.add_parser(
        "audit-disjoint",
        help="Audit a candidate protocol against every prior study pool.",
    )
    disjoint.add_argument("--candidate", required=True, help="Candidate protocol path.")
    disjoint.add_argument(
        "--against",
        action="append",
        required=True,
        help="Prior protocol path; repeat for every development, pilot, and confirmation pool.",
    )
    disjoint.add_argument(
        "--output", required=True, help="Immutable JSON audit report."
    )

    prepare = subparsers.add_parser(
        "prepare",
        help="Create shuffled judge packets and a separate sealed assignment key.",
    )
    prepare.add_argument("--protocol", required=True, help="Validated protocol path.")
    prepare.add_argument(
        "--responses", required=True, help="Complete four-condition response artifact."
    )
    prepare.add_argument(
        "--packet-output", required=True, help="New condition-blinded packet JSON path."
    )
    prepare.add_argument(
        "--key-output", required=True, help="New sealed assignment-key JSON path."
    )
    prepare.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Sealed preregistered integer randomization seed.",
    )
    prepare.add_argument(
        "--authorize-opt-in-study",
        action="store_true",
        help="Trusted-operator attestation that real participants opted in.",
    )

    score = subparsers.add_parser(
        "score", help="Unblind and score frozen independent judgments."
    )
    score.add_argument("--protocol", required=True, help="Preregistered protocol path.")
    score.add_argument(
        "--responses",
        required=True,
        help="Frozen complete four-condition response artifact.",
    )
    score.add_argument(
        "--packets", required=True, help="Frozen blinded packet JSON path."
    )
    score.add_argument("--key", required=True, help="Sealed assignment-key JSON path.")
    score.add_argument("--judgments", required=True, help="Frozen judgment artifact.")
    score.add_argument(
        "--against",
        action="append",
        required=True,
        help="Prior protocol path; repeat for every development and earlier pilot pool.",
    )
    score.add_argument(
        "--attest-untouched-pool",
        action="store_true",
        help="Trusted-operator attestation that no pilot outcome influenced the frozen design.",
    )
    score.add_argument(
        "--attest-consent-active",
        action="store_true",
        help="Trusted-operator attestation that consent remains active at scoring time.",
    )
    score.add_argument(
        "--shadow-metrics",
        default="",
        help="Optional strict bound shadow-metric bundle JSON object.",
    )
    score.add_argument(
        "--outcome-replay",
        default="",
        help="Optional strict JSON Outcome Replay result for bottleneck diagnosis.",
    )
    score.add_argument(
        "--output", required=True, help="New immutable JSON report path."
    )
    score.add_argument(
        "--authorize-unblind",
        action="store_true",
        help="Trusted-operator attestation that judgments are frozen before unblinding.",
    )
    score.add_argument(
        "--require-go",
        action="store_true",
        help="Exit 1 unless every registered confirmation canary gate passes.",
    )
    return parser


def _validate(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = load_protocol(protocol_path)
    _enforce_real_study_paths(
        protocol, inputs=(protocol_path,), outputs=_paths(args.output)
    )
    payload = {
        "valid": True,
        "schema_version": protocol["schema_version"],
        "study_mode": protocol["study_mode"],
        "episode_count": len(protocol.get("episodes", [])),
        "shadow_only": True,
        "mutation_authority": "none",
    }
    if args.output:
        _require_json_output(args.output)
        _atomic_write_json(Path(args.output).expanduser().resolve(), payload)
    _receipt(
        command="validate",
        valid=True,
        study_mode=protocol["study_mode"],
        episode_count=payload["episode_count"],
    )
    return EXIT_SUCCESS


def _audit_disjoint(args: argparse.Namespace) -> int:
    candidate_path = Path(args.candidate).expanduser().resolve()
    candidate = load_protocol(candidate_path)
    reference_paths = tuple(Path(path).expanduser().resolve() for path in args.against)
    output_path = Path(args.output).expanduser().resolve()
    _require_json_output(output_path)
    _enforce_real_study_paths(
        candidate,
        inputs=(candidate_path, *reference_paths),
        outputs=(output_path,),
    )
    references = [load_protocol(path) for path in reference_paths]
    report = audit_study_disjointness(candidate, references)
    _atomic_write_json(output_path, report)
    _receipt(
        command="audit-disjoint",
        disjoint=bool(report["disjoint"]),
        reference_count=len(references),
    )
    return EXIT_SUCCESS if report["disjoint"] else EXIT_NO_GO


def _prepare(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol).expanduser().resolve()
    responses_path = Path(args.responses).expanduser().resolve()
    packet_output = Path(args.packet_output).expanduser().resolve()
    key_output = Path(args.key_output).expanduser().resolve()
    if packet_output == key_output:
        raise ValidationError("packet output and key output must be different files")
    _require_json_output(packet_output)
    _require_json_output(key_output)
    if packet_output.exists() or key_output.exists():
        raise ValidationError("refusing to overwrite an existing output")

    protocol = load_protocol(protocol_path)
    if (
        protocol["study_mode"] != "development"
        and args.authorize_opt_in_study is not True
    ):
        raise ValidationError(
            "real study preparation requires --authorize-opt-in-study"
        )
    _enforce_real_study_paths(
        protocol,
        inputs=(protocol_path, responses_path),
        outputs=(packet_output, key_output),
    )
    responses = load_responses(responses_path)
    bundle = prepare_blinded_packets(protocol, responses, seed=args.seed)
    public_packets, private_key = _bundle_halves(bundle)
    # Publish the text-free sealed key before the packet that contains private
    # prompts and responses. Cross-file atomic commit is unavailable; this
    # ordering makes a rare second-publication failure leave only the less
    # sensitive, unusable half.
    _atomic_write_json(key_output, private_key)
    try:
        _atomic_write_json(packet_output, public_packets)
    except Exception:
        # Publishing only one half makes the set unusable but still leaves raw
        # private data. Do not silently delete it; report the exact failure and
        # let the operator quarantine the immutable orphan.
        raise
    _receipt(
        command="prepare",
        prepared=True,
        packet_count=len(public_packets.get("judge_packets", [])),
        blinded=True,
    )
    return EXIT_SUCCESS


def _score(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol).expanduser().resolve()
    responses_path = Path(args.responses).expanduser().resolve()
    packet_path = Path(args.packets).expanduser().resolve()
    key_path = Path(args.key).expanduser().resolve()
    judgments_path = Path(args.judgments).expanduser().resolve()
    reference_paths = tuple(
        Path(value).expanduser().resolve() for value in args.against
    )
    output_path = Path(args.output).expanduser().resolve()
    _require_json_output(output_path)
    protocol = load_protocol(protocol_path)
    if args.authorize_unblind is not True:
        raise ValidationError("score requires --authorize-unblind")
    if args.attest_consent_active is not True:
        raise ValidationError(
            "score requires --attest-consent-active before private artifacts are read"
        )
    retention_until = datetime.fromisoformat(
        protocol["privacy"]["retention_until"].replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    if retention_until <= datetime.now(timezone.utc):
        raise ValidationError(
            "study retention expired before private artifacts are read"
        )
    optional_inputs = tuple(
        Path(value).expanduser().resolve()
        for value in (args.shadow_metrics, args.outcome_replay)
        if value
    )
    _enforce_real_study_paths(
        protocol,
        inputs=(
            protocol_path,
            responses_path,
            packet_path,
            key_path,
            judgments_path,
            *reference_paths,
            *optional_inputs,
        ),
        outputs=(output_path,),
    )
    public_packets = _load_private_json(packet_path)
    private_key = _load_private_json(key_path)
    responses = load_responses(responses_path)
    judgments = load_judgments(judgments_path)
    references = [load_protocol(path) for path in reference_paths]
    shadow_records = (
        _load_private_json(Path(args.shadow_metrics).expanduser().resolve())
        if args.shadow_metrics
        else None
    )
    outcome_replay = (
        _load_private_json(Path(args.outcome_replay).expanduser().resolve())
        if args.outcome_replay
        else None
    )
    result = score_study(
        protocol,
        {"public": public_packets, "private": private_key},
        judgments,
        responses=responses,
        reference_protocols=references,
        untouched_pool_attested=bool(args.attest_untouched_pool),
        consent_active_attested=bool(args.attest_consent_active),
        shadow_metric_bundle=shadow_records,
        outcome_replay_result=outcome_replay,
    )
    _atomic_write_json(output_path, result)
    go = _result_go(result)
    _receipt(
        command="score",
        scored=True,
        go=go,
        study_mode=protocol["study_mode"],
        mutation_authority="none",
    )
    if args.require_go and not go:
        return EXIT_NO_GO
    return EXIT_SUCCESS


def _bundle_halves(bundle: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(bundle, Mapping):
        raise ValidationError("packet preparation returned an invalid bundle")
    public = bundle.get("public")
    private = bundle.get("private")
    if not isinstance(public, dict) or not isinstance(private, dict):
        raise ValidationError("packet bundle must contain public and private objects")
    return public, private


def _result_go(result: Mapping[str, Any]) -> bool:
    if isinstance(result.get("go"), bool):
        return bool(result["go"])
    decision = result.get("decision")
    if isinstance(decision, Mapping):
        return bool(decision.get("go"))
    return decision == "go"


def _enforce_real_study_paths(
    protocol: Mapping[str, Any],
    *,
    inputs: tuple[Path, ...],
    outputs: tuple[Path, ...],
) -> None:
    if protocol.get("study_mode") == "development":
        return
    if os.name == "nt":
        raise ValidationError(
            "pilot artifacts require hardened POSIX permissions; run in WSL "
            "with an owner-only external directory"
        )
    for path in (*inputs, *outputs):
        _require_external_private_path(path)
    for path in inputs:
        try:
            file_stat = path.stat()
        except FileNotFoundError:
            # The owning schema loader will provide the input-specific
            # missing-file error.
            continue
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValidationError("real study inputs must be regular files")
        if file_stat.st_mode & 0o077:
            raise ValidationError(
                "real study inputs must not be accessible by group or other users"
            )


def _require_external_private_path(path: Path) -> None:
    resolved = path.expanduser().resolve()
    if resolved == _REPO_ROOT or resolved.is_relative_to(_REPO_ROOT):
        raise ValidationError("real study artifacts must stay outside the repository")
    live_roots = {Path.home().joinpath(".hermes").resolve()}
    configured = str(os.environ.get("HERMES_HOME") or "").strip()
    if configured:
        live_roots.add(Path(configured).expanduser().resolve())
    if any(resolved == root or resolved.is_relative_to(root) for root in live_roots):
        raise ValidationError(
            "real study artifacts must stay outside live Hermes state"
        )


def _paths(value: str) -> tuple[Path, ...]:
    return (Path(value).expanduser().resolve(),) if value else ()


def _require_json_output(path: str | Path) -> None:
    if Path(path).suffix.lower() != ".json":
        raise ValidationError("generated artifacts must use a .json filename")


def _load_private_json(path: Path) -> Any:
    try:
        before_path = path.stat()
        if not stat.S_ISREG(before_path.st_mode):
            raise ValidationError("input must be an existing regular JSON file")
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValidationError("input must be an existing regular JSON file")
            if (
                before.st_dev != before_path.st_dev
                or before.st_ino != before_path.st_ino
            ):
                raise ValidationError("input changed before it was read")
            if before.st_size > _MAX_JSON_BYTES:
                raise ValidationError(f"JSON input exceeds {_MAX_JSON_BYTES} bytes")
            encoded = handle.read(_MAX_JSON_BYTES + 1)
            after = os.fstat(handle.fileno())
    except FileNotFoundError as exc:
        raise ValidationError("input must be an existing regular JSON file") from exc
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValidationError(f"JSON input exceeds {_MAX_JSON_BYTES} bytes")
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(encoded) != after.st_size
    ):
        raise ValidationError("input changed while it was being read")
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError("input must be UTF-8 JSON") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
            parse_float=_parse_finite_json_float,
        )
    except RecursionError as exc:
        raise ValidationError("input JSON exceeds the nesting-depth bound") from exc
    _require_bounded_json_depth(value)
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("duplicate JSON key is not allowed")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValidationError(f"non-finite JSON value is not allowed: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValidationError("non-finite JSON number is not allowed")
    return parsed


def _require_bounded_json_depth(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        if not isinstance(current, (dict, list)):
            continue
        if depth > _MAX_JSON_DEPTH:
            raise ValidationError("input JSON exceeds the nesting-depth bound")
        children = current.values() if isinstance(current, dict) else current
        stack.extend((child, depth + 1) for child in children)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a complete private JSON artifact without replacing its path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValidationError("refusing to overwrite an existing output") from exc
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _receipt(**values: Any) -> None:
    print(json.dumps(values, allow_nan=False, sort_keys=True))


def _safe_error_message(exc: Exception) -> str:
    """Return an operator-useful error without echoing private paths."""

    if isinstance(exc, OSError):
        return "private artifact I/O failed"
    if isinstance(exc, ValidationError):
        return str(exc) or "private artifact validation failed"
    if isinstance(exc, json.JSONDecodeError):
        return "private JSON artifact is invalid"
    return "private artifact processing failed"


if __name__ == "__main__":
    raise SystemExit(main())
