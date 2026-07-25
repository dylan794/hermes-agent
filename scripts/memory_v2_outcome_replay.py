#!/usr/bin/env python3
"""Validate and analyze minimized Memory v2 outcome-replay artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plugins.memory.memory_v2.evals.outcome_replay import (
    ABLATION_COMPONENTS,
    ValidationError,
    analyze_dataset,
    audit_dataset_disjointness,
    load_dataset,
    load_private_intake,
)


EXIT_SUCCESS = 0
EXIT_INCONCLUSIVE = 1
EXIT_INVALID = 2


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _validate(args)
        if args.command == "analyze":
            return _analyze(args)
        if args.command == "audit-disjoint":
            return _audit_disjoint(args)
        if args.command == "collect":
            return _collect(args)
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
        description=(
            "Offline, side-effect-free bottleneck diagnostics for minimized Memory v2 "
            "shadow episodes."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate a replay dataset.")
    validate.add_argument("--dataset", required=True, help="Replay dataset JSON or YAML path.")
    validate.add_argument("--output", default="", help="Optional JSON validation report path.")

    analyze = subparsers.add_parser(
        "analyze", help="Analyze paired oracle gaps with participant-cluster uncertainty."
    )
    analyze.add_argument("--dataset", required=True, help="Validated replay dataset path.")
    analyze.add_argument("--output", default="", help="Optional JSON result path.")
    analyze.add_argument(
        "--require-decisive",
        action="store_true",
        help="Exit 1 when the valid dataset does not identify a decisive bottleneck.",
    )

    disjoint = subparsers.add_parser(
        "audit-disjoint",
        help="Check exact participant, project, workstream, query, and corpus overlap.",
    )
    disjoint.add_argument("--candidate", required=True, help="Candidate replay dataset path.")
    disjoint.add_argument(
        "--against",
        required=True,
        action="append",
        help="Reference dataset path; repeat for every prior data pool.",
    )
    disjoint.add_argument("--output", default="", help="Optional JSON audit report path.")

    collect = subparsers.add_parser(
        "collect",
        help="Minimize an operator-controlled private intake into a disjoint pilot dataset.",
    )
    collect.add_argument(
        "--intake",
        required=True,
        help="Private intake JSON/YAML path outside the repository.",
    )
    collect.add_argument(
        "--token-key-file",
        required=True,
        help="External file containing exactly 64 lowercase hexadecimal key characters.",
    )
    collect.add_argument(
        "--against",
        required=True,
        action="append",
        help="Prior minimized dataset; repeat for every previous pool.",
    )
    collect.add_argument(
        "--output",
        required=True,
        help="New minimized JSON dataset path outside the repository.",
    )
    collect.add_argument(
        "--authorize-opt-in-collection",
        action="store_true",
        help="Trusted operator attestation that every intake participant opted in.",
    )
    collect.add_argument(
        "--allow-partial-panel",
        action="store_true",
        help="Allow staging data without every registered oracle variant.",
    )
    collect.add_argument(
        "--allow-underpowered",
        action="store_true",
        help="Allow staging data below its registered episode/participant floors.",
    )
    return parser


def _validate(args: argparse.Namespace) -> int:
    _prepare_output(args.output, inputs=(args.dataset,))
    dataset = load_dataset(args.dataset)
    payload = {
        "valid": True,
        "schema_version": dataset["schema_version"],
        "lab_id": dataset["lab_id"],
        "study_mode": dataset["study_mode"],
        "episode_count": len(dataset["episodes"]),
        "participant_clusters": len(
            {row["participant_ref"] for row in dataset["episodes"]}
        ),
        "project_clusters": len({row["project_ref"] for row in dataset["episodes"]}),
        "raw_content_stored": False,
        "diagnostic_only": True,
    }
    _emit(payload, args.output)
    return EXIT_SUCCESS


def _analyze(args: argparse.Namespace) -> int:
    _prepare_output(args.output, inputs=(args.dataset,))
    result = analyze_dataset(load_dataset(args.dataset))
    _emit(result, args.output)
    decisive = result["diagnosis"]["status"] == "decisive_bottleneck"
    if args.require_decisive and not decisive:
        return EXIT_INCONCLUSIVE
    return EXIT_SUCCESS


def _audit_disjoint(args: argparse.Namespace) -> int:
    _prepare_output(args.output, inputs=(args.candidate, *args.against))
    candidate = load_dataset(args.candidate)
    references = [load_dataset(path) for path in args.against]
    result = audit_dataset_disjointness(candidate, references)
    _emit(result, args.output)
    return EXIT_SUCCESS if result["disjoint"] else EXIT_INCONCLUSIVE


def _collect(args: argparse.Namespace) -> int:
    if args.authorize_opt_in_collection is not True:
        raise ValidationError("collect requires --authorize-opt-in-collection")
    intake_path = _require_external_path(args.intake, label="private intake")
    key_path = _require_external_path(args.token_key_file, label="token key")
    output_path = _require_external_path(args.output, label="collected dataset")
    _prepare_output(
        str(output_path),
        inputs=(str(intake_path), str(key_path), *args.against),
        must_not_exist=True,
    )
    dataset = load_private_intake(
        intake_path,
        token_material=_read_token_key(key_path),
    )
    if dataset["study_mode"] != "pilot":
        raise ValidationError("collected real episodes must use study_mode 'pilot'")
    _require_decision_ready(
        dataset,
        allow_partial_panel=bool(args.allow_partial_panel),
        allow_underpowered=bool(args.allow_underpowered),
    )
    references = [load_dataset(path) for path in args.against]
    audit = audit_dataset_disjointness(dataset, references)
    receipt = {
        "collected": bool(audit["disjoint"]),
        "disjoint": bool(audit["disjoint"]),
        "lab_id": dataset["lab_id"],
        "episode_count": len(dataset["episodes"]),
        "participant_clusters": len(
            {row["participant_ref"] for row in dataset["episodes"]}
        ),
        "raw_content_stored": False,
        "diagnostic_only": True,
        "mutation_authority": "none",
        "disjointness": audit,
    }
    if not audit["disjoint"]:
        _emit(receipt, "")
        return EXIT_INCONCLUSIVE
    _write_json_atomic(output_path, dataset)
    _emit(receipt, "")
    return EXIT_SUCCESS


def _require_decision_ready(
    dataset: dict[str, Any],
    *,
    allow_partial_panel: bool,
    allow_underpowered: bool,
) -> None:
    episodes = dataset["episodes"]
    if not allow_partial_panel:
        expected = set(ABLATION_COMPONENTS)
        for index, episode in enumerate(episodes):
            actual = {row["ablation"] for row in episode["variants"]}
            if actual != expected:
                missing = sorted(expected - actual)
                raise ValidationError(
                    f"episode {index} is missing oracle variants: {missing}; "
                    "use --allow-partial-panel only for non-decision staging"
                )
    if not allow_underpowered:
        thresholds = dataset["thresholds"]
        if len(episodes) < thresholds["minimum_paired_episodes"]:
            raise ValidationError(
                "episode count is below thresholds.minimum_paired_episodes; "
                "use --allow-underpowered only for non-decision staging"
            )
        participant_clusters = len({row["participant_ref"] for row in episodes})
        if participant_clusters < thresholds["minimum_participant_clusters"]:
            raise ValidationError(
                "participant clusters are below thresholds.minimum_participant_clusters; "
                "use --allow-underpowered only for non-decision staging"
            )


def _read_token_key(path: Path) -> bytes:
    try:
        if path.stat().st_size > 256:
            raise ValidationError("token key file is unexpectedly large")
        text = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"cannot read token key file: {exc}") from exc
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValidationError(
            "token key file must contain exactly 64 lowercase hexadecimal characters"
        )
    if os.name != "nt" and path.stat().st_mode & 0o077:
        raise ValidationError("token key file must not be accessible by group or other users")
    return bytes.fromhex(text)


def _require_external_path(value: str, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    try:
        path.relative_to(_REPO_ROOT)
    except ValueError:
        return path
    raise ValidationError(f"{label} must stay outside the repository")


def _prepare_output(
    output: str,
    *,
    inputs: tuple[str, ...],
    must_not_exist: bool = False,
) -> None:
    if not output:
        return
    output_path = Path(output).expanduser().resolve()
    if output_path.suffix.lower() != ".json":
        raise ValidationError("output file must use .json")
    input_paths = {Path(path).expanduser().resolve() for path in inputs}
    if output_path in input_paths:
        raise ValidationError("output path must not replace an input artifact")
    if must_not_exist and output_path.exists():
        raise ValidationError("output path already exists; collected datasets are immutable")


def _emit(payload: dict[str, Any], output: str) -> None:
    if output:
        _write_json_atomic(Path(output).expanduser().resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
