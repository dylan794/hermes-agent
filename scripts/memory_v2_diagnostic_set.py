#!/usr/bin/env python3
"""Prepare, validate, and score an owner-only Memory v2 diagnostic set."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plugins.memory.memory_v2.evals.diagnostic_set import (
    DiagnosticSetError,
    prepare_diagnostic_set,
    public_summary,
    score_labeled_set,
    validate_diagnostic_set,
)
from plugins.memory.memory_v2.shadow_pipeline import (
    ShadowPipelineConfig,
    ShadowRetrievalPipeline,
)
from plugins.memory.memory_v2.shadow_reranker import ShadowRerankerConfig


EXIT_SUCCESS = 0
EXIT_INCOMPLETE = 1
EXIT_INVALID = 2
_MAX_PRIVATE_INPUT_BYTES = 20 * 1024 * 1024


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            return _prepare(args)
        if args.command == "validate":
            return _validate(args)
        if args.command == "score":
            return _score(args)
    except (
        DiagnosticSetError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    return EXIT_INVALID


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a private owner-labeling packet that diagnoses Memory v2 "
            "without modifying live memory."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--raw-events", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--public-summary", required=True)
    prepare.add_argument("--episode-count", type=int, default=30)
    prepare.add_argument("--control-count", type=int, default=8)
    prepare.add_argument("--held-out-count", type=int, default=10)
    prepare.add_argument("--created-at", default="")
    prepare.add_argument("--authorize-private-history", action="store_true")

    validate = commands.add_parser("validate")
    validate.add_argument("--packet", required=True)
    validate.add_argument("--public-summary", default="")

    score = commands.add_parser("score")
    score.add_argument("--packet", required=True)
    score.add_argument("--split", choices=("development", "held_out", "all"), default="development")
    score.add_argument("--output", required=True)
    return parser


def _prepare(args: argparse.Namespace) -> int:
    if args.authorize_private_history is not True:
        raise DiagnosticSetError("prepare requires --authorize-private-history")
    raw_path = _regular_file(args.raw_events, "raw event archive")
    private_output = _private_external_output(args.output)
    public_output = Path(args.public_summary).expanduser().resolve()
    _require_new_outputs(private_output, public_output)
    events = _load_jsonl(raw_path)
    created_at = args.created_at or datetime.now(timezone.utc).isoformat()

    def run_shadow(
        query: str,
        evidence: list[dict[str, Any]],
        cutoff: str,
        gap_days: float,
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="memory-v2-diagnostic-") as directory:
            return ShadowRetrievalPipeline(
                Path(directory) / "derived.sqlite",
                config=ShadowPipelineConfig(
                    enabled=True,
                    reranker=ShadowRerankerConfig(
                        enabled=True,
                        minimum_utility=0.0,
                    ),
                ),
                scratch_root=directory,
            ).run(
                query=query,
                raw_events=evidence,
                profile_id="single-participant-development",
                tenant_id="private-local-development",
                evidence_cutoff=cutoff,
                context={
                    "has_current_context": False,
                    "gap_days": gap_days,
                },
            )

    packet = prepare_diagnostic_set(
        events,
        shadow_runner=run_shadow,
        created_at=created_at,
        episode_count=args.episode_count,
        control_count=args.control_count,
        held_out_count=args.held_out_count,
    )
    summary = public_summary(packet)
    _write_json_exclusive(private_output, packet, private=True)
    _write_json_exclusive(public_output, summary, private=False)
    print(
        json.dumps(
            {
                "prepared": True,
                "episode_count": summary["selection"]["episode_count"],
                "control_candidate_count": summary["selection"][
                    "control_candidate_count"
                ],
                "development_count": summary["split_policy"]["development_count"],
                "held_out_count": summary["split_policy"]["held_out_count"],
                "pending_owner_labels": summary["label_status_counts"].get(
                    "pending", 0
                ),
                "outcome_replay_ready": summary["outcome_replay_ready"],
                "private_packet": True,
                "live_profile_modified": False,
                "mutation_authority": "none",
            },
            sort_keys=True,
        )
    )
    return EXIT_SUCCESS


def _validate(args: argparse.Namespace) -> int:
    packet = validate_diagnostic_set(_load_json(_regular_file(args.packet, "packet")))
    summary = public_summary(packet)
    if args.public_summary:
        output = Path(args.public_summary).expanduser().resolve()
        if output.exists():
            raise DiagnosticSetError("public summary output already exists")
        _write_json_exclusive(output, summary, private=False)
    print(json.dumps(summary, sort_keys=True))
    return EXIT_SUCCESS if summary["owner_labels_complete"] else EXIT_INCOMPLETE


def _score(args: argparse.Namespace) -> int:
    packet = validate_diagnostic_set(_load_json(_regular_file(args.packet, "packet")))
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise DiagnosticSetError("score output already exists")
    score = score_labeled_set(packet, split=args.split)
    _write_json_exclusive(output, score, private=False)
    print(json.dumps(score, sort_keys=True))
    return EXIT_SUCCESS


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.stat().st_size > _MAX_PRIVATE_INPUT_BYTES:
        raise DiagnosticSetError("raw event archive exceeds its byte bound")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise DiagnosticSetError(
                    f"raw event line {line_number} must be a JSON object"
                )
            rows.append(row)
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > _MAX_PRIVATE_INPUT_BYTES:
        raise DiagnosticSetError("packet exceeds its byte bound")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DiagnosticSetError("packet must be a JSON object")
    return value


def _regular_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    try:
        status = path.stat()
    except FileNotFoundError as exc:
        raise DiagnosticSetError(f"{label} must exist") from exc
    if not stat.S_ISREG(status.st_mode):
        raise DiagnosticSetError(f"{label} must be a regular file")
    return path


def _private_external_output(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path == _REPO_ROOT or path.is_relative_to(_REPO_ROOT):
        raise DiagnosticSetError("private packet must stay outside the repository")
    live_root = Path.home().joinpath(".hermes").resolve()
    if path == live_root or path.is_relative_to(live_root):
        raise DiagnosticSetError("private packet must stay outside live Hermes state")
    return path


def _require_new_outputs(*paths: Path) -> None:
    if len(set(paths)) != len(paths):
        raise DiagnosticSetError("output paths must be distinct")
    if any(path.exists() for path in paths):
        raise DiagnosticSetError("diagnostic outputs are immutable and cannot overwrite")


def _write_json_exclusive(
    path: Path,
    payload: dict[str, Any],
    *,
    private: bool,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
        mode=0o700 if private else 0o755,
    )
    if private and os.name != "nt":
        path.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600 if private else 0o644)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise DiagnosticSetError("refusing to overwrite an existing output") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
