#!/usr/bin/env python3
"""Operational Memory v2 archive CLI.

This script is intentionally small and JSON-only. It wraps provider archive
operations without exposing raw provider internals or unbounded raw dumps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_constants import get_hermes_home  # noqa: E402
from plugins.memory.memory_v2 import MemoryV2Provider  # noqa: E402
from plugins.memory.memory_v2.evals.baselines import (  # noqa: E402
    ArchiveOnlyBaseline,
    MemoryV2Baseline,
    NoMemoryBaseline,
    RawFTSBaseline,
    SemanticOnlyBaseline,
)
from plugins.memory.memory_v2.evals.datasets import load_eval_dataset  # noqa: E402
from plugins.memory.memory_v2.evals.runners import run_eval  # noqa: E402
from plugins.memory.memory_v2.health import MemoryHealthChecker  # noqa: E402
from plugins.memory.memory_v2.session_backfill import SESSION_BACKFILL_CONFIRM  # noqa: E402
from scripts import memory_v2_privacy_scan as privacy_scan  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "handler", None):
        parser.print_help(sys.stderr)
        return 2
    try:
        payload, exit_code = args.handler(args)
    except Exception as exc:  # Last-resort operator-friendly failure, no traceback.
        payload = {"success": False, "error": str(exc), "command": _command_name(args)}
        exit_code = 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Memory v2 archive operational CLI (JSON output only).")
    parser.add_argument(
        "--hermes-home",
        default="",
        help="Hermes profile directory. Defaults to HERMES_HOME or ~/.hermes.",
    )
    subparsers = parser.add_subparsers(dest="group")

    archive = subparsers.add_parser("archive", help="Raw archive inspection and derived-index repair.")
    archive_sub = archive.add_subparsers(dest="archive_command")

    status = archive_sub.add_parser("status", help="Show archive status and safe counts.")
    status.set_defaults(handler=cmd_archive_status)

    search = archive_sub.add_parser("search", help="Search bounded raw archive evidence packets.")
    search.add_argument("--query", required=True, help="Keyword query. Output includes query hash, not full query in status fields.")
    search.add_argument("--limit", type=int, default=5, help="Maximum results, capped at 20.")
    search.add_argument("--session-id", default="")
    search.add_argument("--event-type", default="")
    search.add_argument("--created-after", default="")
    search.add_argument("--created-before", default="")
    search.add_argument("--no-verify-integrity", action="store_true")
    search.set_defaults(handler=cmd_archive_search)

    show = archive_sub.add_parser("show", help="Show one bounded raw event packet by id.")
    show.add_argument("event_id")
    show.add_argument("--expect-record-sha256", default="", help="Optional sha256:... record hash pin.")
    show.add_argument("--include-neighbor-ids", action="store_true")
    show.add_argument("--no-verify-integrity", action="store_true")
    show.set_defaults(handler=cmd_archive_show)

    rebuild = archive_sub.add_parser("rebuild-index", help="Rebuild derived SQLite/raw archive indexes from canonical files.")
    rebuild.set_defaults(handler=cmd_archive_rebuild_index)

    verify = archive_sub.add_parser("verify", help="Verify raw archive hash chain and safe manifest counts.")
    verify.set_defaults(handler=cmd_archive_verify)

    backfill = subparsers.add_parser("session-backfill", help="Backfill SessionDB into raw archive; dry-run by default.")
    backfill_sub = backfill.add_subparsers(dest="backfill_command")
    dry_run = backfill_sub.add_parser("dry-run", help="Preview SessionDB import. Never mutates archive/checkpoints.")
    _add_backfill_common_args(dry_run)
    dry_run.set_defaults(handler=cmd_session_backfill_dry_run)
    run = backfill_sub.add_parser("run", help="Run confirmed SessionDB import.")
    _add_backfill_common_args(run)
    run.add_argument("--confirm", default="", help=f"Must equal {SESSION_BACKFILL_CONFIRM}.")
    run.set_defaults(handler=cmd_session_backfill_run)

    scan = subparsers.add_parser("privacy-scan", help="Run Memory v2 privacy scanner with JSON summary.")
    scan.add_argument("--mode", choices=tuple(sorted(privacy_scan.SCAN_PROFILES)), help="Named privacy scan mode/profile.")
    scan.add_argument("--release", choices=("memory-v2",), help="Alias for --mode memory-v2-release-artifacts.")
    scan.add_argument("--paths", nargs="+", type=Path, help="Files/directories to scan instead of git diff, or override mode defaults.")
    scan.add_argument("--base-ref", default="", help="Git ref for diff scan.")
    scan.add_argument("--staged", action="store_true", help="Scan staged diff.")
    scan.add_argument("--format", choices=("json",), default="json", help="Only JSON is emitted by this wrapper.")
    scan.set_defaults(handler=cmd_privacy_scan)

    eval_cmd = subparsers.add_parser("eval", help="Run deterministic Memory v2 eval fixtures and emit compact summary.")
    eval_cmd.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="YAML dataset fixture path. Defaults to local_memory_eval_v1.yaml.",
    )
    eval_cmd.add_argument("--baseline", action="append", choices=["no_memory", "raw_fts", "archive_only", "semantic_only", "memory_v2"], default=[])
    eval_cmd.add_argument("--workdir", default="", help="Directory for temporary baseline stores.")
    eval_cmd.add_argument(
        "--fail-on-acceptance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit nonzero when deterministic acceptance fails.",
    )
    eval_cmd.set_defaults(handler=cmd_eval)
    return parser


def _add_backfill_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", default="", help="Optional SessionDB source filter, e.g. discord.")
    parser.add_argument("--session-id", default="", help="Optional exact session id filter.")
    parser.add_argument("--limit", type=int, default=500, help="Message limit, capped at 5000.")
    parser.add_argument("--batch-size", type=int, default=500, help="Import batch size, capped at 5000.")
    parser.add_argument("--max-batches", type=int, default=None, help="Stop after N batches for canaries/resume testing.")
    parser.add_argument("--since-message-id", type=int, default=None)
    parser.add_argument("--until-message-id", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from matching checkpoint when no explicit since id is set.")
    parser.add_argument("--include-tools", action="store_true", help="Opt in to importing tool-role messages when profile config permits it.")
    parser.add_argument("--no-include-tools", action="store_true", help="Explicitly skip tool-role messages.")
    parser.add_argument("--state-db-path", default="", help="Current-profile state.db path override; must stay under Hermes home.")


def _command_name(args: argparse.Namespace) -> str:
    group = getattr(args, "group", "") or ""
    if group == "archive":
        return f"archive {getattr(args, 'archive_command', '')}".strip()
    if group == "session-backfill":
        return f"session-backfill {getattr(args, 'backfill_command', '')}".strip()
    return group


def _provider(args: argparse.Namespace) -> MemoryV2Provider:
    home = _hermes_home(args)
    provider = MemoryV2Provider()
    provider.initialize("memory-v2-archive-ops", hermes_home=str(home), platform="cli")
    return provider


def _hermes_home(args: argparse.Namespace) -> Path:
    if getattr(args, "hermes_home", ""):
        return Path(args.hermes_home).expanduser().resolve()
    return get_hermes_home().expanduser().resolve()


def _tool(provider: MemoryV2Provider, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(provider.handle_tool_call(name, payload))


def _with_command(payload: dict[str, Any], command: str) -> dict[str, Any]:
    return {"command": command, **payload}


def cmd_archive_status(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    provider = _provider(args)
    payload = _tool(provider, "memory_v2_status", {})
    # Explicitly keep status as path-safe summary only.
    safe = {
        "success": bool(payload.get("success")),
        "provider": payload.get("provider", "memory_v2"),
        "initialized": bool(payload.get("initialized")),
        "platform": payload.get("platform", ""),
        "base_dir": payload.get("base_dir", "memory_v2"),
        "raw_archive": _archive_summary(payload.get("raw_archive") or {}),
        "counts": payload.get("counts") or {},
    }
    return _with_command(safe, "archive status"), 0 if safe["success"] else 1


def cmd_archive_verify(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    provider = _provider(args)
    report = provider.store.verify_raw_archive()
    payload = {
        "success": bool(report.get("success", True)),
        "archive": _archive_summary(report),
        "issue_count": int(report.get("issue_count") or 0),
        "issues": list(report.get("issues") or [])[:20],
    }
    return _with_command(payload, "archive verify"), 0 if payload["success"] and payload["archive"]["status"] == "ok" else 1


def cmd_archive_rebuild_index(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    provider = _provider(args)
    result = provider.index.rebuild_from_store(provider.store)
    manifest = provider.store.read_raw_archive_manifest()
    payload = {
        "success": True,
        "mutated": "derived_index_only",
        "result": {field_name: int(count) for field_name, count in dict(result).items() if isinstance(count, int)},
        "archive": _archive_summary(manifest),
    }
    return _with_command(payload, "archive rebuild-index"), 0


def cmd_archive_search(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    provider = _provider(args)
    payload = _tool(
        provider,
        "memory_v2_archive_search",
        {
            "query": args.query,
            "limit": args.limit,
            "session_id": args.session_id,
            "event_type": args.event_type,
            "created_after": args.created_after,
            "created_before": args.created_before,
            "verify_integrity": not args.no_verify_integrity,
        },
    )
    return _with_command(payload, "archive search"), 0 if payload.get("success") else 1


def cmd_archive_show(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    provider = _provider(args)
    payload = _tool(
        provider,
        "memory_v2_archive_show",
        {
            "id": args.event_id,
            "expected_record_sha256": args.expect_record_sha256,
            "include_neighbor_ids": args.include_neighbor_ids,
            "verify_integrity": not args.no_verify_integrity,
        },
    )
    return _with_command(payload, "archive show"), 0 if payload.get("success") else 1


def cmd_session_backfill_dry_run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    provider = _provider(args)
    payload = _tool(provider, "memory_v2_session_backfill", _backfill_payload(args, dry_run=True))
    return _with_command(_safe_backfill_payload(payload), "session-backfill dry-run"), 0 if payload.get("success") else 1


def cmd_session_backfill_run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if str(args.confirm or "") != SESSION_BACKFILL_CONFIRM:
        return {
            "success": False,
            "command": "session-backfill run",
            "error": f"--confirm must equal {SESSION_BACKFILL_CONFIRM}; dry-run is the default safe mode",
        }, 2
    provider = _provider(args)
    payload = _tool(provider, "memory_v2_session_backfill", _backfill_payload(args, dry_run=False, confirm=args.confirm))
    return _with_command(_safe_backfill_payload(payload), "session-backfill run"), 0 if payload.get("success") else 1


def _backfill_payload(args: argparse.Namespace, *, dry_run: bool, confirm: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dry_run": dry_run,
        "source": args.source,
        "session_id": args.session_id,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "since_message_id": args.since_message_id,
        "until_message_id": args.until_message_id,
        "resume": args.resume,
        "state_db_path": args.state_db_path,
    }
    if args.include_tools:
        payload["include_tools"] = True
    elif args.no_include_tools:
        payload["include_tools"] = False
    if confirm:
        payload["confirm"] = confirm
    return payload


def _safe_backfill_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "success",
        "mode",
        "dry_run",
        "source",
        "session_id",
        "limit",
        "batch_size",
        "batches",
        "max_batches",
        "resume",
        "resumed_from_checkpoint",
        "checkpoint_last_message_id",
        "considered",
        "imported",
        "skipped",
        "skipped_reasons",
        "error_count",
        "errors",
        "imported_ids",
        "skipped_ids",
        "next_since_message_id",
        "stopped_reason",
        "error",
    }
    safe = {field_name: value for field_name, value in payload.items() if field_name in allowed}
    raw_archive = payload.get("raw_archive") or {}
    if isinstance(raw_archive, dict):
        safe["raw_archive"] = _archive_summary(raw_archive)
    return safe


def cmd_privacy_scan(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    mode = args.mode
    if args.release == "memory-v2":
        if mode and mode != "memory-v2-release-artifacts":
            raise ValueError("--release memory-v2 cannot be combined with a different --mode")
        mode = "memory-v2-release-artifacts"
    profile = privacy_scan.profile_for_mode(mode)
    records, scan_scope = privacy_scan.scan_scope_for_args(args, profile)
    findings = privacy_scan.scan_records(records, profile)
    payload = privacy_scan.json_payload(profile, findings, scan_scope)
    payload["findings"] = payload["findings"][:50]
    return _with_command(payload, "privacy-scan"), 1 if findings else 0


def cmd_eval(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    datasets = args.dataset or ["plugins/memory/memory_v2/evals/fixtures/local_memory_eval_v1.yaml"]
    baselines = args.baseline or ["no_memory", "raw_fts", "memory_v2"]
    reports: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="memory-v2-ops-eval-") as temp_dir:
        workdir = Path(args.workdir).expanduser().resolve() if args.workdir else Path(temp_dir)
        workdir.mkdir(parents=True, exist_ok=True)
        for dataset_path in datasets:
            dataset = load_eval_dataset(dataset_path)
            report = run_eval(dataset, baselines=_build_baselines(baselines, workdir / dataset.name)).to_dict()
            reports.append(_compact_eval_report(report))
    if len(reports) == 1:
        payload = {"success": True, **reports[0]}
    else:
        payload = {"success": True, "reports": reports}
    gate_passed = _eval_acceptance_passed(payload)
    exit_code = 0 if (not args.fail_on_acceptance or gate_passed) else 1
    if args.fail_on_acceptance and not gate_passed:
        payload["success"] = False
    return _with_command(payload, "eval"), exit_code


def _build_baselines(names: list[str], workdir: Path) -> list[Any]:
    baselines: list[Any] = []
    for index, name in enumerate(names):
        baseline_dir = workdir / f"{index}_{name}"
        if name == "no_memory":
            baselines.append(NoMemoryBaseline())
        elif name == "raw_fts":
            baselines.append(RawFTSBaseline(baseline_dir / "raw.sqlite"))
        elif name == "archive_only":
            baselines.append(ArchiveOnlyBaseline(baseline_dir / "archive_only"))
        elif name == "semantic_only":
            baselines.append(SemanticOnlyBaseline(baseline_dir / "semantic_only"))
        elif name == "memory_v2":
            baselines.append(MemoryV2Baseline(baseline_dir / "memory_v2"))
        else:
            raise ValueError(f"unknown baseline: {name}")
    return baselines


def _compact_eval_report(report: dict[str, Any]) -> dict[str, Any]:
    acceptance = dict(report.get("acceptance") or {})
    compact_acceptance = {
        "dataset": acceptance.get("dataset", report.get("dataset", "")),
        "target_baseline": acceptance.get("target_baseline", ""),
        "passed": bool(acceptance.get("passed")),
        "check_count": len(acceptance.get("checks") or []),
        "failed_checks": [
            {"name": check.get("name", ""), "baseline": check.get("baseline", ""), "actual": check.get("actual"), "threshold": check.get("threshold")}
            for check in (acceptance.get("checks") or [])
            if not check.get("passed")
        ][:20],
    }
    return {
        "dataset": str(report.get("dataset") or ""),
        "summary": report.get("summary") or {},
        "acceptance": compact_acceptance,
    }


def _eval_acceptance_passed(payload: dict[str, Any]) -> bool:
    if "reports" in payload:
        return all(_eval_acceptance_passed(dict(report)) for report in payload.get("reports") or [])
    return bool((payload.get("acceptance") or {}).get("passed"))


def _archive_summary(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": str(data.get("status") or "unknown"),
        "event_count": int(data.get("event_count") or 0),
        "verified_event_count": int(data.get("verified_event_count") or 0),
        "issue_count": int(data.get("issue_count") or 0),
        "last_record_sha256": str(data.get("last_record_sha256") or ""),
        "derived_index_status": str(data.get("derived_index_status") or ""),
        "indexed_event_count": int(data.get("indexed_event_count") or 0),
        "last_indexed_record_sha256": str(data.get("last_indexed_record_sha256") or ""),
        "raw_index_schema_version": int(data.get("raw_index_schema_version") or 0),
        "byte_size": int(data.get("byte_size") or 0),
    }


if __name__ == "__main__":
    raise SystemExit(main())
