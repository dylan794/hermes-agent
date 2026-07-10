from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "memory_v2_archive_ops.py"
SPEC = importlib.util.spec_from_file_location("memory_v2_archive_ops", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
archive_ops = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = archive_ops
SPEC.loader.exec_module(archive_ops)


def _parse(argv: list[str]):
    return archive_ops.build_parser().parse_args(argv)


def test_session_backfill_payload_omits_include_tools_by_default(tmp_path):
    args = _parse([
        "--hermes-home",
        str(tmp_path),
        "session-backfill",
        "dry-run",
    ])

    payload = archive_ops._backfill_payload(args, dry_run=True)

    assert "include_tools" not in payload


def test_session_backfill_payload_include_tools_is_explicit_opt_in(tmp_path):
    args = _parse([
        "--hermes-home",
        str(tmp_path),
        "session-backfill",
        "dry-run",
        "--include-tools",
    ])

    payload = archive_ops._backfill_payload(args, dry_run=True)

    assert payload["include_tools"] is True


def test_session_backfill_payload_no_include_tools_is_explicit_false(tmp_path):
    args = _parse([
        "--hermes-home",
        str(tmp_path),
        "session-backfill",
        "dry-run",
        "--no-include-tools",
    ])

    payload = archive_ops._backfill_payload(args, dry_run=True)

    assert payload["include_tools"] is False
