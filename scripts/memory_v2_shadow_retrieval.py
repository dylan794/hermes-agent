#!/usr/bin/env python3
"""Run one explicitly authorized, local-only Memory v2 shadow retrieval."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plugins.memory.memory_v2.shadow_pipeline import (
    ShadowPipelineConfig,
    ShadowRetrievalPipeline,
)
from plugins.memory.memory_v2.shadow_reranker import ShadowRerankerConfig


EXIT_SUCCESS = 0
EXIT_INVALID = 2
_MAX_INPUT_BYTES = 10_000_000
_MAX_JSON_DEPTH = 64


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a disposable derived index and run a read-only private shadow "
            "retrieval. This command never changes live Hermes memory."
        )
    )
    parser.add_argument("--input", required=True, help="Private request JSON path.")
    parser.add_argument("--output", required=True, help="New private result JSON path.")
    parser.add_argument(
        "--authorize-private-shadow",
        action="store_true",
        help="Confirm operator authority to process this private snapshot locally.",
    )
    args = parser.parse_args(argv)
    try:
        if not args.authorize_private_shadow:
            raise ValueError("--authorize-private-shadow is required")
        input_path = Path(args.input).expanduser().resolve()
        output_path = Path(args.output).expanduser().resolve()
        _require_external_private_path(input_path, "input")
        _require_external_private_path(output_path, "output")
        payload = _load_private_json(input_path)
        request = _validate_request(payload)
        with tempfile.TemporaryDirectory(prefix="memory-v2-shadow-") as temp_dir:
            pipeline = ShadowRetrievalPipeline(
                Path(temp_dir) / "derived.sqlite",
                config=ShadowPipelineConfig(
                    enabled=True,
                    reranker=ShadowRerankerConfig(enabled=True),
                ),
                scratch_root=temp_dir,
            )
            result = pipeline.run(**request)
        result["private_artifact"] = True
        _atomic_write_json(output_path, result)
        print(
            json.dumps(
                {
                    "schema_version": result["schema_version"],
                    "output": str(output_path),
                    "private_artifact": True,
                    "shadow_only": True,
                    "read_only": True,
                    "mutation_authority": "none",
                    "intent_decision": result["intent"]["decision"],
                    "candidate_count": result["retrieval"]["candidate_count"],
                    "selected_count": len(result["result"]["bundle"]["items"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return EXIT_SUCCESS
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID


def _validate_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("input must be a JSON object")
    required = {
        "query",
        "raw_events",
        "profile_id",
        "tenant_id",
        "evidence_cutoff",
        "context",
    }
    if set(value) != required:
        raise ValueError("input does not match the strict shadow request schema")
    return {
        "query": value["query"],
        "raw_events": value["raw_events"],
        "profile_id": value["profile_id"],
        "tenant_id": value["tenant_id"],
        "evidence_cutoff": value["evidence_cutoff"],
        "context": value["context"],
    }


def _require_external_private_path(path: Path, label: str) -> None:
    if path == _REPO_ROOT or path.is_relative_to(_REPO_ROOT):
        raise ValueError(f"private {label} must be outside the repository")
    if label == "output":
        live_roots = {Path.home().joinpath(".hermes").resolve()}
        configured = str(os.environ.get("HERMES_HOME") or "").strip()
        if configured:
            live_roots.add(Path(configured).expanduser().resolve())
        if any(path == root or path.is_relative_to(root) for root in live_roots):
            raise ValueError("private output must be outside live Hermes state")


def _load_private_json(path: Path) -> Any:
    """Read and decode one bounded regular file through a single handle."""

    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("input must be an existing regular JSON file")
            if before.st_size > _MAX_INPUT_BYTES:
                raise ValueError("input exceeds the private shadow byte bound")
            encoded = handle.read(_MAX_INPUT_BYTES + 1)
            after = os.fstat(handle.fileno())
    except FileNotFoundError as exc:
        raise ValueError("input must be an existing regular JSON file") from exc

    if len(encoded) > _MAX_INPUT_BYTES:
        raise ValueError("input exceeds the private shadow byte bound")
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(encoded) != after.st_size
    ):
        raise ValueError("input changed while it was being read")
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("input must be UTF-8 JSON") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
            parse_float=_parse_finite_json_float,
        )
    except RecursionError as exc:
        raise ValueError("input JSON exceeds the nesting-depth bound") from exc
    _require_bounded_json_depth(value)
    return value


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is not allowed: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value is not allowed: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number is not allowed")
    return parsed


def _require_bounded_json_depth(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        if not isinstance(current, (dict, list)):
            continue
        if depth > _MAX_JSON_DEPTH:
            raise ValueError("input JSON exceeds the nesting-depth bound")
        children = current.values() if isinstance(current, dict) else current
        stack.extend((child, depth + 1) for child in children)


def _atomic_write_json(path: Path, payload: MappingLike) -> None:
    """Publish a complete private artifact without ever replacing ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        try:
            # A same-directory hard link atomically publishes the fully synced
            # inode and fails if any filesystem object already occupies path.
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError("refusing to overwrite an existing output") from exc
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry where the platform exposes directory fsync."""

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
        # Some otherwise valid filesystems do not support directory fsync.
        pass
    finally:
        os.close(descriptor)


MappingLike = dict[str, Any]


if __name__ == "__main__":
    raise SystemExit(main())
