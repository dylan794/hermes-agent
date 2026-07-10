"""Privacy scanner for Memory v2 artifacts and reports.

The scanner is intentionally local, deterministic, and report-only. It reports
bounded metadata about sensitive-looking text without returning raw secret values.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .redaction import sensitive_findings

_TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".csv",
    ".log",
    ".py",
}
_DEFAULT_SCAN_DIRS = (
    "inbox",
    "sources",
    "reports",
    "evals",
    "artifacts/manifests",
    "artifacts/derived",
)


def scan_memory_v2_privacy(
    base_dir: str | Path,
    *,
    paths: Iterable[str | Path] | None = None,
    max_files: int = 500,
    max_findings: int = 100,
) -> Dict[str, Any]:
    """Scan Memory v2 text artifacts for unredacted sensitive-looking values."""
    root = Path(base_dir).expanduser().resolve()
    requested = list(paths or _DEFAULT_SCAN_DIRS)
    files = _iter_scan_files(root, requested, max_files=max_files)
    findings: List[Dict[str, Any]] = []
    scanned = 0
    skipped_binary = 0
    for path in files:
        if len(findings) >= max_findings:
            break
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data:
            skipped_binary += 1
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            skipped_binary += 1
            continue
        scanned += 1
        rel = _safe_rel(path, root)
        for finding in sensitive_findings(text):
            if len(findings) >= max_findings:
                break
            findings.append(
                {
                    "path": rel,
                    "path_sha256": _sha256(rel),
                    "kind": str(finding.get("kind") or "unknown"),
                    "start": int(finding.get("start") or 0),
                    "end": int(finding.get("end") or 0),
                }
            )
    return {
        "success": True,
        "mode": "memory_v2_privacy_scan",
        "report_only": True,
        "scanned_files": scanned,
        "skipped_binary_files": skipped_binary,
        "finding_count": len(findings),
        "truncated": len(findings) >= max_findings,
        "findings": findings,
    }


def _iter_scan_files(root: Path, requested: List[str | Path], *, max_files: int) -> List[Path]:
    out: List[Path] = []
    for entry in requested:
        path = (root / entry).resolve() if not Path(entry).is_absolute() else Path(entry).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if path.is_file():
            out.append(path)
        elif path.is_dir():
            out.extend(sorted(child for child in path.rglob("*") if child.is_file()))
        if len(out) >= max_files:
            return out[:max_files]
    return out[:max_files]


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return f"outside:{_sha256(str(path))}"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
