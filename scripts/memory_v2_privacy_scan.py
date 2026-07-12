#!/usr/bin/env python3
"""CI-capable privacy scanner for Memory v2 diffs and artifacts.

By default this scans added lines in the current git diff. Use --paths to scan
complete files/directories, --staged for the staged diff, or --base-ref to scan
changes against a release/publication base.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]

TEXT_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".env",
    ".ini",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}

SYNTHETIC_BAIT_ALLOW_MARKER = "privacy-scan: synthetic-bait-ok"

ALLOW_MARKERS = (
    "privacy-scan: allow",
    "privacy-scan allow",
    "allowlist secret",
    "allowlisted secret",
)

FAKE_CONTEXT_WORDS = (
    "dummy",
    "example",
    "fake",
    "fixture",
    "mock",
    "placeholder",
    "sample",
    "sandbox",
)

PATH_USER_ALLOWLIST = {
    "alice",
    "bob",
    "charlie",
    "demo",
    "example",
    "fake",
    "fixture",
    "mock",
    "sample",
    "test",
    "user",
    "username",
    "yourname",
}

PRIVATE_PATH_RE = re.compile(
    r"(?P<path>"
    r"(?:file://)?/(?:home|Users)/(?P<unix_user>[A-Za-z0-9._-]+)(?:/[A-Za-z0-9._~+\-@%:,=]+)+"
    r"|(?:file://)?/mnt/[A-Za-z]/Users/(?P<wsl_user>[A-Za-z0-9._-]+)(?:/[A-Za-z0-9._~+\-@%:,=]+)+"
    r"|[A-Za-z]:\\Users\\(?P<win_user>[A-Za-z0-9._-]+)(?:\\[^\\\s'\"<>|]+)+"
    r")"
)
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?P<key_quote>['\"]?)(?P<key>(?:[A-Z][A-Z0-9_]*(?:_KEY|_TOKEN|_SECRET|_PASSWORD|_CREDENTIAL|API_KEY|ACCESS_TOKEN|ACCESS_KEY_ID)|AWS_ACCESS_KEY_ID|[a-z][a-z0-9_]*(?:key|token|secret|password|credential)|api_key|access_token|key|token|secret|password|credential))(?P=key_quote)"
    r"\s*(?:=|:)\s*"
    r"(?P<quote>['\"]?)(?P<value>[^'\"\s#,}]+)(?P=quote)",
    re.IGNORECASE,
)
BEARER_RE = re.compile(r"(?P<header>Authorization\s*:\s*Bearer\s+)(?P<token>[^\s'\"]+)", re.IGNORECASE)
PEM_PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
SNOWFLAKE_RE = re.compile(r"(?<!\d)(?P<id>[1-9]\d{16,20})(?!\d)")

_STRONG_SECRET_VALUE_RE = re.compile(
    r"(?:"
    r"sk-[A-Za-z0-9._-]{8,}"
    r"|ghp_[A-Za-z0-9_]{8,}"
    r"|github_pat_[A-Za-z0-9_]{12,}"
    r"|xox[baprs]-[A-Za-z0-9-]{8,}"
    r"|AKIA[A-Z0-9]{8,}"
    r"|ASIA[A-Z0-9]{8,}"
    r"|eyJ[A-Za-z0-9_-]{12,}"
    r")"
)
_GENERIC_CODE_KEY_NAMES = {
    "key",
    "candidate_key",
    "subject_key",
    "predicate_key",
    "group_key",
    "route_key",
    "type_key",
    "edge_key",
    "cluster_key",
    "safe_key",
    "subkey",
    "raw_key",
    "import_key",
    "provided_import_key",
    "fallback_import_key",
    "bogus_key",
}
_SENSITIVE_KEY_FRAGMENTS = (
    "api",
    "auth",
    "access",
    "bearer",
    "credential",
    "password",
    "passwd",
    "secret",
    "token",
    "openai",
    "anthropic",
    "github",
    "aws",
)


@dataclass(frozen=True)
class ScanProfile:
    mode: str
    description: str
    allow_markers: tuple[str, ...]
    allow_fake_context: bool
    allow_path_user_allowlist: bool
    default_paths: tuple[Path, ...] = ()
    default_source: str = "diff"


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    column: int
    kind: str
    message: str
    snippet: str


RELEASE_ARTIFACT_PATHS = tuple(
    ROOT / path
    for path in (
        "plugins/memory/memory_v2",
        "scripts/memory_v2_archive_ops.py",
        "scripts/memory_v2_eval.py",
        "scripts/memory_v2_privacy_scan.py",
        "docs/memory-v2-archive-release-checklist.md",
        "docs/memory-v2-evals.md",
        "docs/memory-v2-privacy.md",
        "docs/memory-v2-p0-status.md",
        "docs/memory-v2-p1-extraction.md",
        "plugins/memory/memory_v2/README.md",
        "tests/plugins/memory/test_memory_v2_release_docs.py",
    )
)

ADVERSARIAL_FIXTURE_PATHS = (
    ROOT / "tests" / "fixtures" / "memory_v2_privacy_scan" / "adversarial_bait.py",
)

PROFILE_DEFAULT = ScanProfile(
    mode="diff-or-path",
    description="Backward-compatible Memory v2 diff/--paths scan.",
    allow_markers=(*ALLOW_MARKERS, SYNTHETIC_BAIT_ALLOW_MARKER),
    allow_fake_context=True,
    allow_path_user_allowlist=True,
)

SCAN_PROFILES: dict[str, ScanProfile] = {
    "memory-v2-release-artifacts": ScanProfile(
        mode="memory-v2-release-artifacts",
        description="Memory v2 release artifact gate; scans deterministic public release files unless --paths is supplied.",
        allow_markers=(SYNTHETIC_BAIT_ALLOW_MARKER,),
        allow_fake_context=False,
        allow_path_user_allowlist=False,
        default_paths=RELEASE_ARTIFACT_PATHS,
        default_source="mode-default-paths",
    ),
    "full-repo-public-hygiene": ScanProfile(
        mode="full-repo-public-hygiene",
        description="Broad public hygiene scan across the repository. This is not the Memory v2 release gate.",
        allow_markers=(SYNTHETIC_BAIT_ALLOW_MARKER,),
        allow_fake_context=False,
        allow_path_user_allowlist=False,
        default_paths=(ROOT,),
        default_source="mode-default-paths",
    ),
    "intentional-adversarial-fixtures": ScanProfile(
        mode="intentional-adversarial-fixtures",
        description="Scans synthetic adversarial privacy fixtures; every bait line must use privacy-scan: synthetic-bait-ok.",
        allow_markers=(SYNTHETIC_BAIT_ALLOW_MARKER,),
        allow_fake_context=False,
        allow_path_user_allowlist=False,
        default_paths=ADVERSARIAL_FIXTURE_PATHS,
        default_source="mode-default-paths",
    ),
}


def profile_for_mode(mode: str | None) -> ScanProfile:
    if not mode:
        return PROFILE_DEFAULT
    try:
        return SCAN_PROFILES[mode]
    except KeyError as exc:
        known = ", ".join(sorted(SCAN_PROFILES))
        raise ValueError(f"unknown privacy scan mode {mode!r}; expected one of: {known}") from exc


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        # Do not echo absolute caller/artifact paths in CI output; those paths are
        # often exactly what this scanner is meant to catch.
        return f"<external>/{path.name}"


def _line_is_allowlisted(line: str, profile: ScanProfile = PROFILE_DEFAULT) -> bool:
    lowered = line.lower()
    return any(marker in lowered for marker in profile.allow_markers)


def _has_fake_context(line: str) -> bool:
    lowered = line.lower()
    return any(word in lowered for word in FAKE_CONTEXT_WORDS)


def _looks_like_secret_assignment(key: str, value: str, line: str, profile: ScanProfile = PROFILE_DEFAULT) -> bool:
    """Return true for likely committed secrets while ignoring ordinary code keys."""
    normalized_key = key.strip("'\"").lower()
    normalized_value = value.strip().strip("'\"")
    value_lower = normalized_value.lower()
    if not normalized_value:
        return False
    if value_lower in {
        "false",
        "true",
        "none",
        "null",
        "0",
        "1",
        "redacted",
        "xxx",
        "xxxx",
        "changeme",
        "fake",
        "dummy",
        "example",
        "placeholder",
        "test",
        "sample",
    }:
        return False
    if normalized_key in _GENERIC_CODE_KEY_NAMES:
        return False
    # Avoid flagging ordinary Python/JSON structure such as candidate_key=candidate.id
    # or key = normalize_key(value).  Strong token-looking values still trip below.
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?(?:\([^\n]*\))?", normalized_value):
        return False
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\(", normalized_value):
        return False
    key_is_sensitive = any(fragment in normalized_key for fragment in _SENSITIVE_KEY_FRAGMENTS)
    if _STRONG_SECRET_VALUE_RE.search(normalized_value):
        return True
    if not key_is_sensitive:
        return False
    if profile.allow_fake_context and _has_fake_context(line) and value_lower in {"fake", "dummy", "example", "placeholder", "test", "sample"}:
        return False
    # Sensitive-looking keys with non-trivial literal values are worth reviewing,
    # but short identifiers/booleans should not make source scans unusable.
    return len(normalized_value) >= 8 and any(ch in normalized_value for ch in ("-", "_", ".", "/", ":", "="))


def _redact_path(match: re.Match[str]) -> str:
    path = match.group("path")
    user = match.group("unix_user") or match.group("wsl_user") or match.group("win_user")
    return path.replace(user, "<user>", 1) if user else "<path>"


def _redact_snowflake(value: str) -> str:
    if len(value) <= 8:
        return "<id>"
    return f"{value[:4]}…{value[-4:]}"


def _finding(path: str, line_no: int, match: re.Match[str], kind: str, message: str, snippet: str) -> Finding:
    return Finding(
        path=path,
        line=line_no,
        column=match.start() + 1,
        kind=kind,
        message=message,
        snippet=snippet,
    )


def scan_line(path: str, line_no: int, line: str, profile: ScanProfile = PROFILE_DEFAULT) -> list[Finding]:
    """Scan one logical content line and return redacted findings."""
    if _line_is_allowlisted(line, profile):
        return []

    findings: list[Finding] = []

    for match in PRIVATE_PATH_RE.finditer(line):
        user = match.group("unix_user") or match.group("wsl_user") or match.group("win_user") or ""
        if profile.allow_path_user_allowlist and user.lower() in PATH_USER_ALLOWLIST:
            continue
        findings.append(
            _finding(
                path,
                line_no,
                match,
                "private_path",
                "local user/home path may leak private machine details",
                _redact_path(match),
            )
        )

    for match in SECRET_ASSIGNMENT_RE.finditer(line):
        key = match.group("key")
        value = match.group("value")
        if not _looks_like_secret_assignment(key, value, line, profile):
            continue
        findings.append(
            _finding(
                path,
                line_no,
                match,
                "secret_assignment",
                "env-style secret assignment should not be committed or published",
                f"{key}=[REDACTED]",
            )
        )

    for match in BEARER_RE.finditer(line):
        token = match.group("token")
        if (
            token.upper() in {"REDACTED", "TOKEN", "BEARER_TOKEN"}
            or token.lower().startswith("<")
            or not token.strip("*")
        ):
            continue
        findings.append(
            _finding(
                path,
                line_no,
                match,
                "bearer_token",
                "Authorization Bearer token should not be committed or published",
                "Authorization: Bearer ***",
            )
        )

    for match in PEM_PRIVATE_KEY_RE.finditer(line):
        findings.append(
            _finding(
                path,
                line_no,
                match,
                "private_key_pem",
                "private-key PEM material should not be committed or published",
                "-----BEGIN [REDACTED] PRIVATE KEY-----",
            )
        )

    for match in SNOWFLAKE_RE.finditer(line):
        value = match.group("id")
        if (profile.allow_fake_context and _has_fake_context(line)) or len(set(value)) <= 2:
            continue
        findings.append(
            _finding(
                path,
                line_no,
                match,
                "snowflake_id",
                "long Discord/platform snowflake-like ID may leak private identifiers",
                _redact_snowflake(value),
            )
        )

    return findings


def _is_probably_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            chunk = fh.read(4096)
    except OSError:
        return True
    return b"\0" in chunk


def iter_path_lines(paths: Sequence[Path]) -> Iterator[tuple[str, int, str]]:
    files: list[Path] = []
    for input_path in paths:
        path = input_path.resolve()
        if path.is_dir():
            for root, dirnames, filenames in os.walk(path):
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
                for filename in sorted(filenames):
                    candidate = Path(root) / filename
                    if candidate.suffix in TEXT_EXTENSIONS:
                        files.append(candidate)
        elif path.is_file():
            files.append(path)

    for file_path in sorted(set(files), key=lambda p: _repo_relative(p)):
        if _is_probably_binary(file_path):
            continue
        try:
            with file_path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    yield _repo_relative(file_path), line_no, line.rstrip("\n")
        except UnicodeDecodeError:
            continue


def _run_git(args: Sequence[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{proc.stderr.strip()}")
    return proc.stdout


def current_diff_lines(*, base_ref: str | None, staged: bool) -> Iterator[tuple[str, int, str]]:
    args = ["diff", "--no-ext-diff", "--unified=0"]
    if staged:
        args.append("--cached")
    elif base_ref:
        args.append(base_ref)
    diff = _run_git(args)

    current_path: str | None = None
    new_line_no = 0
    for raw in diff.splitlines():
        if raw.startswith("+++ b/"):
            current_path = raw[6:]
            continue
        if raw.startswith("+++ "):
            current_path = raw[4:]
            continue
        if raw.startswith("@@"):
            match = re.search(r"\+(\d+)(?:,(\d+))?", raw)
            new_line_no = int(match.group(1)) if match else 0
            continue
        if current_path is None:
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            yield current_path, new_line_no, raw[1:]
            new_line_no += 1
        elif raw.startswith("-") and not raw.startswith("---"):
            continue
        else:
            new_line_no += 1


def scan_records(records: Iterable[tuple[str, int, str]], profile: ScanProfile = PROFILE_DEFAULT) -> list[Finding]:
    findings: list[Finding] = []
    for path, line_no, line in records:
        findings.extend(scan_line(path, line_no, line, profile))
    return sorted(findings, key=lambda f: (f.path, f.line, f.column, f.kind))


def render_text(findings: Sequence[Finding]) -> str:
    if not findings:
        return "Memory v2 privacy scan: no findings."
    lines = [f"Memory v2 privacy scan: {len(findings)} finding(s)."]
    for item in findings:
        lines.append(
            f"{item.path}:{item.line}:{item.column}: {item.kind}: {item.message} [{item.snippet}]"
        )
    return "\n".join(lines)


def scan_scope_for_args(args: argparse.Namespace, profile: ScanProfile) -> tuple[Iterator[tuple[str, int, str]], dict[str, object]]:
    if args.paths:
        return iter_path_lines(args.paths), {
            "source": "explicit-paths",
            "paths": [_repo_relative(path) for path in args.paths],
        }
    if profile.default_paths:
        return iter_path_lines(profile.default_paths), {
            "source": profile.default_source,
            "paths": [_repo_relative(path) for path in profile.default_paths],
        }
    source = "staged-diff" if args.staged else ("base-ref-diff" if args.base_ref else "worktree-diff")
    scope: dict[str, object] = {"source": source}
    if args.base_ref:
        scope["base_ref"] = args.base_ref
    if args.staged:
        scope["staged"] = True
    return current_diff_lines(base_ref=args.base_ref, staged=args.staged), scope


def json_payload(profile: ScanProfile, findings: Sequence[Finding], scan_scope: dict[str, object]) -> dict[str, object]:
    return {
        "mode": profile.mode,
        "description": profile.description,
        "success": not findings,
        "finding_count": len(findings),
        "scan_scope": scan_scope,
        "findings": [asdict(f) for f in findings],
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=tuple(sorted(SCAN_PROFILES)),
        help="Named scan mode/profile. Omit for the backward-compatible diff/--paths scan.",
    )
    parser.add_argument(
        "--release",
        choices=("memory-v2",),
        help="Convenience alias for --mode memory-v2-release-artifacts.",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        type=Path,
        help="Scan complete files/directories instead of only added diff lines, or override a mode's default paths.",
    )
    parser.add_argument(
        "--base-ref",
        help="Git ref to diff against when --paths/--staged are not used (example: origin/main). Ignored by modes with default paths.",
    )
    parser.add_argument("--staged", action="store_true", help="Scan staged diff instead of worktree diff.")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. JSON is deterministic and CI-friendly.",
    )
    args = parser.parse_args(argv)
    if args.release == "memory-v2":
        if args.mode and args.mode != "memory-v2-release-artifacts":
            parser.error("--release memory-v2 cannot be combined with a different --mode")
        args.mode = "memory-v2-release-artifacts"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    profile = profile_for_mode(args.mode)
    records, scan_scope = scan_scope_for_args(args, profile)
    findings = scan_records(records, profile)

    if args.format == "json":
        print(json.dumps(json_payload(profile, findings, scan_scope), indent=2, sort_keys=True))
    else:
        print(render_text(findings))
    return 1 if findings else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
