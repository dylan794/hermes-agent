"""Shared redaction helpers for Memory v2.

These helpers are intentionally conservative. Memory v2 stores raw evidence and
retrieval logs locally, so every persistence boundary should redact common
credential shapes even when the caller already tried to sanitize input.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

REDACTION = "[REDACTED]"
SENSITIVE_QUERY = "[REDACTED sensitive query]"
REDACTION_VERSION = 2

_PRIVATE_KEY_RE = re.compile(r"(?is)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----")
_URI_CREDENTIAL_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://[^\s:/?#]+:)([^\s@/?#]+)(@)")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
_TOKEN_PREFIX_RE = re.compile(
    r"(?i)\b(gh[pousr]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{12,}|xox[baprs]-[A-Za-z0-9-]{8,}|sk-[A-Za-z0-9][A-Za-z0-9_-]{8,}|sk-proj-[A-Za-z0-9_-]{8,}|AIza[0-9A-Za-z_-]{20,}|AKIA[0-9A-Z]{16})\b"
)
_LABELED_SECRET_RE = re.compile(
    r"(?is)"
    r"("
    r"(?:authorization\s*:\s*bearer\s+)|"
    r"(?:bearer\s+)|"
    r"(?:[A-Z0-9_]*(?:API[_-]?KEY|PRIVATE[_-]?KEY|SECRET[_-]?ACCESS[_-]?KEY|ACCESS[_-]?TOKEN|AUTH[_-]?TOKEN|CLIENT[_-]?SECRET|TOKEN|PASSWORD|PASSWD|SECRET|CREDENTIAL)\s*(?:=|:|is)?\s*)|"
    r"(?:api[_ -]?key\s*(?:=|:|is)?\s*)|"
    r"(?:private\s+key\s*(?:=|:|is)?\s*)|"
    r"(?:client\s+secret\s*(?:=|:|is)?\s*)|"
    r"(?:password\s*(?:=|:|is)?\s*)|"
    r"(?:passwd\s*(?:=|:|is)?\s*)|"
    r"(?:token\s*(?:=|:|is)?\s*)|"
    r"(?:secret\s*(?:=|:|is)?\s*)|"
    r"(?:credential\s*(?:=|:|is)?\s*)|"
    r"(?:(?:openai|anthropic|github|gitlab|aws|azure|google|gcp|slack|discord|stripe|huggingface|hf)\s+(?:api\s+)?key\s*(?:=|:|is)?\s*)"
    r")"
    r"([^\s,;\]\}\)]+)"
)
_HIGH_ENTROPY_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)[A-Z0-9_]*\s*(?:=|:|is)\s*)([^\s,;\]\}\)]+)"
)
_LOCAL_PATH_RE = re.compile(
    r"(?i)(?:/home/[^\s\"'<>]+|/mnt/[a-z]/Users/[^\s\"'<>]+|[A-Z]:\\\\Users\\\\[^\s\"'<>]+)"
)
_COOKIE_SESSION_RE = re.compile(
    r"(?is)\b((?:cookie|set-cookie|sessionid|sid)\s*(?:=|:)?\s*)([^\s,;\]\}\)]+)"
)
_DISCORD_ID_RE = re.compile(r"\b(?:discord(?:[_ -]?(?:user|guild|channel|message))?[_ -]?id\s*(?:=|:)?\s*)?(\d{17,20})\b", re.IGNORECASE)


_FINDING_PATTERNS = [
    ("private_key", _PRIVATE_KEY_RE),
    ("uri_credentials", _URI_CREDENTIAL_RE),
    ("jwt", _JWT_RE),
    ("token_prefix", _TOKEN_PREFIX_RE),
    ("labeled_secret", _LABELED_SECRET_RE),
    ("high_entropy_assignment", _HIGH_ENTROPY_ASSIGNMENT_RE),
    ("local_path", _LOCAL_PATH_RE),
    ("cookie_or_session", _COOKIE_SESSION_RE),
    ("discord_id", _DISCORD_ID_RE),
]

_INSTRUCTION_LIKE_PATTERNS = [
    re.compile(r"(?i)\bSYSTEM\s*:"),
    re.compile(r"(?i)\bDEVELOPER\s*:"),
    re.compile(r"(?i)\bTOOL\s*:"),
    re.compile(r"(?i)ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions"),
    re.compile(r"(?i)promote\s+this\s+memory\s+automatically"),
    re.compile(r"(?i)reveal\s+(?:hidden\s+)?system\s+prompts?"),
    re.compile(r"(?i)\btool_call\b"),
    re.compile(r"(?i)\bfunction_call\b"),
    re.compile(r"(?i)\bmemory_v2_promote\b"),
    re.compile(r"(?i)\bcandidate_id\b"),
    re.compile(r"``+"),
]


def redact_text(text: str) -> str:
    """Redact common credential forms from text."""

    redacted = str(text or "")
    redacted = _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", redacted)
    redacted = _URI_CREDENTIAL_RE.sub(lambda match: f"{match.group(1)}{REDACTION}{match.group(3)}", redacted)
    redacted = _LABELED_SECRET_RE.sub(lambda match: match.group(0) if match.group(2).startswith(REDACTION) else f"{match.group(1)}{REDACTION}", redacted)
    redacted = _HIGH_ENTROPY_ASSIGNMENT_RE.sub(lambda match: match.group(0) if match.group(2).startswith(REDACTION) else f"{match.group(1)}{REDACTION}", redacted)
    redacted = _TOKEN_PREFIX_RE.sub(REDACTION, redacted)
    redacted = _JWT_RE.sub(REDACTION, redacted)
    redacted = _LOCAL_PATH_RE.sub("[REDACTED PATH]", redacted)
    redacted = _COOKIE_SESSION_RE.sub(lambda match: f"{match.group(1)}{REDACTION}", redacted)
    redacted = _DISCORD_ID_RE.sub(lambda match: match.group(0).replace(match.group(1), "[REDACTED DISCORD ID]"), redacted)
    return redacted


def sensitive_findings(text: str) -> List[Dict[str, Any]]:
    """Return bounded metadata about sensitive-looking text without values."""
    raw = str(text or "")
    findings: List[Dict[str, Any]] = []
    for kind, pattern in _FINDING_PATTERNS:
        for match in pattern.finditer(raw):
            findings.append({"kind": kind, "start": int(match.start()), "end": int(match.end())})
            if len(findings) >= 50:
                findings.append({"kind": "omitted", "count": -1})
                return findings
    return findings


def escape_untrusted_evidence_text(text: str) -> tuple[str, int]:
    """Redact instruction-shaped archive text before putting it in packets."""
    safe = redact_text(str(text or ""))
    redactions = 0
    for pattern in _INSTRUCTION_LIKE_PATTERNS:
        safe, count = pattern.subn("[REDACTED INSTRUCTION-LIKE TEXT]", safe)
        redactions += count
    return safe, redactions


def redaction_metadata(value: Any) -> Dict[str, Any]:
    """Summarize redaction findings for JSON/YAML-like data without leaking values."""
    finding_count = 0

    def visit(item: Any) -> None:
        nonlocal finding_count
        if isinstance(item, str):
            finding_count += len(sensitive_findings(item))
        elif isinstance(item, dict):
            for key, child in item.items():
                finding_count += len(sensitive_findings(str(key)))
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return {
        "redaction_version": REDACTION_VERSION,
        "redaction_findings_count": int(finding_count),
        "contains_sensitive_placeholders": finding_count > 0,
    }


def contains_sensitive_text(text: str) -> bool:
    """Return true if text appears to contain a credential-like value."""

    raw = str(text or "")
    return redact_text(raw) != raw


def redacted_query_for_log(query: str) -> str:
    """Return a safe retrieval-log query string."""

    text = str(query or "")
    if contains_sensitive_text(text):
        return SENSITIVE_QUERY
    return text[:500]


def redacted_query_hash_input(query: str) -> str:
    """Return canonical query text to hash for retrieval logs.

    Sensitive queries intentionally hash the redacted sentinel, not the raw
    secret-bearing string, so low-entropy secret guesses cannot be verified by
    comparing hashes in exported logs.
    """

    return redacted_query_for_log(query)


def redact_data(value: Any) -> Any:
    """Recursively redact strings in JSON/YAML-like data."""

    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_data(item) for item in value)
    if isinstance(value, dict):
        return {key: redact_data(item) for key, item in value.items()}
    return value
