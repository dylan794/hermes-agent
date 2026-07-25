"""Privacy hardening tests for Memory v2 raw archives."""

from __future__ import annotations

import json

from plugins.memory.memory_v2.index import MemoryV2Index
from plugins.memory.memory_v2.privacy_scan import scan_memory_v2_privacy
from plugins.memory.memory_v2.redaction import redact_text, redaction_metadata
from plugins.memory.memory_v2.store import MemoryV2Store


def _fake_openai_key() -> str:
    return "sk-proj-" + ("a" * 28)


def _fake_github_token() -> str:
    return "ghp_" + ("b" * 36)


def _fake_jwt() -> str:
    return "eyJ" + ("a" * 12) + "." + ("b" * 12) + "." + ("c" * 12)


def _fake_private_key() -> str:
    return "-----BEGIN PRIVATE KEY-----\nabc123\n-----END PRIVATE KEY-----"


SECRET_VALUES = [
    "OPENAI_API_KEY=" + _fake_openai_key(),
    "authorization: " + "bearer " + _fake_jwt(),
    _fake_github_token(),
    _fake_private_key(),
    "https://user:" + "pa" + "ss" + "@example.com/private",
    "cookie: supersecretcookie",
    "/home/dylan_kinsman/private/file.txt",
    r"C:\Users\Dylan\Documents\private-notes.txt",
    "discord user id: 123456789012345678",
]


def _store(tmp_path):
    store = MemoryV2Store(tmp_path / "memory_v2")
    store.initialize()
    return store


def _index(store):
    index = MemoryV2Index(store.base_dir / "indexes" / "memory.sqlite")
    index.initialize()
    return index


def test_redaction_covers_common_secret_path_and_id_patterns():
    joined = "\n".join(SECRET_VALUES)
    redacted = redact_text(joined)

    assert _fake_openai_key() not in redacted
    assert _fake_jwt() not in redacted
    assert _fake_github_token() not in redacted
    assert "BEGIN PRIVATE KEY" not in redacted
    assert "user:pass" not in redacted
    assert "supersecretcookie" not in redacted
    assert "/home/dylan_kinsman" not in redacted
    assert r"C:\Users\Dylan" not in redacted
    assert "123456789012345678" not in redacted
    meta = redaction_metadata({"text": joined})
    assert meta["redaction_version"] >= 2
    assert meta["redaction_findings_count"] >= len(SECRET_VALUES)
    assert meta["contains_sensitive_placeholders"] is True


def test_raw_event_normalization_adds_privacy_metadata_and_never_stores_secret_values(tmp_path):
    store = _store(tmp_path)
    event = store.append_raw_event(
        {
            "type": "turn",
            "session_id": "privacy-session",
            "user_content": "token=" + _fake_openai_key() + " path /mnt/c/Users/Dylan/secret.txt",
            "assistant_content": b"binary\x00payload",
            "discord_user_id": "123456789012345678",
        }
    )
    raw_json = store.raw_events_path.read_text(encoding="utf-8")

    assert event["redaction_version"] >= 2
    assert event["redaction_findings_count"] >= 3
    assert event["contains_sensitive_placeholders"] is True
    assert event["blocked_reason"] == ""
    assert _fake_openai_key() not in raw_json
    assert "/mnt/c/Users/Dylan" not in raw_json
    assert "123456789012345678" not in raw_json
    assert "binary\\u0000payload" not in raw_json
    assert "rejected binary content" in raw_json


def test_tool_raw_events_are_capped_and_marked_lower_trust(tmp_path):
    store = _store(tmp_path)
    event = store.append_raw_event(
        {
            "type": "tool",
            "session_id": "tool-session",
            "tool": "terminal",
            "content": "x" * 20_000,
            "trust_level": "trusted",
            "can_instruct": True,
        }
    )

    assert event["trust_level"] == "tool_output_untrusted"
    assert event["can_instruct"] is False
    assert event["tool_output_capped"] is True
    assert len(event["content"]) < 9_000
    assert "tool-output chars" in event["content"]


def test_privacy_scanner_reports_metadata_without_secret_values(tmp_path):
    store = _store(tmp_path)
    leak_path = store.base_dir / "reports" / "leaky.json"
    leak_path.write_text(json.dumps({"token": _fake_github_token()}), encoding="utf-8")

    report = scan_memory_v2_privacy(store.base_dir, paths=["reports"], max_findings=10)
    serialized = json.dumps(report, sort_keys=True)

    assert report["success"] is True
    assert report["finding_count"] >= 1
    assert report["findings"][0]["kind"] in {"token_prefix", "labeled_secret", "high_entropy_assignment"}
    assert _fake_github_token() not in serialized
    assert "leaky.json" in serialized
    assert "path_sha256" in report["findings"][0]


def test_tombstone_raw_event_suppresses_archive_hydration_and_redacts_audit(tmp_path):
    store = _store(tmp_path)
    event = store.append_raw_event(
        {
            "type": "turn",
            "session_id": "delete-me",
            "user_content": "please forget secret value " + _fake_github_token(),
        }
    )
    index = _index(store)
    index.rebuild_raw_archive_index(store)

    result = store.tombstone_raw_event(
        event["id"],
        reason="user requested deletion of " + _fake_github_token(),
        actor="privacy-test",
    )
    shown = store.get_raw_event_by_id(event["id"], index=index)
    source = store.read_source_ref(event["id"])
    audit_json = store.operations_path.read_text(encoding="utf-8")
    tombstone_json = store.raw_event_tombstones_path.read_text(encoding="utf-8")

    assert result["status"] == "tombstoned"
    assert store.raw_event_exists(event["id"], index=index) is False
    assert shown is not None
    assert shown["archive_status"] == "tombstoned"
    assert _fake_github_token() not in json.dumps(shown)
    assert source is not None
    assert source.quote == "[TOMBSTONED RAW EVENT]"
    assert _fake_github_token() not in audit_json
    assert _fake_github_token() not in tombstone_json
