# Memory v2 privacy and tombstone policy

Memory v2 is designed to operate on real personal history without turning the raw archive into an uncontrolled dump.

## Persistence boundaries

- Raw archive events are redacted before they are written and hash-chained.
- Raw archive events are always non-instructing evidence (`can_instruct: false`); normal turns use `trust_level: untrusted`, and tool-result events use `trust_level: tool_output_untrusted`.
- Tool-result events get an additional output cap before persistence.
- Binary-looking values are replaced with an omission marker instead of being serialized into JSONL.
- Source refs and archive tools return bounded evidence packets, not full raw dumps.

Each raw event records privacy metadata:

- `redaction_version`
- `redaction_findings_count`
- `contains_sensitive_placeholders`
- `blocked_reason`

These fields are meant for audits and scanner gating. They should not be treated as proof that text is safe to export publicly.

## Privacy scanner

`plugins.memory.memory_v2.privacy_scan.scan_memory_v2_privacy()` is report-only. It scans selected Memory v2 text artifacts such as inbox files, source refs, reports, evals, and derived artifact metadata for credential/path/session-id patterns.

The scanner intentionally returns only:

- relative path
- path hash
- finding kind
- character offsets
- counts and truncation state

It must not return raw secret values.

## Tombstones vs hard delete

Raw archive JSONL is append-only so integrity checks and source provenance remain auditable. A tombstone is therefore the default deletion primitive:

1. `MemoryV2Store.tombstone_raw_event(event_id, reason=..., actor=...)` writes privacy-safe metadata under `privacy/raw_event_tombstones.yaml`.
2. The tombstone stores no raw event text.
3. The matching source ref is overwritten with an unavailable/tombstoned quote.
4. Bounded raw archive hydration returns a tombstone stub instead of the original event.
5. Source existence checks treat tombstoned raw events as unavailable.
6. Rebuild Memory v2 indexes after batches of tombstones so derived search state catches up.

This is not full GDPR-style physical deletion yet. Physical deletion would require rewriting the append-only archive, rebuilding indexes/source refs, invalidating candidate/source references, and retaining only non-sensitive audit metadata. Until that exists, tombstone is the safe default because it prevents re-serving deleted text without silently breaking provenance.

## Audit-log rule

Deletion/tombstone audit records may include ids, timestamps, reason summaries after redaction, actor names after redaction, and hashes. They must not copy raw user turns, assistant text, tool output, file contents, or secret values.

## Export rule

Before exporting Memory v2 artifacts outside the local profile:

1. Run the privacy scanner over `inbox`, `sources`, `reports`, `evals`, and artifact metadata.
2. Review findings locally.
3. Exclude `inbox/raw_events.jsonl` unless there is explicit user approval and a separate export redaction pass.
4. Prefer reports and derived metadata over raw archive data.

## Known limitations

- Regex redaction catches common credentials and paths, not every possible secret.
- Tombstones suppress serving/hydration but do not erase historical bytes from append-only JSONL.
- Existing stale derived indexes may still contain searchable tokens until rebuilt.
- Unsalted hashes can confirm guesses for low-entropy values if exported; keep scanner reports local unless sanitized further.
