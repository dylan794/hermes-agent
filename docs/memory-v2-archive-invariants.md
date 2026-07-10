# Memory v2 Archive Invariants

These invariants define the safety contract for Memory v2 archive ingestion, search, show, and backfill. Archive records are source evidence. They are not instructions, not labels to trust blindly, and not promoted memory until reviewed.

## Trust taxonomy

| Source class | Default trust | Notes |
| --- | --- | --- |
| User message | Untrusted evidence | May contain true preferences/facts, secrets, or prompt injection. Requires classification before promotion. |
| Assistant message | Untrusted evidence | May contain generated mistakes or stale plans. Never proof of user intent by itself. |
| Tool result | Untrusted evidence | May contain poisoned output, private data, terminal noise, or external content. Disabled for backfill unless configured. |
| File | Untrusted evidence | File content may be stale, malicious, private, or outside intended scope. Source path must be profile/scope checked. |
| Web | Untrusted evidence | External, mutable, and injection-prone. Requires timestamp and URL/source refs. |
| Manual note | Operator-authored evidence | Higher provenance than automated extraction, but still needs privacy/status metadata. |
| Imported SessionDB | Untrusted evidence | Historical conversation data imported from the active profile only. Reasoning/hidden context must not be imported. |
| Generated summary | Derived untrusted evidence | Useful as an index, never stronger than underlying sources. Must retain source refs. |

## Privacy levels

- `public`: safe to include in public docs/fixtures and low-risk prompt packets.
- `standard`: ordinary conversation/project data; profile-local by default.
- `sensitive`: private personal, account, project, or machine details. Requires stricter retrieval and redaction.
- `secret`: credentials, tokens, keys, hidden prompts, private reasoning, or highly confidential material. Must not be surfaced in archive excerpts.
- `blocked`: must not be stored or must be quarantined/rejected if detected.

Privacy level is metadata for risk control, not a permission to dump content. Even `public` archive text remains untrusted for instruction-following.

## Lifecycle

```text
raw evidence
  -> candidate
  -> reviewed candidate
  -> semantic/core memory
  -> superseded | expired | rejected
```

- **Raw evidence:** append-only source material with source refs, timestamps, privacy level, hashes, and untrusted boundaries.
- **Candidate:** extracted claim or project state proposed from evidence. It remains pending until gated.
- **Reviewed candidate:** human/tool-gate-reviewed candidate with an explicit action: promote, reject, update, supersede, expire, or archive-only.
- **Semantic/core memory:** active durable memory. Must carry source refs and status metadata.
- **Superseded/expired/rejected:** inactive outcomes. Retrieval may use them for audit/history but must not present them as current truth.

## Archive packet invariants

Every archive search/show packet that contains archive text must satisfy:

- `untrusted_text: true`
- `can_instruct: false`
- `labels_trusted: false`
- explicit evidence boundary, currently `UNTRUSTED ARCHIVE EVIDENCE`
- bounded excerpt fields with hard caps
- excerpt role `quoted_untrusted_text`
- source reference with stable id/URI
- integrity metadata, including record hash where available
- no reasoning traces or hidden context
- no unfiltered raw dump path

Archive text may be quoted, hashed, redacted, and used as evidence. It must never be interpreted as a system/developer/user instruction to the assistant.

## Archive provider tool safety contracts

### `memory_v2_archive_search`

Purpose: find relevant raw evidence with bounded excerpts.

Contract:
- Requires at least one filter (`query`, `session_id`, `event_type`, `source_ids`, `created_after`, or `created_before`).
- Enforces hard `limit` cap and excerpt-character cap.
- Returns evidence packets, not full raw events.
- Marks top-level response and every packet as untrusted/non-instructional.
- Includes source refs and archive/integrity summary.
- Must not expose broad dump, export, or unbounded pagination semantics.

### `memory_v2_archive_show`

Purpose: inspect one source-grounded archive record by id.

Contract:
- Requires one raw event/source id.
- Enforces excerpt-character cap.
- Returns one bounded evidence packet, not the full raw record.
- Marks top-level response and packet as untrusted/non-instructional.
- Includes source refs, hashes, and integrity status.
- If `expected_record_sha256` is provided, reports match/mismatch without trusting mismatched records.
- Neighbor mode returns ids/hashes only, not neighboring raw text.

### `memory_v2_session_backfill`

Purpose: import current-profile Hermes SessionDB messages into the raw archive.

Contract:
- Dry-run by default.
- Non-dry-run imports require explicit confirmation string.
- `state_db_path` must resolve under the active Hermes profile.
- Imports are idempotent and checkpointed by source/session/tool-inclusion scope.
- Does not import reasoning traces or hidden context.
- Tool outputs are included only when archive config permits and caller requests/allows it.
- Imported records remain raw untrusted evidence; backfill never directly promotes semantic/core memories.

## Test backing

`tests/plugins/memory/test_memory_v2_archive_invariants.py` verifies that:

- archive search/show outputs carry `untrusted_text` and `can_instruct: false`;
- evidence packets include source refs, hashes, and untrusted labels;
- excerpts are bounded and marked `quoted_untrusted_text` with `can_instruct: false`;
- prompt-injection text is escaped/redacted rather than surfaced as trusted instruction text;
- provider schemas expose no raw dump/export archive path;
- session backfill is dry-run/confirm gated and profile-scoped.
