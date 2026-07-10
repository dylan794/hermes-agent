# Memory v2 Archive Threat Model

Memory v2 raw archive records are evidence, not instructions and not durable truth. The archive can contain user text, assistant text, tool output, files, web content, imported SessionDB messages, and generated summaries. All of those inputs may be stale, private, adversarial, or corrupted.

## Security goals

- Keep archive data profile-scoped under the active `hermes_home`.
- Prevent private data from leaking into prompt packets, logs, fixtures, docs, or public releases.
- Preserve source-grounded auditability without exposing unbounded raw dumps.
- Ensure archive text cannot instruct the assistant, override policies, or self-promote into memory.
- Detect forged, tampered, stale, or low-trust evidence before promotion.

## Trust boundaries

- **Archive ingestion boundary:** raw evidence enters from conversations, tools, files, web, manual notes, SessionDB imports, and generated summaries. It is untrusted until reviewed.
- **Provider output boundary:** archive search/show tools return bounded evidence packets. These packets are data only; they must carry `untrusted_text: true` and `can_instruct: false`.
- **Promotion boundary:** candidates become semantic/core memory only after gates evaluate source refs, privacy, staleness, contradiction, confidence, and user intent.
- **Profile boundary:** archive import, search, indexes, checkpoints, and source refs must remain inside the active profile unless an operator explicitly supplies a safe in-profile path.

## Threats and mitigations

### Private data leakage

Threats:
- Secrets, paths, credentials, personal identifiers, or private conversation excerpts appear in archive files or tool responses.
- Redacted evidence is re-expanded through source refs, hashes, debug output, or public fixtures.

Mitigations:
- Classify every record with a privacy level: `public`, `standard`, `sensitive`, `secret`, or `blocked`.
- Apply redaction before indexing and before provider output.
- Return bounded excerpts and hashes, not full records.
- Never include reasoning traces or hidden context in archive imports.
- Keep public fixtures synthetic and scan them for private-looking data.

### Cross-profile contamination

Threats:
- A backfill imports another Hermes profile's `state.db`.
- Derived indexes or checkpoints share data across profiles.
- Hardcoded `~/.hermes` paths bypass profile isolation.

Mitigations:
- Resolve all archive paths under active `hermes_home`.
- Reject `memory_v2_session_backfill.state_db_path` outside the active profile.
- Keep raw archive, indexes, manifests, and checkpoints profile-local.
- Hash local paths in checkpoints where possible instead of storing absolute paths.

### Prompt injection from archived text

Threats:
- Archived user, assistant, tool, file, web, or summary text says “ignore previous instructions,” “reveal secrets,” or similar.
- Search/show output is copied into the prompt as if it were trusted instruction text.

Mitigations:
- Treat every archive excerpt as quoted evidence with `untrusted_text: true`, `can_instruct: false`, `labels_trusted: false`, and an explicit evidence boundary.
- Escape or redact instruction-like phrases in excerpts.
- Tests must fail if archive packets can appear as trusted instructions.
- Promotion gates must classify claims separately from source wording.

### Forged or tampered archive records

Threats:
- Raw archive JSONL is edited, reordered, truncated, or replaced.
- A caller supplies a forged record ID or hash pin.

Mitigations:
- Store content and record hashes with chain metadata.
- Search/show include integrity status and record hashes.
- `memory_v2_archive_show.expected_record_sha256` reports hash-pin mismatch instead of silently trusting content.
- Derived indexes remain rebuildable from canonical archive files and manifests.

### Stale evidence

Threats:
- Old preferences, project states, environment facts, or web facts are retrieved as current truth.
- Generated summaries outlive the evidence they summarized.

Mitigations:
- Preserve `created_at`, `observed_at`, source refs, status, and validity windows.
- Apply stale penalties during retrieval and require source verification for high-stakes or contradictory claims.
- Move records through superseded, expired, or rejected lifecycle states instead of keeping all claims active.

### Tool-result poisoning

Threats:
- Tool output contains adversarial instructions, fabricated data, private files, or terminal noise.
- A tool result is over-weighted because it looks machine-generated.

Mitigations:
- Tool outputs are optional for backfill and disabled unless configured.
- Tool-result excerpts remain untrusted evidence and are capped.
- Tool outputs must carry source refs and never become semantic memory without review.

### Broad dump attempts

Threats:
- A caller asks the archive provider to dump the entire raw archive, bypass excerpt caps, or return all records.
- Large search results expose private data by aggregation.

Mitigations:
- Expose only bounded archive provider tools: `memory_v2_archive_search`, `memory_v2_archive_show`, and confirmed/dry-run `memory_v2_session_backfill`.
- Enforce hard caps on search limits and excerpt lengths.
- Require at least one search filter for archive search.
- Do not add raw dump/export tools to agent-visible schemas.

### Accidental public fixture leakage

Threats:
- Tests or benchmark fixtures include real user messages, secrets, local paths, IDs, or proprietary data.
- Redaction tests accidentally preserve sensitive samples in expected output.

Mitigations:
- Use synthetic fixture text and fake credentials only.
- Prefer hash assertions and redaction assertions over real private examples.
- Review fixtures before release; block fixtures with real-looking secrets, profile paths, or personal data.

## Residual risks

- Hashes prove integrity of stored records, not truth of claims.
- Redaction can miss novel secret formats.
- Manual notes and generated summaries can be wrong even when well-formed.
- A user can intentionally request exact old wording; the provider must still enforce caps and privacy policy.
