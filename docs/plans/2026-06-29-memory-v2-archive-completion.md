# Memory v2 Archive Completion Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Finish the Memory v2 archive/session-backfill/retrieval layer so it is production-grade: source-grounded, privacy-safe, adversarially hardened, low-compute at large archive scale, operationally usable, and eval-gated.

**Architecture:** Keep raw JSONL as the canonical append-only evidence log, and make all fast lookup/search state derived and rebuildable. Hot paths should use bounded/indexed lookup and evidence packets; full archive scans belong only in explicit verify/rebuild/repair jobs. Archive content is always untrusted evidence, never instructions or current belief.

**Tech Stack:** Python, SQLite/FTS5, JSONL/YAML canonical stores, Hermes MemoryProvider plugin tools, pytest, deterministic local eval fixtures, optional CLI wrappers.

---

## Planning participants

This plan synthesizes three independent viewpoints:

1. **Locke / parent planner:** prioritizes the minimum end-to-end product definition: archive evidence, source refs, backfill, retrieval, CLI/runbook, and release gates.
2. **Subagent A — architecture/performance:** focused on raw archive metadata indexes, O(1)/bounded lookup, source-resolution complexity, and large-store performance gates.
3. **Subagent B — trust/safety/evals:** focused on threat model, privacy, adversarial memory, source integrity, docs, rollout, and eval proof.

## Current state as of `1037450e8`

Implemented:

- `memory_v2_session_backfill`
  - dry-run by default
  - confirmation-gated mutation via `IMPORT_SESSIONDB_TO_MEMORY_V2`
  - read-only SQLite access to `state.db`
  - active-profile path guard
  - deterministic raw event ids for idempotency
  - redacted user/assistant/tool SessionDB messages as raw archive evidence
- `memory_v2_archive_search`
  - bounded evidence packets
  - filters by query/session/event type/source ids/date
  - source refs, hashes, chain metadata, integrity status
  - untrusted/non-instruction labels
- `memory_v2_archive_show`
  - single raw event packet by id
  - optional hash pin
  - optional neighbor ids/hashes only
- local-path redaction added to Memory v2 redaction helpers
- tests added for backfill, idempotency, source-grounded retrieval, no broad dump, cross-profile rejection, and golden tool schemas

Known gaps:

- Archive search/show/backfill/source-resolution still use full archive scans in normal paths.
- Integrity checks can accidentally become expensive if run per result.
- No raw-event offset/index table dedicated to archive lookup.
- No documented CLI/operator workflow for backfill/search/show/rebuild.
- Tool-output import is capped/redacted/untrusted, but still needs stronger policy/evals.
- No formal archive threat model or adversarial conversation-loop tests.
- No complete longitudinal/source-grounded benchmark proving Memory v2 beats raw FTS.
- Full repo suite has at least one unrelated ACP approval failure; Memory v2 targeted tests are the reliable gate for this work until that is separated.

## Definition of complete

The archive layer is complete only when all of these are true:

### Architecture and data model

- Raw JSONL remains canonical append-only evidence.
- A rebuildable raw archive metadata/search index exists with ids, offsets, hashes, type/session/time filters, import keys, and source metadata.
- Derived index health can detect stale/missing/corrupt state.
- Raw archive index can be rebuilt from JSONL from scratch.
- Source refs and raw events have O(1) or indexed lookup paths.
- Full archive scans are confined to explicit verify/rebuild/repair operations.

### Performance

- `memory_v2_archive_show` is bounded: one indexed lookup plus one record hydration.
- `memory_v2_archive_search` is bounded: indexed query plus `limit + 1` record hydrations.
- SessionDB backfill idempotency does not scan the archive.
- Review/dream/extraction source validation does not preload all raw events.
- Structural perf tests fail if hot paths call `read_raw_events()` unbounded.
- Large synthetic perf gates pass reliably.

### Safety/privacy

- Backfilled content is redacted before storage, indexing, source refs, reports, and prompt-visible packets.
- Tool outputs are capped, redacted, and always lower-trust/untrusted.
- Archive retrieval never returns full raw dumps.
- `can_instruct=false` is preserved for archive/tool/imported evidence.
- Cross-profile imports are rejected unless an explicitly tested override exists.
- Non-dry-run backfill requires exact confirmation.
- Tombstone/delete/export/rebuild policy exists for future privacy control.

### Source grounding

- Every durable semantic/core memory can be traced to valid source refs or is explicitly marked manual/low-confidence.
- Source refs resolve to bounded evidence packets.
- Hash-chain/integrity status is surfaced without misleading the caller.
- Tampered/dangling source refs block unsafe promotion.
- Summaries are treated as indexes, not evidence.

### Adversarial safety

- Raw archive, assistant logs, and tool output are never instruction-bearing.
- Prompt-injection red-team tests pass through provider and conversation-loop paths.
- Unsafe memory poisoning attempts are rejected or held for review.
- Retrieved archive packets are strongly fenced as `UNTRUSTED ARCHIVE EVIDENCE`.

### Evals and release

- Deterministic evals cover longitudinal recall, stale facts, source recall, irrelevant suppression, contradiction, privacy, adversarial memory, latency, and token budget.
- Memory v2 beats raw FTS on the agreed source-grounded benchmark, not just on storage completeness.
- Privacy leakage and adversarial instruction-following rates are zero for release gates.
- Operational docs/CLI exist for dry-run import, confirmed import, resume, search, show, verify, rebuild, privacy scan, eval, and rollback.

---

## Phase 0: Freeze threat model and invariants

**Objective:** Make safety and correctness criteria explicit before deeper implementation.

**Files:**

- Create: `docs/memory-v2-archive-threat-model.md`
- Create: `docs/memory-v2-archive-invariants.md`
- Modify: `docs/plans/memory-v2-spec.md`
- Test: `tests/plugins/memory/test_memory_v2_archive_invariants.py`

**Tasks:**

1. Write the threat model covering private data leakage, cross-profile contamination, prompt injection, forged/tampered archive records, stale evidence, tool-result poisoning, broad dump attempts, and accidental public fixture leakage.
2. Define trust taxonomy: user message, assistant message, tool result, file, web, manual note, imported SessionDB, generated summary.
3. Define privacy levels: `public`, `standard`, `sensitive`, `secret`, `blocked`.
4. Define lifecycle: raw evidence → candidate → reviewed candidate → semantic/core memory → superseded/expired/rejected.
5. Add invariant tests proving archive packets always carry `untrusted_text`, `can_instruct=false`, bounded excerpts, source refs, and no unfiltered dump path.

**Verification:**

```bash
source venv/bin/activate
python -m pytest tests/plugins/memory/test_memory_v2_archive_invariants.py -q
python -m pytest tests/plugins/memory/test_memory_v2_session_backfill.py -q
```

**Acceptance criteria:**

- Invariants are documented and test-backed.
- Every archive provider tool has a written safety contract.
- Tests fail if archive text can appear as trusted instructions.

---

## Phase 1: Build a rebuildable raw archive index

**Objective:** Stop treating JSONL as the hot-path query structure while keeping it canonical.

**Files:**

- Modify: `plugins/memory/memory_v2/index.py`
- Modify: `plugins/memory/memory_v2/store.py`
- Create: `tests/plugins/memory/test_memory_v2_raw_archive_index.py`

**Design:**

Add derived SQLite tables, either in the existing `memory.sqlite` or a sibling derived DB:

```sql
CREATE TABLE IF NOT EXISTS raw_events (
  id TEXT PRIMARY KEY,
  event_type TEXT,
  source_system TEXT,
  session_id TEXT,
  provider_session_id TEXT,
  message_id INTEGER,
  created_at TEXT,
  observed_at TEXT,
  archive_status TEXT,
  trust_level TEXT,
  privacy_level TEXT,
  can_instruct INTEGER,
  chain_index INTEGER,
  record_sha256 TEXT,
  previous_record_sha256 TEXT,
  content_sha256 TEXT,
  byte_offset INTEGER,
  byte_length INTEGER,
  line_no INTEGER,
  source_ref_id TEXT,
  import_key_sha256 TEXT,
  indexed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_raw_events_session_created
  ON raw_events(session_id, created_at, id);
CREATE INDEX IF NOT EXISTS idx_raw_events_type_created
  ON raw_events(event_type, created_at, id);
CREATE INDEX IF NOT EXISTS idx_raw_events_created
  ON raw_events(created_at, id);
CREATE INDEX IF NOT EXISTS idx_raw_events_import_key
  ON raw_events(import_key_sha256);
CREATE INDEX IF NOT EXISTS idx_raw_events_message
  ON raw_events(source_system, provider_session_id, message_id);
CREATE INDEX IF NOT EXISTS idx_raw_events_record_hash
  ON raw_events(record_sha256);
```

Add FTS:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS raw_events_fts USING fts5(
  id UNINDEXED,
  user_content,
  assistant_content,
  content,
  tool,
  session_id UNINDEXED,
  event_type UNINDEXED
);
```

**Tasks:**

1. Add raw archive index schema and migrations.
2. Add `index_raw_archive_event(event, byte_offset, byte_length, line_no)`.
3. Add `rebuild_raw_archive_index(store)` that streams JSONL and records offsets.
4. Update `append_raw_event` flow to update derived raw index after canonical append.
5. Add manifest/index status fields: `derived_index_status`, `indexed_event_count`, `last_indexed_record_sha256`, `raw_index_schema_version`.

**Verification:**

```bash
python -m pytest tests/plugins/memory/test_memory_v2_raw_archive_index.py -q
python -m pytest tests/plugins/memory/test_memory_v2_store.py -q
```

**Acceptance criteria:**

- Raw event by id can be resolved without reading the full archive.
- Import key existence can be checked without scanning JSONL.
- Search/filter can return matching ids/offsets via SQLite.
- Index rebuild from JSONL is deterministic.
- Raw JSONL remains canonical and sufficient for recovery.

---

## Phase 2: Add bounded raw archive store APIs

**Objective:** Make the safe path the easy path in code.

**Files:**

- Modify: `plugins/memory/memory_v2/store.py`
- Modify: `plugins/memory/memory_v2/index.py`
- Test: `tests/plugins/memory/test_memory_v2_raw_archive_index.py`
- Test: `tests/plugins/memory/test_memory_v2_raw_archive_perf.py`

**APIs to add:**

```python
iter_raw_events(...)
get_raw_event_by_id(event_id)
get_raw_events_by_ids(ids)
read_raw_event_at_offset(byte_offset, byte_length)
search_raw_events(query='', session_id='', event_type='', source_ids=None, created_after='', created_before='', limit=5)
raw_event_exists(event_id)
raw_import_key_exists(import_key_sha256)
get_raw_event_neighbors(event_id)
```

**Tasks:**

1. Implement exact lookup through SQLite metadata + JSONL byte slice.
2. Implement search through raw FTS/filter tables, hydrating only `limit + 1` records.
3. Implement neighbor lookup via indexed `chain_index`.
4. Keep existing `read_raw_events(limit=N)` for bounded tail/debug reads.
5. Rename or clearly fence unbounded read paths for repair-only usage.

**Verification:**

```bash
python -m pytest tests/plugins/memory/test_memory_v2_raw_archive_index.py -q
python -m pytest tests/plugins/memory/test_memory_v2_raw_archive_perf.py -q
```

**Acceptance criteria:**

- `archive_show` dependencies can be O(1).
- `archive_search` dependencies are O(log n + limit)`.
- Source existence checks are O(1).
- Full scans are clearly limited to verify/rebuild/repair.

---

## Phase 3: Rewrite provider archive search/show to use bounded APIs

**Objective:** Make the public/provider archive tools production-safe in complexity and payload semantics.

**Files:**

- Modify: `plugins/memory/memory_v2/__init__.py`
- Modify: `tests/plugins/memory/test_memory_v2_session_backfill.py`
- Modify: `tests/plugins/memory/test_memory_v2_provider_tool_golden_schema.py`
- Create: `tests/plugins/memory/test_memory_v2_raw_archive_perf.py`

**Tasks:**

1. Rewrite `_archive_search` to query raw archive index and hydrate only `limit + 1` events.
2. Rewrite `_archive_show` to use `get_raw_event_by_id`.
3. Replace `has_more = len(matches) == limit` with `limit + 1` based pagination.
4. Change integrity semantics from repeated full-chain verification to explicit modes:
   - `manifest`
   - `event`
   - `full` only in health/repair or explicit expensive mode
5. Ensure neighbor ids use indexed chain neighbors, not full-list positions.
6. Preserve current JSON shapes unless intentionally versioned.

**Verification:**

```bash
python -m pytest tests/plugins/memory/test_memory_v2_session_backfill.py -q
python -m pytest tests/plugins/memory/test_memory_v2_provider_tool_golden_schema.py -q
python -m pytest tests/plugins/memory/test_memory_v2_raw_archive_perf.py -q
```

**Acceptance criteria:**

- `memory_v2_archive_search` never calls unbounded `read_raw_events()`.
- `memory_v2_archive_show` never calls unbounded `read_raw_events()`.
- Integrity status says exactly what was checked.
- Search/show still return bounded, redacted, untrusted evidence packets.

---

## Phase 4: Productionize SessionDB backfill

**Objective:** Make backfill resumable, large-scale, and operationally safe.

**Files:**

- Modify: `plugins/memory/memory_v2/session_backfill.py`
- Modify: `plugins/memory/memory_v2/__init__.py`
- Create: `plugins/memory/memory_v2/backfill_state.py` if useful
- Modify: `tests/plugins/memory/test_memory_v2_session_backfill.py`
- Modify: `tests/plugins/memory/test_memory_v2_raw_archive_perf.py`

**Tasks:**

1. Replace `existing_ids = {event.id for event in store.read_raw_events()}` with raw-index checks.
2. Add backfill checkpoint state, e.g. `memory_v2/backfill/sessiondb.yaml`:

```yaml
sessiondb:
  source: discord
  session_id: ''
  last_message_id: 12345
  imported_count: 10000
  skipped_count: 200
  last_run_at: '...'
  state_db_fingerprint: 'sha256:...'
```

3. Add `resume=true` behavior.
4. Add `batch_size` and optional `max_batches` for operator workflows.
5. Add lock file or equivalent guard against concurrent backfills.
6. Keep dry-run default and confirmation-gated mutation.
7. Keep report text private: counts, cursors, safe ids only.

**Verification:**

```bash
python -m pytest tests/plugins/memory/test_memory_v2_session_backfill.py -q
python -m pytest tests/plugins/memory/test_memory_v2_raw_archive_perf.py -q
```

**Acceptance criteria:**

- Re-running backfill against large archives is idempotent without full scans.
- Interrupted import resumes safely.
- Concurrent import is rejected or serialized.
- Dry run never mutates archive, index, or checkpoint.
- Cross-profile path protection remains enforced.

---

## Phase 5: Harden source-ref resolution across Memory v2

**Objective:** Remove hidden full-archive scans from source validation/review/dream/extraction paths.

**Files:**

- Modify: `plugins/memory/memory_v2/extraction.py`
- Modify: `plugins/memory/memory_v2/review.py`
- Modify: `plugins/memory/memory_v2/review_actions.py`
- Modify: `plugins/memory/memory_v2/operations.py`
- Modify: `plugins/memory/memory_v2/consolidation.py`
- Modify: `plugins/memory/memory_v2/health.py`
- Test: `tests/plugins/memory/test_memory_v2_load_perf.py`
- Test: `tests/plugins/memory/test_memory_v2_raw_archive_perf.py`

**Tasks:**

1. Replace source-existence fallbacks like `any(event.id == source_id for event in read_raw_events())` with `read_source_ref` + raw index existence.
2. Update review queue to hydrate only source refs needed by reviewed candidates, not all raw events.
3. Update dream/extraction/operation validators to use bounded source APIs.
4. Keep full archive source inventory only for explicit health/repair flows.

**Verification:**

```bash
python -m pytest tests/plugins/memory/test_memory_v2_load_perf.py -q
python -m pytest tests/plugins/memory/test_memory_v2_provider.py -q
python -m pytest tests/plugins/memory/test_memory_v2_raw_archive_perf.py -q
```

**Acceptance criteria:**

- Candidate review/dream/source validation does not preload all raw events.
- Missing/dangling source detection remains correct.
- Review `limit` bounds source hydration.

---

## Phase 6: Add privacy hardening and tombstone policy

**Objective:** Make archive safe to operate on real personal history.

**Files:**

- Modify: `plugins/memory/memory_v2/redaction.py`
- Modify: `plugins/memory/memory_v2/store.py`
- Create: `plugins/memory/memory_v2/privacy_scan.py`
- Create: `tests/plugins/memory/test_memory_v2_archive_privacy.py`
- Create: `docs/memory-v2-privacy.md`

**Tasks:**

1. Expand redaction regression coverage for API keys, bearer tokens, GitHub tokens, private keys, URLs with credentials, cookies/session IDs, local paths, Discord IDs where policy says sensitive.
2. Add raw event fields: `redaction_version`, `redaction_findings_count`, `contains_sensitive_placeholders`, `blocked_reason`.
3. Add tool-output caps and binary rejection rules.
4. Add privacy scanner for archive artifacts, source refs, reports, docs, eval outputs, and fixtures.
5. Design tombstone/delete/export workflow:
   - tombstone raw event or mark unavailable
   - rebuild indexes
   - invalidate source refs
   - avoid re-leaking deleted text in audit logs

**Verification:**

```bash
python -m pytest tests/plugins/memory/test_memory_v2_archive_privacy.py -q
python -m pytest tests/plugins/memory/test_memory_v2_session_backfill.py -q
```

**Acceptance criteria:**

- No raw secret/path leaks into archive, index, source refs, provider outputs, reports, or docs fixtures.
- Tool outputs are capped and marked lower trust.
- There is a documented deletion/tombstone path even if full GDPR-style deletion is a later phase.

---

## Phase 7: Add adversarial memory and prompt-injection tests

**Objective:** Prove archive evidence cannot become instructions.

**Files:**

- Create: `tests/plugins/memory/test_memory_v2_adversarial_archive.py`
- Create: `tests/agent/test_memory_v2_archive_prompt_injection.py` if conversation-loop integration is feasible
- Modify: `plugins/memory/memory_v2/retrieval.py`
- Modify: `plugins/memory/memory_v2/__init__.py`
- Create: `docs/memory-v2-adversarial-memory.md`

**Tasks:**

1. Add fixtures with fake system/developer messages, tool-output injections, delimiter-breaking Markdown, unsafe memory requests, stale exfiltration claims, and keyword-stuffed poison records.
2. Harden packet formatting with explicit `UNTRUSTED ARCHIVE EVIDENCE` boundaries.
3. Ensure packet text is escaped or structured so it cannot masquerade as system/developer/tool calls.
4. Add write-gate protections for security-policy and external-action preference claims.
5. Add conversation-loop tests where retrieved memory tries to override policy.

**Verification:**

```bash
python -m pytest tests/plugins/memory/test_memory_v2_adversarial_archive.py -q
python -m pytest tests/agent/test_memory_v2_archive_prompt_injection.py -q
```

**Acceptance criteria:**

- Raw/tool/archive text cannot instruct the agent.
- Unsafe memory poisoning is rejected or pending review.
- Delimiter-breaking attempts remain quoted evidence.
- Current system/developer/user hierarchy always wins.

---

## Phase 8: Expand deterministic evals and LoCoMo-style benchmark

**Objective:** Prove Memory v2 improves memory quality, not just storage completeness.

**Files:**

- Modify: `docs/memory-v2-evals.md`
- Modify/Create: `tests/fixtures/memory_v2/*.yaml`
- Modify/Create: `plugins/memory/memory_v2/evals/*.py` or existing eval harness paths
- Create: `tests/plugins/memory/test_memory_v2_archive_evals.py`

**Tasks:**

1. Add longitudinal fixtures: multi-session facts, preference changes, stale facts, project continuity, source recall, irrelevant suppression, contradiction, temporal questions, multi-hop source lookup.
2. Add baselines:
   - no memory
   - raw FTS
   - archive-only
   - semantic-only
   - routed Memory v2
3. Score:
   - answer correctness
   - temporal correctness
   - source attribution precision/recall
   - stale rejection
   - privacy leakage
   - adversarial instruction-following
   - irrelevant injection
   - token budget
   - latency
4. Add pass/fail release thresholds.

**Verification:**

```bash
python -m pytest tests/plugins/memory/test_memory_v2_archive_evals.py -q
# plus any existing memory eval command documented in docs/memory-v2-evals.md
```

**Acceptance criteria:**

- Memory v2 beats raw FTS on source-grounded longitudinal benchmark.
- Privacy leakage is zero in eval outputs.
- Adversarial instruction-following is zero.
- Token and latency budgets are enforced.

---

## Phase 9: Add operational CLI/runbook

**Objective:** Make archive operations usable without provider JSON spelunking.

**Files:**

- Modify/Create: `hermes_cli/commands.py` or appropriate CLI subcommand module
- Create: `tests/plugins/memory/test_memory_v2_cli.py` or CLI test path
- Create: `docs/memory-v2-archive-ops.md`
- Create: `docs/memory-v2-rollout.md`

**Commands to support:**

```bash
hermes memory-v2 archive status
hermes memory-v2 archive search --query "..." --limit 5
hermes memory-v2 archive show EVENT_ID --expect-record-sha256 sha256:...
hermes memory-v2 archive rebuild-index
hermes memory-v2 archive verify
hermes memory-v2 session-backfill dry-run --source discord --limit 5000
hermes memory-v2 session-backfill run --source discord --resume --confirm IMPORT_SESSIONDB_TO_MEMORY_V2
hermes memory-v2 privacy-scan
hermes memory-v2 eval
```

If full CLI is too much for the first implementation pass, create a script under `scripts/` with docs, then migrate to CLI.

**Verification:**

```bash
python -m pytest tests/plugins/memory/test_memory_v2_cli.py -q
python -m pytest tests/plugins/memory/test_memory_v2_session_backfill.py -q
```

**Acceptance criteria:**

- Dry-run is default for imports.
- Mutating import requires exact confirmation.
- Commands print safe summaries only.
- Operators can resume, verify, rebuild, and rollback.
- Docs include exact commands and failure recovery steps.

---

## Phase 10: Rollout gates and final release checklist

**Objective:** Move from implemented to safely dogfooded/releasable.

**Files:**

- Create: `docs/memory-v2-archive-release-checklist.md`
- Create/Modify: `plugins/memory/memory_v2/config.py` if feature flags are centralized
- Modify: `plugins/memory/memory_v2/README.md`

**Rollout stages:**

1. Synthetic fixtures only.
2. Isolated dogfood profile.
3. Limited personal-profile dry-run.
4. Small confirmed import with health checks.
5. Read-only archive search/show enabled.
6. Candidate extraction enabled.
7. Semantic promotion with review gate.
8. Limited automatic prefetch.
9. Broader release.

**Feature flags:**

```yaml
memory_v2:
  archive:
    enabled: true
    backfill_enabled: false
    search_tools_enabled: true
    show_tools_enabled: true
    include_tool_outputs: false
  consolidation:
    enabled: false
  prefetch:
    enabled: false
  auto_promote:
    enabled: false
```

**Release gate commands:**

```bash
source venv/bin/activate
python -m pytest tests/plugins/memory -q
python -m pytest tests/plugins/memory/test_memory_v2_raw_archive_perf.py -q
python -m pytest tests/plugins/memory/test_memory_v2_adversarial_archive.py -q
python -m pytest tests/plugins/memory/test_memory_v2_archive_privacy.py -q
python -m pytest tests/plugins/memory/test_memory_v2_archive_evals.py -q
python -m py_compile plugins/memory/memory_v2/*.py
```

Full repo:

```bash
python -m pytest tests/ -o 'addopts=' -q
```

If full suite still fails due to unrelated ACP approval behavior, document exact failing test and keep Memory v2 release gate separate.

**Acceptance criteria:**

- Dogfood report proves safety/quality before default enablement.
- Privacy scan passes.
- Eval thresholds pass.
- Archive health is `ok` after backfill and rebuild.
- Docs are current and synthetic-only.
- Branch/release artifacts contain no private data.

---

## Implementation order recommendation

Do not jump directly to semantic intelligence. Finish the archive substrate first:

1. Phase 0 threat model/invariants.
2. Phase 1 raw archive index.
3. Phase 2 bounded APIs.
4. Phase 3 rewrite archive search/show.
5. Phase 4 productionize backfill.
6. Phase 5 remove hidden full-scan source-resolution paths.
7. Phase 6 privacy hardening.
8. Phase 7 adversarial tests.
9. Phase 8 evals.
10. Phase 9 CLI/runbook.
11. Phase 10 rollout gates.

## Immediate next task if executing

Start with a small first PR/commit:

**Task:** Add archive threat model + invariant tests.

**Why first:** It prevents later performance work from accidentally weakening the safety contract.

**Initial tests to write:**

- `test_archive_search_rejects_unfiltered_dump`
- `test_archive_show_packet_is_untrusted_and_non_instructional`
- `test_archive_packet_has_source_ref_and_hashes`
- `test_backfill_dry_run_is_non_mutating`
- `test_backfill_rejects_cross_profile_state_db`

Then proceed to raw archive index design.

## Notes for future context recovery

If context runs out, resume from this file and inspect current code with:

```bash
git status --short
git log --oneline -5
python -m pytest tests/plugins/memory/test_memory_v2_session_backfill.py tests/plugins/memory/test_memory_v2_provider_tool_golden_schema.py -q
```

The most important unresolved engineering issue is **full archive scans on hot paths**. The most important unresolved trust issue is **proving archive evidence cannot become instruction/policy**. The most important product issue is **an operator CLI/runbook for safe backfill and verification**.
