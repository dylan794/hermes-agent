# Memory v2 P0 hardening status

Date: 2026-07-11

## What P0 now guarantees

- The plugin implementation, fixtures, scripts, tests, and documentation are versioned together on a dedicated branch.
- Local evaluation ingests through `MemoryV2Provider.sync_turn()`, consolidates through provider hooks, and retrieves through `MemoryV2Provider.prefetch()`.
- Eval retrieval does not inspect expected answers or expected source IDs. Query session authority must be supplied independently through query metadata.
- Raw archive rows remain excluded from generic semantic search. Exact/deep prefetch can hydrate a bounded number of raw events only when `archive.enabled` and `archive.prefetch_raw_enabled` are explicitly enabled, and only for the provider's active session.
- Full Hermes turn trajectories may be passed to `sync_turn(messages=...)`. Tool results are redacted, bounded, deduplicated, archived as canonical raw evidence, and represented only as pending episode candidates.
- Provider-owned session state is authoritative. Caller-supplied `session_id` values do not widen raw access or change captured provenance.
- Promotion grounding requires canonical raw evidence, an existing durable memory record, or an existing artifact record. A writable source-reference sidecar alone is not canonical evidence.
- Direct promote/reject tools require a fresh review-plan ID, candidate fingerprint, and exact confirmation token. Forced promotion is rejected. Direct open-loop mutation is hidden until it has an equivalent review-plan contract. Contradiction auto-supersession is disabled; it may create review candidates only.
- Durable review mutations use a cross-process profile lock and a prepared/committed journal. A failure after canonical mutation begins leaves `recovery_required`, blocks later mutations, and is not auto-repaired.
- Recalled memory is wrapped as untrusted context/evidence, not as authoritative instructions.

## Fail-closed defaults

These remain disabled unless explicitly configured:

```yaml
memory_v2:
  archive:
    enabled: false
    capture_enabled: false
    search_tools_enabled: false
    show_tools_enabled: false
    prefetch_raw_enabled: false
    include_tool_outputs: false
  extraction:
    enabled: false
    candidate_creation_enabled: false
  consolidation:
    enabled: false
  prefetch:
    enabled: false
  review_apply:
    enabled: false
```

## Evaluation contract

`memory_v2_eval.py --chronological-contracts` produces deterministic local 30-, 90-, and 365-day scenarios and evaluates acceptance per horizon. The current benchmark reports measured results and exits non-zero when any gate is missed.

**The existence of these contracts is not a claim that Memory v2 beats human recall or the baseline today.** The current longitudinal gates remain unmet. P0 makes that failure truthful and reproducible; improving retrieval quality is subsequent work.

The legacy hard benchmark likewise reports its current miss rather than using expected source labels or fixture-only precision filters to manufacture a pass.

## Retired pre-P0 contracts

Recovered tests that assert deliberately removed behavior are retained as exact, strict expected failures. They cover contracts such as:

- raw archive rows in generic semantic search;
- direct/forced mutation without a confirmed review plan;
- automatic contradiction supersession;
- frontier/dogfood workflows before core readiness;
- benchmark tests that require a win regardless of measured output.

Exact node IDs live in `tests/plugins/memory/conftest.py`. New failures are not dynamically quarantined.

## Verification commands

```bash
venv/bin/python -m pytest tests/plugins/memory/test_memory_v2_*.py \
  tests/plugins/memory/evals \
  tests/agent/test_memory_v2_conversation_loop_session_prefetch.py \
  tests/agent/test_memory_provider.py -q

venv/bin/python scripts/memory_v2_eval.py \
  --dataset plugins/memory/memory_v2/evals/fixtures/hard_longitudinal_memory_v2_v1.yaml \
  --baseline raw_fts --baseline memory_v2 \
  --output /tmp/memory-v2-hard.json
venv/bin/python scripts/memory_v2_eval.py \
  --chronological-contracts --baseline memory_v2 \
  --output /tmp/memory-v2-longitudinal.json
```

For benchmark commands, non-zero exit status is expected whenever acceptance gates are not met; inspect the emitted report rather than treating process success as the metric.
