# Memory v2 shadow retrieval

Memory v2 now has an offline path for measuring whether derived, scoped
evidence would improve an answer. It is experimental, disabled by default,
read-only, and not connected to the live Hermes provider.

## What the path does

One authorized run performs these bounded stages:

1. A memory-need router skips acknowledgements, simple questions, and queries
   already answerable from the current conversation. Ambiguous continuity
   queries receive only a five-candidate search.
2. Corpus hygiene removes machine prompts, compaction artifacts, and copied
   fork/subagent turns while retaining lineage.
3. A deterministic builder derives typed exact-span evidence and resolves
   project/workstream scope. Assistant completion statements remain
   unverified until a later user or tool event corroborates them through an
   explicit event link, or bounded workstream and semantic overlap.
4. A disposable SQLite/FTS index applies profile, tenant, evidence-cutoff,
   privacy, visibility, current/history, and workstream filters before
   ranking. An optional pinned local embedding adapter may add dense recall.
5. A utility reranker can abstain with `NONE` and returns at most five
   untrusted evidence items. An optional pinned local model may score the
   bounded set, but cannot call tools or authorize any mutation.

The output retains exact source spans and evidence timestamps. Every result
declares `shadow_only`, `read_only`, `untrusted_data`, and
`mutation_authority: none`.

Every source event must carry explicit profile and tenant IDs matching the
request. Future, tombstoned, retrieval-disabled, disallowed-privacy, and
disallowed-visibility events are removed before deterministic extraction or
any optional adapter. Per-event, total-snapshot, context, candidate, and
result bounds are enforced. Evidence timestamps must be timezone-aware and are
ordered by their UTC instant. Extraction stops at bounded anchor, span, and
node budgets and reports overflow reasons and rejected-node counts.

If an operator-reviewed source event carries a recognized `memory_status`,
current queries suppress explicit historical/stale/superseded states while
history queries may retain them. The shadow path never invents or applies a
supersession relationship automatically.

## Private local command

Prepare a JSON object outside the repository:

```json
{
  "query": "What did we previously decide for the alpha migration?",
  "raw_events": [
    {
      "id": "event-1",
      "type": "turn",
      "user_content": "Decision: use SQLite FTS for alpha migration.",
      "observed_at": "2026-05-01T10:00:00Z",
      "project_id": "alpha",
      "workstream_id": "migration",
      "profile_id": "private-profile",
      "tenant_id": "private-tenant"
    }
  ],
  "profile_id": "private-profile",
  "tenant_id": "private-tenant",
  "evidence_cutoff": "2026-06-01T00:00:00Z",
  "context": {
    "has_current_context": false,
    "gap_days": 30,
    "project_id": "alpha",
    "workstream_id": "migration"
  }
}
```

Run:

```bash
python scripts/memory_v2_shadow_retrieval.py \
  --input /private/path/request.json \
  --output /private/path/result.json \
  --authorize-private-shadow
```

The command refuses repository-local private paths and existing outputs. It
also refuses live-Hermes output paths. Input is read once through a bounded
regular-file handle; duplicate JSON keys, excessive nesting, and non-finite
numbers are rejected. It uses an explicitly scoped scratch index, verifies the
evidence/FTS/edge/vector integrity manifest, deletes the index before
returning, publishes the private result with an atomic no-clobber operation,
and prints only a content-free receipt to stdout. Files use mode `0600` on
POSIX; on Windows, the operator must choose a directory whose inherited ACL is
private to the operator.

## Local-model boundary

The library accepts two optional local adapters:

- a versioned, fixed-dimension embedding adapter for candidate generation;
- a versioned utility scorer/abstainer with a pinned artifact digest and hard
  timeout.

Model output is strict-schema data, not authority. Invalid dimensions,
non-finite vectors, model/version/dimension/artifact identity mismatches,
timeouts, malformed responses, or model requests for authority fail closed or
use the configured deterministic fallback. No adapter can promote, supersede,
delete, execute, or invoke a tool. Timed-out in-process adapters open a
pipeline-lifetime circuit so repeated calls cannot accumulate workers. The CLI
enables no learned adapters; deploying adapter code remains a trusted-host
decision and should use a credential-free, network-disabled subprocess
boundary when real models are introduced.

## Evaluation gates

Do not connect shadow results to live answer context until a disjoint,
operator-labeled replay demonstrates all of the following:

- memory-needed routing recall is high and unnecessary-memory precision is
  acceptable;
- top-1 usefulness and complete causal-bundle rates beat the FTS baseline;
- `NONE` precision/recall is calibrated;
- irrelevant-injection, stale-current, and invalid-citation rates remain near
  zero;
- gains persist across people, projects, query classes, and 30/90/365-day
  checkpoints;
- local-model latency and failure fallback remain within the configured
  bounds.

Metrics with no eligible labeled cases are reported as unavailable, not as
perfect scores, and include explicit coverage/eligibility fields.

Even after those gates pass, automatic promotion, automatic supersession,
broad raw prefetch, and model-driven promotion remain separate decisions and
stay disabled.
