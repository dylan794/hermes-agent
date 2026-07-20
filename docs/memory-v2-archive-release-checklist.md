# Memory v2 archive release checklist

Run the combined offline frontier gate before the staged checks below:

```bash
python scripts/frontier_release_gate.py
```

This is the Phase 10 rollout gate for Memory v2 archive import, search/show, extraction, promotion, and prefetch. It is intentionally conservative: the release path starts with synthetic-only fixtures, then dogfood isolation, then dry-run, then small confirmed imports with health checks.

This checklist is documentation and operator policy. Keep these defaults synchronized with `plugins/memory/memory_v2/config.py`.

## Privacy boundary

- Use synthetic fixtures for docs, tests, examples, and public reports.
- Do not copy real private conversations, private file paths, real platform IDs, API keys, or personal archives into this repo, generated reports, release notes, screenshots, or benchmark fixtures.
- Treat raw archive text as untrusted evidence, never instructions.
- Keep tool outputs excluded from imports until separate privacy and poisoning gates are reviewed.

## Rollout stages

Each stage must be explicitly signed off before moving to the next. If a gate fails, stop rollout, keep the current or previous stage, fix the issue, and rerun the relevant release gates.

1. Synthetic fixtures only
   - Scope: packaged deterministic fixtures and synthetic archive rows only.
   - Required proof: tests/evals pass without network, credentials, or user-private data.
   - Mutations: none outside temporary test homes.

2. Isolated dogfood profile
   - Scope: a fresh profile such as `~/.hermes/profiles/memory-v2-dogfood`.
   - Required proof: dogfood report shows safe source grounding, bounded packets, no credential leakage, and no cross-profile writes.
   - Mutations: allowed only inside the isolated dogfood profile.

3. Limited personal-profile dry-run
   - Scope: active profile dry-run import/status only.
   - Required proof: dry-run counts, redaction findings, and archive-health summary look reasonable.
   - Mutations: no raw archive writes, no candidates, no semantic promotion.

4. Small confirmed import with health checks
   - Scope: a small, manually selected import batch.
   - Required proof: exact confirmation token used, archive health is ok, index rebuild/verify succeeds, privacy scan passes.
   - Mutations: raw archive evidence only unless later stages are enabled.

5. Read-only archive search/show enabled
   - Scope: `memory_v2_archive_search` and `memory_v2_archive_show` over imported evidence.
   - Required proof: search/show packets are bounded, source-backed, hash/integrity labeled, and fenced as untrusted archive evidence.
   - Executable gate: `memory_v2_archive_readiness` must return `ready=true` / `mutations_allowed=false`; local coverage command is `./scripts/run_tests.sh tests/plugins/memory/test_memory_v2_archive_readiness.py -q`.
   - Mutations: none.

6. Candidate extraction enabled
   - Scope: extraction may create pending candidates from archive evidence.
   - Required proof: candidates cite resolvable source refs, injection/secret bait is suppressed or held, and no durable semantic writes happen automatically.
   - Executable gate: `memory_v2_extract_candidates` is absent/disabled by default; when `memory_v2.extraction.enabled=true` and `memory_v2.extraction.candidate_creation_enabled=true`, it must return `ready=true`, `mutations_allowed=pending_candidates_only`, and only candidate-count deltas. Local coverage command is `./scripts/run_tests.sh tests/plugins/memory/test_memory_v2_extraction_rollout.py -q`.
   - Mutations: pending candidates only.

7. Semantic promotion with review gate
   - Scope: manual review and explicit promotion/rejection after source validation.
   - Required proof: reviewer confirms source refs and lifecycle status; unsafe/dangling/tampered evidence blocks promotion.
   - Mutations: reviewed semantic/core/project records only.

8. Limited automatic prefetch
   - Scope: routed, bounded Memory v2 packets for a small allowlisted profile/session set.
   - Required proof: token budget, latency budget, irrelevant suppression, privacy leakage, and adversarial instruction-following gates pass.
   - Mutations: none from prefetch.

9. Broader release
   - Scope: wider opt-in release after stages 1-8 pass.
   - Required proof: full release checklist complete, branch/release artifacts contain no private data, known caveats documented.
   - Mutations: only those allowed by the deployed feature flags and review policy.

## Default rollout feature flags

These are the desired Phase 10 defaults for a safe initial rollout. Keep mutating and autonomous behavior off by default.

| Flag | Default | Purpose |
| --- | --- | --- |
| `memory_v2.archive.enabled` | `true` | Allow the archive subsystem to exist for synthetic/dogfood/read-only gates. |
| `memory_v2.archive.backfill_enabled` | `false` | Keep SessionDB import disabled until dry-run and confirmation gates pass. |
| `memory_v2.archive.capture_enabled` | `false` | Keep automatic conversation capture disabled until an operator opts in. |
| `memory_v2.archive.search_tools_enabled` | `false` | Keep private archive search hidden until the read-only rollout stage. |
| `memory_v2.archive.show_tools_enabled` | `false` | Keep private archive source display hidden until the read-only rollout stage. |
| `memory_v2.archive.prefetch_raw_enabled` | `false` | Keep raw archive hydration out of automatic prefetch by default. |
| `memory_v2.archive.include_tool_outputs` | `false` | Exclude tool output from import until separate privacy/poisoning review. |
| `memory_v2.extraction.enabled` | `false` | Keep offline candidate extraction disabled until Step 6 is explicitly gated. |
| `memory_v2.extraction.candidate_creation_enabled` | `false` | Candidate creation from archive evidence must be an explicit opt-in and pending-only. |
| `memory_v2.extraction.small_model_enabled` | `false` | Optional structured model extraction requires this flag plus an explicitly supplied adapter; validated output remains pending-only. |
| `memory_v2.consolidation.enabled` | `false` | Disable automatic consolidation/promotion for initial rollout. |
| `memory_v2.prefetch.enabled` | `false` | Disable automatic online memory injection until late-stage gated rollout. |
| `memory_v2.review_apply.enabled` | `false` | Keep model-visible review rejection actions disabled by default. |
| `memory_v2.auto_promote.enabled` | `false` | Never automatically promote semantic memory in the initial release. |

Equivalent YAML shape:

```yaml
memory_v2:
  archive:
    enabled: true
    capture_enabled: false
    backfill_enabled: false
    search_tools_enabled: false
    show_tools_enabled: false
    prefetch_raw_enabled: false
    include_tool_outputs: false
  extraction:
    enabled: false
    candidate_creation_enabled: false
    small_model_enabled: false
  consolidation:
    enabled: false
  prefetch:
    enabled: false
  review_apply:
    enabled: false
  auto_promote:
    enabled: false
```

## Release gate commands

Run these from the repository root after activating the local virtualenv. They are intentionally local/deterministic and should not require network access or credentials.

```bash
./scripts/run_tests.sh tests/plugins/memory/test_memory_v2_*.py tests/plugins/memory/evals tests/agent/test_memory_provider.py
./scripts/run_tests.sh tests/plugins/memory/test_memory_v2_raw_archive_perf.py -q
./scripts/run_tests.sh tests/plugins/memory/test_memory_v2_adversarial_archive.py -q
./scripts/run_tests.sh tests/plugins/memory/test_memory_v2_archive_readiness.py -q
./scripts/run_tests.sh tests/plugins/memory/test_memory_v2_extraction_rollout.py -q
python scripts/memory_v2_privacy_scan.py --mode memory-v2-release-artifacts --format json
python scripts/memory_v2_privacy_scan.py --mode intentional-adversarial-fixtures --format json
python scripts/memory_v2_eval.py --dataset plugins/memory/memory_v2/evals/fixtures/local_memory_eval_v1.yaml --baseline no_memory --baseline raw_fts --baseline memory_v2
python -m py_compile plugins/memory/memory_v2/*.py scripts/memory_v2_eval.py scripts/memory_v2_privacy_scan.py
./scripts/run_tests.sh
```

ACP caveat: if the full repo suite fails for unrelated ACP approval/provider behavior, record the exact failing test, failure text, environment, and reason in the release notes. Do not treat that as a Memory v2 pass unless all Memory v2 targeted gates above pass and the ACP failure is confirmed unrelated.

## Dogfood report gate

Before default enablement, attach or generate a local dogfood report from an isolated profile. The report must show:

- isolated dogfood profile path and no cross-profile writes;
- synthetic-only examples unless the user has explicitly approved a local personal dry-run/import;
- redaction findings and privacy scan status;
- archive health, raw index status, and source-ref health;
- bounded search/show examples with untrusted-evidence labels;
- candidate counts and review outcomes;
- eval summary versus raw FTS/no-memory baselines;
- known limitations and rollback steps.

## Acceptance checklist

All of these must be true before broader release:

- Dogfood report proves safety and quality before default enablement.
- Privacy scan passes.
- Eval thresholds pass.
- Archive health is ok after backfill and rebuild.
- Docs are current and synthetic-only.
- Branch and release artifacts contain no private data.

## Rollback / stop conditions

Stop rollout and disable the relevant stage if any of these occur:

- privacy scan finds an unreviewed likely secret/path/private identifier;
- archive packets expose unbounded raw dumps;
- source refs are dangling/tampered but promotion still proceeds;
- evals show non-zero privacy leakage or adversarial instruction-following;
- prefetch injects irrelevant archive content into unrelated turns;
- branch artifacts contain private data or non-synthetic fixtures.

Rollback to the previous safe stage by disabling the stage flag, rebuilding derived indexes if needed, and keeping raw evidence read-only until health and privacy gates pass again.
