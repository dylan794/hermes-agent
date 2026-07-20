# Memory v2 Stage 8 isolated canary

This is the authoritative Stage 8 feature-flag matrix and go/no-go contract. It is for a synthetic, temporary profile only. It does not authorize editing or enabling a live Hermes profile.

Run from the repository root:

```bash
python scripts/memory_v2_stage8_canary.py
```

The command accepts no Hermes-home argument. It copies the checked-in [`stage_8_canary_config.example.yaml`](../plugins/memory/memory_v2/stage_8_canary_config.example.yaml) into a fresh temporary directory, exercises the canary, removes that directory, and emits one deterministic JSON object. A zero exit status and `"go":true` are both required.

## Authoritative Stage 8 feature-flag matrix

The runtime-default column is defined by `MemoryV2FeatureFlags`. The canary column is the complete checked-in Stage 8 profile. The rollback column returns to the Stage 5 read-only archive surface while preserving evidence for diagnosis.

| Flag | Runtime default | Stage 8 canary | Rollback | Stage 8 policy |
| --- | --- | --- | --- | --- |
| `memory_v2.archive.enabled` | `true` | `true` | `true` | Keep the local archive available for integrity inspection. |
| `memory_v2.archive.capture_enabled` | `false` | `true` | `false` | Capture only inside the isolated allowlisted canary profile. |
| `memory_v2.archive.backfill_enabled` | `false` | `false` | `false` | No broad or historical import during this canary. |
| `memory_v2.archive.search_tools_enabled` | `false` | `true` | `true` | Retain the bounded Stage 5 read-only evidence surface. |
| `memory_v2.archive.show_tools_enabled` | `false` | `true` | `true` | Retain the bounded Stage 5 read-only evidence surface. |
| `memory_v2.archive.prefetch_raw_enabled` | `false` | `false` | `false` | Broad raw archive hydration stays off. |
| `memory_v2.archive.include_tool_outputs` | `false` | `false` | `false` | Tool outputs remain outside archive capture. |
| `memory_v2.extraction.enabled` | `false` | `true` | `false` | Deterministic offline extraction is canary-scoped. |
| `memory_v2.extraction.candidate_creation_enabled` | `false` | `true` | `false` | Extraction may create pending candidates only. |
| `memory_v2.extraction.small_model_enabled` | `false` | `false` | `false` | Model-driven extraction stays off. |
| `memory_v2.consolidation.enabled` | `false` | `true` | `false` | Expose report-only consolidation; no automatic authority. |
| `memory_v2.prefetch.enabled` | `false` | `true` | `false` | Enable bounded semantic packets for the isolated canary. |
| `memory_v2.review_apply.enabled` | `false` | `false` | `false` | Model-visible mutation remains disabled. |
| `memory_v2.auto_promote.enabled` | `false` | `false` | `false` | Model-driven and automatic promotion stay off. |
| `memory_v2.contradictions.auto_supersede` | `false` | `false` | `false` | Automatic supersession stays off. |
| `memory_v2.contradictions.create_candidates` | `false` | `false` | `false` | Contradiction handling remains report-only. |
| `memory_v2.working_memory.enabled` | `false` | `false` | `false` | Working-memory prompt injection is outside this canary. |

The only canonical mutation in the verifier is one explicit trusted-host call to `MemoryOperationService.promote_candidate`. It uses the candidate fingerprint from a freshly generated read-only review plan. The model-facing promotion tool is exercised and must reject the same request without mutation.

## Measurable go criteria

All checks below must pass in the same run:

- The profile scope is `fresh_temporary_directory_only`.
- Capture produces exactly 1 hash-chain-verified raw event, 0 candidates, 0 memories, and 0 operation records.
- Archive readiness returns `ready=true` with no blockers; bounded search/show packets are marked `untrusted_text=true` and `can_instruct=false`.
- Deterministic extraction considers the synthetic event and creates exactly 1 pending candidate, 0 model candidates, 0 memories, and 0 operation records.
- Daily reporting, review queue generation, and review planning produce no count changes.
- The model-facing promotion request is rejected for lack of external operator authority and produces no count changes.
- The fingerprint-bound trusted-host promotion produces exactly 1 active semantic memory and leaves 0 pending candidates.
- Semantic prefetch returns the approved preference and its source reference while excluding raw-event records, raw event fields, and archived assistant text.
- Runtime health is `ok`, with no dangling source, interrupted-operation, or recovery-required issue.
- Every JSON value under `checks` is `true`, exit status is zero, and the complete JSON output is byte-for-byte identical across two consecutive runs.

## No-go and rollback

Do not advance or widen the allowlist if the command exits nonzero, returns `"go":false`, changes output between identical runs, or any measured criterion fails. Also stop for any privacy/adversarial release-gate failure, unbounded packet, raw-prefetch evidence, model-authorized mutation, dangling source, interrupted operation, or recovery-required journal.

Rollback only the isolated canary configuration to the matrix's Rollback column: disable capture, extraction, candidate creation, consolidation, and semantic prefetch; keep automatic promotion, automatic supersession, raw prefetch, model extraction, and model review-apply off. Do not delete raw evidence, candidates, operation journals, or indexes during incident triage. Keep bounded archive search/show read-only, and require manual recovery before any further canonical mutation when a recovery-required or interrupted-operation issue exists.

After rollback, rerun the Stage 5 archive-readiness gate and the complete repository test wrapper before considering another canary:

```bash
./scripts/run_tests.sh tests/plugins/memory/test_memory_v2_archive_readiness.py -q
./scripts/run_tests.sh
```
