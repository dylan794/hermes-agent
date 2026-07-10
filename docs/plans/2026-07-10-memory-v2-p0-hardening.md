# Memory v2 P0 hardening plan

## Goal

Turn the current prototype into a coherent, versioned, production-path-tested Memory v2 baseline capable of supporting a future claim about year-long recall. This phase does **not** claim human-level or superhuman memory.

## Scope

1. Recover and version the Memory v2 provider, tests, eval CLI, fixtures, and truthful documentation.
2. Remove benchmark-only retrieval behavior; evals must exercise initialized provider lifecycle and production packet composition.
3. Add chronological 30/90/365-day benchmark contracts and a human-baseline methodology.
4. Add bounded raw-evidence retrieval for exact/deep/source-verification routes while generic search remains semantic-only.
5. Enforce structured `provider_session_id` authority before raw hydration.
6. Capture bounded, redacted completed-turn tool trajectories as source-backed work episodes; tool text remains untrusted and cannot directly promote beliefs.
7. Require canonical raw-event or artifact evidence for verified promotion.
8. Add cross-process locks and prepared/committed/failed operation journaling around canonical mutations; health checks must detect interrupted operations.
9. Ensure direct mutation tools cannot bypass review-plan confirmation.

## Non-goals

- No embedding/vector backend.
- No persisted graph database.
- No broad automatic semantic promotion.
- No public push or PR in this phase.
- No mutation of the live profile memory store during tests.

## Required test gates

- Restored Memory v2 unit/integration/eval suite is present and executable.
- Every behavior change follows RED → GREEN with a focused test.
- Generic search never returns raw archive rows.
- Route-specific raw recall fails closed without archive flags, active provider authority, canonical source evidence, or usable raw index.
- Cross-session/provider-session raw evidence is excluded before hydration.
- Tool episode capture is bounded, redacted, source-backed, primary-context-only, and ignores interrupted turns.
- Eval adapter packet equals the production provider packet for identical profile/query state.
- No eval-only precision/filtering code can affect the system under test.
- 30/90/365-day streams run chronologically through provider lifecycle and survive restart/index rebuild.
- Concurrent raw append yields a valid single hash chain.
- Interrupted canonical mutations are journaled and detected/recoverable without silent partial success.
- Promotions reject mere/forged SourceRef files lacking canonical raw/artifact backing.
- Direct mutation entrypoints enforce the same review confirmation contract.
- Ruff, compileall, targeted Memory v2 suite, adjacent memory-manager tests, privacy scan, and package smoke pass.

## Completion evidence

- Clean scoped Git status on branch `memory-v2-p0-hardening`.
- Commands and results recorded in updated docs.
- Independent spec/security reviewers report no critical or important issues.
- One verified local commit; no push.
