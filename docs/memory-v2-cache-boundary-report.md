# Memory v2 cache-boundary report

## Scope

This report covers frontier task packet 05 only: Memory v2 stable prompt content,
dynamic prefetch rendering, working/artifact section placement, and the existing
current-turn conversation-loop injection.

## Implemented boundary

- `plugins/memory/memory_v2/context_packets.py` renders static Memory v2 contracts
  and safe, source-backed core records separately from dynamic recall.
- Stable rendering omits source identifiers and rejects unsourced or
  session/channel/tool/path/credential-shaped core statements.
- `plugins/memory/memory_v2/__init__.py` places the stable rendering in
  `system_prompt_block()`.
- The same provider builds one dynamic `MemoryPacket` containing routed recall
  plus any working-memory and artifact sections, then applies one explicit
  untrusted boundary and one token budget.
- `agent/conversation_loop.py` already injects the prefetched result into an API
  copy of the current user turn through `build_memory_context_block()`. Dynamic
  recall is not added to the cached system prompt or persisted conversation.

## Deterministic evidence

The packet acceptance command is:

```bash
scripts/run_tests.sh \
  tests/plugins/memory/test_memory_v2_cache_aware_context.py \
  tests/agent/test_memory_v2_conversation_loop_session_prefetch.py -q
```

The focused provider regression command is:

```bash
scripts/run_tests.sh tests/plugins/memory/test_memory_v2_provider.py -q
```

Acceptance result: 8 passed. Combined acceptance, provider, retrieval, and
privacy regression result: 130 passed.

These tests use temporary Hermes homes and visibly synthetic record, session, and
channel values. They verify stable/dynamic separation, source grounding,
cache-stable system content, current-turn placement, intact trust markers, and
dynamic token bounds.

## Residual risks

- The token estimator is intentionally approximate and conservative; it is not a
  provider-specific tokenizer.
- Cache breakpoints are transport-specific. This work preserves a byte-stable
  system prefix but does not add or change transport breakpoint syntax.
- Cache-safety screening is defense in depth. Core promotion gates remain
  responsible for preventing private or volatile statements from becoming core
  records.

## Privacy confirmation

No live profile memory, session database, logs, credentials, provider data,
gateway messages, cookies, or user files were read or used as fixtures. The
implementation and report contain repo-relative paths only; tests use temporary
directories and synthetic values.
