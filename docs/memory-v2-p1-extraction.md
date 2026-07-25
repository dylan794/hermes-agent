# Memory v2 P1: candidate-only offline extraction

Status: implemented on `memory-v2-p1-extraction`.

P1 broadens offline extraction beyond explicit `remember that ...` language while preserving the P0 authority boundary: extraction creates **pending `CandidateMemory` records only**. It never writes active memory items, project cards, open loops, core memory, or operation records.

## Deterministic preprocessing

The extractor reads a bounded set of canonical raw events through the raw-event index, sorts them deterministically by chain position and timestamp, and considers only evidence-bearing fields:

- user turn text
- assistant text only when paired with a later adjacent user acceptance
- tool `result`, `stdout`, `stderr`, or `content`

Secrets, redaction markers, instruction-shaped bait, unsupported event types, ephemeral chatter, and unresolved source references fail closed.

## Typed candidates

`CandidateMemory` now carries:

- `claim_kind`
- `confidence`
- `durability`
- `evidence_spans`
- `negative_evidence`
- `extraction_method`
- `extractor_version`

Each evidence span contains a canonical `source_id`, source field, role, exact start/end offsets, and the exact source slice. Stored spans are schema-validated. Candidate indexing includes claim kind and extraction method, but promotion remains a separate confirmed review-plan mutation.

The deterministic extractor recognizes conservative forms of:

- debugging/project decisions
- test failures that contradict an assumption
- environment state established by command output
- blockers found in user or tool evidence
- completed actions and passing test results
- authoritative/source-of-truth artifacts
- assistant proposals followed by adjacent explicit user acceptance
- existing labeled project, preference, environment, skill, and open-loop forms

Assistant statements alone are never treated as user facts. An assistant proposal requires a separate exact user acceptance span such as `do it`, `go ahead`, or `sounds good`.

## Negative evidence and contradictions

Failed tests or user statements that a result proved an assumption wrong create pending contradiction candidates. The same exact source slice is preserved in `negative_evidence`. No existing memory is automatically superseded.

## Deduplication

Candidates are compared by memory type, claim kind, destination, and normalized semantic token overlap. Equivalent pending candidates merge source references and exact evidence spans; decided candidates and already-active equivalent memories are not duplicated.

## Optional structured small-model adapter

The provider accepts a callable `extraction_model_adapter` during initialization. It is invoked only when all three conditions hold:

```yaml
memory_v2:
  extraction:
    enabled: true
    candidate_creation_enabled: true
    small_model_enabled: true
```

and an adapter was explicitly supplied by the host. Memory v2 does not silently make a network/model call.

The adapter receives bounded, redacted, instruction-escaped event fields and must return schema version `1` with typed candidates, unit-interval confidence/durability scores, and exact evidence spans. Exact citation alone is insufficient: each claim kind has a deterministic support rule (for example, a preference requires an explicit first-person preference span, a constraint requires constraint language, and a contradiction requires matching negative evidence). Model output is rejected when:

- a span is fabricated, out of bounds, points outside the selected evidence, or has the wrong role
- evidence contains secrets or instruction-shaped text
- a claim cites assistant text without an exact user acceptance span
- the source is not canonical and integrity-valid
- scores/types/schema are malformed
- more than 25 candidates or 12 spans per field are returned

Validated model claims are capped at confidence `0.82`, marked `extraction_method: structured_model`, deduplicated against deterministic/existing candidates, and remain pending.

## Verification

Focused coverage:

```bash
scripts/run_tests.sh \
  tests/plugins/memory/test_memory_v2_p1_extraction.py \
  tests/plugins/memory/test_memory_v2_extraction.py \
  tests/plugins/memory/test_memory_v2_extraction_rollout.py \
  tests/plugins/memory/test_memory_v2_feature_flags.py -q
```

The full Memory v2 suite and privacy scans remain release gates. This P1 work improves proposal quality and evidence structure; it does not claim extracted candidates are true or ready for automatic promotion.
