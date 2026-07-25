# Memory v2 evals

Memory v2 evals are deterministic, local regression checks for Hermes's Memory v2 provider. They are meant to catch retrieval regressions before dogfood or release work: source-grounded recall, stale/irrelevant memory suppression, and bounded memory packet behavior.

The [North Star specification](memory-v2-north-star.md) defines the separate, preregistered human-baseline evidence required before claiming superiority at one-year work continuity. Passing the local harness does not establish that claim.

The default packaged fixture is:

```text
plugins/memory/memory_v2/evals/fixtures/local_memory_eval_v1.yaml
```

## Purpose

Use these evals to answer practical questions:

- Does Memory v2 retrieve the right facts for preference and project-continuity queries?
- Does it preserve source references for recalled memory?
- Does it suppress irrelevant queries instead of injecting unrelated memory?
- Does it stay deterministic and cheap enough to run in local regression tests?
- Does it treat raw/archive text as untrusted evidence rather than executable instructions?
- Does it survive longitudinal / LoCoMo-shaped questions with stale facts, multi-hop project recall, and adversarial rows?

The eval harness is not a full agent benchmark. It exercises the memory ingestion, consolidation, routing, retrieval, and scoring layers without making model or network calls.

## Baselines

Local deterministic baselines are implemented under `plugins/memory/memory_v2/evals/`:

- `no_memory` — returns no memory. This is the floor for recall and a useful control for suppression.
- `raw_fts` — indexes raw event text with SQLite FTS and retrieves lexical matches. This tests whether routed Memory v2 recall adds value over simple log search on a given fixture; reports must not turn one fixture result into a broad win claim.
- `archive_only` — indexes bounded raw archive evidence without promoted candidates; useful for separating archive recall from consolidation behavior.
- `semantic_only` — runs the Memory v2 write/consolidation path under a separate label for longitudinal comparisons.
- `memory_v2` — initializes the actual `MemoryV2Provider` under an explicit temporary `config.yaml`, ingests events through `sync_turn(event_id=..., created_at=...)`, runs consolidation through the provider tool, and retrieves the production `prefetch()` packet. It does not use fixture route/oracle labels or eval-only precision filters.

External adapter status helpers live in `plugins/memory/memory_v2/evals/adapters.py`, but local regression commands do not require external providers, API keys, or network access.

## Metrics

Each query produces a score row with:

- `source_recall` — fraction of expected source refs found in retrieved refs.
- `text_contains` — fraction of expected answer substrings present in the answer or memory packet.
- `suppression` — 1.0 when a query that should not retrieve memory retrieves nothing; 0.0 when it leaks memory into irrelevant queries.
- `retrieved_count` — number of records returned.
- `token_estimate` — rough packet-size estimate for budget checks.
- `latency_ms` — local retrieval time for the query.
- `privacy_leakage` — 1.0 if credential-like markers survive into answer/packet text, otherwise 0.0.
- `adversarial_instruction_following` — 1.0 if instruction-shaped bait survives as executable-looking text, otherwise 0.0.
- `irrelevant_injection` — 1.0 if no-retrieve rows still inject memory content, otherwise 0.0.

Reports summarize metric averages per baseline.

## Human-baseline harness

The human-baseline harness supports a blinded, paired comparison between human
and Memory v2 responses. It is deliberately separate from deterministic fixture
scoring:

Use the [operational pilot runbook](memory-v2-human-baseline-pilot.md) for the
six-participant pilot design, disjointness audit, collection controls, and
power-planning diagnostics.

```text
plugins/memory/memory_v2/evals/human_baseline.py
scripts/memory_v2_human_baseline.py
```

Its versioned protocol schema is `memory-v2-human-baseline-protocol/v1`;
responses, public packets, private keys, judgments, and results each use a
separate versioned schema. A protocol identifies the study and `study_mode`;
participants; paired queries; checkpoints; task strata; six fixed judgment
dimensions; safety gates; coverage thresholds; superiority margin; judge
minimums; and participant-cluster bootstrap settings. `study_mode` is
`development`, `pilot`, or `confirmatory`; only the last can produce a
superiority claim.

The fixed dimension identifiers are `factual_correctness`,
`temporal_correctness`, `completeness`, `source_grounding`, `actionability`, and
`calibration_restraint`. The protocol registers `score_min`, `score_max`, and
`dimension_minimum`; each query names its nonempty `applicable_dimensions`. A
response is a work-continuity success only when its across-judge mean reaches
the minimum on every applicable dimension and that response has no safety
failure.

For the North Star claim, `primary_checkpoint` is 365 days. The primary
estimate is the paired Memory-v2-minus-human success-rate difference at that
checkpoint. The 95% percentile interval resamples participant clusters while
retaining all nested workstreams and response pairs. Superiority requires its
lower bound to exceed the registered margin, every required 365-day stratum to
meet its registered non-inferiority margin, complete coverage, and every Memory
v2 hard gate to pass. Confirmatory mode enforces the North Star study floors,
including a superiority margin of at least `0.05` and at least 10,000 bootstrap
samples; lower settings remain available only for development or pilot runs.

The workflow has three phases:

```bash
python scripts/memory_v2_human_baseline.py validate --protocol <protocol.json> --output <validation.json>
python scripts/memory_v2_human_baseline.py prepare --protocol <protocol.json> --human-responses <human.json> --memory-responses <memory-v2.json> --packet-output <judge-packets.json> --key-output <private-key.json> --seed <sealed-integer-seed>
python scripts/memory_v2_human_baseline.py score --protocol <protocol.json> --packets <judge-packets.json> --key <private-key.json> --judgments <judgments.json> --output <report.json> --require-superiority
```

- `validate` rejects incomplete schema, checkpoint/stratum coverage, participant
  or item counts, judge requirements, thresholds, and safety gates before the
  study is prepared.
- `prepare` randomizes the paired responses into A/B judge packets without
  structured condition labels, strips participant metadata, rejects configured
  raw participant IDs found in public answer text, and writes the identity
  mapping and seed separately as a sealed private key. It does not rewrite
  answer text or guarantee that other stylistic/self-identifying clues are
  absent.
- `score` consumes frozen blinded judgments, resamples whole participant
  clusters, reports the paired success-rate difference and confidence interval,
  and applies the preregistered superiority and hard-gate decision. Without
  `--require-superiority`, a valid, complete no-claim result exits successfully;
  use the flag for a confirmatory superiority gate.

Exit code `0` means the requested operation was valid and complete. With
`--require-superiority`, exit code `1` means a valid study did not earn the
claim. Exit code `2` means an artifact was invalid, incomplete, inconsistent,
or tampered; do not interpret it as a measured no-claim result.

Judges receive a condition-neutral reference-evidence and rubric pack, never the
private mapping key. At least two blinded judges must score each response;
primary-outcome disagreement requires blinded adjudication under the registered
protocol. The question text, reference judgments, oracle source IDs, private
mapping key, and judge comments must never enter Memory v2's archive or index
before response generation. Commit to the seed at preregistration, keep it
sealed through judging, and publish it only after judgments freeze.

Every judgment supplies per-slot `material_error_notes` (`null` when absent, or
an object containing `reason` and `evidence_refs`) and a boolean
`potentially_identifiable` flag, in addition to scores, gate failures, and
condition guesses. It also declares `judge_role` as `primary` or `adjudicator`.
Only primary judges count toward `min_judges_per_item` and inter-rater
agreement. An adjudicator is allowed only at the primary checkpoint and only
when the primary judges disagree on binary work-continuity success.

Pilot reports add `pilot_diagnostics.item_difficulty` and
`pilot_diagnostics.participant_cluster_variance`, while inter-rater agreement
is reported overall and by checkpoint. Use `audit-disjoint` before collection
to reject exact overlap with development or prior-study protocols; the
operational runbook documents the command and its limits.

Before judging, a condition-blind operator applies the preregistered
semantics-preserving formatting normalization and audits public packets. Do not
silently rewrite or exclude self-identifying content. Judges record a
post-score condition guess; publish non-`unsure` guess accuracy, the `unsure`
rate, and flagged packet counts as a blinding-effectiveness check.

The harness can establish that a protocol is internally complete and compute a
decision. It cannot make a study representative, prove that evidence access was
fair, or turn synthetic rows into a human comparison. Those requirements,
including the 30/90/180/365 checkpoints, held-out cohort, preregistration,
population scale, and allowed claim language, are normative in the [North Star
specification](memory-v2-north-star.md).

## Outcome replay and bottleneck diagnostics

The [Outcome Replay & Bottleneck Lab](memory-v2-outcome-replay-lab.md) analyzes
minimized opt-in shadow episodes against offline, single-change oracle variants.
It attributes paired work-continuity regret to archive/extraction, candidate
recall, routing, ranking, temporal resolution, packet composition, or answer
synthesis before a team invests in a learned component.

```bash
python scripts/memory_v2_outcome_replay.py validate \
  --dataset plugins/memory/memory_v2/evals/fixtures/outcome_replay_synthetic_v1.yaml

python scripts/memory_v2_outcome_replay.py analyze \
  --dataset plugins/memory/memory_v2/evals/fixtures/outcome_replay_synthetic_v1.yaml
```

The bundled dataset is an engineered synthetic rehearsal, not evidence that
ranking is the real bottleneck. Replay outputs are diagnostic-only, cannot
support a human-superiority claim, and carry no memory or skill mutation
authority.

Real pilot pools must pass through the lab's external `collect` command. It
requires explicit operator attestation, a protected HMAC key, every prior pool
for disjointness checking, a complete oracle panel, and registered sample-size
floors by default. It never reads the live Hermes profile and refuses to place
private intake, keys, or real minimized datasets inside the repository.

## Earn the Canary

The [Earn-the-Canary runbook](memory-v2-earn-canary.md) defines the next
offline decision study. It compares the same frozen answerer under four
condition-blinded evidence arms: no memory, bounded raw FTS, bounded Memory v2,
and bounded operator-selected oracle evidence. The workflow measures paired
work-continuity improvement, safety and temporal errors, judge agreement,
blinding diagnostics, cluster uncertainty, and remaining oracle headroom.

The CLI validates preregistration, audits exact disjointness, creates shuffled
judge packets plus a separate sealed key, and scores frozen judgments:

```bash
python scripts/memory_v2_earn_canary.py validate --protocol <external-protocol.yaml>
python scripts/memory_v2_earn_canary.py audit-disjoint \
  --candidate <external-untouched-pilot-protocol.yaml> \
  --against <external-development-or-pilot-protocol.yaml> \
  --output <external-disjointness.json>
```

Real artifacts remain outside the repository and live Hermes state.
Development results are ineligible for `go`. An untouched pilot may earn only
a separately authorized limited answer-injection canary; its report has no
mutation authority.

## Command examples

Run the packaged local fixture with all deterministic local baselines:

```bash
python scripts/memory_v2_eval.py \
  --dataset plugins/memory/memory_v2/evals/fixtures/local_memory_eval_v1.yaml \
  --baseline no_memory \
  --baseline raw_fts \
  --baseline memory_v2
```

Run the built-in chronological 30/90/365-day contracts. These contracts ingest events in timestamp order and include restart plus index-rebuild checkpoints for provider-backed baselines:

```bash
python scripts/memory_v2_eval.py \
  --chronological-contracts \
  --baseline raw_fts \
  --baseline memory_v2 \
  --no-fail-on-acceptance
```

Write a JSON report to a file:

```bash
python scripts/memory_v2_eval.py \
  --dataset plugins/memory/memory_v2/evals/fixtures/local_memory_eval_v1.yaml \
  --output memory-v2-eval-report.json
```

Run eval unit tests:

```bash
scripts/run_tests.sh tests/plugins/memory/evals -q
```

## Dogfood status

Live/fresh dogfood and frontier workflows are deliberately deferred until the P0 benchmark gates improve. Recovered pre-P0 tests for those workflows remain strict expected failures; do not treat them as release gates or enable live dogfood from this branch.

## Adding fixtures

Add new YAML fixtures under:

```text
plugins/memory/memory_v2/evals/fixtures/
```

A fixture should include:

- `version`, `name`, and `description` metadata.
- `events`: deterministic user/assistant events with stable `id`, `session_id`, `role`, and `text` fields.
- `queries`: query cases with stable `id`, expected route label, query `text`, `expected_answer_contains`, `expected_source_refs`, and optional `should_retrieve: false` for suppression tests.

Keep fixtures small, synthetic, and privacy-safe. Do not copy real private conversations or machine-specific paths into fixtures.

After adding a fixture, add or update tests under `tests/plugins/memory/evals/` and run the CLI against the packaged fixture path.

## Adding an external adapter

External adapters should be opt-in and separate from deterministic local regression tests.

Recommended process:

1. Add a readiness/status entry in `plugins/memory/memory_v2/evals/adapters.py` that checks import and required environment variables without making network calls.
2. Implement the adapter behind an explicit CLI or test flag so local tests remain offline by default.
3. Normalize adapter results into the same report shape as local baselines.
4. Document required packages, environment variables, and limitations without including real keys or account identifiers.
5. Add skipped-by-default tests that verify adapter availability checks without requiring credentials.

## Limitations

- The harness is deterministic and local; it does not measure LLM answer quality or full multi-turn agent behavior.
- The human-baseline example artifacts demonstrate mechanics in `development`
  mode only. The harness can validate, blind, and score a study, but no
  confirmatory human rows have been collected here; do not claim a human,
  raw-FTS, or Memory-v2 win from methodology or example output alone.
- Substring scoring can miss semantically correct paraphrases and can over-reward copied text.
- The fixtures are intentionally small, so passing them is a regression signal, not proof of broad memory quality.
- Latency numbers are local-machine dependent and should be interpreted as rough smoke signals.
- External provider comparisons are not part of default local regression because they may require credentials, network access, provider-specific setup, and non-deterministic behavior.
