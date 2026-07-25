# Memory v2 Outcome Replay & Bottleneck Lab

## Purpose and status

The lab identifies which part of Memory v2 limits real work-continuity
performance before investing in a learned router, reranker, larger index, or
answer checker. It consumes minimized, opt-in shadow-study artifacts and runs
offline paired diagnostics.

The lab is development infrastructure, not production telemetry. It does not
modify a live Hermes profile, execute tools, repeat external side effects,
promote memories, supersede facts, activate skills, or support a human
superiority claim.

Implementation:

```text
plugins/memory/memory_v2/evals/outcome_replay.py
scripts/memory_v2_outcome_replay.py
plugins/memory/memory_v2/evals/fixtures/outcome_replay_synthetic_v1.yaml
```

## What the lab measures

Every episode pairs the current system outcome with one or more single-change
offline variants:

| Variant | Component isolated | Question |
|---|---|---|
| `no_memory` | Memory contribution | Did Memory v2 improve over the same answerer without memory? |
| `oracle_archive` | Archive/extraction | Would the task succeed if the needed evidence had been captured and extracted? |
| `oracle_candidate` | Candidate recall | Would a larger or corrected candidate pool rescue the task? |
| `oracle_route` | Routing | Would the correct retrieval plan rescue the task? |
| `oracle_rerank` | Ranking | Would correct ordering within the same candidate pool rescue the task? |
| `oracle_temporal` | Temporal resolution | Would correct current/history/conflict handling rescue the task? |
| `oracle_packet` | Packet composition | Would better bounded evidence selection or structure rescue the task? |
| `oracle_synthesis` | Answer synthesis | Would better evidence use, citation, calibration, or abstention rescue the task? |

The current and oracle executions must use the same frozen evidence cutoff,
corpus, code, configuration, and answerer unless that exact component is the
registered intervention. The schema records this assertion but cannot prove
the operator changed only one component.

## Artifact contract

The versioned dataset schema is
`memory-v2-outcome-replay-dataset/v1`. It stores:

- opaque episode, participant, project, workstream, query, profile, and corpus
  references;
- a query hash, query class, checkpoint, and evidence cutoff;
- exact archive, index, configuration, code, and answerer digests;
- route, candidate-set digest, ranked reference fingerprints, evidence times,
  packet digest, packet size, citation fingerprints, and safety-filter digest;
- externally verified success, corrections, rework, latency, tokens, citation
  correctness, stale-conflict errors, and safety failures.

It rejects raw prompts, raw answers, raw tool output, unknown fields,
model-self-reported labels, non-opaque identity fields, moving artifacts,
side-effect-enabled replay, evidence timestamps after the frozen cutoff,
judgments dated outside the frozen episode, citations to references that were
not retrieved, and successful outcomes without positively verified citation
correctness or with material safety or temporal failures.

Opaque identity references and query/reference fingerprints must be produced by
a keyed tokenization service or secret-salted HMAC whose key stays outside the
dataset. Do not use bare hashes of names, ticket IDs, short prompts, or other
low-entropy values: they are vulnerable to dictionary recovery. Use the same
protected audit key across pools that must be checked for overlap, and rotate it
according to the registered revocation policy.

Allowed label sources are:

- `independent_judge`;
- `trusted_operator`;
- `deterministic_harness`.

Execution receipts and model assessments remain untrusted and cannot serve as
labels without independent verification.

## Collection workflow

The `collect` command is the one-way privacy boundary for real episodes. Its
private intake and 256-bit tokenization key must stay outside the repository.
The intake contains the consent record plus private participant, project,
workstream, corpus, and query values; the emitted dataset contains only
domain-separated HMAC references and the already-minimized replay traces.

Collection is intentionally not wired to the live Hermes profile. An operator
must curate the private intake from an explicitly enrolled shadow run and must
run the command from a trusted host context. A model-authored flag is not proof
of consent or operator authority.

### 1. Obtain opt-in consent

Register the consent artifact, retention deadline, revocation policy, and
profile scope before collecting an episode. Keep participant/project identity
mapping outside the replay dataset. Withdrawal deletes or quarantines the
external mapping and every associated dataset according to the registered
policy; do not use withdrawal as outcome selection.

### 2. Freeze the eligible world

At the episode's evidence cutoff, freeze and hash:

- canonical archive/corpus;
- derived index;
- Memory v2 and host configuration;
- source code/build;
- answerer model and prompt configuration;
- safety-filter configuration.

Future evidence must be inaccessible to every variant.

### 3. Record the current trace

Record only hashes, opaque pointers, absolute evidence timestamps, counts, and
bounded metrics. Do not copy query text, response text, source text, tool
output, secrets, or authority credentials into the dataset.

### 4. Freeze an external outcome label

An independent judge, trusted operator, or deterministic harness records
work-continuity success, corrections, rework, latency, tokens, citation
correctness, stale/conflict errors, and registered safety failures. A model
cannot grade or authorize itself.

### 5. Construct offline single-change variants

Use read-only snapshots or captured tool results. Never replay a write, send,
purchase, deployment, deletion, or other external side effect. Change one
component, freeze its replay artifact digest, and label its outcome using the
same rules as the current trace.

Not every early episode needs every variant. Missing coverage is reported and
cannot be interpreted as a zero gap. An individual component can show a partial
signal, but selecting the largest bottleneck requires every oracle component to
meet coverage floors on the same episode panel.

### 6. Validate and audit separation

Create a new immutable, disjoint pilot dataset before validation. The key file
must contain exactly 64 lowercase hexadecimal characters (32 bytes). Protect
it with owner-only permissions or an equivalent Windows ACL and retain the same
protected key for pools that must support exact overlap detection.

```bash
python scripts/memory_v2_outcome_replay.py collect \
  --intake <external-private-intake.yaml> \
  --token-key-file <external-owner-only-key.txt> \
  --against <every-prior-minimized-pool.yaml> \
  --output <external-new-pilot-dataset.json> \
  --authorize-opt-in-collection
```

The default collection gate requires `study_mode: pilot`, a complete current
plus oracle panel for every episode, and the registered episode and participant
floors. `--allow-partial-panel` and `--allow-underpowered` exist only for
non-decision staging. Collection refuses repository-local private inputs, keys,
or outputs; refuses to overwrite an existing dataset; and emits no dataset
when the exact disjointness audit finds overlap.

The private intake uses schema
`memory-v2-outcome-replay-private-intake/v1`. Its top-level keys are
`schema_version`, `lab_id`, `study_mode`, `created_at`, `consent`, `thresholds`,
and `episodes`. Each episode supplies private `participant_id`, `project_id`,
`workstream_id`, `query_text`, and snapshot `corpus_id` values alongside the
same minimized query class, checkpoint, frozen digests, traces, and externally
verified outcomes required by the public dataset. Never commit, publish, or
attach a real private intake or token key to an issue or model conversation.

```bash
python scripts/memory_v2_outcome_replay.py validate \
  --dataset <shadow-dataset.yaml> \
  --output <validation.json>

python scripts/memory_v2_outcome_replay.py audit-disjoint \
  --candidate <shadow-dataset.yaml> \
  --against <development-dataset.yaml> \
  --against <prior-pilot-dataset.yaml> \
  --output <disjointness.json>
```

The exact audit checks participant, project, workstream, query, and archive
hash overlap. Related people, project lineage, paraphrases, and shared upstream
evidence still require an operator audit.

### 7. Analyze

```bash
python scripts/memory_v2_outcome_replay.py analyze \
  --dataset <shadow-dataset.yaml> \
  --output <bottleneck-report.json>
```

Add `--require-decisive` when using the command as an engineering gate. Exit
codes are:

- `0`: valid result, and decisive when required;
- `1`: valid but inconclusive, or an overlap was found;
- `2`: invalid artifact or unsafe operation request.

## Statistical decision

For each available variant, the lab reports:

- paired current and variant success rates;
- paired success-rate difference;
- rescue and regression rates;
- participant-cluster percentile-bootstrap interval;
- participant and project coverage;
- safety-regression count;
- latency, token, and rework deltas.

A component is called the decisive largest bottleneck only when:

1. the registered minimum paired-episode and participant-cluster floors pass;
2. the participant-cluster confidence-interval lower bound is above zero;
3. the oracle variant has no observed safety regression;
4. every component is evaluated on the same complete, coverage-eligible episode
   panel.

The largest qualifying paired gap becomes the recommended next engineering
target. A positive gap in a partial panel is labeled a partial bottleneck
signal, not the winner. If no component qualifies, the lab requests more
disjoint episodes or missing oracle variants rather than guessing.

This diagnostic does not correct for testing many components and is not a
confirmatory product claim. Preregister component priorities or apply a
multiple-comparison procedure before using the same machinery for formal
inference.

## Synthetic rehearsal

The bundled fixture is mechanics-only development data. It contains four
synthetic participant/project clusters engineered so `oracle_rerank` rescues
every current failure while `oracle_route` does not. A valid analysis should
therefore identify `ranking` with a paired gap of `1.0`.

Run its tests through the required wrapper:

```bash
./scripts/run_tests.sh \
  tests/plugins/memory/evals/test_memory_v2_outcome_replay.py \
  tests/plugins/memory/evals/test_memory_v2_outcome_replay_cli.py -q
```

This result is not evidence that ranking is the real Memory v2 bottleneck.

## Safety and authority

- Collection is opt-in and profile-scoped.
- Raw content and authority credentials never enter replay datasets.
- Dataset outputs are diagnostic and have `mutation_authority: none`.
- Automatic promotion, automatic supersession, skill activation, and learned
  online changes remain disabled.
- Hard temporal and safety filters remain outside any future learned ranker.
- Real replay datasets stay outside repository fixtures and publication
  artifacts.
- Derived reports may be rebuilt; frozen source artifacts and labels remain
  authoritative.

## What happens after real data

Improve only the largest measured safe oracle gap:

- archive/extraction gap: improve capture and grounded extraction;
- candidate gap: improve indexing/candidate generation;
- route gap: improve the query planner;
- ranking gap: calibrate a shadow-only reranker;
- temporal gap: improve conflict and supersession resolution;
- packet gap: improve evidence selection/structure;
- synthesis gap: improve grounded response use and citation checking.

Only repeated operator-verified successful procedures may emit fingerprinted
SkillRegistry candidates. Lab outcomes cannot install, activate, or execute a
skill.
