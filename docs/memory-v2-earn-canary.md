# Memory v2: Earn the Canary

## Purpose and status

Earn the Canary is the offline, opt-in longitudinal decision study for Memory
v2. It tests whether bounded Memory v2 evidence materially improves real work
against no long-term memory and raw lexical search, while measuring the
remaining gap to an operator-selected evidence oracle.

The study does not inject memory into live answers, change a Hermes profile,
replay tools, promote candidates, supersede facts, activate skills, or grant
mutation authority. A report is evidence for a later operator decision, never
authorization by itself.

The four answer conditions are:

| Condition | Evidence available to the frozen answerer |
|---|---|
| `no_memory` | No long-term-memory packet |
| `raw_fts` | A bounded raw SQLite/FTS result |
| `memory_v2` | The bounded, routed Memory v2 evidence packet |
| `oracle_evidence` | A bounded evidence packet selected after the episode by a trusted operator |

The answerer, prompt, evidence cutoff, safety policy, and tool-disabled replay
environment must otherwise be frozen and identical. Oracle evidence cannot use
events after the cutoff.

## Artifact boundary

Real protocols, prompts, answers, judge packets, assignment keys, judgments,
and reports are private artifacts. Keep them outside both the repository and
live Hermes state. Generated artifacts are immutable and are published with
atomic no-clobber writes. Pilot commands fail closed on native Windows because
the CLI cannot prove owner-only artifact permissions there. Run a real pilot
inside hardened WSL with every input and output in an owner-only (`0700`)
external directory and regular input files set to `0600`. Native Windows
remains supported for synthetic `development` rehearsals only.

Repository fixtures are synthetic development rehearsals only. They cannot be
included in a pilot, confirmation pool, performance claim, or canary decision.

Inputs use strict, versioned schemas. JSON rejects duplicate keys, non-finite
numbers, excessive nesting, mutation during read, and oversized files. Protocol
and response loaders also accept strictly parsed YAML. Generated packet, key,
judgment, metric, replay, and report artifacts use JSON.

The public half of a packet bundle means “condition-blinded for authorized
judges,” not safe for public release. It still contains private prompts and
answers. The separate sealed key contains condition assignments and opaque
participant/project clustering information but no response text.

## Study pools and consent

Use separate immutable pools:

1. `development` for synthetic fixtures, rubric repair, judge calibration,
   blinding repair, and engineering choices;
2. `pilot` as the preregistered, untouched limited-canary decision pool.

Every real participant must explicitly opt in under a frozen consent,
retention, revocation, and profile-scope record. Use the same protected HMAC
key for pools that must support exact overlap auditing. Never store that key in
the repository or give it to an answerer or judge.

The protocol binds the protected identity namespace with a digest and every
episode carries its consent-receipt digest. Disjointness is not meaningful
across mismatched tokenization namespaces and fails closed rather than treating
incomparable opaque IDs as disjoint.

Before collecting or preparing a new pool, audit it against every development,
pilot, prior confirmation, and tuning pool. Exact opaque participant, project,
workstream, query, and corpus identities are checked without printing the
overlapping values. A trusted operator must separately audit related people,
project lineage, paraphrased questions, and shared upstream evidence.

Every pilot participant, project, query, and upstream evidence source must be
disjoint from development and earlier pilot pools. Do not tune the system,
rubric, thresholds, or packet format after opening pilot outcomes.

## Freeze each episode

For each query, freeze and fingerprint:

- the archive/corpus and evidence cutoff;
- the derived index and raw FTS index;
- Memory v2 configuration, safety filters, source-span attestation, and code;
- the answerer model, prompt template, generation settings, and tool-disabled
  replay boundary;
- the rubric, judge instructions, judge calibration, adjudication policy, and
  packet renderer;
- a commitment to the sealed blinding seed, which `prepare` verifies when the
  integer seed is revealed;
- all four evidence packets and generated answers.

The protocol must be created before any response is generated. The response
set, packet set, and frozen judgments bind transitively to the complete
protocol fingerprint, including these preregistered study mechanics.

Each episode preregisters one `condition_input_digest` per arm. Every response
must bind to the matching digest as well as the frozen answerer, configuration,
code, and evidence cutoff. Scoring revalidates the complete response set
against the sealed key; packet and key files alone are insufficient.

Include memory-needed questions and externally labeled no-memory controls.
Cover multiple participants, projects, workstreams, query classes, and
30/90/365-day gaps. Participant clustering is the primary uncertainty unit;
many questions from one person do not count as independent people.

Do not execute captured tool calls or external side effects. The answerer and
retrieved content are untrusted data and have `mutation_authority: none`.

## CLI workflow

Validate a protocol:

```bash
python scripts/memory_v2_earn_canary.py validate \
  --protocol <external-protocol.yaml> \
  --output <external-validation.json>
```

Check a candidate pool against every prior pool:

```bash
python scripts/memory_v2_earn_canary.py audit-disjoint \
  --candidate <external-untouched-pilot-protocol.yaml> \
  --against <external-development-protocol.yaml> \
  --against <external-pilot-protocol.yaml> \
  --output <external-disjointness.json>
```

Prepare shuffled A/B/C/D packets and a separate sealed key:

```bash
python scripts/memory_v2_earn_canary.py prepare \
  --protocol <external-protocol.yaml> \
  --responses <external-four-arm-responses.json> \
  --packet-output <external-blinded-packets.json> \
  --key-output <external-private-key.json> \
  --seed <sealed-integer-seed> \
  --authorize-opt-in-study
```

The seed and key stay sealed until responses and judgments are frozen. Slot
order is deterministically shuffled per episode. Response IDs, slot labels,
formatting metadata, and packet IDs must not reveal the condition.

Have at least two qualified judges score independently. Judges rate task
success, usefulness, correctness, temporal correctness, source grounding,
distraction, registered safety failures, stale-as-current errors, irrelevant
injection, and citation validity before making a condition guess. They also
flag potentially identifiable packets. Resolve registered primary-outcome
disagreements only through the preregistered blinded adjudication rule.

Score only after judgments freeze:

```bash
python scripts/memory_v2_earn_canary.py score \
  --protocol <external-protocol.yaml> \
  --responses <external-four-arm-responses.json> \
  --packets <external-blinded-packets.json> \
  --key <external-private-key.json> \
  --judgments <external-frozen-judgments.json> \
  --against <external-development-protocol.yaml> \
  --against <external-prior-pilot-protocol.yaml> \
  --shadow-metrics <external-shadow-metric-bundle.json> \
  --outcome-replay <external-outcome-replay-result.json> \
  --output <external-canary-report.json> \
  --authorize-unblind \
  --attest-untouched-pool \
  --attest-consent-active \
  --require-go
```

Exit codes are:

- `0`: valid workflow result, and `go` when `--require-go` is present;
- `1`: valid no-go/inconclusive result or a disjointness failure;
- `2`: invalid schema, unsafe path, missing authority attestation, or attempted
  overwrite.

Standard output is a content-free receipt. Private prompts, answers,
assignments, judgments, paths, and report bodies are never printed.

The score command recomputes exact disjointness from the scored protocol and
every supplied prior protocol. The separate `--attest-untouched-pool` records a
trusted operator's assertion that no pilot outcome influenced the frozen
design. Exact HMAC overlap measurement cannot prove that broader procedural
claim, and the attestation cannot waive a measured overlap.

`--attest-consent-active` is a separate scoring-time assertion. The scorer also
requires the registered retention deadline to remain in the future. Both are
processing preconditions: the CLI checks them before reading packet, key,
response, judgment, metric, or reference artifacts, and the core API rejects
rather than emitting a no-go report. Neither the original receipt nor an
unexpired retention date alone proves that consent has not been revoked.

Optional shadow metrics use schema
`memory-v2-earn-canary-shadow-metrics/v1` and must bind the study ID, protocol
fingerprint, and response-set fingerprint to an exact, full episode panel. Each
wrapper row contains an opaque episode ID, its query hash, and one strict
`compute_shadow_metrics` input record; missing, duplicate, extra, or
cross-study rows fail closed.

## Failure taxonomy and oracle gaps

The full-oracle comparison establishes whether useful evidence was available;
it does not by itself identify which component failed. Diagnose positive
Memory-v2-to-oracle gaps with frozen traces and, where needed, the existing
single-change Outcome Replay lab.

Use the fixed taxonomy:

- archive/extraction did not represent the evidence;
- candidate recall missed represented evidence;
- routing chose the wrong project, workstream, or temporal plan;
- ranking selected the wrong candidate;
- temporal resolution treated stale or superseded evidence as current;
- packet composition omitted a required multi-event bundle;
- answer synthesis failed to use available evidence or abstain;
- unnecessary memory was injected;
- unresolved when available evidence does not support one category.

A failure label requires an independent judge, deterministic harness, or
trusted-operator evidence reference. Model self-reports cannot label a failure
or authorize an engineering or rollout decision.

## Canary decision

Preregister thresholds before seeing pilot outcomes. A `go` earns only a
limited, separately authorized answer-injection canary and requires
all registered coverage, quality, and safety gates, including:

- measured exact disjointness from every supplied prior protocol plus a
  separate trusted-operator attestation that the `pilot` decision pool remained
  untouched;
- preregistered minimum rows per checkpoint and query class, bounded maximum
  episode share for any participant or project, and minimum effective
  participant/project cluster counts;
- paired crossed participant/project pigeonhole-bootstrap confidence bounds
  showing the registered Memory v2 improvement over raw FTS and no memory;
- registered 30/90/365-day and query-class coverage without a dominant
  participant or project;
- a bounded Memory-v2-to-oracle gap;
- zero unattested citations and registered hard safety failures;
- stale-as-current, irrelevant-injection, and latency rates within their
  registered limits;
- acceptable primary-judge agreement and no unresolved structural blinding
  leak, including registered raw-agreement, chance-corrected kappa,
  identifiability, and condition-guess bounds. Agreement and condition-guess
  decision bounds resample whole judge packets so correlated A-D slots do not
  count as independent evidence.

Pilot protocols enforce conservative outer bounds even when the
preregistration is stricter: identifiability at most 0.05, irrelevant injection
at most 0.02, stale-as-current at most 0.01, oracle headroom at most 0.15,
top-one usefulness at least 0.70, evidence completeness at least 0.80, and
`NONE` precision and recall at least 0.90. Pilot success also fixes the
per-dimension rubric cutoff at 3/5 and forbids checkpoint or query-class
noninferiority margins below -0.05.

Condition-guess accuracy is a diagnostic, not proof of a leak: a genuinely
better oracle answer can be recognizable from quality alone. Structural
metadata leakage and judge identifiability flags require an audit.

Missing denominators are unavailable, never perfect. Development results are
always ineligible for `go`, even when their numerical thresholds pass. A pilot
`go` is not a production or human-superiority claim. The report has
`mutation_authority: none`; the operator must still make and document the
separate limited-canary decision.

## Relationship to other evaluation paths

- The deterministic eval harness remains the fast regression floor.
- Shadow retrieval produces the bounded Memory v2 arm and retrieval metrics.
- Human-baseline pilot tooling measures judge mechanics and one-year human
  comparison; it is not replaced by this four-arm systems study.
- Outcome Replay provides single-change diagnosis after the full oracle exposes
  a material gap.

Automatic promotion, automatic supersession, broad raw prefetch, model-driven
promotion, and live answer injection remain disabled throughout this study.
