# Memory v2 owner diagnostic set

## Purpose

The owner diagnostic set identifies the first failing stage in shadow
retrieval before changing the index, reranker, packet builder, or answerer. It
is development-only and cannot authorize a pilot, canary, memory mutation, or
product claim.

The private packet contains query and cited evidence text so the owner can
label usefulness. It must remain outside the repository and live Hermes
profile. The public summary contains counts and digests only.

## 1. Prepare a frozen packet

```bash
python scripts/memory_v2_diagnostic_set.py prepare \
  --raw-events <private-raw-events.jsonl> \
  --output <external-owner-only-packet.json> \
  --public-summary <content-free-summary.json> \
  --episode-count 30 \
  --control-count 8 \
  --held-out-count 10 \
  --authorize-private-history
```

Preparation runs shadow retrieval at a zero utility threshold so the packet
retains every candidate that passed hard safety validation. It separately
reconstructs the frozen `0.45` baseline result. The final chronological ten
episodes are marked `held_out`; do not inspect or tune against their labels
until the development configuration is frozen.

## 2. Complete owner labels

For every development episode, change `owner_labels.status` from `pending` to
`complete` and fill:

- `memory_needed`;
- exact `required_source_refs`;
- useful, stale, and irrelevant candidate IDs;
- the smallest complete bundle of useful candidate IDs;
- current-answer success and citation correctness;
- one assessment for each oracle component;
- optional notes.

Candidate classifications must be disjoint. Minimal-bundle candidates must be
useful candidates. Required source references and candidate IDs must already
exist in that episode. Oracle assessments are `rescues`,
`does_not_rescue`, or `not_applicable`; `pending` is valid only while the
episode remains incomplete.

Validate without exposing private content:

```bash
python scripts/memory_v2_diagnostic_set.py validate \
  --packet <external-owner-only-packet.json> \
  --public-summary <new-content-free-summary.json>
```

Exit code `1` means the packet is valid but owner labels remain incomplete.
Exit code `2` means the packet is invalid.

## 3. Calibrate on development only

```bash
python scripts/memory_v2_diagnostic_set.py score \
  --packet <external-owner-only-packet.json> \
  --split development \
  --output <content-free-development-score.json>
```

Use the development labels to compare fixed threshold and component variants.
After freezing the chosen configuration and artifact digests, label and score
the held-out split exactly once. Never use held-out outcomes to choose another
threshold or component.

## 4. Run Outcome Replay

The diagnostic score is a routing artifact for the existing
[Outcome Replay & Bottleneck Lab](memory-v2-outcome-replay-lab.md), not a
replacement for it. An oracle assessment identifies which offline
single-change replay to construct. Outcome Replay still requires frozen traces,
artifact digests, and trusted-operator or deterministic outcome receipts for
the current and oracle variants.

Do not convert an owner opinion directly into a successful oracle outcome.
Generate the registered offline variant, attach its verified receipt, collect
the minimized replay dataset, and run:

```bash
python scripts/memory_v2_outcome_replay.py validate \
  --dataset <minimized-replay-dataset.json>

python scripts/memory_v2_outcome_replay.py analyze \
  --dataset <minimized-replay-dataset.json> \
  --output <content-free-bottleneck-report.json>
```

Improve only the largest measured safe oracle gap. If labels or variant
receipts are incomplete, the correct result is `not ready`, not a guessed
bottleneck.

## Safety invariants

- Shadow and diagnostic results have `mutation_authority: none`.
- The workflow never changes the live memory profile.
- Private packets are immutable, owner-only, and external to the repository.
- Public summaries contain no query, evidence, answer, or tool-output text.
- The held-out split is chronological, frozen, and prohibited from tuning.
- Single-participant development remains ineligible for pilot or canary `go`.
