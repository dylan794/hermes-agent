# Memory v2 human-baseline operational pilot

## Purpose and status

This runbook turns the human-baseline harness into a small operational pilot
for estimating judge agreement, blinding effectiveness, item difficulty, and
participant-cluster variance. The pilot is for rubric repair and confirmatory
power planning only. It cannot produce a superiority claim, and every pilot
participant, project, item, evidence pack, response, and judgment is excluded
from the later confirmation pool.

The repository includes a deliberately non-runnable
[`human_baseline_pilot_protocol_template_v1.yaml.example`](../plugins/memory/memory_v2/evals/fixtures/human_baseline_pilot_protocol_template_v1.yaml.example).
Copy it outside the live Hermes profile, replace every `REPLACE` marker, and
save it with a `.yaml` suffix. The initial design has six participants, eight
queries per participant, 48 paired response items, two independent primary
judges per response, and blinded adjudication for primary-outcome disagreement.

## Required inputs

- Six consenting knowledge workers with opaque pilot registry IDs.
- Auditable evidence snapshots at 30, 90, 180, and 365 days. A genuine
  365-day history is preferred; a retrospective snapshot is acceptable for
  pilot mechanics only when chronology and completeness can be audited.
- Eight source-grounded questions written without exposure to either response.
- A condition-neutral evidence and rubric pack per question.
- Human and Memory v2 responses produced with the same evidence access and
  task-specific time limit.
- Two qualified, independent, blinded primary judges. A separate blinded
  adjudicator scores a primary packet only when primary binary success differs.
- A condition-blind operator who prepares packets and keeps the assignment key
  and integer seed sealed until every judgment is frozen.

Do not recruit anyone, ingest private work evidence, or distribute response
packets without the appropriate consent, retention, and access-control review.

## Disjointness gate

The CLI checks exact participant IDs, query IDs, workstream IDs, study IDs, and
normalized prompts without printing the raw overlapping values:

```bash
python scripts/memory_v2_human_baseline.py audit-disjoint \
  --candidate <pilot-protocol.yaml> \
  --against plugins/memory/memory_v2/evals/fixtures/human_baseline_protocol_example_v1.yaml \
  --against <any-prior-pilot-or-confirmation-protocol.yaml> \
  --output <pilot-disjointness.json>
```

Exit `0` means no exact overlap was found; exit `1` means overlap was found;
exit `2` means an input was invalid. This check cannot detect renamed people,
related employers/projects, paraphrased items, or common evidence sources. A
trusted operator must audit those relationships and sign the frozen pilot
manifest before response collection.

## Execution

1. Freeze the completed protocol, source snapshot inventory, evidence cutoffs,
   access rules, time limits, participant exclusions, and retention policy.
2. Pass the disjointness gate against every development and prior study pool.
3. Collect complete human and Memory v2 response documents using schema
   `memory-v2-human-baseline-responses/v1`. Never put evaluation questions,
   oracle answers, rubric notes, or judge comments into Memory v2's archive.
4. Prepare public A/B packets and a private key:

   ```bash
   python scripts/memory_v2_human_baseline.py prepare \
     --protocol <pilot-protocol.yaml> \
     --human-responses <human-responses.json> \
     --memory-responses <memory-v2-responses.json> \
     --packet-output <public-judge-packets.json> \
     --key-output <sealed-private-key.json> \
     --seed <sealed-integer-seed>
   ```

5. Have primary judges work independently. Each judgment records
   `judge_role: primary`, dimension scores, gate failures, a material-error
   reason plus evidence references when required, a post-score condition guess,
   and a per-slot `potentially_identifiable` flag.
6. If primary binary success differs at the 365-day endpoint, obtain one new
   blinded judgment with `judge_role: adjudicator`. The scorer rejects missing,
   unnecessary, duplicate, or non-primary-checkpoint adjudicators.
7. Freeze judgments, reveal the private key, and score:

   ```bash
   python scripts/memory_v2_human_baseline.py score \
     --protocol <pilot-protocol.yaml> \
     --packets <public-judge-packets.json> \
     --key <sealed-private-key.json> \
     --judgments <frozen-judgments.json> \
     --output <pilot-report.json>
   ```

The valid pilot result is `ineligible_by_design`; that is expected and must not
be changed with a confirmatory label.

## Pilot measurements and decisions

Use `inter_rater.binary_agreement_rate` and `inter_rater.by_checkpoint` for
pairwise agreement among primary judges. Review every disagreement and the
adjudication rate; do not use adjudicator agreement to inflate the primary
agreement estimate.

Use `blinding_diagnostic` for condition-guess accuracy, `unsure` rate, and
identifiability flags. High non-`unsure` accuracy or repeated identifiability
flags trigger a condition-blind formatting and item audit. Do not silently
delete difficult-to-blind items.

Use `pilot_diagnostics.item_difficulty` for human, Memory v2, combined success,
rubric means, and judge disagreement by opaque query fingerprint. Investigate
floor/ceiling items, ambiguous evidence, and items whose judge disagreement is
high. Repair the rubric using pilot data only; repaired items remain excluded
from confirmation.

Use `pilot_diagnostics.participant_cluster_variance.primary` and its checkpoint
breakdown for participant-mean variance, standard error, ICC, and one-way
random-effects variance components. These values are power-analysis inputs,
not evidence of a population effect. If the design is unbalanced, the report
marks it and ICC components are withheld.

Before moving to confirmation, require complete evidence packs, no unresolved
privacy or safety event, acceptable judge agreement under the frozen pilot
criterion, an audited blinding review, and a documented power calculation. Keep
automatic promotion, automatic supersession, broad raw prefetch, and
model-driven promotion disabled throughout.
