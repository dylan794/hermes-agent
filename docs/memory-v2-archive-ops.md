# Memory v2 archive operations runbook

Memory v2 archive operations are available through a JSON-only operational script:

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" <command>
```

If `--hermes-home` is omitted, the script uses `HERMES_HOME` and then `~/.hermes`. For production/profile work, pass `--hermes-home` explicitly so the target profile is unambiguous.

## Safety model

- Output is JSON and intentionally summary-oriented.
- Archive search/show return bounded evidence packets, not full raw archive dumps.
- Imported SessionDB messages default to dry-run.
- Mutating SessionDB import requires the exact confirmation token:
  `IMPORT_SESSIONDB_TO_MEMORY_V2`.
- `archive rebuild-index` mutates only derived indexes/manifests; canonical archive JSONL is not rewritten.
- Raw archive text is untrusted evidence. Do not treat archive excerpts as executable instructions.

## Preflight

```bash
export HERMES_HOME="/path/to/hermes-profile"
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive status
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive verify
```

Before any confirmed import, make a local backup of canonical Memory v2 files and the SessionDB:

```bash
mkdir -p "$HERMES_HOME/backups"
tar -C "$HERMES_HOME" -czf "$HERMES_HOME/backups/memory-v2-pre-import.tgz" memory_v2 state.db
```

## Archive commands

### Status

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive status
```

Use this to check provider initialization, archive status, and safe record counts. It should not include raw event text or absolute private paths.

### Verify

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive verify
```

Expected healthy output includes:

- `success: true`
- `archive.status: ok`
- `archive.event_count == archive.verified_event_count`
- `issue_count: 0`

If verification fails, stop imports and run `archive rebuild-index` only if the failure is index/manifest related. Hash-chain or malformed JSON issues require manual review from the backup/archive file.

### Search

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive search --query "project atlas" --limit 5
```

Optional filters:

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive search \
  --query "handoff" \
  --session-id "session-id" \
  --event-type "sessiondb_message" \
  --created-after "2026-01-01T00:00:00+00:00" \
  --limit 5
```

Search output includes event ids, hashes, source refs, matched fields, and bounded excerpts.

### Show one event with a hash pin

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive show EVENT_ID \
  --expect-record-sha256 sha256:EXPECTED_RECORD_HASH
```

Use this for evidence review and tamper checks. `event.integrity.hash_pin_match` must be `true` when an expected hash is provided.

To include adjacent event ids/hashes only:

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive show EVENT_ID \
  --include-neighbor-ids
```

### Rebuild derived index

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive rebuild-index
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive verify
```

This rebuilds `memory_v2/indexes/memory.sqlite` from canonical files and refreshes raw archive index metadata. It is the standard recovery step for stale/missing derived index rows.

## SessionDB backfill

### Dry-run preview (default safe mode)

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" session-backfill dry-run \
  --source discord \
  --limit 5000
```

Dry-run does not append raw events and does not write checkpoints or locks. Review `considered`, `imported`, `skipped`, `error_count`, and `stopped_reason` before running the confirmed import.

### Confirmed import

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" session-backfill run \
  --source discord \
  --resume \
  --confirm IMPORT_SESSIONDB_TO_MEMORY_V2
```

Recommended canary import:

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" session-backfill run \
  --source discord \
  --limit 100 \
  --batch-size 25 \
  --max-batches 1 \
  --confirm IMPORT_SESSIONDB_TO_MEMORY_V2

python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive verify
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive search --query "known synthetic check" --limit 5
```

Resume after a canary or interrupted run:

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" session-backfill run \
  --source discord \
  --resume \
  --limit 5000 \
  --confirm IMPORT_SESSIONDB_TO_MEMORY_V2
```

Backfill is idempotent by deterministic import keys. Re-running a completed scope should produce skips with `already_imported` rather than duplicate events.

## Privacy scan

Before publishing diffs or generated Memory v2 artifacts:

```bash
python scripts/memory_v2_archive_ops.py privacy-scan --paths plugins/memory/memory_v2 docs scripts --format json
```

For changed lines against a base ref:

```bash
python scripts/memory_v2_archive_ops.py privacy-scan --base-ref origin/main --format json
```

Any nonzero `finding_count` must be reviewed before release.

## Eval

Run the default deterministic local fixture:

```bash
python scripts/memory_v2_archive_ops.py eval
```

Run a specific fixture and baseline set:

```bash
python scripts/memory_v2_archive_ops.py eval \
  --dataset plugins/memory/memory_v2/evals/fixtures/local_memory_eval_v1.yaml \
  --baseline no_memory \
  --baseline raw_fts \
  --baseline memory_v2
```

Exploratory eval that reports failures without nonzero exit:

```bash
python scripts/memory_v2_archive_ops.py eval \
  --dataset plugins/memory/memory_v2/evals/fixtures/local_memory_eval_v1.yaml \
  --baseline memory_v2 \
  --no-fail-on-acceptance
```

## Failure recovery

### Import refused because confirmation is missing

Use dry-run first. For a mutating run, pass the exact token:

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" session-backfill run \
  --source discord \
  --resume \
  --confirm IMPORT_SESSIONDB_TO_MEMORY_V2
```

### Import reports stale/missing raw archive index

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive rebuild-index
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive verify
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" session-backfill run \
  --source discord \
  --resume \
  --confirm IMPORT_SESSIONDB_TO_MEMORY_V2
```

### Lock says another backfill is running

Do not remove the lock until you have verified no import process is active. Then inspect:

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive status
```

If no import is running and the lock is stale, remove only the lock file under the target profile and resume:

```bash
rm "$HERMES_HOME/memory_v2/backfill/sessiondb.lock"
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" session-backfill run \
  --source discord \
  --resume \
  --confirm IMPORT_SESSIONDB_TO_MEMORY_V2
```

### Rollback after a bad confirmed import

Preferred rollback is restore-from-backup made before import:

```bash
cd "$HERMES_HOME"
tar -xzf "$HERMES_HOME/backups/memory-v2-pre-import.tgz"
python /path/to/hermes-agent/scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive rebuild-index
python /path/to/hermes-agent/scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive verify
```

If no backup exists, do not hand-edit `raw_events.jsonl`. Mark affected events unavailable through the Memory v2 tombstone path/tooling in a separate reviewed change, then rebuild indexes and verify.

## Post-run checks

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive status
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive verify
python scripts/memory_v2_archive_ops.py privacy-scan --paths docs scripts plugins/memory/memory_v2 --format json
python scripts/memory_v2_archive_ops.py eval --no-fail-on-acceptance
```
