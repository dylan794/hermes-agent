# Memory v2 archive rollout checklist

This checklist is the short operational companion to `docs/memory-v2-archive-ops.md`.

## Stage 0: local validation

```bash
scripts/run_tests.sh tests/plugins/memory/test_memory_v2_cli.py -q
python scripts/memory_v2_archive_ops.py privacy-scan --paths docs scripts plugins/memory/memory_v2 tests/plugins/memory --format json
python scripts/memory_v2_archive_ops.py eval --no-fail-on-acceptance
```

Do not proceed with privacy findings or failing targeted CLI tests.

## Stage 1: profile preflight

```bash
export HERMES_HOME="/path/to/hermes-profile"
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive status
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive verify
mkdir -p "$HERMES_HOME/backups"
tar -C "$HERMES_HOME" -czf "$HERMES_HOME/backups/memory-v2-pre-import.tgz" memory_v2 state.db
```

Proceed only when archive verification is healthy or any degraded state has a documented recovery plan.

## Stage 2: dry-run import

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" session-backfill dry-run \
  --source discord \
  --limit 5000
```

Review safe counts only:

- `considered`
- `imported`
- `skipped`
- `skipped_reasons`
- `error_count`
- `stopped_reason`

Dry-run must not create archive records, checkpoints, or locks.

## Stage 3: canary import

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" session-backfill run \
  --source discord \
  --limit 100 \
  --batch-size 25 \
  --max-batches 1 \
  --confirm IMPORT_SESSIONDB_TO_MEMORY_V2

python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive verify
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive search --query "known check" --limit 5
```

If index errors appear, run:

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive rebuild-index
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive verify
```

## Stage 4: full resumable import

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" session-backfill run \
  --source discord \
  --resume \
  --limit 5000 \
  --confirm IMPORT_SESSIONDB_TO_MEMORY_V2
```

Repeat with `--resume` until `stopped_reason` is `completed` or no new imports appear.

## Stage 5: post-import acceptance

```bash
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive status
python scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive verify
python scripts/memory_v2_archive_ops.py eval --no-fail-on-acceptance
```

Keep the pre-import backup until the archive has survived normal use and verification after at least one subsequent session.

## Rollback gate

If a confirmed import needs rollback, restore the backup rather than editing JSONL:

```bash
cd "$HERMES_HOME"
tar -xzf "$HERMES_HOME/backups/memory-v2-pre-import.tgz"
python /path/to/hermes-agent/scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive rebuild-index
python /path/to/hermes-agent/scripts/memory_v2_archive_ops.py --hermes-home "$HERMES_HOME" archive verify
```

If no backup exists, stop rollout. Use a separately reviewed tombstone/remediation procedure; do not hand-edit the append-only archive.
