# Memory v2 Dream Cycle Pre-Deployment Test Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Prove the Memory v2 dream cycle is safe, source-grounded, reversible, observable, and non-destructive before enabling any real-profile cron deployment.

**Architecture:** Treat dream cycle deployment as a safety-critical rollout. Automated tests and sandbox/shadow runs must mechanically prove invariants before any human approval. The first real deployment may only use `--auto-apply off`; `safe_rejections` is a later, separately-approved rollout.

**Tech Stack:** Python, pytest, Memory v2 provider/plugin, profile-scoped filesystem stores, SQLite FTS index, Hermes cron/CLI, `flock`, JSON/YAML audit artifacts.

---

## Non-negotiable invariants

1. `auto_apply=off` must never promote candidates, reject candidates, create durable memory items, apply review-plan actions, or touch profile config/skills/cron files.
2. `auto_apply=safe_rejections` may only reject candidates that are already in the scoped review queue and have `review_lane == "reject_or_archive"`.
3. Dream cycle must never auto-promote memory.
4. Every mutation must have an operation/audit record and source-grounded explanation.
5. Reports/logs must not leak secrets or raw private context unnecessarily.
6. Every run must write profile-local artifacts only under the selected `--hermes-home`.
7. Cron must never run overlapping dream cycles.
8. Rollback must be tested before live deployment.

---

## Phase 0: Freeze and inspect implementation

**Objective:** Ensure the candidate being tested is known and not already deployed.

**Commands:**

```bash
cd /path/to/hermes-agent
git status --short
git diff --name-only
grep -R "memory_v2_dream_cycle\|plugins.memory.memory_v2.dream" \
  "$HOME/.hermes/cron" \
  "$HOME/.hermes/config.yaml" \
  2>/dev/null || true
python -m plugins.memory.memory_v2.dream --help > /tmp/memory-v2-dream-help.out 2> /tmp/memory-v2-dream-help.err
```

**Pass gates:**
- No active real-profile cron job exists.
- Working tree changes are understood.
- CLI help works.
- CLI warning is either fixed or formally tracked as a blocker/waiver before cron.

**Abort if:** dream cycle is already scheduled, target files are unclear, or CLI help has unexplained warnings.

---

## Phase 1: Automated regression and eval gate

**Objective:** Prove current tests/evals pass before adding deployment risk.

**Commands:**

```bash
cd /path/to/hermes-agent
source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || true
python -m pytest tests/plugins/memory -q
python -m pytest tests/plugins/memory/test_memory_v2_dream_cycle.py \
  tests/plugins/memory/test_memory_v2_review_queue.py \
  tests/plugins/memory/test_memory_v2_review_actions.py \
  tests/plugins/memory/test_memory_v2_operations_health.py \
  tests/plugins/memory/test_memory_v2_extraction.py -q
python scripts/memory_v2_eval.py \
  --dataset plugins/memory/memory_v2/evals/fixtures/local_memory_eval_v1.yaml \
  --dataset plugins/memory/memory_v2/evals/fixtures/local_memory_eval_project_v1.yaml \
  --dataset plugins/memory/memory_v2/evals/fixtures/local_memory_eval_adversarial_v1.yaml \
  --baseline no_memory --baseline raw_fts --baseline memory_v2
```

**Pass gates:**
- 100% pytest pass.
- Eval acceptance passes for every fixture.
- Memory v2 source recall >= 0.95 average.
- Suppression >= 0.90 average.
- No secret/adversarial fixture leaks.

**Abort if:** any test/eval fails or memory_v2 underperforms raw FTS by more than 0.05 on source recall.

---

## Phase 2: Add missing deterministic hardening tests

**Objective:** Convert debate findings into automated tests before live deployment.

**Required tests to add:**

1. Adversarial candidate matrix:
   - prompt injection text
   - fake system/developer instructions
   - secret-looking strings
   - missing/fabricated source refs
   - duplicate groups
   - contradictions
   - skill/procedure/config/cron mutation bait
   - weird IDs, path traversal, unicode, control chars
   - huge queues

2. Provider abuse tests through `MemoryV2Provider.handle_tool_call`:
   - invalid `auto_apply`: `promote_all`, `true`, objects, whitespace, uppercase
   - invalid dates/modes
   - absurd limits: negative, zero, huge
   - assert invalid args cause no partial mutation or report write

3. Golden report/schema tests:
   - normalize `run_id` and timestamps
   - validate required fields
   - assert paths are relative and profile-local

4. Property/invariant tests:
   - for generated candidate sets, `off` mode changes no gate decisions and creates no memory items
   - `safe_rejections` applies only reject actions on scoped reject-lane candidates
   - no promotion ever occurs

5. Idempotence tests:
   - repeated identical runs converge
   - candidate count does not grow unbounded
   - reports have unique paths

6. Crash/concurrency tests:
   - lock contention produces one winner / one safe skip
   - interrupted writes leave parseable prior artifacts
   - no corruption in JSON/YAML/index

7. Performance/load tests:
   - 0 raw/0 candidates
   - 50 raw/50 candidates
   - 500 raw/500 candidates
   - 500 raw/500 candidates/100 rejection candidates

**Pass gates:**
- All new tests pass repeatedly.
- 500/500 load run finishes under 10s locally unless explicitly waived.
- Report JSON < 2MB and episode YAML < 512KB for bounded fixture.

**Abort if:** any invariant depends on manual review instead of code.

---

## Phase 3: Backup and rollback drill

**Objective:** Prove recovery before touching live memory.

**Commands:**

```bash
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_ROOT="$HOME/.hermes/backups/memory-v2-dream-predeploy"
BACKUP_DIR="$BACKUP_ROOT/$STAMP"
mkdir -p "$BACKUP_DIR"
tar --xattrs --acls -czf "$BACKUP_DIR/hermes-profile.tar.gz" -C "$HOME" .hermes
sha256sum "$BACKUP_DIR/hermes-profile.tar.gz" > "$BACKUP_DIR/hermes-profile.tar.gz.sha256"
sha256sum -c "$BACKUP_DIR/hermes-profile.tar.gz.sha256"
RESTORE_TEST="/tmp/hermes-restore-test-$STAMP"
mkdir -p "$RESTORE_TEST"
tar -xzf "$BACKUP_DIR/hermes-profile.tar.gz" -C "$RESTORE_TEST"
test -d "$RESTORE_TEST/.hermes"
```

**Pass gates:**
- Backup exists and checksum verifies.
- Restore into temp path succeeds.
- Restored profile has `memory_v2` and config files.

**Abort if:** backup or restore is untested.

---

## Phase 4: Sandbox clone dry runs

**Objective:** Run exact command path against an isolated profile clone.

**Commands:**

```bash
SANDBOX="$HOME/.hermes-predeploy-sandbox-memory-v2"
rm -rf "$SANDBOX"
mkdir -p "$SANDBOX"
rsync -a --delete --exclude 'logs/' --exclude 'backups/' "$HOME/.hermes/" "$SANDBOX/"
cd /path/to/hermes-agent
python -m plugins.memory.memory_v2.dream --hermes-home "$SANDBOX" --auto-apply off > /tmp/memory-v2-dream-sandbox.json
python -m json.tool /tmp/memory-v2-dream-sandbox.json > /tmp/memory-v2-dream-sandbox.pretty.json
```

**Validation script:**

```bash
python - <<'PY'
import json
from pathlib import Path
r = json.loads(Path('/tmp/memory-v2-dream-sandbox.json').read_text())
assert r['success'] is True
assert r['auto_apply'] == 'off'
assert r['review_apply'] is None
assert r['policy'] == 'source_grounded_review_plan_first'
assert r['before_counts']['memory_items'] == r['after_counts']['memory_items']
assert r['before_counts']['rejected_candidates'] == r['after_counts']['rejected_candidates']
assert r['before_counts']['operation_records'] == r['after_counts']['operation_records']
print('sandbox-off-pass')
PY
```

**Repeat:** run three times and verify counts converge except report/episode artifact growth.

**Pass gates:**
- JSON/YAML parse.
- No promotions, rejections, operation records, or memory item changes under `off`.
- Writes are only under sandbox `memory_v2` allowlist.

**Abort if:** default profile changes or unexpected paths change.

---

## Phase 5: Shadow real-profile copy and diff allowlist

**Objective:** Test against realistic data without touching the real profile.

**Commands:**

```bash
SHADOW="/tmp/hermes-memory-v2-dream-shadow-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$SHADOW"
rsync -a --delete --exclude '.env' --exclude 'logs/' --exclude 'backups/' "$HOME/.hermes/" "$SHADOW/"
python -m plugins.memory.memory_v2.dream --hermes-home "$SHADOW" --auto-apply off > /tmp/memory-v2-dream-shadow.json
```

**Pass gates:**
- Same `off` invariants as sandbox.
- Machine-readable diff only includes allowed `memory_v2` report/episode/index/pending-candidate paths.
- Secret scanner finds no obvious credentials in report/log output.

**Abort if:** anything outside allowlist changes.

---

## Phase 6: Cron-like execution and lock gate

**Objective:** Prove actual deployment command is safe under cron constraints.

**Wrapper pattern:**

```bash
LOCK="$HOME/.hermes/run/memory-v2-dream.lock"
mkdir -p "$HOME/.hermes/logs/memory-v2-dream" "$(dirname "$LOCK")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
JSON_OUT="$HOME/.hermes/logs/memory-v2-dream/sandbox-$STAMP.json"
ERR_OUT="$HOME/.hermes/logs/memory-v2-dream/sandbox-$STAMP.err"
flock -n "$LOCK" bash -lc 'cd /path/to/hermes-agent && python -m plugins.memory.memory_v2.dream --hermes-home "$HOME/.hermes-predeploy-sandbox-memory-v2" --auto-apply off' > "$JSON_OUT" 2> "$ERR_OUT"
python -m json.tool "$JSON_OUT" > /dev/null
```

**Pass gates:**
- Lock contention test: one process runs, second fails/skips cleanly.
- Logs are local, parseable, redacted, and include run id/path/exit status.
- Cron-like env uses correct Python/cwd/profile.

**Abort if:** no lock, unparseable logs, wrong profile, or stderr warning is unexplained.

---

## Phase 7: First real-profile one-shot, off mode only

**Objective:** One manual production run after all automated/sandbox gates pass.

**Required approval phrase before running:**

```text
Approved to run Memory v2 dream cycle once against ~/.hermes with --auto-apply off.
```

**Command:**

```bash
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$HOME/.hermes/logs/memory-v2-dream"
LOCK="$HOME/.hermes/run/memory-v2-dream.lock"
mkdir -p "$LOG_DIR" "$(dirname "$LOCK")"
flock -n "$LOCK" bash -lc 'cd /path/to/hermes-agent && python -m plugins.memory.memory_v2.dream --hermes-home "$HOME/.hermes" --auto-apply off' > "$LOG_DIR/prod-manual-$STAMP.json" 2> "$LOG_DIR/prod-manual-$STAMP.err"
```

**Pass gates:**
- `success == true`
- `auto_apply == off`
- `review_apply is null`
- no memory item delta
- no rejected candidate delta
- no operation record delta
- artifacts exist and parse

**Abort/rollback if:** any mutation beyond allowed report/pending extraction/index artifacts occurs.

---

## Phase 8: Disabled cron staging, then off-only cron

**Objective:** Stage cron without accidentally enabling risky behavior.

**Policy:**
- Cron entry starts disabled/commented.
- Exact command must include `--auto-apply off`.
- Use `flock -n`.
- Write local JSON/stderr logs.
- No Discord success messages.

**Enable only after:**
- manual one-shot real-profile off run passes
- the profile owner explicitly approves enabling cron

**Observation window:**
- 7 clean scheduled `off` runs before considering any mutation mode.

---

## Phase 9: Safe-rejections is a separate rollout

**Objective:** Keep mutation rollout separate from dream-cycle reporting rollout.

Do not enable `safe_rejections` until:
- at least 7 clean off-mode production reports
- adversarial/property tests pass
- shadow `safe_rejections` canary has zero false-positive rejections
- the profile owner explicitly approves

First `safe_rejections` run must be sandbox/shadow, then one manual live run, never immediate nightly cron.

---

## Absolute deployment blockers

- Any promotion by dream cycle.
- Any mutation under `auto_apply=off` outside allowlist.
- Any unredacted secret in report/log/eval artifact.
- Any invalid tool args causing mutation.
- Any source-less or fabricated-source candidate promoted/rejected automatically.
- Any write outside explicit `--hermes-home`.
- Any config/skills/cron/profile file touched by dream cycle.
- Any concurrency corruption or overlapping cron run.
- Any rollback drill failure.
- Any unexplained CLI warning in production path.

---

## Final recommendation

Deployment is not ready until Phase 2 hardening tests and Phase 3 rollback drill exist and pass. After that, run sandbox/shadow off-mode cycles, then a single live off-mode manual run, then only later a disabled/staged cron. `safe_rejections` should be treated as a second deployment, not part of initial rollout.
