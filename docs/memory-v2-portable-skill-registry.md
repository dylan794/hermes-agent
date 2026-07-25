# Portable SkillRegistry Interface

## Status

This document fixes the first portable SkillRegistry contract for Memory v2.
It is an interface and threat-model decision, not an enabled skill installer or
execution engine. The dependency-free Python contract is in
`plugins/memory/memory_v2/skill_registry.py`.

All automatic skill creation, installation, activation, supersession, and
execution remain disabled. No live Hermes profile is changed by this design.

## Boundary

Memory and skills solve different problems and have different authority:

| Surface | Purpose | Trust treatment | May execute? |
|---|---|---|---|
| Raw memory event | Append-only record of what happened | Untrusted evidence | No |
| Semantic memory/project card | Reviewed claim about work | Bounded, source-grounded, untrusted on retrieval | No |
| Skill candidate | Proposal that a repeatable procedure may be useful | Untrusted proposal with a fingerprint | No |
| Skill bundle | Versioned procedure, scripts, references, and assets | Privileged content only after host review | Only through a separate host sandbox |
| Skill execution receipt | Record of a host-run outcome | Untrusted raw evidence for later analysis | No |

A skill is never represented as a `MemoryItem`. Semantic memory may keep a
`procedure_ref` pointing to an exact `SkillRef`, but it cannot activate or
execute that reference. A `skill_candidate` extracted from repeated successful
work is only a proposal.

## Package boundary

The interface module imports only the Python standard library. It does not
import Hermes, Memory v2 storage, SQLite, model tools, or a host executor. It
can therefore move into a standalone package without changing its public data
model.

The first implementation may live beside Memory v2, but adapters must depend
on this contract rather than the contract depending on adapters:

```text
external catalogs ── SkillSourceAdapter ──> quarantine/staging
                                               │
                                               v
host/operator ── opaque authority ──> SkillRegistryWriter
                                               │
                                               v
                               canonical bundles + lifecycle log
                                               │
                              rebuildable metadata index
                                               │
                                               v
agent host <── bounded metadata ── SkillRegistry (read only)
     │
     └── sandboxed host execution ──> SkillExecutionReceipt ──> raw archive
```

The registry has no `execute()` method. Execution, capability grants, network
policy, filesystem mounts, and sandboxing belong to a host adapter.

## Core types

### `SkillRef`

Every reference contains all three fields:

- `skill_id`: stable portable identity such as `memory-v2/source-audit`;
- `version`: source-declared or registry-assigned immutable version;
- `bundle_digest`: exact `sha256:` digest of the canonical bundle tree.

There is intentionally no name-only or implicit-`latest` reference. A UI may
offer an update, but it must resolve the selected update to a new `SkillRef`
before review and authorization.

### `SkillProvenance`

Every staged record identifies the source type, source identifier, source URI,
evidence observation time, acquisition time, optional evidence references, and
optional signer references. Times are absolute ISO-8601 instants. The source
URI is provenance, not proof of trust.

### `SkillDescriptor`

This is the only object returned by search. It contains bounded metadata,
compatibility, required capabilities, provenance, trust review state, and
lifecycle state. It never contains `SKILL.md`, scripts, references, or assets.

`metadata_is_untrusted` is fixed to `true`. Review or signature verification
does not make skill-supplied titles and descriptions safe to concatenate into
an instruction channel.

Each descriptor has a deterministic `record_fingerprint()`. Mutations use it
for optimistic concurrency and operator review binding.

### `SkillBundle`

A bundle is an exact mapping of portable relative paths to bytes and must
contain `SKILL.md`. The v1 bounds are:

- at most 512 files;
- at most 8 MiB per file;
- at most 32 MiB total.

Absolute paths, drive paths, traversal, duplicate canonical paths, and digest
mismatches fail closed. The tree digest includes each normalized path, byte
length, and exact content in sorted-path order.

### `SkillExecutionReceipt`

The host can archive a receipt containing the exact `SkillRef`, host, outcome,
times, granted capabilities, evidence references, and an optional output
digest. It contains no authority token and no output body. The corresponding
raw event may later support a source-grounded `skill_candidate`, but success
does not automatically promote or update a skill.

## Read interface

`SkillRegistry` exposes four operations:

| Operation | Contract |
|---|---|
| `search(query)` | Return at most 50 bounded descriptors after lifecycle and capability filtering. |
| `describe(ref)` | Return the descriptor for one exact pinned reference. |
| `load_bundle(ref)` | Explicitly load and re-hash one exact bundle. No broad body prefetch. |
| `verify(ref)` | Return artifact/signature verification results without activating anything. |

The default search lifecycle is `active`. Revoked and superseded versions are
excluded from current discovery but remain addressable by exact reference for
audit/history subject to host policy.

The canonical bundle store and append-only lifecycle log are authoritative.
Search indexes and SQLite/FTS tables are derived and rebuildable. Index damage
must never change lifecycle or trust state.

## Source adapters

`SkillSourceAdapter` supports bounded discovery and exact bundle fetch. Source
results are `SourceSkillCandidate` values and remain untrusted even when a
catalog labels itself official. The candidate fingerprint binds displayed
metadata, provenance, version, and expected bundle digest.

Initial adapters should support filesystem Agent Skills bundles. Hermes,
Codex/OpenAI, Claude, Git repositories, or hosted catalogs should be thin
translations into this common model. Source-specific trust labels may be kept
as metadata but cannot bypass registry review.

## Mutation interface

`SkillRegistryWriter` is a separate capability from `SkillRegistry`. A normal
agent/retrieval component receives only the read interface.

The mutation lane has four explicit operations:

1. `stage`: store a digest-verified bundle in quarantine;
2. `activate`: make one reviewed pinned version available to a host;
3. `supersede`: link an active record to a distinct reviewed replacement;
4. `revoke`: disable discovery and future activation while preserving history.

Every operation receives a `SkillMutationIntent` with:

- an exact subject reference;
- `confirm=true`;
- a reviewed candidate or record fingerprint;
- a reason, requester, and timestamp;
- the expected current record fingerprint for every canonical-state change;
- a distinct replacement reference for supersession.

Stage and activation bind the candidate fingerprint to the bundle digest.
Supersession binds it to the replacement bundle digest. Revocation binds it to
the reviewed current-record fingerprint.

The writer owns a `SkillAuthorityVerifier`. The caller passes an opaque host
authority object; model-visible strings, booleans, tool arguments, and memory
records cannot implement the verifier or mint authority. The verifier returns a
decision bound to the exact mutation-request fingerprint, trusted host/operator
principal, scope, issue time, and expiry. Verification exceptions, stale
fingerprints, wrong scope, absent authority, or expired authority fail closed.

The committed receipt records the principal and audit references but never the
opaque authority credential.

## Lifecycle rules

```text
untrusted source candidate
          │ explicit stage + authority
          v
       staged/unverified
          │ review + verify + explicit activate + authority
          v
       active/reviewed-or-trusted
          ├── explicit supersede + authority ──> superseded (history retained)
          └── explicit revoke + authority ─────> revoked (history retained)
```

There are no automatic transitions in v1. Revocation and supersession are
append-only lifecycle events rather than overwrite/delete operations. A crash
after preparing but before committing a transition leaves the registry in a
recovery-required state; readers must continue using the last fully committed
state and writers must fail closed until recovery completes.

## Host integration and prompt caching

A host chooses active pinned references at session construction. Registry
changes apply to the next session by default. An explicit host-level “apply
now” operation may rebuild a session only if that host already has a safe cache
invalidation boundary; the registry itself never edits an active prompt or
toolset.

Hosts should progressively disclose:

1. bounded name/description/version metadata;
2. `SKILL.md` only after selection;
3. referenced scripts, references, and assets only when needed.

Before execution, the host independently checks required capabilities against
its own policy, constructs a least-privilege sandbox, and records the actual
grants in the receipt. Agent Skills `allowed-tools` or catalog trust labels may
inform compatibility but cannot replace host capability policy.

## First implementation slice

The safe first slice after this interface is:

1. content-addressed local bundle storage plus an append-only lifecycle log;
2. a rebuildable SQLite metadata index;
3. read-only filesystem Agent Skills discovery and exact bundle validation;
4. descriptor search, exact load, verification, and export adapters;
5. crash/tamper/rebuild tests.

Activation UI, remote installs, automatic skill creation, and all execution
remain out of scope until the storage and authority boundaries pass adversarial
tests.

## Required contract tests

- bundle order does not change a digest; any path/content change does;
- traversal, absolute paths, duplicate canonical paths, oversize content, and
  missing `SKILL.md` fail closed;
- active, superseded, and revoked descriptors require complete lifecycle data;
- search is bounded and defaults to active metadata only;
- no API accepts “latest” in place of a pinned reference;
- absent, expired, wrong-scope, mismatched, or verifier-error authority fails;
- model-originated requests cannot construct or substitute a trusted verifier;
- current-record fingerprints prevent stale activation/supersession/revocation;
- registry search/load expose no execution method;
- execution receipts preserve exact version, capabilities, provenance times,
  and outcome without carrying authority secrets;
- derived-index loss/rebuild cannot change canonical state;
- revoked/superseded versions are suppressed from current discovery but remain
  available for authorized history/audit.
