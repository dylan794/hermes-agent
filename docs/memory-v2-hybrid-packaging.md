# Memory v2 hybrid packaging boundary

## Status

Memory v2 remains in the Hermes fork while its real-work evaluation contract is
still changing. At the same time, its dependency-light retrieval and procedure
contracts are kept behind an explicit portable boundary so they can become a
standalone plugin package after the development study stabilizes the API.

This is a hybrid development boundary, not a claim that a standalone package
has already been released.

## Portable core

`plugins/memory/memory_v2/portable_manifest.json` is the machine-readable
extraction manifest. It identifies the modules that must import and initialize
without loading Hermes runtime, provider, gateway, tool, or session modules:

- redaction and untrusted-evidence escaping;
- raw-preserving workstream evidence derivation;
- scope-first disposable retrieval;
- memory-need routing;
- bounded reranking and abstention;
- read-only shadow orchestration;
- version-pinned SkillRegistry contracts.

These modules accept ordinary mappings, dataclasses, protocols, paths, and
callables. They do not own the live Hermes profile, provider lifecycle, canonical
memory mutation, tool execution, or model-facing authorization.

## Hermes adapter

The current Hermes adapter remains rooted at
`plugins/memory/memory_v2/__init__.py`. It owns:

- `MemoryProvider` lifecycle integration;
- profile-scoped configuration and paths;
- canonical archive, candidate, memory, and project-card storage;
- provider tools and feature flags;
- trusted host/operator mutation checks;
- live prefetch packet integration.

The offline shadow pipeline is deliberately not connected to that live adapter.
Its output remains bounded, untrusted, read-only evidence.

## Enforced boundary

`tests/plugins/memory/test_memory_v2_portable_boundary.py` loads every portable
module in an isolated synthetic package. This bypasses the Hermes adapter
entrypoint and fails if importing the declared core loads any forbidden Hermes
runtime namespace.

The test executes the modules; it does not inspect source text. The manifest
also fails validation when it names a missing module, duplicates a module, or
uses an unexpected schema.

## Extraction plan

After the single-participant development study:

1. Freeze the smallest portable API actually exercised by the study.
2. Move the manifest-listed modules into a standalone `memory-v2-core` package.
3. Keep thin compatibility re-exports in the fork for one transition release.
4. Package the Hermes provider lifecycle as the standalone plugin adapter.
5. Run the same core contract suite against both the in-fork adapter and the
   independently installed plugin.
6. Version and publish only after the privacy scan, release matrix, and
   installation/removal smoke tests pass.

No study result may weaken the boundary: portable code has no mutation
authority, and a standalone adapter must preserve the same fail-closed defaults
as the in-fork implementation.
