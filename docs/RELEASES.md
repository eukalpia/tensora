# Tensora Release Process

This document defines how Tensora releases should be prepared, validated, versioned, and published.

The release process exists to ensure that a published version corresponds to a specific, tested repository state with clear compatibility claims.

## 1. Versioning

Stable public Dart packages follow Semantic Versioning.

Tensora also maintains independent versioned compatibility surfaces where necessary:

- public Dart API;
- native C ABI;
- `.tmodel` format;
- optional backend/plugin interfaces;
- serialized checkpoint formats where a stable contract exists.

A package version does not imply that every internal compatibility surface shares the same numeric version.

## 2. Pre-1.0 policy

Before 1.0, public APIs may evolve more quickly, but breaking changes must still be:

- intentional;
- documented;
- accompanied by migration notes when users are expected to be affected.

Experimental APIs should be visibly marked so that iteration does not accidentally create a stability promise.

## 3. Release types

### Patch release

For compatible bug fixes, security fixes, documentation corrections, or narrowly scoped performance improvements that do not intentionally change stable public semantics.

### Minor release

For backward-compatible features, new APIs, new supported backend/device combinations, and additive model-format/runtime capabilities.

### Major release

For intentional breaking changes to stable public contracts.

## 4. Release candidate

Before a stable release, create a release candidate from an exact revision.

The candidate must not be considered validated if additional code changes are introduced after validation without re-running affected gates.

## 5. Required release evidence

A release candidate should include evidence for all relevant supported surfaces:

- Dart formatting/static analysis;
- Dart unit/integration tests;
- native unit tests;
- ABI integration tests;
- numerical reference tests;
- supported backend/device tests;
- sanitizer results where applicable;
- model-format validation/security tests;
- stress/leak tests for changed resource-sensitive components;
- Flutter integration tests for supported Flutter surfaces;
- example builds/runs;
- benchmark baseline;
- compatibility matrix update;
- changelog/release notes.

## 6. Real hardware requirement

A release must not claim stable hardware-dependent functionality based solely on compilation.

Examples:

- CUDA claims require real NVIDIA execution;
- iOS/Android runtime claims require appropriate runtime/device validation;
- camera pipeline claims require platform integration testing;
- hardware acceleration claims require tested provider behavior.

## 7. Release branch/tag strategy

The exact branch/tag strategy may evolve once implementation begins, but releases should preserve these principles:

- release tags are immutable;
- release artifacts map to one exact commit;
- generated artifacts are reproducible where practical;
- checksums are published for native binaries/model tooling where applicable;
- signed artifacts should be introduced when distribution infrastructure is mature.

## 8. Changelog

Every release should summarize:

```text
Added
Changed
Fixed
Performance
Security
Deprecated
Removed
Compatibility
Known issues
```

Only include sections that have relevant changes.

## 9. Compatibility notes

Release notes must state material changes to:

- supported platforms;
- supported architectures;
- native provider requirements;
- CUDA/runtime versions;
- `.tmodel` compatibility;
- public Dart API;
- native ABI;
- model/checkpoint loading behavior.

## 10. Breaking-change checklist

Before a breaking release:

- identify every stable API/ABI/format break;
- link relevant RFCs;
- publish migration guidance;
- document replacement APIs;
- identify last compatible release;
- update examples and documentation;
- test migrations where tooling exists.

## 11. Security releases

Security fixes may require an accelerated release path.

Requirements remain:

- fix validated on exact candidate;
- regression test added when disclosure constraints allow;
- affected versions identified as accurately as possible;
- disclosure coordinated according to `SECURITY.md`;
- public details avoid unnecessarily exposing users before a fix is available.

## 12. Performance releases

If a release highlights a performance improvement, provide reproducible benchmark context:

- hardware;
- OS;
- backend;
- model/operator;
- shapes/batch sizes;
- dtype;
- warmup;
- measurement method;
- before/after versions.

Do not attribute upstream provider improvements to Tensora unless Tensora changed the relevant layer.

## 13. Native binary distribution

When Tensora begins publishing native binaries:

- publish only supported target combinations;
- include checksums;
- document build inputs/toolchain versions;
- avoid packaging unused large backends into all distributions;
- validate runtime loading on clean representative systems;
- preserve third-party notices/licenses required by dependencies.

## 14. Dart/Flutter package release

Before publishing a package:

- public API docs are complete enough to use the package;
- package metadata is correct;
- license references are correct;
- unsupported platforms are not advertised;
- native asset/plugin packaging is tested;
- examples use the release candidate versions.

## 15. `.tmodel` format release

Before declaring a `.tmodel` format version stable:

- schema is published;
- parser has malformed-input and fuzz coverage;
- compatibility behavior is documented;
- resource limits exist;
- integrity behavior exists;
- CLI inspect/verify supports the version;
- at least one real deployment workflow consumes it.

## 16. Release blocking conditions

Do not release with:

- known memory corruption in supported paths;
- unresolved data race in supported paths;
- known model-parser vulnerability without mitigation;
- failing required CI on the release revision;
- broken supported platform packaging;
- placeholder/fake implementation exposed as stable;
- undocumented breaking API/ABI/format change;
- unexplained major performance regression in a core supported workload.

## 17. Post-release verification

After publication:

- verify packages/artifacts are downloadable;
- verify checksums/signatures where applicable;
- run a minimal smoke install from public distribution channels;
- confirm documentation references the correct version;
- create follow-up issues for any non-blocking known limitations.

## 18. Rollback/yank policy

If a release is materially unsafe or unusable, prioritize protecting users over preserving release appearance.

Depending on the distribution channel:

- publish a fixed patch rapidly;
- mark the release as affected/deprecated;
- yank/withdraw artifacts only where ecosystem policy and user safety justify it;
- document the recommended safe version.

Never rewrite an existing immutable release tag to hide a mistake.

## 19. Release ownership

A release should have a clearly identified maintainer responsible for ensuring that:

- the candidate revision is known;
- validation evidence corresponds to that revision;
- compatibility/release notes are complete;
- publishing steps are executed consistently.

## 20. Tensora 1.0 gate

A 1.0 release should represent a genuine stability boundary, not a marketing date.

At minimum, 1.0 should have:

- stable core Tensor/Device/DType semantics;
- documented native ABI/version behavior;
- proven resource lifecycle;
- at least one production-capable inference path;
- supported Flutter deployment path;
- documented compatibility matrix;
- reliable tests and release automation;
- stable `.tmodel` V1 if included in 1.0 scope;
- no major public subsystem whose behavior is primarily placeholder/experimental while presented as stable.