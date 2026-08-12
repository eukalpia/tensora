# Tensora Compatibility Policy

This document defines how Tensora communicates platform, backend, device, dtype, API, ABI, and model-format support.

The central rule is simple:

> **Support is a tested contract, not a build flag or roadmap item.**

## 1. Support levels

Every platform/backend feature should use one of these states.

### Stable

The feature:

- is covered by automated tests;
- has been exercised on representative real hardware where hardware behavior matters;
- has documented semantics and limitations;
- is covered by compatibility policy for the relevant release line;
- has no known blocker preventing production use within documented constraints.

### Beta

The feature is functional and meaningfully tested, but one or more of these may still change:

- public API details;
- packaging;
- performance characteristics;
- supported operator/device combinations;
- operational limitations.

Breaking changes should still be documented and minimized.

### Experimental

The feature is available for evaluation but is not a compatibility commitment.

Experimental status must be obvious in documentation and diagnostics.

### Unsupported

The feature is not supported. Do not imply otherwise because an upstream provider might theoretically support it.

## 2. Machine-readable matrix

Tensora should eventually maintain a machine-readable support matrix consumed by documentation and tests.

Dimensions should include:

```text
Tensora version
platform
architecture
backend
provider version range
dtype
training/inference
feature/operator group
support state
known limitations
```

Generated documentation should derive from this source where practical to avoid drift.

## 3. Initial platform direction

The intended platform direction includes:

```text
Linux x64
Windows x64
macOS arm64
Android arm64
iOS arm64
```

This list is directional only until concrete matrix entries are implemented and tested.

Additional architectures and platforms may be added through the normal compatibility process.

## 4. Backend direction

Planned backend categories include:

- portable CPU execution;
- ATen/LibTorch-backed tensor/training execution;
- CUDA-capable native execution;
- ONNX Runtime inference;
- platform-specific acceleration providers where justified.

A public compatibility matrix must distinguish backend support from platform support. For example, “Android supported” does not mean every Android device/provider/operator is supported.

## 5. DType compatibility

A dtype is supported only for a specific operation/backend/device combination where tested.

Planned dtype vocabulary:

```text
bool
uint8
int8
int16
int32
int64
float16
bfloat16
float32
float64
```

Do not claim universal float16 or bfloat16 support across CPU/GPU/mobile providers.

Capability queries should allow callers to inspect support before execution where practical.

## 6. Public Dart API compatibility

Stable public Dart packages follow semantic versioning.

Within a stable major version:

- avoid removing public symbols;
- avoid changing method meaning silently;
- avoid changing default semantics in behaviorally significant ways;
- deprecate before removal where feasible;
- document migrations.

Experimental namespaces may have weaker guarantees but must be clearly marked.

## 7. Native ABI compatibility

The C ABI is versioned independently where necessary.

The runtime must expose an ABI version and reject incompatible callers cleanly.

ABI changes must define:

- old/new version relationship;
- whether compatibility is additive or breaking;
- minimum runtime expected by Dart bindings;
- migration implications.

Never rely on C++ class layout compatibility across releases.

## 8. `.tmodel` compatibility

The `.tmodel` container has its own format version.

A bundle declares at least:

- format version;
- minimum supported Tensora runtime;
- relevant backend/provider requirements;
- model/task metadata.

The loader must fail clearly when the current runtime cannot satisfy the bundle contract.

Breaking format changes require a new format version and documented migration/conversion tooling where practical.

## 9. Compiler/IR compatibility

Internal compiler IR is not automatically a public compatibility surface.

Do not serialize or expose internal IR as a long-lived public format unless a deliberate compatibility contract is established.

## 10. Provider versions

Upstream provider versions can affect behavior materially.

The support matrix should record tested version ranges for major native dependencies such as:

- CUDA toolkit/runtime;
- cuDNN or related libraries where used;
- LibTorch/ATen;
- ONNX Runtime;
- platform SDKs.

Avoid accepting arbitrary provider versions without validation if known incompatibilities exist.

## 11. Hardware claims

Hardware-specific support requires real execution evidence.

Examples:

- CUDA support requires actual NVIDIA GPU tests;
- mobile accelerator support requires representative real-device tests;
- camera zero/minimum-copy paths require real platform integration testing.

Cross-compilation alone proves only that a build artifact can be produced.

## 12. Backend fallback

Fallback behavior must be part of compatibility semantics.

A policy may allow:

```text
requested backend unavailable
        ↓
explicitly allowed fallback
        ↓
alternative backend
```

But fallback must be observable in diagnostics/profiling.

Large implicit GPU↔CPU transfers must not be hidden.

Callers should be able to configure strict failure when fallback is undesirable.

## 13. Model/operator compatibility

A model loader should distinguish:

- format unsupported;
- operator unsupported;
- dtype unsupported;
- shape unsupported;
- provider unavailable;
- resource limit exceeded;
- model/runtime version incompatible.

Avoid collapsing these into a generic “model failed” error.

## 14. Reproducibility compatibility

Tensora should document that identical seeds do not necessarily guarantee bitwise-identical floating-point results across:

- different hardware;
- different providers;
- different kernel implementations;
- different runtime versions.

Where deterministic modes exist, their scope and cost must be documented.

## 15. Deprecation policy

For stable APIs, prefer this sequence:

```text
introduce replacement
 ↓
deprecate old API
 ↓
document migration
 ↓
retain through an appropriate compatibility window
 ↓
remove in a major release if breaking
```

Security-critical or fundamentally incorrect behavior may require faster action; such exceptions must be documented clearly.

## 16. Platform end-of-support

Removing a stable platform/backend requires:

- documented reason;
- advance notice when feasible;
- final supported release identification;
- migration alternatives where available.

## 17. Compatibility review

Any pull request affecting one of these requires explicit compatibility review:

- public Dart API;
- C ABI;
- `.tmodel` schema;
- serialization/checkpoints;
- provider requirements;
- device semantics;
- default fallback behavior;
- package names/dependency boundaries.

## 18. Documentation rule

Never write statements such as:

```text
Supports GPU
Supports Android
Supports training
```

without enough qualification to identify the tested backend/platform scope.

Prefer statements such as:

```text
CUDA training: beta on Linux x64 with tested runtime versions X–Y
ONNX CPU inference: stable on listed desktop targets
```

once those facts are actually established.

## 19. Current state

During the foundation phase, compatibility entries are **planned, not supported**. The repository should not publish a stable support matrix until executable implementations and tests exist.