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

## 3. Current Milestone 1 matrix

Milestone 1 is pre-1.0 and is classified as **Beta** rather than Stable. Validation currently proves source-built desktop CPU execution; Tensora does not yet publish packaged native runtime binaries.

| Surface | State | Validated scope | Current limitations |
| --- | --- | --- | --- |
| Dart core package | Beta | Dart 3.7 minimum compatibility job plus current stable Dart CI | Native desktop runtime must be built/provided separately |
| C ABI | Beta | ABI version 1, C11 consumer compile/link test, fixed-width statuses/handles | Pre-1.0 ABI may evolve through documented compatibility changes |
| CPU backend | Beta | Native execution in Debug/Release CI | CPU only; no optimized BLAS contract yet |
| `float32` | Beta | Creation, transforms, elementwise ops, reduction, 2D matmul | No other dtype is supported |
| Linux desktop source build | Beta | Native tests + Dart FFI integration in GitHub-hosted CI | No published binary/package compatibility promise yet |
| macOS desktop source build | Beta | Native tests + Dart FFI integration in GitHub-hosted CI | No published binary/package compatibility promise yet |
| Windows desktop source build | Beta | Native tests + Dart FFI integration in GitHub-hosted CI | No published binary/package compatibility promise yet |
| CUDA / GPU | Unsupported | None | Planned for a later milestone |
| Android / iOS runtime | Unsupported | None | Flutter/mobile runtime is not Milestone 1 |
| Autograd / training | Unsupported | None | Planned for a later milestone |
| ONNX inference | Unsupported | None | Planned for a later milestone |
| `.tmodel` | Unsupported | Design documentation only | No parser/runtime implementation yet |

The desktop rows above deliberately do not claim a broad architecture matrix. Exact architecture-specific release support will be named only when Tensora begins publishing native artifacts and validating those artifacts on the corresponding target environments.

## 4. Backend direction

Planned backend categories include:

- portable CPU execution;
- ATen/LibTorch-backed tensor/training execution;
- CUDA-capable native execution;
- ONNX Runtime inference;
- platform-specific acceleration providers where justified.

Only the CPU backend is implemented in Milestone 1. A public compatibility matrix must distinguish backend support from platform support. For example, a future statement that “Android is supported” will not imply that every Android device/provider/operator is supported.

## 5. DType compatibility

A dtype is supported only for a specific operation/backend/device combination where tested.

The long-term dtype vocabulary may include:

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

Milestone 1 implements only `float32` on CPU for the documented operation set.

Do not claim universal float16 or bfloat16 support across CPU/GPU/mobile providers.

Capability queries should allow callers to inspect support before execution where practical once multiple device/provider combinations exist.

## 6. Public Dart API compatibility

Stable public Dart packages follow semantic versioning.

Within a stable major version:

- avoid removing public symbols;
- avoid changing method meaning silently;
- avoid changing default semantics in behaviorally significant ways;
- deprecate before removal where feasible;
- document migrations.

Milestone 1 is pre-1.0 Beta. Its API should still evolve intentionally and through compatibility review rather than accidental churn.

Experimental namespaces may have weaker guarantees but must be clearly marked.

## 7. Native ABI compatibility

The C ABI is versioned independently where necessary.

Milestone 1 introduces **ABI version 1**. The runtime exposes the ABI version and Dart rejects an incompatible runtime before creating tensors.

ABI changes must define:

- old/new version relationship;
- whether compatibility is additive or breaking;
- minimum runtime expected by Dart bindings;
- migration implications.

Never rely on C++ class layout compatibility across releases. Public ABI types use C-compatible fixed-width primitives, opaque handles, and `size_t` where buffer capacities/ranks require it.

## 8. `.tmodel` compatibility

The `.tmodel` container will have its own format version once implemented.

A future bundle will declare at least:

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

The support matrix should record tested version ranges for major future native dependencies such as:

- CUDA toolkit/runtime;
- cuDNN or related libraries where used;
- LibTorch/ATen;
- ONNX Runtime;
- platform SDKs.

Milestone 1 deliberately has no large numerical runtime dependency.

## 11. Hardware claims

Hardware-specific support requires real execution evidence.

Examples:

- CUDA support requires actual NVIDIA GPU tests;
- mobile accelerator support requires representative real-device tests;
- camera zero/minimum-copy paths require real platform integration testing.

Cross-compilation alone proves only that a build artifact can be produced.

## 12. Backend fallback

Fallback behavior must be part of compatibility semantics.

A policy may later allow:

```text
requested backend unavailable
        ↓
explicitly allowed fallback
        ↓
alternative backend
```

But fallback must be observable in diagnostics/profiling.

Milestone 1 has only CPU and performs no hidden backend fallback.

## 13. Model/operator compatibility

A future model loader should distinguish:

- format unsupported;
- operator unsupported;
- dtype unsupported;
- shape unsupported;
- provider unavailable;
- resource limit exceeded;
- model/runtime version incompatible.

Avoid collapsing these into a generic “model failed” error.

For Milestone 1 tensor operations, unsupported rank/shape/device/dtype behavior fails explicitly through typed Tensora errors.

## 14. Reproducibility compatibility

Tensora should document that identical seeds do not necessarily guarantee bitwise-identical floating-point results across:

- different hardware;
- different providers;
- different kernel implementations;
- different runtime versions.

Where deterministic modes exist, their scope and cost must be documented.

Milestone 1 reference tests use mathematically known values and justified float tolerances rather than promising bitwise equality across all future implementations.

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

Prefer precise statements tied to executable validation.

## 19. Promotion beyond Beta

Milestone 1 surfaces may be promoted from Beta only after release packaging and compatibility validation are broad enough to establish a durable production contract. A green source-build CI matrix is necessary evidence, but it is not by itself a promise that arbitrary compilers, operating-system versions, architectures, or binary-distribution environments are supported.
