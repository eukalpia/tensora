# Tensora Backend Model

This document defines how execution backends integrate with Tensora and what they may or may not expose to the public API.

## 1. Purpose

Tensora must use mature native numerical infrastructure early while preserving the ability to replace, combine, or specialize providers later.

The backend layer exists to prevent the public Dart API from becoming permanently coupled to one execution engine.

## 2. Backend responsibilities

A backend may provide some or all of:

- device discovery;
- tensor allocation;
- tensor operations;
- model loading;
- model execution;
- autograd;
- compilation;
- streams/events;
- synchronization;
- memory telemetry;
- provider-specific capability information.

Backends must advertise capabilities instead of relying on hard-coded assumptions in application code.

## 3. Backend-neutral public API

Public code should use Tensora concepts:

```text
Tensor
Device
DType
Model
Session
ExecutionPolicy
```

not provider concepts such as specific C++ tensor/session classes.

Provider identity may be exposed as diagnostic metadata without becoming the main application programming model.

## 4. Initial backend strategy

### CPU

Purpose:

- portable correctness baseline;
- test environment;
- fallback where explicitly permitted;
- small/portable workloads.

### ATen / LibTorch integration

Purpose:

- broad tensor functionality;
- autograd;
- training primitives;
- access to optimized CPU/CUDA execution.

Tensora should use it as an implementation provider, not expose its object model publicly.

### CUDA

Purpose:

- NVIDIA GPU tensor computation;
- GPU-backed training;
- high-performance model workloads.

Initial CUDA support should leverage mature libraries/providers. Custom kernels are introduced only for measured Tensora workloads where they provide clear benefit.

### ONNX Runtime

Purpose:

- portable inference;
- model/provider abstraction;
- deployment across supported targets.

Tensora remains responsible for consistent Dart-facing lifecycle, errors, model packaging, profiling, and Flutter integration.

## 5. Capability model

A backend capability response should eventually be able to describe:

```text
backend identity
backend version
available devices
supported dtypes
training availability
inference availability
supported operator/model groups
async support
stream support
memory telemetry support
zero-copy/import/export capabilities
```

Capabilities must reflect the actual installed/runtime environment, not only compile-time support.

## 6. Selection

Backend selection may be:

- explicit;
- model-recommended;
- execution-policy driven;
- `bestAvailable`.

Selection must remain explainable.

Diagnostics should be able to report:

```text
requested backend/device
selected backend/device
selection reason
fallbacks considered
fallbacks taken
```

## 7. Fallback

Fallback is a policy decision, not a hidden convenience.

Possible policies:

```text
strict — fail if requested backend cannot execute
allowCompatible — use a compatible alternative
```

If fallback creates a host/device transfer, dtype change, or material performance impact, that must be observable.

Silent semantic changes are prohibited.

## 8. Backend errors

Provider errors are translated into Tensora error categories.

Preserve useful diagnostic context without requiring callers to parse provider-specific error strings.

Examples:

```text
BackendUnavailable
UnsupportedOperation
UnsupportedDType
DeviceOutOfMemory
DeviceLost
InvalidModel
InternalBackendFailure
```

A backend error must not cross the C ABI as an uncaught C++ exception.

## 9. Tensor ownership

Backend-owned tensor state remains behind the native runtime.

The backend must define:

- allocation owner;
- release mechanism;
- view/alias semantics;
- device synchronization requirements;
- behavior if a backend shuts down before tensors are released.

The native core should present consistent handle semantics to Dart.

## 10. Streams and asynchronous devices

GPU backends should avoid global synchronization after every operation.

The architecture should support backend streams/events while preserving simple public semantics.

Where a Dart API requires completion before returning a result, synchronization may occur at the correct boundary rather than after each internal operator.

## 11. Model sessions

Inference backends should distinguish:

- loaded immutable model state;
- mutable execution/session state;
- per-request buffers;
- shared provider resources.

For stateful generation workloads, per-session state such as KV cache must not become accidentally global.

## 12. Provider-specific optimization

A backend may implement:

- fused operators;
- provider graph optimization;
- memory arenas;
- pinned host memory;
- asynchronous transfer;
- model compilation.

These optimizations must preserve documented Tensora semantics.

## 13. Custom kernels

Custom Tensora kernels are justified when profiling shows that:

- current provider overhead is a meaningful bottleneck;
- a fused path materially improves a flagship workload;
- provider functionality is unavailable;
- platform integration needs a specialized data path.

Custom code must add correctness tests and backend parity tests where applicable.

## 14. Mobile packaging

Large backend dependencies should be modular.

A Flutter application using a small inference backend should not automatically bundle full desktop training infrastructure.

Backend packaging should consider:

- binary size;
- architecture slices;
- platform store constraints;
- native dependency licensing/notices;
- initialization cost.

## 15. Backend registration

The native runtime should eventually support an internal registry resembling:

```text
backend id
factory
capabilities
priority
platform constraints
```

The public API should not depend on registry implementation details.

## 16. Testing a backend

A backend is not complete until it passes relevant:

- backend unit tests;
- ABI integration tests;
- numerical reference tests;
- parity tests;
- invalid-input tests;
- repeated lifecycle tests;
- memory stress;
- real-device execution where hardware-specific;
- benchmarks.

## 17. New backend checklist

Before accepting a new backend, document:

1. What unique platform/workload does it enable?
2. Which Tensora contracts does it implement?
3. Which dtypes/operators/models are supported?
4. How does it allocate/release resources?
5. How does it report capabilities?
6. How does it fail?
7. What are its threading rules?
8. What does it add to package size?
9. What license/distribution obligations exist?
10. How is it tested on real hardware?
11. Does it require a new public API? If yes, why is the existing abstraction insufficient?

## 18. Long-term direction

Tensora should gradually own more execution logic where doing so creates measurable product value:

```text
provider-backed eager execution
        ↓
Tensora graph/runtime
        ↓
backend partitioning
        ↓
Tensora compiler
        ↓
select custom kernels
```

Owning a layer is a means to improve correctness, portability, deployment, or performance—not an objective by itself.