# Tensora Architecture

This document defines the architectural direction and non-negotiable invariants of Tensora.

Tensora is intended to expose an idiomatic Dart API while executing performance-critical machine-learning workloads through optimized native backends. The architecture must make it possible to evolve backends and compiler technology without forcing application developers to rewrite stable code.

## 1. System layers

```text
Dart / Flutter application
          ↓
Public Tensora Dart API
          ↓
Execution and graph abstractions
          ↓
Dart FFI bridge
          ↓
Stable C ABI
          ↓
Tensora Native Core
          ↓
Backend interface
          ↓
CPU / ATen / CUDA / ONNX / platform providers
```

Each layer owns a distinct responsibility.

### Public Dart API

Owns:

- tensors and shapes;
- dtypes and devices;
- modules, parameters, optimizers;
- datasets and pipelines;
- model/session APIs;
- async/streaming interfaces;
- structured Dart errors;
- user-visible diagnostics.

It must not expose backend implementation types.

### Dart FFI bridge

Owns:

- ABI bindings;
- opaque native handles;
- conversion between C ABI status values and Dart exceptions;
- native resource wrappers;
- finalizer safety mechanisms;
- carefully bounded transfer of metadata and small values.

It must not become the location for numerical kernels.

### Stable C ABI

Owns the compatibility boundary between Dart and the native runtime.

Requirements:

- C-compatible types only;
- opaque handles instead of C++ object layout;
- explicit ABI version;
- structured status codes;
- validated pointer/length pairs;
- defined ownership for every returned resource;
- no exceptions escaping across the ABI.

### Native Core

Owns:

- handle registry/lifetime management;
- devices;
- streams/events where applicable;
- memory accounting;
- backend dispatch;
- backend-independent native validation;
- model/session runtime coordination;
- profiler event emission;
- conversion of provider-specific failures into Tensora status values.

### Backends

Own actual computation and provider integration.

Examples:

- CPU backend;
- ATen/LibTorch backend;
- CUDA-enabled execution through native libraries;
- ONNX Runtime provider;
- future Metal/NPU backends.

The backend interface is internal and may evolve faster than stable public APIs, but its responsibilities must remain explicit.

## 2. Dependency direction

Dependencies must flow toward the core.

```text
tensora_flutter
      ↓
tensora_edge / task-specific packages
      ↓
tensora_nn / tensora_model / tensora_data
      ↓
tensora
      ↓
native ABI/runtime
```

Prohibited examples:

- `tensora` importing Flutter;
- `tensora_nn` depending on `tensora_flutter`;
- native core containing Flutter widget or application lifecycle logic;
- stable API packages depending directly on a provider-specific C++ interface.

## 3. Tensor representation

A Dart `Tensor` is a typed owner/reference to native tensor state, not a container of large Dart scalar collections.

Conceptually:

```text
Dart Tensor
  ├── handle
  ├── cached immutable metadata where safe
  └── lifecycle state
        ↓
Native tensor object
  ├── dtype
  ├── shape
  ├── strides
  ├── layout
  ├── device
  ├── storage
  └── provider-specific representation
```

Large tensor payloads remain in native memory or device memory.

## 4. Ownership model

Every native object must have a defined owner and release path.

Required model:

1. native objects use deterministic native lifetime management;
2. a Dart wrapper owns or references an opaque handle;
3. explicit `dispose`/close semantics release expensive resources deterministically;
4. Dart finalizers exist only as a safety net;
5. duplicate wrappers must not cause double-free;
6. views must retain the storage they reference;
7. sessions must retain shared model weights safely;
8. runtime shutdown must not invalidate still-live objects without a defined error.

GPU memory must never depend solely on eventual garbage collection.

## 5. Tensor views and aliasing

Operations such as transpose, slice, and reshape may return views if the selected backend can represent them safely.

The semantics must define:

- whether an operation returns a view or an independent tensor;
- storage lifetime;
- mutability/aliasing rules;
- behavior under autograd;
- when contiguity is required;
- when a copy occurs.

Copies should be observable in profiling when material.

## 6. Device model

Public code uses Tensora devices rather than provider-specific devices.

Initial concepts:

```text
CPU
CUDA(index)
BestAvailable
```

Future concepts may include Metal or NPU devices.

A device capability query should answer facts such as:

- supported dtypes;
- available memory where reliable;
- supported execution features;
- provider identity;
- training/inference capability.

`bestAvailable` must be explainable: diagnostics should report what was selected and why.

## 7. Backend neutrality

A public API is acceptable only if it remains sensible when a current provider is replaced.

For example, the public type should be:

```text
Tensor
```

not:

```text
TorchTensor
```

Provider-specific escape hatches may exist in clearly separated expert packages, but ordinary APIs remain portable.

## 8. Error architecture

Errors cross the native boundary as structured statuses, never as uncaught C++ exceptions.

Native error information should include where possible:

- stable Tensora error code;
- operation;
- backend/provider;
- device;
- human-readable reason;
- structured diagnostic context.

The Dart layer maps those errors to typed exceptions.

User/model-input failures must not crash the process when recovery is technically possible.

## 9. Async and threading model

Public APIs must distinguish fast metadata operations from potentially blocking compute.

Heavy operations include:

- model loading;
- inference;
- training steps;
- compilation;
- model conversion;
- downloads in optional distribution packages.

Flutter-facing heavy work must not execute on the UI isolate by default.

The implementation may use native worker pools, backend-managed threads, isolates for appropriate CPU-side work, and device streams. The exact mechanism may vary, but public semantics must define cancellation, ordering, and object safety.

## 10. Isolate boundaries

Dart isolates do not imply that arbitrary native tensor handles are safe to move between isolates.

The architecture must explicitly define:

- which handles are isolate-local;
- which native resources may be shared safely;
- whether transfer requires serialization, a shared native registry, or a new wrapper;
- behavior after source-isolate termination;
- synchronization requirements.

Until proven safe, prefer keeping device tensors and mutable training state within one execution domain.

## 11. Execution modes

### Eager mode

Initial mode for research, debugging, and direct execution.

```dart
final y = x.matmul(w).add(b).gelu();
```

### Graph/compiled mode

Future mode for optimized execution.

```dart
final compiled = await Tensora.compile(model);
final y = await compiled(input);
```

The public model definition should not require a separate implementation for compiled mode unless a technical limitation is documented.

## 12. Internal graph and IR

A future Tensora IR should model:

- typed values;
- shape information;
- layouts;
- device constraints;
- operators and attributes;
- constants;
- side effects;
- backend constraints.

Compiler passes should only be introduced when they solve measured workloads. Likely passes include:

- constant folding;
- dead-node elimination;
- fusion;
- layout propagation;
- memory planning;
- backend partitioning.

The compiler is not a prerequisite for the first useful Tensora releases.

## 13. Training architecture

Dart owns training orchestration and high-level semantics.

Initially, autograd and many numerical operations may be supplied by a mature native backend.

```text
Dart model / optimizer
        ↓
Tensora API
        ↓
Native autograd/provider
        ↓
CPU/GPU kernels
```

The public API must not expose the provider's autograd object model.

## 14. Inference architecture

Portable inference should use a backend adapter capable of loading established model formats.

Model sessions should separate:

- immutable/shareable model state;
- mutable request/session state;
- provider execution state;
- application-visible result objects.

This is especially important for local language models where KV cache belongs to a session rather than globally mutable model weights.

## 15. Pipeline architecture

A `Pipeline` represents reusable preprocessing, model execution, and postprocessing.

```text
Input
 ↓
Preprocess
 ↓
Tensor/model
 ↓
Postprocess
 ↓
Typed result
```

The long-term objective is to reuse the same pipeline semantics across training/evaluation, server inference, and Flutter deployment to reduce training-serving skew.

## 16. Flutter architecture

`tensora_flutter` is an adapter package over core runtime APIs.

It may own:

- asset loading;
- camera/audio adapters;
- lifecycle-aware controllers;
- optional widgets;
- platform plugin integration.

It must not own tensor semantics or numerical kernels.

For camera workloads:

```text
Camera/native frame
      ↓
Frame adapter
      ↓
Native preprocessing
      ↓
Tensor
      ↓
Model
      ↓
Postprocessing
      ↓
Small typed Dart result
      ↓
Flutter overlay
```

Large frame data should not be copied through Dart merely for convenience when a safe native path exists.

## 17. Backpressure

Realtime sources require bounded queues.

Supported policy concepts may include:

- latest;
- drop oldest;
- block producer where appropriate.

For live camera inference, the default should generally prioritize freshness rather than processing stale accumulated frames.

## 18. `.tmodel`

`.tmodel` is a deployment bundle, not a new universal tensor/weight format.

It may package established artifacts such as:

- ONNX models;
- SafeTensors;
- tokenizers;
- labels;
- preprocessing/postprocessing descriptions;
- metadata;
- integrity data;
- embedded golden samples.

The format must be versioned, strictly validated, resource-limited, and incapable of executing arbitrary packaged code.

See `docs/MODEL_FORMAT.md`.

## 19. Security boundary

Treat all external model artifacts and metadata as untrusted.

The architecture must defend against:

- path traversal;
- malformed archives;
- integer overflow;
- shape/numel overflow;
- oversized allocations;
- malformed graphs;
- tokenizer/resource bombs;
- unsupported operations;
- invalid native handles;
- use-after-free;
- double-release;
- race conditions around shared state.

## 20. Performance architecture

Performance is measured at boundaries:

- pure backend baseline;
- Tensora dispatch overhead;
- Dart/FFI overhead;
- host/device transfers;
- synchronization;
- model loading;
- Flutter pipeline overhead.

Tensora should minimize its own overhead, but benchmark reports must clearly distinguish backend performance from Tensora-specific improvements.

## 21. Extensibility

Long-term extension points may include:

- backend providers;
- custom operators;
- preprocessing operations;
- model loaders;
- profiler exporters;
- execution policies.

Public extension interfaces should remain small and capability-driven.

## 22. Architectural non-goals

Do not turn Tensora core into:

- a vector database;
- a cloud hosting platform;
- a general application framework;
- a generic orchestration framework;
- a replacement for every established model format;
- a full CUDA library before real workloads require custom kernels.

## 23. Architecture review checklist

Before introducing a new subsystem, answer:

1. What responsibility does it own?
2. What does it depend on?
3. Who owns each resource?
4. Which operations can block?
5. What happens on cancellation?
6. What crosses the Dart/native boundary?
7. What crosses isolate/thread boundaries?
8. What is the failure model?
9. Does this expose a provider-specific concept publicly?
10. How is it tested?
11. How is performance measured?
12. How can the implementation be replaced without breaking callers?

If these answers are unclear, the subsystem is not ready to become part of the stable architecture.