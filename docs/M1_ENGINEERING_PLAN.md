# Milestone 1 Engineering Implementation Plan

Milestone 1 establishes the first production Tensor Core / CPU vertical slice. It intentionally implements one narrow, complete path rather than reserving future framework surface.

## 1. Repository structure

The milestone introduces only:

- `packages/tensora` for the public Dart API and private FFI bridge;
- `native` for the stable C ABI, runtime primitives, handle registry, tensor representation, and CPU backend;
- `examples/tensor_basics` for a runnable end-to-end example;
- `benchmarks` plus the Dart package benchmark for reproducible baseline measurements;
- CI configuration for native, Dart/FFI, sanitizer, stress, release, benchmark-smoke, and example gates.

No CUDA, neural-network, Flutter, ONNX, model-format, compiler, or transformer package is created.

The older roadmap mentions `slice` in its broad Milestone 1 direction. The approved Milestone 1 vertical-slice task and its exact Definition of Done do not include `slice`; this implementation therefore defers it rather than expanding the stable ABI beyond the current acceptance contract.

## 2. Dart public API

Milestone 1 exposes:

- `Tensor`
- `Shape`
- `DType.float32`
- `Device.cpu`
- typed `TensoraException` subclasses

`Tensor` supports `fromList`, `zeros`, `ones`, `full`, `reshape`, 2D `transpose`, equal-shape `add`, equal-shape `multiply`, `sum`, 2D `matmul`, `toList`, and deterministic `dispose`.

`Shape` is defensively copied and validated. A non-const public constructor is deliberately used so immutability and eager validation are guaranteed even when callers supply mutable lists.

## 3. Internal Dart/native boundary

The Dart package keeps all `Pointer`, native handle, dynamic-library, ABI-code, and allocation details under `lib/src/native`.

Tensor payloads cross the FFI boundary only for explicit host import/export. Tensor-to-tensor operations cross once per operation and keep numerical data in native memory.

The native library path can be supplied with `TENSORA_NATIVE_LIBRARY`. Otherwise the bridge attempts only the canonical platform library name.

## 4. C ABI

ABI version 1 uses C-compatible primitives and an opaque `uint64_t` tensor handle.

The ABI provides:

- version and thread-local error diagnostics;
- float32 tensor creation/fill;
- metadata queries;
- reshape and 2D transpose;
- add, multiply, sum, and 2D matmul;
- explicit host copy;
- retain/release;
- live-object/live-storage diagnostics for lifecycle validation;
- a minimal no-op call for FFI-overhead measurement.

Every exported function validates output pointers and user-controlled pointer/length or handle inputs. No C++ exception may escape the boundary.

## 5. Native tensor representation

A native Tensor contains:

- `DType`;
- `Device`;
- validated shape;
- contiguous row-major strides;
- element count;
- byte size through storage;
- RAII-owned CPU storage.

Milestone 1 reshape and transpose return independent contiguous tensors. This avoids exposing aliasing/view semantics before those semantics are explicitly introduced.

## 6. Memory ownership

A newly created operation result returns a handle with one owning reference.

The process-wide handle registry stores typed entries and never reuses handle identifiers during process lifetime. Retain/release is reference-counted and mutex-protected.

Dart wrappers own one native reference:

- `dispose()` releases it deterministically;
- double dispose is a no-op;
- every public operation checks disposed state before FFI;
- a Dart finalizer releases only as a safety net;
- failure while adopting a newly returned handle releases that handle before rethrowing.

Native tensor storage uses RAII and allocation accounting.

## 7. Error propagation

Native functions return stable status codes:

- `OK`
- `INVALID_ARGUMENT`
- `INVALID_SHAPE`
- `OUT_OF_MEMORY`
- `UNSUPPORTED`
- `INVALID_HANDLE`
- `INTERNAL_ERROR`

A thread-local diagnostic string carries operation-specific context.

The Dart bridge immediately converts a non-OK status into a typed exception. Ordinary callers never receive unexplained integer status values.

## 8. CPU backend interface

Tensor operations are routed through a small backend interface and dispatcher. CPU is the only implementation and the only public device in Milestone 1.

No plugin framework or fake future backend is introduced.

## 9. Threading assumptions

Native Tensor values are immutable after creation in Milestone 1.

- separate tensors may be operated on concurrently;
- the same tensor may participate in concurrent read-only operations;
- handle lookup/retain/release is thread-safe;
- an operation that obtains its native shared reference before a concurrent release may complete safely;
- computation does not run under the global handle-registry mutex;
- Dart Tensor wrappers are not transferable across isolates in Milestone 1.

## 10. Tests

Native tests cover direct C ABI and internal lifecycle behavior:

- ABI version;
- allocation/release/retain;
- invalid, stale, and double-released handles;
- null pointers;
- negative/zero/overflowing dimensions;
- creation and metadata;
- reshape/transpose;
- add/multiply/sum/matmul reference values;
- incompatible shapes;
- concurrent read-only execution;
- 10,000+ create/execute/dispose cycles;
- allocation accounting returning to baseline.

Dart tests cover:

- Shape immutability, validation, equality, hashing, and overflow;
- DType and Device;
- public Tensor creation/metadata/lifecycle;
- factory operations;
- reshape/transpose;
- add/multiply/sum/matmul;
- invalid input and native exception mapping;
- deterministic property-style invariants;
- repeated lifecycle use through the real FFI path.

## 11. Benchmarks

The Dart benchmark reports:

- environment and build configuration;
- minimal FFI no-op overhead;
- tensor creation;
- elementwise add;
- matmul for selected matrix sizes;
- native-to-Dart extraction;
- warmup, iterations, median, and p95.

A native benchmark supplies a CPU-only baseline for matmul so FFI/runtime overhead is not conflated with the kernel baseline.

Performance measurements are evidence, not support claims.

## 12. Milestone 1 Definition of Done

Milestone 1 is complete only when the exact PR head passes:

- Dart formatting and static analysis;
- Dart unit/integration tests through the real native library;
- native debug configure/build/tests;
- native release configure/build/tests;
- ASan/UBSan validation where supported;
- lifecycle stress validation with no unbounded Tensora allocation growth;
- benchmark smoke execution;
- tensor basics example execution;
- documentation/support-matrix updates that claim only CPU + float32 functionality actually validated.

The completed path must be real:

`Dart Tensor API -> Dart FFI -> Tensora C ABI v1 -> native Tensor -> CPU backend -> validated result`.
