# RFC 0001: Tensor Core ABI and Lifetime

- Status: Accepted
- Created: 2026-08-12
- Last updated: 2026-08-12

## Summary

Define the first stable native boundary needed to implement Tensora's Milestone 1 CPU Tensor vertical slice.

The design uses opaque non-pointer handles, typed status codes, deterministic reference-counted release, native immutable float32 tensors, an internal CPU backend dispatcher, and explicit host copies.

## Motivation

Milestone 1 must prove that an idiomatic Dart Tensor can own native memory and execute a real CPU matrix multiplication without exposing C++ implementation types or relying on Dart scalar storage.

The first ABI also needs enough lifetime and error discipline that future backends do not require a public Tensor API redesign.

## Goals

- real Dart-to-native tensor execution;
- stable C-compatible ABI versioning;
- safe stale-handle and duplicate-release behavior;
- deterministic cleanup with a finalizer safety net;
- structured error mapping;
- CPU backend isolation;
- concurrency that does not serialize numerical work behind a global runtime lock;
- measured FFI overhead.

## Non-goals

- CUDA or other accelerator devices;
- autograd/training;
- ONNX/model loading;
- Flutter integration;
- `.tmodel`;
- graph execution/compiler work;
- in-place tensor mutation;
- tensor views/aliasing;
- NumPy/PyTorch broadcasting compatibility.

## Detailed design

### Public Tensor semantics

Milestone 1 tensors are immutable values backed by native storage.

Elementwise operations require exactly equal shapes. `matmul` accepts two rank-2 float32 tensors with compatible inner dimensions. `sum` returns a scalar tensor represented by rank zero.

`reshape` and 2D `transpose` return new independent contiguous tensors in Milestone 1. View/aliasing semantics are intentionally deferred.

### ABI handles

`ts_tensor_t` is `uint64_t`.

The native registry issues monotonically increasing process-lifetime identifiers and does not reinterpret handles as pointers. Registry entries carry object type and reference count.

Released identifiers are not recycled. Therefore a stale handle cannot later alias an unrelated tensor during ordinary process lifetime.

### ABI version

`ts_abi_version()` returns `1`.

Dart bindings require ABI version 1 and fail with a typed runtime exception before creating tensors when another version is loaded.

### Error state

Functions return `ts_status_t`. A per-thread diagnostic string can be retrieved immediately after failure.

Provider/internal exceptions are caught at every exported ABI boundary and mapped into a stable Tensora status. No C++ exception crosses the C ABI.

### Shape contract

Rank zero represents a scalar and contains one element.

For rank greater than zero:

- rank must not exceed 32;
- every dimension must be positive;
- each dimension must fit `int64_t`;
- total element count must fit `uint64_t`;
- byte count must fit native `size_t`;
- overflow is rejected before allocation.

Zero-sized dimensions are unsupported in Milestone 1.

### Storage

CPU storage is RAII-owned native memory. Tensor payloads are never retained as ordinary Dart collections after import.

Host import and `toList()` are explicit copies. Tensor operations stay native.

### Backend dispatch

The C ABI delegates backend-independent validation and handle management to native core code. Actual computation is dispatched through a small backend interface.

CPU is the only registered implementation for Milestone 1. Unsupported devices fail explicitly.

## API / ABI impact

This RFC introduces ABI v1. There is no prior executable ABI to migrate.

The public Dart API exposes Tensora concepts only and contains no C pointer, handle, or provider object.

## Ownership and lifetime

Every ABI function that creates a tensor returns one owned native reference.

`ts_tensor_retain` adds one reference. `ts_tensor_release` removes one. The native object is destroyed when the final reference is released.

Each Dart Tensor wrapper owns one reference and exposes synchronous `dispose()`.

The finalizer is only a fallback for wrappers that were not disposed explicitly.

Operation implementation acquires a native shared reference during handle lookup. Concurrent release cannot free storage already acquired by an in-flight operation.

## Concurrency

The handle registry is mutex-protected for lookup and reference-count mutation only.

Tensor computation occurs after the registry lock is released.

Milestone 1 tensors are immutable, so separate tensors and repeated read-only operations using the same tensor may execute concurrently.

Cross-isolate Tensor transfer is not supported in Milestone 1.

## Error model

Stable statuses:

- `TS_OK`
- `TS_INVALID_ARGUMENT`
- `TS_INVALID_SHAPE`
- `TS_OUT_OF_MEMORY`
- `TS_UNSUPPORTED`
- `TS_INVALID_HANDLE`
- `TS_INTERNAL_ERROR`

Dart maps them to typed `TensoraException` subclasses.

## Compatibility

ABI v1 is additive from a repository that previously had no executable ABI.

Future ABI changes must preserve version checking and must not rely on C++ layout compatibility.

## Security considerations

The ABI treats all handles, pointers, lengths, ranks, dimensions, and capacities as untrusted.

Tests cover null pointers, invalid/stale handles, duplicate release, rank limits, negative/zero dimensions, multiplication overflow, incompatible operation shapes, and repeated allocation/release.

## Performance considerations

Tensor operations use one FFI call per tensor operation.

No scalar-level FFI loop is used for numerical work.

The benchmark suite measures both a minimal FFI call and actual tensor operations so fixed bridge cost is distinguishable from native compute.

## Testing strategy

Native tests exercise the C ABI directly and include sanitizer-compatible stress and concurrency cases.

Dart integration tests cross the real FFI boundary for every supported public operation.

Matmul and other numerical operations are checked against analytically known values and deterministic generated cases.

## Alternatives considered

### Native pointer handles

Rejected because arbitrary/stale integers could become unsafe pointer dereferences and object-type validation would be weak.

### Reusable slot/generation handles

Viable, but monotonic non-reused identifiers are simpler for the first runtime and eliminate stale-handle aliasing without packing generation bits.

### Dart-owned numerical arrays

Rejected because it would invalidate the native-storage objective and make future accelerator backends expensive to introduce.

### Full broadcasting in Milestone 1

Rejected because partial NumPy/PyTorch compatibility would create semantic debt. Exact-shape elementwise semantics are explicit and complete.

### View-based reshape/transpose

Deferred until aliasing, contiguity, mutation, and future autograd semantics can be designed together.

## Migration plan

No migration is required because this is the first executable Tensor/ABI implementation.

## Rollout plan

Land the complete Milestone 1 vertical slice only after exact-head native, Dart/FFI, sanitizer, stress, example, and benchmark-smoke validation.

## Open questions

None that block Milestone 1. Future view semantics, accelerator devices, isolate sharing, and asynchronous execution require separate design work before becoming supported contracts.
