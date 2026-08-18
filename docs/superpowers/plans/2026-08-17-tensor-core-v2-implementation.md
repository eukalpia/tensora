# Tensor Core V2 Implementation Plan

> Execute this plan slice-by-slice. Every task uses test-first changes and preserves all P0/NN V2 gates.

## Phase boundary

This plan implements P1 from `docs/superpowers/specs/2026-08-17-tensor-core-v2-design.md`. P1A must be fully green before P1B begins; P1B before P1C; P1C before P1D.

## P1A — dtype storage, typed host I/O and casting

### Task 1: Lock ABI v6 contracts with failing tests

Files:
- modify `native/include/tensora.h`
- modify `native/tests/c_abi_test.c`
- add `native/tests/dtype_storage_test.cc`
- modify `native/CMakeLists.txt`

Steps:
1. Add compile-time/runtime tests expecting ABI version 6.
2. Add tests for generic host import/export/full/cast symbols.
3. Add malformed dtype, byte-size, null-pointer, bool canonicality, output-clearing and overflow cases.
4. Add golden round-trips for every dtype.
5. Run core native tests and confirm failure before implementation.

### Task 2: Centralize native dtype semantics

Files:
- add `native/src/tensor/dtype.h`
- add `native/src/tensor/dtype.cc`
- modify `native/src/tensor/tensor.h`
- modify `native/CMakeLists.txt`

Steps:
1. Expand native `DType` to all stable ABI codes.
2. Implement code validation, byte width, categories, accumulator dtype and promotion.
3. Implement half and bfloat conversion helpers with round-to-nearest-even.
4. Add exhaustive ordered-pair promotion tests and conversion edge vectors.
5. Keep helpers allocation-free and exception-contained.

### Task 3: Replace float-only CPU storage with dtype-aware storage

Files:
- modify `native/src/memory/tensor_storage.h`
- modify `native/src/memory/cpu_storage.h`
- modify `native/src/memory/cpu_storage.cc`
- modify storage/internal tests

Steps:
1. Introduce canonical raw host copy in `TensorStorage`.
2. Store a closed typed-vector variant in `CpuStorage`.
3. Preserve checked float32 accessors for existing autograd/NN V2 code.
4. Add generic typed construction, full, raw copy, element read/write and cast helpers.
5. Canonicalize bool bytes and track actual typed live bytes.
6. Verify constructors are transactional under injected allocation failure.

### Task 4: Make Tensor metadata and views dtype-correct

Files:
- modify `native/src/tensor/tensor.h`
- modify `native/src/tensor/tensor.cc`
- modify `native/src/tensor/shape.cc` if byte-size ownership must move
- modify `native/tests/tensor_view_test.cc`

Steps:
1. Derive tensor byte size from dtype-aware storage rather than float32 shape assumptions.
2. Add generic raw host copy respecting strides and storage offsets in elements.
3. Keep `CopyToHostF32` as a strict compatibility wrapper.
4. Qualify reshape/transpose views for every CPU dtype.
5. Add non-contiguous typed view round-trip and bounds tests.

### Task 5: Implement ABI v6 generic creation/export/full/cast

Files:
- modify `native/src/c_api.cc`
- modify `native/include/tensora.h`
- modify `native/src/backends/backend.h`
- modify `native/src/backends/cpu/cpu_backend.h`
- modify `native/src/backends/cpu/cpu_backend.cc`

Steps:
1. Implement generic import/export with exact byte validation.
2. Implement generic full using one canonical scalar value.
3. Implement CPU cast with fail-before-publish semantics.
4. Reimplement f32 APIs as compatibility wrappers.
5. Make reshape/transpose dtype-generic; keep unsupported numerical kernels explicit.
6. Verify every output handle/count is cleared on failure.

### Task 6: Wire Dart FFI and public Tensor API

Files:
- modify `packages/tensora/lib/src/native/native_bindings.dart`
- modify `packages/tensora/lib/src/native/native_runtime.dart`
- add `packages/tensora/lib/src/dtype/half_codec.dart`
- modify `packages/tensora/lib/src/dtype/dtype.dart`
- modify `packages/tensora/lib/src/tensor/tensor.dart`
- modify Dart tests

Steps:
1. Require ABI v6.
2. Add generic raw import/export/full/cast bindings.
3. Encode/decode all Dart dtype representations deterministically.
4. Make `Tensor.fromList`, `zeros`, `ones`, `full`, `toList`, `toTypedData`, and `cast` dtype-aware.
5. Preserve deterministic native-handle adoption/release on all errors.
6. Add Dart integration tests for every dtype and cast boundary.

### Task 7: Requalify training/inference boundaries

Files:
- modify `native/src/autograd/autograd.h`
- modify training bridge/storage files only as required
- modify ONNX bridge tests only as required

Steps:
1. Keep native autograd explicitly float32-only until additional backward kernels are qualified.
2. Reject integer/bool `requiresGrad` and non-float32 backward with structured errors.
3. Preserve float32 NN V2 behavior bit-for-bit.
4. Explicitly reject unsupported non-float32 accelerator transfers instead of fallback.
5. Run training, MPS, ONNX and CoreML acceptance suites.

### Task 8: Close coverage and platform matrix

Steps:
1. Merge core, LibTorch and ONNX coverage traces.
2. Restore native owned-code lines/functions to 100%.
3. Run Dart merged production coverage >=99.9%.
4. Run ASan/UBSan, TSan, fuzz and lifecycle soaks.
5. Run Linux/macOS/Windows exact-SHA matrix.
6. Update compatibility and API documentation.

## P1B — promotion, broadcasting and scalar arithmetic

### Task 9: Broadcast planner

Files:
- add `native/src/tensor/broadcast.h/.cc`
- add `native/tests/broadcast_test.cc`

Implement checked trailing-dimension planning, output shape, aligned strides and scalar rank-zero semantics. Prove incompatible-shape and overflow paths.

### Task 10: Promoted add/multiply

Modify backend interface and CPU kernels to cast inputs to the promoted output dtype, iterate broadcast plans, reject bool arithmetic, and preserve float32 autograd.

### Task 11: Dart scalar/operator API

Add `+`, `*`, scalar overload handling, explicit temporary ownership, and end-to-end promotion tests.

### Task 12: Broadcast-aware backward

Reduce float32 broadcast gradients to parent shapes and preserve alias-version validation. Add finite-difference and repeated-parent tests.

### Task 13: P1B qualification

Restore hard coverage thresholds and full exact-SHA platform matrix before continuing.

## P1C — axis reductions

### Task 14: Reduction planner and ABI

Add normalized axis lists, duplicate rejection, keep-dimension output planning, accumulator dtype and checked indexing.

### Task 15: Sum/mean/min/max kernels

Implement deterministic CPU kernels across approved numerical dtypes, explicit empty-input behavior and output dtype selection.

### Task 16: Supported backward rules

Implement float32 sum/mean backward first, then qualify additional floating paths only with finite differences.

### Task 17: Dart reduction API and qualification

Add named arguments, negative-axis normalization, integration tests, coverage and full matrix.

## P1D — indexing and composition

### Task 18: Slice/index view planner

Add half-open positive-stride slicing, index-axis removal, checked offsets and view identity/version tests.

### Task 19: Concat and stack

Validate dtype/device/rank/shape, allocate transactionally, and copy logical element order for all CPU dtypes.

### Task 20: Gather and scatter

Require int64 indices, bounds-check before publication, implement deterministic last-write-wins scatter, and preserve source immutability.

### Task 21: Dart indexing API

Add `Slice`, `index`, `concat`, `stack`, `gather`, `scatter` with Flutter-style named arguments and ownership-safe temporary handling.

### Task 22: P1 final qualification

Run the full acceptance gate from the design document on one exact SHA, update roadmap/compatibility docs, and promote only after all checks are green.
