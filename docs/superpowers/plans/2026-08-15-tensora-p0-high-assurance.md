# Tensora P0 High-Assurance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Converge the newest Tensora training engine with the strongest dtype/ABI and coverage guarantees, then make the exact P0 head green across the hosted Linux/macOS/Windows matrix without weakening existing autograd, ownership, or accelerator contracts.

**Architecture:** Start from `feature/training-engine-v1` and port only proven semantic/assurance pieces from `feature/dtype-abi-assurance`. Keep the native storage implementation float32-only during P0 while publishing stable dtype semantics/ABI codes. Use test-first changes, exact-head GitHub Actions validation, and hard coverage gates; later P1 storage work is explicitly out of scope.

**Tech Stack:** Dart 3.7+, C++20/C11 ABI, CMake, LibTorch 2.13 CPU, ONNX Runtime 1.26, GitHub Actions, gcovr 8.3, ASan/UBSan/TSan.

## Global Constraints

- `main` must not be modified during P0.
- `feature/training-engine-v1` behavior for autonomous CPU autograd, optimizers, checkpoints, views, aliases, and ONNX must not regress.
- Dart merged production coverage must be >= 99.9%; the threshold must not be lowered.
- Native owned-code merged line coverage must be 100.0% for the P0 owned-code set.
- No new broad coverage exclusions are permitted; the existing validated Win32 loader exclusion budget may not grow.
- Non-float32 dtype metadata is semantic/ABI-only in P0; native allocation must reject unsupported storage explicitly in Debug and Release.
- Explicit accelerator requests must never silently fall back to CPU.
- Every ABI output handle must be cleared on failure before returning a non-success status.

---

### Task 1: Restore stable dtype semantics without claiming storage support

**Files:**
- Modify: `packages/tensora/lib/src/dtype/dtype.dart`
- Modify: `packages/tensora/lib/src/tensor/tensor.dart`
- Modify: `native/include/tensora.h`
- Create: `packages/tensora/test/dtype_semantics_test.dart`
- Modify: `native/tests/c_abi_test.c`

**Interfaces:**
- Produces: `DType.byteWidth`, `DType.nativeCode`, `DType.isFloatingPoint`, `DType.isInteger`, `DType.isBoolean`, `DType.supportsArithmetic`, `DType.supportsGradients`, `DType.reductionAccumulator`, `DType.nativeStorageImplemented`, `DType.promote(DType,DType)`.
- Produces stable ABI constants `TS_DTYPE_FLOAT32=1` through `TS_DTYPE_BOOL=10`.
- Preserves native tensor creation as float32-only.

- [ ] **Step 1: Add failing Dart semantic tests**

Add `dtype_semantics_test.dart` that asserts all ten dtype metadata rows, promotion symmetry, reduction accumulators, stable ABI codes, and explicit rejection of non-float32 tensor creation.

- [ ] **Step 2: Add failing C ABI constant assertions**

Extend `native/tests/c_abi_test.c` with compile/runtime assertions that the public dtype constants are exactly 1..10 and ABI version remains 4.

- [ ] **Step 3: Implement public dtype semantics**

Use the proven semantic table from `feature/dtype-abi-assurance`:

```dart
enum DType {
  float16(byteWidth: 2, nativeCode: 2),
  bfloat16(byteWidth: 2, nativeCode: 3),
  float32(byteWidth: 4, nativeCode: 1),
  float64(byteWidth: 8, nativeCode: 4),
  int8(byteWidth: 1, nativeCode: 5),
  uint8(byteWidth: 1, nativeCode: 6),
  int16(byteWidth: 2, nativeCode: 7),
  int32(byteWidth: 4, nativeCode: 8),
  int64(byteWidth: 8, nativeCode: 9),
  boolean(byteWidth: 1, nativeCode: 10);

  const DType({required this.byteWidth, required this.nativeCode});
  final int byteWidth;
  final int nativeCode;
  bool get nativeStorageImplemented => this == float32;
}
```

- [ ] **Step 4: Make unsupported creation fail in Release**

Replace assertion-only creation validation with an explicit structured error:

```dart
static void _validateCreation({
  required DType dtype,
  required String operation,
}) {
  if (!dtype.nativeStorageImplemented) {
    throw UnsupportedOperationException(
      'Tensor creation currently supports only DType.float32 native storage.',
      operation: 'tensor.$operation',
    );
  }
}
```

Call with `operation: 'fromList'` and `operation: 'full'`.

- [ ] **Step 5: Publish stable C ABI dtype constants**

Keep `TS_ABI_VERSION 4u` and expand only the enum constants; do not add unsupported creation entry points.

- [ ] **Step 6: Push exact-head validation**

Expected required workflows: Dart FFI CI, Native CI, CPU Training Engine CI, Tensor Alias CI, C ABI fuzz, sanitizers, workspace CI.

- [ ] **Step 7: Commit**

Commit message: `feat: restore stable dtype semantics`

---

### Task 2: Restore 99.9% Dart high-assurance gate and fault-runtime coverage

**Files:**
- Modify: `.github/workflows/high-assurance-ci.yml`
- Create: `native/tests/ffi_contract_fixture.c`
- Create: `packages/tensora/integration_test/native_abi_fault_integration_test.dart`
- Create: `packages/tensora/integration_test/native_runtime_fault_integration_test.dart`
- Create: `packages/tensora/integration_test/native_training_fault_integration_test.dart`
- Create: `packages/tensora/integration_test/native_inference_fault_integration_test.dart`
- Create: `packages/tensora/integration_test/support/fault_runtime_control.dart`

**Interfaces:**
- Produces a deterministic fake native runtime controlled through test-only fault modes.
- High Assurance discovers every `lib/src/**/*.dart` production source automatically.

- [ ] **Step 1: Port fault-runtime tests before changing the threshold**

Port the proven deterministic C fault runtime and Dart integration tests from `feature/dtype-abi-assurance`, retaining current ABI v4 symbols used by the training-engine line.

- [ ] **Step 2: Verify failure contracts covered by the fixture**

Cover at minimum ABI mismatch, unknown status, missing diagnostics, corrupted tensor metadata, null-success handles, partial output adoption, training output failures, inference output failures, UTF-8 capacity races, and provider/session inconsistencies.

- [ ] **Step 3: Raise merged production coverage threshold**

Change the workflow job name and final threshold from 90% to 99.9%.

- [ ] **Step 4: Require all production Dart sources**

Use:

```python
required_sources = {
    path.as_posix()
    for path in Path('lib/src').rglob('*.dart')
}
```

Fail when any source is missing from merged LCOV.

- [ ] **Step 5: Freeze coverage exclusion budget**

Allow ignored lines only in `lib/src/native/native_runtime.dart` for the validated Win32 loader and fail if the existing budget exceeds 48 lines.

- [ ] **Step 6: Push exact-head High Assurance validation**

Expected: merged production Dart coverage >= 99.900000%, no missing production sources, no exclusion-budget growth.

- [ ] **Step 7: Commit**

Commit message: `test: restore high-assurance ABI fault coverage`

---

### Task 3: Reconcile native library resolution and finalizer boundary hardening

**Files:**
- Compare/port as applicable: `packages/tensora/lib/src/native/native_library_resolver.dart`
- Compare/port as applicable: `packages/tensora/lib/src/native/finalizer_callbacks.dart`
- Modify: `packages/tensora/lib/src/native/native_runtime.dart`
- Modify: `packages/tensora/lib/src/native/native_training_runtime.dart`
- Modify: `packages/tensora/lib/src/inference/onnx.dart`
- Modify: `packages/tensora/lib/src/tensor/tensor.dart`
- Modify: `packages/tensora/lib/src/training/training.dart`
- Create/modify focused tests under `packages/tensora/test/`

**Interfaces:**
- Central native library resolution returns one deterministic platform path/override decision.
- Finalizer callbacks contain no Dart-object captures and release by opaque handle only.

- [ ] **Step 1: Add tests for Linux/macOS/Windows library naming, explicit override, blank override, and unsupported platform behavior.**
- [ ] **Step 2: Add tests for deterministic tensor/module/optimizer/session finalizer callbacks.**
- [ ] **Step 3: Port only the resolver/finalizer pieces that do not conflict with the newer training-engine adoption helpers.**
- [ ] **Step 4: Run/push Dart unit + FFI + High Assurance matrices.**
- [ ] **Step 5: Commit** with `refactor: centralize native runtime boundaries`.

---

### Task 4: Finish tensor-view, alias, and autograd mutation contracts

**Files:**
- Modify as required: `native/src/tensor/tensor.h`
- Modify as required: `native/src/tensor/tensor.cc`
- Modify as required: `native/src/autograd/autograd.h`
- Modify as required: `native/src/backends/cpu/cpu_backend.cc`
- Modify tests: `native/tests/tensor_view_test.cc`
- Modify tests: `native/tests/tensor_alias_state_test.cc`
- Modify tests: `native/tests/autograd_finite_difference_test.cc`

**Interfaces:**
- Views share backing storage when layout permits.
- All aliases share one mutation epoch.
- Backward must fail before publishing any leaf gradient when a saved alias is stale.

- [ ] **Step 1: Add/confirm failing tests for contiguous zero-copy reshape and rank-2 transpose aliasing.**
- [ ] **Step 2: Add/confirm logical-order tests for non-contiguous elementwise/reduction/matmul.**
- [ ] **Step 3: Add/confirm materialization-only-when-required reshape tests.**
- [ ] **Step 4: Add/confirm finite-difference gradients through views.**
- [ ] **Step 5: Add/confirm stale saved-alias rollback test requiring zero published leaf gradients.**
- [ ] **Step 6: Make the minimal native changes required by failures.**
- [ ] **Step 7: Push Native CI, Tensor Alias CI, CPU Training Engine CI, sanitizer and TSan validation.**
- [ ] **Step 8: Commit** with `fix: complete tensor alias autograd contracts`.

---

### Task 5: Drive native owned-code coverage to 100% without exclusions

**Files:**
- Modify focused tests under `native/tests/`
- Modify only if coverage tooling is incorrect: `.github/workflows/native-coverage.yml`

**Interfaces:**
- `gcovr` merges core-only, LibTorch CPU, and ONNX CPU traces over `native/src/` and enforces `--fail-under-line 100`.

- [ ] **Step 1: Run Native Coverage on the exact branch head and download the generated JSON/XML/HTML artifact.**
- [ ] **Step 2: Enumerate each uncovered owned source line by file and contract.**
- [ ] **Step 3: Add a focused test that reaches each legitimate error/success path; do not add source exclusions to satisfy the gate.**
- [ ] **Step 4: Repeat until merged line coverage reports 100.0000%.**
- [ ] **Step 5: Confirm no regression in sanitizer/fuzz/TSan suites.**
- [ ] **Step 6: Commit** with `test: close native owned-code coverage`.

---

### Task 6: Exact-head cross-platform and compatibility qualification

**Files:**
- Modify as failures require: `.github/workflows/native-ci.yml`, `.github/workflows/dart-ci.yml`, `.github/workflows/workspace-ci.yml`, platform loader tests, and compatibility docs.
- Modify: `docs/COMPATIBILITY.md`
- Modify: `docs/engineering/TRAINING_ENGINE_V1_STATUS.md`

**Interfaces:**
- Hosted Linux/macOS/Windows claims are source-build/runtime claims only.
- CUDA/XPU/HIP remain separate physical qualification gates.

- [ ] **Step 1: Trigger all P0 hosted workflows on one exact SHA.**
- [ ] **Step 2: Fix platform-specific build/runtime failures without platform-specific semantic fallback.**
- [ ] **Step 3: Verify Windows sidecar/native-library resolution and macOS/Linux default discovery.**
- [ ] **Step 4: Verify Debug/Release and sanitizers where supported.**
- [ ] **Step 5: Update compatibility/status docs to match only proven evidence.**
- [ ] **Step 6: Commit** with `docs: qualify unified P0 runtime`.

---

### Task 7: P0 release gate and integration handoff

**Files:**
- Modify: `docs/engineering/TRAINING_ENGINE_V1_STATUS.md`
- Create: `docs/engineering/P0_UNIFIED_RELEASE_EVIDENCE.md`

**Interfaces:**
- Produces one exact commit SHA that satisfies all P0 gates.

- [ ] **Step 1: Record exact SHA and every required workflow conclusion.**
- [ ] **Step 2: Record Dart merged coverage numerator/denominator and native merged coverage numerator/denominator.**
- [ ] **Step 3: Record sanitizer/fuzz/TSan/lifecycle results and supported platform matrix.**
- [ ] **Step 4: Explicitly list unqualified hardware paths so no GPU support is inferred.**
- [ ] **Step 5: Open/update a draft PR targeting the stability line; do not merge to `main`.**
- [ ] **Step 6: Only after the exact-head matrix is green, mark P0 ready for review and begin a separate P1 design/plan.**
