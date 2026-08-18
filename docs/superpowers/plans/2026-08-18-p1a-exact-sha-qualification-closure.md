# P1A Exact-SHA Qualification Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote Tensor Core V2 P1A only after one owner-authored exact revision proves typed CPU execution, ABI v6, Dart compatibility, cross-platform safety, and every inherited P0/NN V2 gate without reducing any contract.

**Architecture:** P1A keeps the existing eager Tensor runtime and replaces float32-only CPU storage with exact typed storage for the approved ten dtypes. Typed Dart APIs cross ABI v6 into Tensora-owned CPU storage; reshape, transpose, casting, and host materialization remain native and deterministic, while autograd and accelerator training remain explicitly float32-only. This plan closes qualification only; broadcasting, promoted arithmetic, axis reductions, indexing, compiler work, CUDA ownership, and distributed execution remain separate vertical slices.

**Tech Stack:** Dart 3.7 and current stable, `dart:ffi`, C11 public ABI, C++20 native runtime, CMake/CTest, Clang/GCC/MSVC, ASan/UBSan, ThreadSanitizer, GitHub Actions, optional LibTorch and ONNX Runtime providers.

**Spec:** `docs/superpowers/specs/2026-08-17-tensor-core-v2-design.md`; strategic parent: `docs/superpowers/specs/2026-08-18-core-superiority-compatibility-design.md`

## Global Constraints

- P1A supports real CPU storage for `float16`, `bfloat16`, `float32`, `float64`, `int8`, `uint8`, `int16`, `int32`, `int64`, and `bool`.
- ABI version is exactly `6`; all ABI v5 float32 entry points remain available.
- Typed host import/export uses exact byte counts and canonical representations; no implicit Dart collection round-trip is allowed during tensor-to-tensor operations.
- Integer and boolean tensors never acquire gradients. P1A backward execution remains explicitly float32-only.
- No device, dtype, provider, or compatibility fallback may be silent.
- Failure outputs and newly-created handles are cleared or rolled back before returning an error.
- Existing P0 and NN V2 behavior, lifecycle semantics, parameter identity, training, MPS, ONNX, and CoreML gates must remain unchanged.
- Native owned-code line and function coverage remain `100%`; existing Dart high-assurance thresholds may not be lowered.
- No coverage exclusion, warning suppression, test deletion, assertion weakening, or status-contract rewrite may be used to obtain green CI.
- Windows, Linux, and macOS warnings remain errors for Tensora-owned code.
- Temporary self-modifying workflows are forbidden in the final candidate. Commits used for exact-SHA qualification must be authored through the authenticated repository owner path so GitHub does not classify the matrix as `action_required`.
- P1B does not start until the complete P1A exact-SHA matrix is green.

---

### Task 1: Establish an owner-authored immutable qualification candidate

**Files:**
- Create: `docs/superpowers/plans/2026-08-18-p1a-exact-sha-qualification-closure.md`
- Verify absent: `.github/workflows/p1-status-contract-repair.yml`
- Verify absent: `.github/workflows/p1-status-contract-cleanup.yml`
- Verify absent: any workflow that commits or pushes production changes after tests

**Interfaces:**
- Consumes: current `feature/p1-tensor-core-v2` head and the approved P1A design.
- Produces: an owner-authored branch head that can trigger the normal pull-request matrix without `action_required`.

- [ ] **Step 1: Verify no temporary self-modifying workflow remains**

```bash
set -euo pipefail
for path in \
  .github/workflows/p1-status-contract-repair.yml \
  .github/workflows/p1-status-contract-cleanup.yml; do
  test ! -e "$path"
done
if grep -R --line-number --include='*.yml' --include='*.yaml' \
  -E 'git push|contents:[[:space:]]*write' .github/workflows; then
  echo 'unexpected self-modifying workflow' >&2
  exit 1
fi
```

Expected: exit code `0`; no production workflow writes to the branch.

- [ ] **Step 2: Verify the branch is based directly on merged P0/NN V2 main**

```bash
git merge-base --is-ancestor b802aceb4128a3f80709e47e157088abd21c6539 HEAD
git rev-list --left-right --count b802aceb4128a3f80709e47e157088abd21c6539...HEAD
```

Expected: the merge-base command succeeds; the branch is not behind the recorded NN V2 base.

- [ ] **Step 3: Commit the qualification plan through the authenticated owner path**

```bash
git add docs/superpowers/plans/2026-08-18-p1a-exact-sha-qualification-closure.md
git commit -m 'docs(p1): add exact-SHA qualification closure plan'
```

Expected: the resulting GitHub Actions actor is the authenticated repository owner rather than `github-actions[bot]`.

---

### Task 2: Reprove native typed-storage and legacy status contracts

**Files:**
- Verify: `native/src/memory/cpu_storage.cc`
- Verify: `native/src/backends/cpu/cpu_backend.cc`
- Verify: `native/src/tensor/tensor.cc`
- Verify: `native/src/tensor/dtype.h`
- Test: `native/tests/dtype_storage_test.cc`
- Test: `native/tests/training_core_internal_test.cc`
- Test: `native/tests/tensor_view_test.cc`
- Test: `native/tests/tensor_core_test.cc`

**Interfaces:**
- Consumes: `CpuStorage::{FromRaw,Full,Cast}`, `Tensor::{CopyToHostRaw,CopyToHostF32}`, `typed_tensor_abi::{MakeCpuTensor,CastCpuTensor}`.
- Produces: exact typed CPU storage and preserved P0/NN V2 status behavior for all native callers.

- [ ] **Step 1: Run the dependency-light Debug matrix**

```bash
rm -rf build/p1a-native-debug
cmake -S native -B build/p1a-native-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTENSORA_BUILD_TESTS=ON \
  -DTENSORA_BUILD_BENCHMARKS=OFF \
  -DTENSORA_WITH_TORCH=OFF \
  -DTENSORA_WITH_ONNXRUNTIME=OFF
cmake --build build/p1a-native-debug --parallel 2
ctest --test-dir build/p1a-native-debug --output-on-failure
```

Expected: all 12 native CTest suites pass.

- [ ] **Step 2: On failure, preserve established contracts at the source**

The four historical defensive expectations are:

```text
EnsureCpuFloat32(invalid dtype metadata)        -> TS_UNSUPPORTED
CpuStorage::Filled(impossible float32 numel)    -> TS_OUT_OF_MEMORY
CpuStorage::FromData(impossible float32 numel)  -> TS_OUT_OF_MEMORY
contiguous legacy CPU storage bad write count   -> TS_INTERNAL_ERROR
```

Do not change these assertions. Add or adjust production branching so invalid device/dtype capability is rejected before storage use, legacy allocation wrappers map impossible allocations to `TS_OUT_OF_MEMORY`, and legacy contiguous `CopyToHostF32` validates the storage-reported element count.

- [ ] **Step 3: Re-run the focused contracts**

```bash
ctest --test-dir build/p1a-native-debug \
  -R 'tensora_(dtype_storage|training_core_internal|tensor_view|native)_tests' \
  --output-on-failure
```

Expected: every focused test passes.

- [ ] **Step 4: Run the complete native matrix again**

```bash
ctest --test-dir build/p1a-native-debug --output-on-failure
```

Expected: `100% tests passed`.

- [ ] **Step 5: Commit only if production code was required**

```bash
git add native/src native/tests
git commit -m 'fix(p1): preserve typed storage and legacy status contracts'
```

Expected: no test expectation or threshold is weakened.

---

### Task 3: Prove public ABI v6 and Windows export correctness

**Files:**
- Verify: `native/include/tensora.h`
- Verify: `native/src/tensor/tensor.cc`
- Verify: `native/src/tensor/typed_tensor_abi.h`
- Verify: `native/src/c_api.cc`
- Test: `native/tests/c_abi_test.c`
- Test: `native/tests/dtype_abi_contract_test.c`

**Interfaces:**
- Consumes: `ts_tensor_from_host`, `ts_tensor_full`, `ts_tensor_cast`, `ts_tensor_copy_to_host` and legacy ABI v5 symbols.
- Produces: a C11-consumable ABI v6 shared library whose symbols export correctly on ELF, Mach-O, and PE/COFF.

- [ ] **Step 1: Compile the standalone public C contract**

```bash
cc -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -Inative/include \
  native/tests/dtype_abi_contract_test.c \
  -o build/p1a-dtype-abi-contract
./build/p1a-dtype-abi-contract
```

Expected: compile and execution succeed with `TS_ABI_VERSION == 6u`.

- [ ] **Step 2: Verify shared-library exports on Unix**

```bash
cmake --build build/p1a-native-debug --target tensora_native --parallel 2
nm -D build/p1a-native-debug/libtensora_native.so | grep -E \
  ' ts_tensor_(from_host|full|cast|copy_to_host)$'
```

Expected: all four ABI v6 symbols are globally exported.

- [ ] **Step 3: Preserve `TS_API` on every new definition**

Definitions in `native/src/tensor/tensor.cc` must use these exact forms:

```cpp
TS_API ts_status_t ts_tensor_from_host(...);
TS_API ts_status_t ts_tensor_full(...);
TS_API ts_status_t ts_tensor_cast(...);
TS_API ts_status_t ts_tensor_copy_to_host(...);
```

Expected: MSVC emits no C4273 inconsistent-DLL-linkage warning under `/WX`.

- [ ] **Step 4: Run the hosted Windows Debug and Release jobs**

Expected: both `Native windows-latest / Debug` and `Native windows-latest / Release` build and execute their CTest matrices successfully.

- [ ] **Step 5: Commit only if an ABI defect was corrected**

```bash
git add native/include native/src/tensor native/src/c_api.cc native/tests
git commit -m 'fix(p1): qualify ABI v6 exports across platforms'
```

---

### Task 4: Prove Dart 3.7 and current-stable typed semantics

**Files:**
- Verify: `packages/tensora/lib/src/dtype/half_codec.dart`
- Verify: `packages/tensora/lib/src/native/native_bindings.dart`
- Verify: `packages/tensora/lib/src/native/native_runtime.dart`
- Verify: `packages/tensora/lib/src/tensor/tensor.dart`
- Test: `packages/tensora/test/dtype_semantics_test.dart`
- Test: `packages/tensora/test/typed_tensor_integration_test.dart`
- Test: `packages/tensora/test/value_types_test.dart`

**Interfaces:**
- Consumes: ABI v6 typed host functions and `DType` stable codes.
- Produces: deterministic `Tensor.fromList`, `zeros`, `ones`, `full`, `cast`, `toList`, and `toTypedData` behavior on Dart 3.7 and current stable.

- [ ] **Step 1: Resolve workspace dependencies**

```bash
dart pub get
```

Expected: dependency resolution succeeds under the selected SDK.

- [ ] **Step 2: Apply and verify stable formatting**

```bash
dart format packages/tensora/lib packages/tensora/test \
  packages/tensora/integration_test packages/tensora_train/integration_test
git diff --exit-code -- \
  packages/tensora/lib packages/tensora/test \
  packages/tensora/integration_test packages/tensora_train/integration_test
```

Expected: the second command reports no diff.

- [ ] **Step 3: Analyze under current stable**

```bash
dart analyze --fatal-infos --fatal-warnings
```

Expected: `No issues found!`.

- [ ] **Step 4: Run typed semantic and integration tests**

```bash
TENSORA_NATIVE_LIBRARY="$PWD/build/p1a-native-debug/libtensora_native.so" \
  dart test packages/tensora/test/dtype_semantics_test.dart \
    packages/tensora/test/typed_tensor_integration_test.dart \
    packages/tensora/test/value_types_test.dart \
    --reporter expanded
```

Expected: all typed tests pass for all ten dtypes, canonical bool bytes, half/bfloat round-trips, casts, non-contiguous views, and failure rollback.

- [ ] **Step 5: Repeat analysis and package tests under Dart 3.7.0**

```bash
dart analyze --fatal-infos --fatal-warnings
dart test packages/tensora/test --reporter expanded
```

Expected: no generic return-type inference errors and all package tests pass.

- [ ] **Step 6: Commit only formatter or compatibility fixes**

```bash
git add packages/tensora packages/tensora_train/integration_test
git commit -m 'fix(p1): preserve Dart 3.7 typed tensor compatibility'
```

---

### Task 5: Requalify sanitizers, races, fuzzing, and lifecycle behavior

**Files:**
- Verify: `.github/workflows/native-ci.yml`
- Verify: `.github/workflows/thread-sanitizer.yml`
- Verify: `.github/workflows/c-abi-fuzz.yml`
- Verify: `.github/workflows/training-soak.yml`
- Verify: `fuzz/`
- Verify: native lifecycle tests and Dart soak tests

**Interfaces:**
- Consumes: the same production sources used by ordinary Debug/Release builds.
- Produces: memory, undefined-behavior, data-race, malformed-input, and long-run lifecycle evidence.

- [ ] **Step 1: Run ASan and UBSan locally or in the existing hosted job**

```bash
rm -rf build/p1a-sanitized
CC=clang CXX=clang++ cmake -S native -B build/p1a-sanitized \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTENSORA_ENABLE_SANITIZERS=ON \
  -DTENSORA_BUILD_TESTS=ON \
  -DTENSORA_BUILD_BENCHMARKS=OFF \
  -DTENSORA_WITH_TORCH=OFF \
  -DTENSORA_WITH_ONNXRUNTIME=OFF
cmake --build build/p1a-sanitized --parallel 2
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir build/p1a-sanitized --output-on-failure
```

Expected: all 12 tests pass with no sanitizer report.

- [ ] **Step 2: Run ThreadSanitizer**

```bash
rm -rf build/p1a-tsan
CC=clang CXX=clang++ cmake -S native -B build/p1a-tsan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTENSORA_ENABLE_THREAD_SANITIZER=ON \
  -DTENSORA_BUILD_TESTS=ON \
  -DTENSORA_BUILD_BENCHMARKS=OFF \
  -DTENSORA_WITH_TORCH=OFF \
  -DTENSORA_WITH_ONNXRUNTIME=OFF
cmake --build build/p1a-tsan --parallel 2
TSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build/p1a-tsan --output-on-failure
```

Expected: no race report.

- [ ] **Step 3: Run the existing C ABI fuzz workflow**

Expected: malformed dtype, pointer, rank, dimension, capacity, bool canonicality, handle, and output-clearing cases pass under sanitizers.

- [ ] **Step 4: Run core and training lifecycle soaks**

Expected: live tensor count and live storage bytes return to baseline after every soak; no unbounded memory growth occurs.

---

### Task 6: Requalify inherited training and inference providers

**Files:**
- Verify: `native/src/training/`
- Verify: `native/src/inference/`
- Verify: `packages/tensora/integration_test/accelerator_training_integration_test.dart`
- Verify: `packages/tensora/integration_test/cuda_training_integration_test.dart`
- Verify: `packages/tensora/integration_test/training_integration_test.dart`
- Verify: `packages/tensora_train/integration_test/declarative_nn_v2_integration_test.dart`

**Interfaces:**
- Consumes: P1A typed metadata while retaining float32 provider execution.
- Produces: unchanged core-only CPU, LibTorch CPU, real MPS, ONNX Runtime, and CoreML behavior.

- [ ] **Step 1: Run core-only declarative MLP training**

Expected: deterministic loss decreases, parameters change, state assignment remains transactional, and no LibTorch dependency is loaded.

- [ ] **Step 2: Run LibTorch CPU training policy and integration jobs**

Expected: float32 training remains bit-for-bit compatible with the pre-P1A contract; non-float32 training fails explicitly.

- [ ] **Step 3: Run real Apple MPS qualification**

Expected: physical MPS execution is observed and CPU fallback is forbidden.

- [ ] **Step 4: Run ONNX Runtime and CoreML inference jobs**

Expected: representative outputs match trusted references and provider selection remains inspectable.

- [ ] **Step 5: Run NN V2 state, optimizer, activation, and parameter-control gates**

Expected: native parameter identity, `StateDict`, SGD/Adam/AdamW, GELU/SiLU/SwiGLU, train/eval, freeze/unfreeze, and deterministic disposal remain green.

---

### Task 7: Restore hard coverage without exclusions

**Files:**
- Verify: `.github/workflows/native-coverage.yml`
- Verify: `.github/workflows/high-assurance.yml`
- Verify: `tools/merge_dart_coverage.py`
- Add focused tests only when a production branch is genuinely unexecuted

**Interfaces:**
- Consumes: all core, training, inference, and typed-storage test traces.
- Produces: hard native and Dart production coverage artifacts for the exact candidate SHA.

- [ ] **Step 1: Collect merged native coverage**

Expected: Tensora-owned native code reports `100.0%` lines and `100.0%` functions across core, LibTorch, and ONNX configurations.

- [ ] **Step 2: Collect merged Dart coverage**

Expected: the existing high-assurance threshold remains at least `99.9%`; no production source is omitted.

- [ ] **Step 3: Inspect each uncovered line before changing code**

For every residual, identify one of:

```text
reachable production behavior -> add a focused behavioral or white-box contract
unreachable compiler artifact  -> refactor without changing semantics, then test
third-party code               -> keep outside Tensora-owned source accounting
```

No ordinary executable production line may be excluded merely to satisfy the gate.

- [ ] **Step 4: Commit focused coverage tests separately**

```bash
git add native/tests packages/tensora/test packages/tensora_nn/test \
  packages/tensora_optim/test packages/tensora_train/integration_test
git commit -m 'test(p1): close exact typed execution coverage'
```

---

### Task 8: Complete the exact-SHA hosted matrix and publish evidence

**Files:**
- Modify after success: `docs/superpowers/specs/2026-08-18-p1a-qualification-candidate.md`
- Modify after success: PR #9 body if the recorded SHA or gate summary changes

**Interfaces:**
- Consumes: the final owner-authored candidate SHA and every required workflow result.
- Produces: an immutable qualification record and a P1A branch eligible for review/merge or the separately-approved P1B continuation.

- [ ] **Step 1: Require every pull-request workflow to execute**

The matrix must contain real jobs. `action_required`, `skipped`, `cancelled`, `neutral`, or a missing workflow is not success.

- [ ] **Step 2: Require all repository gates to succeed on one SHA**

Required workflow families:

```text
Native CI
Native Contract CI
DType Contract CI
Dart FFI CI
Flutter CI
Workspace CI
High Assurance CI
Native Coverage
Coverage Snapshot
ThreadSanitizer CI
C ABI Fuzz CI
Tensor Alias CI
CPU Training Engine CI
Training CI
Training Soak CI
NN V2 CI
Torch Backend Policy CI
Inference CI
ONNX Policy CI
Source Snapshot
```

Expected: every conclusion is `success` for the same head SHA.

- [ ] **Step 3: Update the qualification record with exact evidence**

Record:

```text
base SHA
candidate SHA
PR integration tree SHA
workflow count and conclusions
native line/function/branch coverage
Dart production coverage
supported platforms/providers
physical MPS result
outstanding review threads
explicit unsupported P1A behavior
```

- [ ] **Step 4: Commit the evidence record**

```bash
git add docs/superpowers/specs/2026-08-18-p1a-qualification-candidate.md
git commit -m 'docs(p1): record exact-SHA P1A qualification'
```

- [ ] **Step 5: Re-run the matrix on the evidence commit**

Expected: documentation-only final head also has the complete green matrix. Do not cite the parent SHA as the final qualified revision.

---

### Task 9: Gate the transition to P1B

**Files:**
- Existing design: `docs/superpowers/specs/2026-08-17-tensor-core-v2-design.md`
- Existing plan: `docs/superpowers/plans/2026-08-17-tensor-core-v2-implementation.md`
- Future separate plan: `docs/superpowers/plans/2026-08-18-p1b-promotion-broadcasting.md`

**Interfaces:**
- Consumes: exact-SHA green P1A.
- Produces: a reviewable P1B branch or continuation that adds only promotion, broadcasting, scalar arithmetic, and broadcast-aware float32 backward.

- [ ] **Step 1: Confirm P1A is green before creating P1B work**

```bash
test "$(git rev-parse HEAD)" = "<recorded-qualified-sha>"
```

Expected: the current head exactly matches the qualification record.

- [ ] **Step 2: Create the P1B plan before touching P1B production code**

The P1B plan must define:

```text
checked broadcast planner
ordered-pair dtype promotion table
promoted add/multiply kernels
Dart scalar/operator ownership
broadcast-gradient reduction
full exact-SHA qualification
```

- [ ] **Step 3: Keep compiler, CUDA ownership, distributed, and quantization work out of P1B**

Expected: P1B remains one independently reviewable vertical slice.
