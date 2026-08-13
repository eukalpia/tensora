# Tensora Testing Strategy

Tensora testing must prove numerical correctness, ABI safety, native ownership, platform portability, provider/device behavior, and failure semantics. Compilation alone is never sufficient evidence for a runtime or hardware claim.

## 1. Validation layers

The repository uses these complementary layers:

```text
Dart unit tests
Native unit tests
C ABI tests
Dart ↔ native FFI tests
Training integration tests
ONNX integration tests
Real accelerator tests
Lifecycle/soak tests
Concurrency tests
Sanitizers
Fuzzing
Benchmarks
```

A feature is complete only when the relevant layers for its failure modes are present.

## 2. Hosted workflows

### Native CI

Native CI validates the dependency-light core across Linux, macOS, and Windows in Debug and Release configurations. It includes registered CTest suites, warnings-as-errors, benchmark smoke, and sanitizer coverage where supported.

Core native tests cover:

- tensor creation and metadata;
- reshape/transpose;
- add/multiply/sum/matmul;
- malformed shapes and arithmetic overflow;
- handle type/stale/duplicate-release behavior;
- public C ABI behavior from a C11 consumer;
- generic device-code mapping;
- repeated lifecycle stress;
- documented immutable-tensor concurrency.

### Dart FFI CI

Dart CI validates:

- canonical formatting;
- strict analyzer settings;
- minimum Dart 3.7 compatibility;
- current stable Dart;
- Linux/macOS/Windows FFI against a real compiled native runtime;
- examples;
- isolated lifecycle stress;
- line-coverage threshold;
- benchmark smoke.

Mock-only tests are not accepted as proof for native-backed public behavior.

### Training CI

Training CI builds the optional LibTorch runtime and runs the native + Dart training suites on:

- Linux CPU;
- macOS CPU plus real Apple MPS;
- Windows CPU.

The training acceptance surface includes:

- autograd and gradient retrieval;
- activation references;
- loss references;
- `Linear` forward/training;
- SGD/Adam/AdamW ownership and updates;
- checkpoint save/load;
- device discovery/transfer;
- deterministic disposal and live-handle accounting.

The Apple MPS job additionally requires real MPS availability and executes accelerator tensor operations, module transfer, forward/loss/backward, optimizer steps, loss reduction, and strict lifecycle checkpoints without CPU fallback.

Windows training stages the pinned LibTorch DLL set beside `tensora_native.dll`. The Dart tests intentionally consume that sidecar layout rather than depending on an ambient PATH entry.

### Inference CI

Inference CI validates ONNX Runtime 1.26.0 against deterministic generated fixtures on:

- Linux CPU;
- Windows CPU;
- Apple CoreML on hosted Apple Silicon.

Acceptance covers:

- model load and missing-model errors;
- provider discovery/selection;
- input/output metadata;
- named float32 inference;
- invalid name/count/shape behavior;
- profiling lifecycle;
- repeated inference;
- concurrent read-only session execution;
- deterministic session/tensor release.

The CoreML job requires an Apple Silicon host and verifies the actual selected provider. The Windows job stages the pinned ONNX Runtime DLL beside `tensora_native.dll` and runs both native and Dart tests through that exact runtime layout.

## 3. High-assurance workflows

Dedicated workflows cover failure modes that should not be hidden inside ordinary integration jobs.

### ASan + UBSan

Native sanitizer builds check memory safety and undefined behavior. Genuine ownership, bounds, or UB failures are blockers.

### ThreadSanitizer

A dedicated TSan workflow exercises supported concurrency paths. TSan evidence applies only to code actually executed by that workflow and does not imply arbitrary mutable training-state thread safety.

### C ABI fuzzing

The C ABI fuzz workflow targets untrusted native boundary inputs and is intended to expose crashes, out-of-bounds access, exception leakage, and malformed-input handling defects.

### Training soak

The training soak workflow repeatedly creates/trains/checkpoints/releases training state and checks live resource counters for unbounded Tensora-owned growth.

### High Assurance CI

The high-assurance workflow groups additional stress and regression gates that are intentionally more expensive than the main per-platform smoke matrix.

## 4. Real hardware qualification

Hardware claims require actual device/provider execution.

The automated hosted matrix currently supplies real hardware evidence for:

- Apple MPS training;
- Apple CoreML inference.

Other vendor targets use manual physical-hardware qualification workflows. A queued job or an unexecuted workflow is **not** evidence of support.

Hardware acceptance must:

- detect the expected physical vendor/device;
- require the backend/provider to be available;
- execute real numerical work;
- verify the selected Tensora device/provider identity;
- reject CPU fallback;
- validate correctness;
- validate deterministic release/lifecycle counters where available.

## 5. Numerical correctness

Every mathematical operation needs an analytical reference or a trusted reference implementation.

Representative core reference:

```text
[1 2]   [5 6]   [19 22]
[3 4] × [7 8] = [43 50]
```

Property/invariant tests cover cases such as:

- transpose twice restores the original matrix;
- adding zeros preserves values;
- multiplying by ones preserves values;
- reshape preserves element count and values.

Training tests require finite losses and observable loss reduction on deterministic small problems. Tolerances must be justified for float32 rather than widened to hide defects.

## 6. Device tests

Device coverage checks:

- CPU/CUDA/MPS/XPU/HIP ABI codes;
- Dart device value semantics;
- negative-index rejection;
- generic device counts;
- unavailable-device structured failure;
- explicit transfer;
- factory `device:` staging and failure cleanup;
- no mixed-device binary operations;
- device index equality, not only vendor kind.

`TensoraRuntime.preferredDevice` is tested as a deterministic query. It is not allowed to change CPU-default factory semantics implicitly.

## 7. ONNX provider tests

Provider-selection tests distinguish:

- `auto` policy;
- explicit CPU;
- explicit CUDA;
- explicit DirectML;
- explicit CoreML;
- explicit OpenVINO;
- explicit MIGraphX.

An explicit unavailable provider must fail. Tests must not reinterpret a successful CPU session as proof that an accelerator request succeeded.

The current portable binding copies input tensor values to host memory before constructing ONNX Runtime inputs, so provider tests must not claim zero-copy device binding.

## 8. ABI tests

The C ABI is a security and compatibility boundary. Current ABI version is **4**.

Tests cover:

- ABI version reporting;
- fixed-width public types;
- C11 compilation/linkage;
- null pointers;
- pointer/length/capacity mismatches;
- malformed shapes;
- unknown, stale, wrong-type and already-released handles;
- duplicate release;
- output zeroing on failure;
- exception containment;
- device-code validation;
- training and inference handle type separation.

A C++ exception must never cross the exported C ABI.

## 9. Ownership and leak testing

Resource lifetime is a correctness property.

Core stress repeatedly performs create → execute → release cycles and requires live tensor/storage counters to return to baseline.

Training tests additionally track module and optimizer handles. Inference tests track session and output tensor handles.

Counters cover Tensora-owned wrappers/storage accounting. Third-party allocator caches are not automatically classified as Tensora leaks.

Finalizers are tested as fallback behavior, but deterministic `dispose()` remains the primary contract.

## 10. Windows dependency-resolution regression

Windows source builds with optional backends must be tested using the dependency layout intended for deployment:

```text
runtime-directory/
  tensora_native.dll
  backend dependency DLLs
```

The Dart bridge receives an explicit native-library path. Regression tests must succeed even when the machine contains an unrelated backend DLL elsewhere in the system search path. This prevents accidental success against an incompatible globally installed runtime.

## 11. Concurrency

Concurrency evidence is scoped.

Validated examples include:

- immutable tensor operations from multiple native threads;
- synchronized reusable ONNX sessions under concurrent read-only execution;
- handle-registry access under contention.

Do not infer that arbitrary mutable modules/optimizers are safe for concurrent mutation unless a dedicated test establishes that contract.

## 12. Fuzz and hostile-input coverage

Untrusted boundary tests include:

- invalid ranks/dimensions;
- arithmetic overflow;
- null pointers;
- undersized buffers;
- arbitrary handles;
- wrong handle types;
- invalid device/provider values;
- invalid model paths;
- invalid ONNX names/counts/shapes;
- invalid optimizer/module arguments.

The goal is safe rejection before unsafe memory access or ambiguous backend behavior.

## 13. Benchmarks

Benchmarks are reproducible measurements, not marketing claims.

Native and Dart/FFI smoke harnesses record enough environment information to interpret results. Performance evidence must include:

- hardware;
- OS;
- compiler/runtime;
- build mode;
- tensor shapes;
- warmup;
- iteration count;
- reported statistics.

CI smoke ensures benchmark paths remain runnable; it does not fail on noisy latency thresholds.

## 14. Exact local core commands

```bash
cmake -S native -B build/native \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORA_BUILD_TESTS=ON \
  -DTENSORA_BUILD_BENCHMARKS=ON
cmake --build build/native --config Release --parallel
ctest --test-dir build/native --build-config Release --output-on-failure

cd packages/tensora
dart pub get
dart format --output=none --set-exit-if-changed lib test benchmark integration_test
dart analyze --fatal-infos --fatal-warnings
dart test --reporter expanded
```

Optional-backend local builds additionally require their native dependencies and the appropriate CMake options described in the development guide.

## 15. Completion rule

Never report a feature complete from an older green SHA.

For a release candidate or pull request readiness claim:

1. identify the exact head SHA;
2. run every required workflow against that SHA;
3. verify each required job conclusion;
4. report hardware qualification separately from hosted portability;
5. keep unsupported/unqualified paths explicitly labeled.
