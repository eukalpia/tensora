# Tensora Testing Strategy

Tensora testing must prove more than compilation. The project spans numerical semantics, native memory, FFI, backends, devices, asynchronous execution, model parsing, and Flutter lifecycle behavior. Each layer requires evidence appropriate to its failure modes.

## 1. Test pyramid

Tensora should maintain these test classes as the corresponding subsystems become real:

```text
Dart unit tests
Native unit tests
C ABI tests
Dart ↔ native integration tests
Backend parity tests
Numerical reference tests
Autograd gradient checks
Serialization/model-format tests
Flutter integration tests
Real-device tests
Stress and leak tests
Failure-injection tests
Security regression tests
Fuzzing
Benchmarks/regression measurements
```

No single class is sufficient by itself.

Milestone 1 implements the classes relevant to the CPU Tensor vertical slice: Dart tests, native tests, C ABI tests, real FFI integration, numerical references, malformed-input/security regressions, concurrency checks, lifecycle stress, sanitizers, and benchmark harnesses.

## 2. Milestone 1 CI gates

Two GitHub Actions workflows validate the current implementation.

### Native CI

The native workflow runs:

- Debug and Release builds on `ubuntu-latest`;
- Debug and Release builds on `macos-latest`;
- Debug and Release builds on `windows-latest`;
- all registered CTest suites in every build;
- Release benchmark smoke on each desktop runner;
- a dedicated Clang AddressSanitizer + UndefinedBehaviorSanitizer job on Ubuntu.

The registered native tests include:

- `tensora_native_tests` — tensor semantics, malformed inputs, lifecycle, concurrency, and 10,000-cycle stress;
- `tensora_handle_registry_tests` — wrong-object-type and stale handle rejection;
- `tensora_c_abi_tests` — compile/link/run through the public header as C11 and fixed-width ABI checks.

### Dart FFI CI

The Dart workflow runs:

- current stable Dart formatting and strict static analysis;
- a dedicated Dart 3.7 minimum compatibility build/analyze/test/example job;
- real Dart FFI tests on Linux, macOS, and Windows;
- the `tensor_basics` example on all three desktop platforms;
- Dart/FFI benchmark smoke on Linux.

Every FFI test uses a real compiled Tensora native shared library. Mock-only native coverage is not accepted as proof of runtime behavior.

## 3. Numerical correctness

Every mathematical operation should be checked against a trusted reference implementation or analytically known values.

Milestone 1 currently covers:

- `add` known values;
- `multiply` known values;
- `sum` known values;
- 2D transpose known values;
- canonical 2D matrix multiplication:

```text
[1 2]   [5 6]   [19 22]
[3 4] × [7 8] = [43 50]
```

Dart property-style tests additionally check deterministic invariants including:

- transpose twice restores the original matrix;
- adding zeros preserves values;
- multiplying by ones preserves values;
- reshape preserves element count and values.

Floating-point comparisons use narrow `float32`-appropriate tolerances. A broad tolerance must be justified rather than used to hide errors.

## 4. ABI tests

The C ABI is a compatibility and security boundary.

Milestone 1 validates:

- ABI version reporting;
- fixed-width status and handle types from a C11 consumer;
- C header compilation and linkage;
- invalid/null handles;
- wrong handle type;
- duplicate release;
- use after release;
- pointer/length/capacity validation;
- rank and shape validation;
- unknown status-code naming;
- exception containment through all exported guarded entry points;
- concurrent calls for the documented immutable Tensor contract.

A native exception must never escape the C ABI.

## 5. FFI integration

For each public native-backed Dart feature, tests cross the real:

```text
Dart public API
  ↓
Dart FFI bindings
  ↓
C ABI
  ↓
native runtime
  ↓
CPU backend
```

The suite covers:

- `fromList` native ownership;
- `zeros`, `ones`, `full`;
- metadata;
- `reshape`;
- 2D `transpose`;
- `add`;
- `multiply`;
- `sum`;
- 2D `matmul`;
- explicit host extraction;
- typed error mapping;
- deterministic disposal;
- double dispose;
- use-after-dispose;
- invalid native handle conversion;
- isolate-local Tensor enforcement;
- repeated invocation and lifecycle accounting.

## 6. Memory and lifetime testing

Resource lifetime is a first-class correctness property.

Native tests run at least 10,000 repeated cycles of:

```text
create inputs
 ↓
execute matmul
 ↓
release result
 ↓
release inputs
```

The test reads Tensora's live Tensor and live CPU-storage counters before and after the stress loop and requires both to return to the baseline.

Dart FFI tests additionally run repeated public-API create/execute/dispose cycles and require the same native counters to return to baseline.

These counters detect Tensora-owned lifecycle regressions; sanitizer/leak validation remains a separate required gate.

## 7. Sanitizers

Native CI uses:

- AddressSanitizer;
- UndefinedBehaviorSanitizer.

The sanitizer build runs the same registered CTest suites, including lifecycle stress, C ABI, and handle type validation.

Sanitizer failures are blockers. Do not add suppressions for genuine ownership, bounds, or undefined-behavior defects.

ThreadSanitizer is a future dedicated gate when its platform/toolchain cost and runtime compatibility are established; Milestone 1 does not claim TSan validation.

## 8. Malformed and hostile input coverage

Milestone 1 tests reject safely:

- negative dimensions;
- zero dimensions;
- rank above 32;
- overflowing element counts;
- null dimension/data/output pointers where invalid;
- too-small shape/output buffers;
- invalid handle zero;
- arbitrary unknown handles;
- stale/released handles;
- wrong native object type;
- duplicate release;
- incompatible elementwise shapes;
- incompatible matmul dimensions;
- non-2D transpose/matmul;
- invalid reshape element count.

The objective is to reject invalid ABI input before unsafe allocation or memory access wherever the ABI has enough information to validate it.

## 9. Concurrency

Milestone 1 tensors are immutable after creation.

Native tests execute repeated matrix multiplication from multiple threads against the same input tensors. Handle registry lookup/refcount operations are synchronized while computation occurs outside the registry mutex.

Dart Tensor wrappers are not cross-isolate shareable in Milestone 1. A Dart test verifies that sending a Tensor through a port to another isolate is rejected.

Future mutable state, streams, device queues, or training state require their own concurrency tests before support is claimed.

## 10. Test determinism

Tests should minimize unnecessary nondeterminism.

Use fixed deterministic inputs where appropriate. Avoid timing thresholds that create flaky CI. Benchmarks report measurements but smoke jobs do not fail on a noisy latency threshold.

## 11. Benchmarks

Milestone 1 includes two reproducible harnesses.

Native:

```bash
./build/native-release/tensora_native_benchmark --smoke
```

Dart/FFI:

```bash
cd packages/tensora
dart run benchmark/tensor_benchmark.dart --smoke
```

The Dart harness measures:

- minimal FFI no-op overhead;
- tensor creation;
- elementwise add;
- matrix multiplication;
- native-to-Dart extraction;
- median and p95 after warmup.

A benchmark report intended as performance evidence must record hardware, operating system, compiler/runtime, build mode, shapes, warmup, iterations, and measured statistics. Smoke runs prove the benchmark path executes; they are not performance marketing claims.

## 12. Exact local commands

Native Debug:

```bash
cmake -S native -B build/native-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTENSORA_BUILD_TESTS=ON \
  -DTENSORA_BUILD_BENCHMARKS=ON
cmake --build build/native-debug --config Debug --parallel
ctest --test-dir build/native-debug --build-config Debug --output-on-failure
```

Native Release:

```bash
cmake -S native -B build/native-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORA_BUILD_TESTS=ON \
  -DTENSORA_BUILD_BENCHMARKS=ON
cmake --build build/native-release --config Release --parallel
ctest --test-dir build/native-release --build-config Release --output-on-failure
```

Dart quality:

```bash
cd packages/tensora
dart pub get
dart format --output=none --set-exit-if-changed lib test benchmark
dart analyze --fatal-infos --fatal-warnings
```

Dart FFI tests after setting `TENSORA_NATIVE_LIBRARY` to the Release shared library:

```bash
cd packages/tensora
dart test --reporter expanded
```

Example:

```bash
cd examples/tensor_basics
dart pub get
dart analyze --fatal-infos --fatal-warnings
dart run bin/main.dart
```

## 13. Future backend parity

A feature claiming multiple backends must have parity tests across those backends.

Parity means compatible public semantics, not necessarily bit-identical floating-point results.

Test:

- outputs;
- dtype support;
- shape behavior;
- errors for unsupported combinations;
- transfer semantics;
- device selection.

Milestone 1 has only CPU, so there is no multi-backend parity claim yet.

## 14. Future model, Flutter, and training tests

Autograd gradient checks, `.tmodel` hostile-input/fuzz tests, Flutter lifecycle/device tests, camera backpressure stress, and local-language-model session tests begin only when those subsystems contain real implementation.

Do not create passing placeholder tests for unimplemented roadmap capabilities.

## 15. Release validation

Before release, validate the **exact candidate revision**:

- formatting/static analysis;
- unit/integration suites;
- native Debug and Release tests;
- C ABI consumer test;
- supported desktop FFI tests;
- minimum Dart compatibility;
- sanitizers;
- stress/leak behavior;
- examples;
- compatibility docs;
- benchmark harness.

A release should never rely on the assumption that earlier commits were green if the candidate changed afterward.
