# Tensora Development Guide

This guide defines how engineering work should be performed in the Tensora repository.

Tensora is systems infrastructure spanning Dart, native code, device runtimes, model formats, and Flutter integration. A change is considered complete only when its behavior, ownership, failure modes, compatibility, tests, and relevant performance characteristics are understood.

## 1. Work from an explicit scope

Before changing code, define:

- the problem being solved;
- the affected package/runtime layer;
- public behavior;
- ownership/lifetime implications;
- device/backend implications;
- expected failure behavior;
- tests required for acceptance;
- whether an RFC is required.

Avoid opportunistic redesign of unrelated subsystems.

## 2. Prefer vertical slices

Implement end-to-end slices that can be validated.

Example:

```text
Dart Tensor API
  ↓
FFI binding
  ↓
C ABI entry point
  ↓
Native validation
  ↓
CPU implementation
  ↓
Correctness test
  ↓
Leak test
  ↓
Benchmark
```

A vertical slice proves the architecture. A directory full of declarations does not.

## 3. Current Milestone 1 toolchain

Minimum source requirements:

- Dart 3.7+ for the `tensora` package;
- CMake 3.20+;
- C11 compiler;
- C++20 compiler.

GitHub Actions additionally runs the current stable Dart SDK and a dedicated Dart 3.7 compatibility job.

The native runtime is intentionally dependency-light in Milestone 1: it uses the C/C++ standard libraries and platform threading support rather than a large external numerical runtime.

## 4. Branch workflow

Create a focused branch from up-to-date `main`.

Examples:

```text
feature/cpu-tensor-storage
feature/onnx-session
fix/native-handle-release
perf/camera-frame-path
docs/model-format
```

Keep the branch limited to one coherent objective. Do not develop broad incomplete features directly on `main`.

## 5. Public contract first

For a new subsystem, define the public or internal contract before filling in implementation details.

Review:

- type names;
- method semantics;
- async behavior;
- disposal behavior;
- invalid-state behavior;
- backend independence;
- extensibility;
- compatibility cost.

Do not stabilize an API merely because an implementation already exists.

## 6. Test before expanding breadth

The first implementation of a feature should include the tests needed to prove its fundamental behavior before adding variants.

For numerical code, include reference values or a trusted implementation comparison.

For native resource code, include repeated allocation/disposal stress.

For parsers, include malformed input.

For Flutter lifecycle behavior, include repeated mount/dispose and background/resume scenarios where test infrastructure allows.

## 7. Build the native runtime

### Debug

```bash
cmake -S native -B build/native-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTENSORA_BUILD_TESTS=ON \
  -DTENSORA_BUILD_BENCHMARKS=ON
cmake --build build/native-debug --config Debug --parallel
ctest --test-dir build/native-debug --build-config Debug --output-on-failure
```

### Release

```bash
cmake -S native -B build/native-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORA_BUILD_TESTS=ON \
  -DTENSORA_BUILD_BENCHMARKS=ON
cmake --build build/native-release --config Release --parallel
ctest --test-dir build/native-release --build-config Release --output-on-failure
```

The CMake targets compile with warnings treated as errors (`-Werror` on GCC/Clang and `/WX` on MSVC).

### ASan + UBSan

On a Clang/GCC environment where the sanitizers are supported:

```bash
CC=clang CXX=clang++ cmake -S native -B build/native-sanitizers \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTENSORA_ENABLE_SANITIZERS=ON \
  -DTENSORA_BUILD_TESTS=ON \
  -DTENSORA_BUILD_BENCHMARKS=OFF
cmake --build build/native-sanitizers --parallel
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ctest --test-dir build/native-sanitizers --output-on-failure
```

Do not suppress genuine sanitizer defects.

## 8. Dart quality gate

From the core package:

```bash
cd packages/tensora
dart pub get
dart format --output=none --set-exit-if-changed lib test benchmark
dart analyze --fatal-infos --fatal-warnings
```

The package enables strict casts, strict inference, and strict raw-type analysis.

CI also resolves and analyzes the package on the minimum supported Dart 3.7 SDK so the `pubspec` floor is an executable compatibility claim.

## 9. Dart ↔ native integration

Build a native Release library, then point the Dart bridge at its absolute path.

Linux:

```bash
export TENSORA_NATIVE_LIBRARY="$PWD/build/native-release/libtensora_native.so"
```

macOS:

```bash
export TENSORA_NATIVE_LIBRARY="$PWD/build/native-release/libtensora_native.dylib"
```

Windows PowerShell with a multi-config generator:

```powershell
$env:TENSORA_NATIVE_LIBRARY = "$PWD\build\native-release\Release\tensora_native.dll"
```

Then run:

```bash
cd packages/tensora
dart test --reporter expanded
```

Do not replace native integration tests with mocks. Public native-backed behavior must cross the real Dart → FFI → C ABI → native runtime path.

## 10. Run the example

With `TENSORA_NATIVE_LIBRARY` still pointing at the built runtime:

```bash
cd examples/tensor_basics
dart pub get
dart analyze --fatal-infos --fatal-warnings
dart run bin/main.dart
```

The example should print a `Shape([2, 2])` result and matrix multiplication values `[19.0, 22.0, 43.0, 50.0]`.

## 11. Benchmarks

Native benchmark smoke:

```bash
./build/native-release/tensora_native_benchmark --smoke
```

Windows multi-config build:

```powershell
.\build\native-release\Release\tensora_native_benchmark.exe --smoke
```

Dart/FFI benchmark:

```bash
cd packages/tensora
dart run benchmark/tensor_benchmark.dart --smoke
```

A standard Dart run omits `--smoke`; `--large` includes the larger configured matrix case.

Performance conclusions must come from optimized builds and must record hardware, OS, runtime/compiler configuration, shapes, warmup, iterations, and statistics. Do not report debug timings as performance evidence.

## 12. FFI rules

Every FFI entry point needs:

- exact ownership semantics;
- nullability rules;
- pointer/length validation;
- error return behavior;
- ABI-safe types;
- defined thread behavior;
- tests from Dart through the native boundary.

Avoid chatty scalar-level FFI APIs for tensor computation. Milestone 1 crosses the boundary at tensor-operation granularity.

## 13. Native handles

Opaque handles must be validated before use.

The Milestone 1 registry rejects:

- zero/unknown handles;
- already released handles;
- wrong object types;
- duplicate release.

Handles are identifiers, not reinterpret-cast native pointers. Released identifiers are not recycled during process lifetime.

## 14. Memory engineering

For changes that allocate memory, document:

- allocator/provider;
- owner;
- release point;
- alias/view behavior;
- peak memory considerations;
- what happens on partial failure.

Prefer RAII in native internals.

Use explicit cleanup in Dart wrappers for expensive resources, with finalizers as a fallback only.

Milestone 1 exposes native live tensor/storage counters for lifecycle validation; they are diagnostic infrastructure, not a replacement for sanitizer/leak tooling.

## 15. Concurrency and isolates

Before making a type shareable across threads or isolates, define:

- whether it is immutable;
- whether operations are reentrant;
- synchronization strategy;
- lifetime while requests are active;
- cancellation interaction;
- shutdown behavior.

Milestone 1 native tensors are immutable. Registry lookup/retain/release is synchronized, while numerical computation occurs outside the registry lock.

Dart `Tensor` wrappers are isolate-local and intentionally unsendable in Milestone 1. Cross-isolate tensor sharing requires a future explicit ownership/transfer design.

## 16. Device transfers

A change that moves tensor data between host and device must make that behavior explicit.

Milestone 1 has CPU storage only. `fromList` is an explicit host import and `toList()` an explicit host extraction. Normal tensor operations do not shuttle numerical payloads through Dart.

Future profiler instrumentation should record:

- source device;
- destination device;
- byte count;
- synchronization introduced;
- transfer duration where measurable.

## 17. Flutter development

Flutter-specific code belongs in Flutter-facing packages and is outside Milestone 1.

For any future inference path verify:

- heavy work is outside the UI isolate;
- cancellation is safe;
- disposal releases native resources;
- queues are bounded;
- lifecycle transitions do not leak;
- large images/audio buffers are not copied through Dart without necessity.

## 18. Model-format development

Model bundle parsing is security-sensitive and is outside Milestone 1.

Any future change to `.tmodel` must include:

- schema update;
- format-version implications;
- malformed-input tests;
- resource-limit tests;
- path/archive safety tests;
- compatibility documentation;
- migration notes for breaking format changes.

## 19. Dependency policy

Before adding a dependency, document why Tensora should own the integration instead of implementing a smaller boundary itself.

Evaluate:

- license;
- platform availability;
- release cadence;
- security maintenance;
- ABI/API stability;
- binary size;
- transitive dependencies;
- packaging implications.

Native backend dependencies should be modular so unused large runtimes do not automatically ship to every application.

## 20. Performance workflow

Performance work follows this sequence:

1. establish a reproducible benchmark;
2. measure the current baseline;
3. identify the bottleneck with profiling;
4. change one relevant layer;
5. compare before/after;
6. verify correctness and memory behavior;
7. record hardware/runtime details.

Do not optimize from intuition alone when measurement is feasible.

## 21. Documentation workflow

When public behavior changes, update documentation in the same pull request.

Relevant documentation may include:

- README;
- architecture;
- compatibility matrix;
- API docs;
- examples;
- benchmarks;
- release notes.

## 22. Review checklist

Before requesting review, verify:

- [ ] scope is focused;
- [ ] public contract is documented;
- [ ] no provider-specific type leaked into stable APIs;
- [ ] ownership is explicit;
- [ ] failure behavior is tested;
- [ ] relevant correctness tests pass;
- [ ] relevant stress/leak tests pass;
- [ ] compatibility impact is documented;
- [ ] benchmarks are present for performance claims;
- [ ] no placeholder code is presented as supported behavior;
- [ ] documentation is updated;
- [ ] no credentials or accidental large artifacts are committed;
- [ ] validation corresponds to the current pull-request head revision.

## 23. Completion standard

Do not use “done” to mean “compiles locally.”

A Tensora change is complete when the supported behavior is demonstrably correct at the relevant boundaries and the repository contains enough evidence for another engineer to understand, reproduce, test, and maintain it.
