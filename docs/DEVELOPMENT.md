# Tensora Development Guide

This guide defines the engineering workflow for Tensora's native/Dart runtime.

Tensora spans Dart APIs, a C ABI, native tensor storage, optional training and inference backends, device/provider selection, and platform-specific dependency loading. A change is complete only when behavior, ownership, failure modes, compatibility, and relevant platform evidence are understood.

## 1. Current toolchain

Minimum source requirements:

- Dart 3.7+;
- CMake 3.20+;
- C11 compiler;
- C++20 compiler.

The current native C ABI is version **4**.

Hosted CI additionally exercises the current stable Dart SDK and platform toolchains on Linux, macOS, and Windows.

## 2. Build modes

Tensora keeps optional large native dependencies behind explicit CMake options.

Dependency-light core:

```text
TENSORA_WITH_TORCH=OFF
TENSORA_WITH_ONNXRUNTIME=OFF
```

Training-enabled runtime:

```text
TENSORA_WITH_TORCH=ON
```

Inference-enabled runtime:

```text
TENSORA_WITH_ONNXRUNTIME=ON
```

Provider-specific ONNX integrations use their dedicated build flags. Never infer provider support merely because the underlying ONNX Runtime distribution was built with that provider.

## 3. Core native build

Release example:

```bash
cmake -S native -B build/native \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORA_BUILD_TESTS=ON \
  -DTENSORA_BUILD_BENCHMARKS=ON
cmake --build build/native --config Release --parallel
ctest --test-dir build/native --build-config Release --output-on-failure
```

Debug uses the same structure with `CMAKE_BUILD_TYPE=Debug` on single-config generators and `--config Debug` for the build/test steps.

Native targets compile with warnings treated as errors. Do not weaken the global warning policy to accommodate a defect in Tensora source.

Third-party headers may require narrowly scoped compiler handling when an upstream warning cannot be fixed in Tensora. Such exceptions must stay local to the translation units that include the dependency.

## 4. LibTorch training build

Resolve the LibTorch CMake prefix from the selected PyTorch/LibTorch installation, then configure:

```bash
cmake -S native -B build/training \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX" \
  -DTENSORA_WITH_TORCH=ON \
  -DTENSORA_BUILD_TESTS=ON \
  -DTENSORA_BUILD_BENCHMARKS=OFF
cmake --build build/training --config Release --parallel
ctest --test-dir build/training --build-config Release --output-on-failure
```

The training runtime exposes CPU and whichever accelerator capabilities are actually available in the linked LibTorch build and current host.

An unavailable device request must fail explicitly. Do not add automatic CPU fallback for an explicitly selected training device.

## 5. ONNX Runtime build

Point CMake at a concrete ONNX Runtime distribution:

```bash
cmake -S native -B build/inference \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORA_WITH_ONNXRUNTIME=ON \
  -DTENSORA_ONNXRUNTIME_ROOT="$ORT_ROOT" \
  -DTENSORA_BUILD_TESTS=ON \
  -DTENSORA_BUILD_BENCHMARKS=OFF
cmake --build build/inference --config Release --parallel
ctest --test-dir build/inference --build-config Release --output-on-failure
```

Provider-specific flags are additive to this configuration.

An explicit ONNX provider request must never silently become CPU. Use the public `auto` policy only when deterministic provider fallback is desired.

## 6. Windows sidecar dependency contract

Windows optional backends must be distributed and tested using sidecar DLLs:

```text
runtime-directory/
  tensora_native.dll
  backend DLLs required by that build
```

When `TENSORA_NATIVE_LIBRARY` names an existing Windows DLL, the Dart runtime loads it with a restricted search policy that prioritizes the runtime directory for dependencies.

For a LibTorch build, stage the required LibTorch DLLs beside `tensora_native.dll`.

For an ONNX Runtime build, stage the exact ONNX Runtime/provider DLLs beside `tensora_native.dll`.

Do not depend on an unrelated globally installed DLL or an ambient PATH entry. This is a correctness requirement as well as a dependency-hijacking hardening measure.

## 7. Dart quality gate

From `packages/tensora`:

```bash
dart pub get
dart format --output=none --set-exit-if-changed lib test benchmark integration_test
dart analyze --fatal-infos --fatal-warnings
dart test --reporter expanded
```

The package uses strict casts, strict inference, and strict raw-type analysis.

The minimum Dart 3.7 job is part of the compatibility contract, not merely a syntax smoke test.

## 8. Dart ↔ native integration

Point `TENSORA_NATIVE_LIBRARY` at an absolute built library path.

Linux:

```bash
export TENSORA_NATIVE_LIBRARY="$PWD/build/native/libtensora_native.so"
```

macOS:

```bash
export TENSORA_NATIVE_LIBRARY="$PWD/build/native/libtensora_native.dylib"
```

Windows PowerShell:

```powershell
$env:TENSORA_NATIVE_LIBRARY = "$PWD\build\native\Release\tensora_native.dll"
```

Then run the relevant Dart tests.

Native-backed public behavior must cross the real Dart → FFI → C ABI → native runtime path. Mocks are useful for isolated logic but are not runtime evidence.

## 9. Device semantics

Public device values are:

- CPU;
- CUDA with index;
- MPS;
- XPU with index;
- HIP/ROCm with index.

Rules for development:

- factories remain CPU-default;
- `device:` is an explicit request;
- `preferredDevice` is a query, not a global default mutation;
- device transfers return independently owned tensor handles;
- binary ops reject device-kind or device-index mismatches;
- failed transfers/factory staging must release temporary CPU storage;
- no hidden cross-device copies to make an operation succeed.

## 10. Current accelerator creation behavior

Host values enter Tensora through a CPU staging tensor. When a factory receives an accelerator `device:`, the runtime transfers the native tensor and releases the staging handle deterministically.

This is intentionally simple and correct. Do not describe it as zero-copy.

Future direct device allocation/import paths require their own ownership, stream, synchronization, and benchmark contracts before replacing this behavior.

## 11. ONNX input behavior

The current portable ONNX bridge reads Tensora input values into host memory and creates ONNX Runtime input tensors from that host buffer.

Therefore:

- accelerated execution providers are real provider execution;
- input binding is still host-materialized;
- provider selection does not imply zero-copy input transfer.

A future device-I/O binding path must be introduced as a separate tested optimization.

## 12. Ownership rules

Every successful native creation call must satisfy:

```text
success => exactly one owned reference escapes
failure => zero owned references escape
```

Every fallible Dart adoption path must release a native handle if metadata validation fails.

Finalizers are backup cleanup only. Tests and examples should use deterministic `dispose()`.

## 13. Handles and ABI

Opaque handles are identifiers, not raw C++ object pointers.

The registry rejects:

- zero/unknown handles;
- stale handles;
- wrong object types;
- duplicate release.

Do not expose C++ layout through the public ABI. Additive or breaking ABI work requires compatibility review and an ABI version decision.

## 14. Error handling

Native exceptions must not cross the C ABI.

Validate untrusted boundary values before backend work whenever possible:

- handles;
- device/provider codes and indices;
- dimensions/ranks;
- pointer lengths and capacities;
- module dimensions;
- optimizer hyperparameters;
- model paths and names.

Do not catch broad errors in Dart merely to convert backend failure into apparent success.

## 15. Sanitizers, fuzzing and concurrency

Run the repository workflows appropriate to the changed subsystem.

Memory/bounds changes require sanitizer evidence. C ABI parsing/validation changes require fuzz/regression evidence. Synchronization changes require concurrency/TSan evidence where supported.

Do not generalize a green concurrency test beyond the state it actually exercises.

## 16. Hardware qualification

Real accelerator support is qualified separately from ordinary hosted portability.

The manual `Accelerator Hardware Qualification` workflow contains explicit targets for vendor/platform combinations. Each target must execute on matching physical hardware and must verify that the requested Tensora device/provider is actually active.

A workflow definition, a queued job, or successful compilation without the hardware test is not support evidence.

Apple MPS and CoreML currently have real automated hosted hardware evidence. Other vendor paths remain qualification-pending until their physical workflows pass.

## 17. Branch and pull-request workflow

Keep changes focused and preserve a clean stacked diff when milestones build on one another.

Before declaring a pull request ready:

1. identify the exact branch head SHA;
2. run every required hosted workflow on that SHA;
3. check the actual conclusion of every required job;
4. run relevant physical hardware qualification where the support claim requires it;
5. document unqualified paths explicitly;
6. check repository hygiene and generated artifacts.

Do not reuse green results from an older revision after moving the branch head.

## 18. Performance work

Performance claims must use optimized builds and record:

- hardware;
- OS;
- compiler/runtime;
- dependency versions;
- tensor/model sizes;
- warmup;
- iteration count;
- reported statistics.

Benchmark smoke in CI proves the harness remains executable; it is not a performance guarantee.

## 19. Scope discipline

Prefer complete vertical slices over breadth.

For a new backend/device/provider path, finish:

```text
public contract
→ ABI mapping
→ native implementation
→ failure semantics
→ deterministic ownership
→ hosted portability
→ physical qualification when required
→ documentation
```

before claiming support or expanding to another variant.
