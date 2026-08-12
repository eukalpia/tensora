# Tensora

**Tensora** is a native machine-learning framework and AI runtime for **Dart and Flutter**.

Its goal is to provide one coherent path from model development and training to server, desktop, and on-device deployment without forcing application developers to leave the Dart ecosystem for ordinary production workflows.

> **Train in Dart. Deploy anywhere. Run natively in Flutter.**

## Project status

Tensora now contains its first executable vertical slice: **Milestone 1 — Tensor Core / CPU**.

The implemented scope is deliberately narrow:

- Dart `Tensor`, `Shape`, `DType.float32`, and `Device.cpu` APIs;
- Tensora-owned native CPU tensor storage;
- stable C ABI version 1 with opaque handles and structured status codes;
- `fromList`, `zeros`, `ones`, `full`, `reshape`, 2D `transpose`, `add`, `multiply`, `sum`, and 2D `matmul`;
- deterministic `dispose()` with a finalizer only as a safety net;
- invalid/stale/wrong-type handle rejection;
- checked shape/allocation arithmetic;
- Dart-to-native integration tests on Linux, macOS, and Windows in GitHub Actions;
- native Debug/Release tests, C ABI tests, lifecycle stress, ASan/UBSan validation, and benchmark smoke runs.

The project is still pre-1.0. CUDA, autograd, training modules, ONNX, Flutter runtime integration, `.tmodel`, transformers, and other roadmap capabilities are **not implemented or supported yet**.

A capability is considered supported only after it is implemented, tested, documented, and represented accurately in [the compatibility policy](docs/COMPATIBILITY.md).

## Tensor Core example

```dart
import 'package:tensora/tensora.dart';

void main() {
  final a = Tensor.fromList(
    [1.0, 2.0, 3.0, 4.0],
    shape: Shape([2, 2]),
  );
  final b = Tensor.fromList(
    [5.0, 6.0, 7.0, 8.0],
    shape: Shape([2, 2]),
  );

  final result = a.matmul(b);
  print(result.toList()); // [19.0, 22.0, 43.0, 50.0]

  result.dispose();
  b.dispose();
  a.dispose();
}
```

Tensor payloads remain in native memory during tensor-to-tensor operations. `Tensor.fromList` performs an explicit host-to-native import and `toList()` performs an explicit native-to-Dart copy.

Tensor wrappers are isolate-local in Milestone 1 and cannot be sent through Dart isolate ports.

## Build and run Milestone 1

Prerequisites:

- Dart 3.7 or newer;
- CMake 3.20 or newer;
- a C11 compiler and a C++20 compiler.

Linux/macOS example:

```bash
cmake -S native -B build/native \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORA_BUILD_TESTS=ON \
  -DTENSORA_BUILD_BENCHMARKS=ON
cmake --build build/native --config Release --parallel
ctest --test-dir build/native --build-config Release --output-on-failure

export TENSORA_NATIVE_LIBRARY="$PWD/build/native/libtensora_native.so"
# macOS: use $PWD/build/native/libtensora_native.dylib

cd examples/tensor_basics
dart pub get
dart run bin/main.dart
```

On Windows, build with the same CMake commands and set `TENSORA_NATIVE_LIBRARY` to the generated `tensora_native.dll` (normally under `build/native/Release/` for a multi-config generator).

See [Development](docs/DEVELOPMENT.md) and [Testing](docs/TESTING.md) for exact validation commands.

## Why Tensora

Dart is well suited to building typed, cross-platform applications, while Flutter provides a strong application runtime across mobile and desktop. The missing layer is a serious ML systems stack designed around those strengths.

Tensora is intended to make workflows such as these first-class:

```text
Dart data pipeline
    ↓
Dart model definition
    ↓
Native CPU / GPU training
    ↓
Model packaging
    ↓
Flutter / server deployment
    ↓
Local accelerated inference
```

The framework will focus on:

- native tensor storage and execution;
- automatic differentiation and training interfaces;
- CPU and accelerator backends;
- portable model inference;
- model packaging and validation;
- reusable preprocessing and postprocessing pipelines;
- Flutter-native camera, audio, text, and local-model integration;
- predictable memory ownership and asynchronous execution;
- profiling, benchmarking, diagnostics, and production reliability.

## Strategic scope

The first adoption target is **production on-device AI for Dart and Flutter**, including:

- computer vision and live-camera inference;
- OCR and document processing;
- speech and audio workloads;
- embeddings and semantic search;
- local RAG primitives;
- local language-model inference;
- privacy-sensitive and offline AI;
- device-side personalization where technically appropriate.

The second major pillar is **real training from Dart**, backed initially by mature native numerical infrastructure rather than performance-critical scalar computation in Dart.

## Architecture direction

Tensora is designed as a layered system:

```text
Dart / Flutter application
          ↓
      Tensora API
          ↓
Execution / graph abstractions
          ↓
      Dart FFI bridge
          ↓
      Stable C ABI
          ↓
   Tensora Native Core
          ↓
Pluggable execution backends
```

Milestone 1 proves the lower half of that design with a real CPU backend. The public Dart API remains independent of native C++ implementation types.

See [Architecture](docs/ARCHITECTURE.md) for the full design principles.

## Repository structure

Only subsystems containing real functionality are created. The current implementation includes:

```text
tensora/
├── packages/
│   └── tensora/
├── native/
├── examples/
│   └── tensor_basics/
├── benchmarks/
├── docs/
└── .github/workflows/
```

Future packages are introduced only when they have a clear responsibility, dependency boundary, acceptance criteria, and working vertical slice.

## Engineering principles

Tensora development follows several non-negotiable rules:

1. **Correctness before breadth.** A small, thoroughly tested feature is preferred over a large collection of placeholders.
2. **Dart-first API, native execution.** Dart owns the developer-facing model; heavy numerical work stays in optimized native backends.
3. **Backend neutrality.** Public APIs must not permanently depend on one execution engine.
4. **Explicit ownership.** Native memory and device resources need deterministic lifecycle semantics.
5. **Zero-copy where practical.** Large transfers must never be hidden or accidental.
6. **Asynchronous by design.** Heavy future Flutter work must not block the UI isolate.
7. **Observable fallbacks.** Device changes, copies, backend fallbacks, and synchronization must be visible in diagnostics and profiling.
8. **Measured performance.** Benchmark claims must be reproducible and separate Tensora overhead from backend performance.
9. **Real support only.** Compilation alone is not proof of runtime support.
10. **Production behavior over demos.** Error handling, memory pressure, malformed inputs, and lifecycle transitions are part of the feature.

## Roadmap

The high-level sequence is:

1. Foundation and public contracts.
2. Native tensor core and CPU execution.
3. Native training backend and CUDA execution.
4. Portable model inference.
5. `.tmodel` deployment bundle.
6. Flutter runtime integration.
7. Vision and live-camera pipelines.
8. Text, embeddings, and local-model APIs.
9. Production edge features.
10. Expanded training ecosystem.
11. Graph compiler and custom optimization where real workloads justify it.

See [ROADMAP.md](ROADMAP.md) for milestones, acceptance gates, and non-goals.

## Working on Tensora

Start with:

- [CONTRIBUTING.md](CONTRIBUTING.md) — contribution rules and pull-request workflow;
- [Development Guide](docs/DEVELOPMENT.md) — local engineering workflow and quality gates;
- [Architecture](docs/ARCHITECTURE.md) — system boundaries and invariants;
- [Testing Strategy](docs/TESTING.md) — correctness, integration, device, stress, and fuzz testing;
- [Compatibility Policy](docs/COMPATIBILITY.md) — platform/backend support policy;
- [RFC Process](docs/RFC_PROCESS.md) — how architectural changes are proposed and reviewed;
- [Release Process](docs/RELEASES.md) — versioning and release requirements;
- [Security Policy](SECURITY.md) — security reporting and secure engineering expectations.

## Contribution philosophy

Tensora grows through **complete vertical slices**.

For example, prefer fully implementing:

```text
Tensor creation
+ native storage
+ one operation
+ error mapping
+ explicit disposal
+ correctness tests
+ leak tests
+ benchmark
+ documentation
```

before adding dozens of unsupported operations or empty packages.

Large architectural changes should begin with an RFC rather than immediately changing stable interfaces.

## License

Tensora is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE).

Unless explicitly stated otherwise, contributions intentionally submitted for inclusion in this repository are provided under the same license terms.
