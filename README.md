# Tensora

**Tensora** is a planned native machine-learning framework and AI runtime for **Dart and Flutter**.

Its goal is to provide one coherent path from model development and training to server, desktop, and on-device deployment without forcing application developers to leave the Dart ecosystem for ordinary production workflows.

> **Train in Dart. Deploy anywhere. Run natively in Flutter.**

## Project status

Tensora is currently in its **foundation and architecture phase**. The public API, native ABI, tensor ownership model, backend boundaries, model packaging format, platform support policy, and engineering workflow are being defined before broad implementation begins.

No backend, device, platform, operator, or performance characteristic should be considered supported until it is implemented, tested, documented, and listed in the compatibility matrix.

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

The public Dart API must remain independent of a specific native provider. Initial implementations may use mature projects such as ATen/LibTorch, ONNX Runtime, CUDA libraries, and platform acceleration APIs, but their implementation types must not leak into the stable Tensora API.

See [Architecture](docs/ARCHITECTURE.md) for the full design principles.

## Planned repository structure

```text
tensora/
├── packages/
│   ├── tensora/
│   ├── tensora_nn/
│   ├── tensora_optim/
│   ├── tensora_data/
│   ├── tensora_train/
│   ├── tensora_edge/
│   ├── tensora_flutter/
│   ├── tensora_vision/
│   ├── tensora_audio/
│   ├── tensora_text/
│   └── tensora_transformers/
├── native/
├── compiler/
├── tools/
├── examples/
├── benchmarks/
├── fuzz/
├── docs/
└── tests/
```

This structure is a direction, not a requirement to create empty packages. New modules should be introduced only when they have a clear responsibility, dependency boundary, acceptance criteria, and working vertical slice.

## Engineering principles

Tensora development follows several non-negotiable rules:

1. **Correctness before breadth.** A small, thoroughly tested feature is preferred over a large collection of placeholders.
2. **Dart-first API, native execution.** Dart owns the developer-facing model; heavy numerical work stays in optimized native backends.
3. **Backend neutrality.** Public APIs must not permanently depend on one execution engine.
4. **Explicit ownership.** Native memory and device resources need deterministic lifecycle semantics.
5. **Zero-copy where practical.** Large transfers must never be hidden or accidental.
6. **Asynchronous by design.** Heavy work must not block Flutter's UI isolate.
7. **Observable fallbacks.** Device changes, copies, backend fallbacks, and synchronization must be visible in diagnostics and profiling.
8. **Measured performance.** Benchmark claims must be reproducible and separate Tensora overhead from backend performance.
9. **Real support only.** Compilation alone is not proof of runtime support.
10. **Production behavior over demos.** Error handling, cancellation, memory pressure, malformed inputs, and lifecycle transitions are part of the feature.

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

Tensora should grow through **complete vertical slices**.

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