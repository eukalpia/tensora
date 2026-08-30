# Tensora Monorepo Architecture

## Goal

Tensora is organized as a layered monorepo so tensor/runtime infrastructure, neural-network APIs, optimization, data, training orchestration, Flutter integration, native backends, examples, benchmarks, and cross-package acceptance tests can evolve independently without circular dependencies.

A package exists here only when it delivers what its name claims. Reserving a name for future work is not a reason to create a package.

## Repository layout

```text
tensora/
├── packages/
│   ├── tensora/
│   ├── tensora_nn/
│   ├── tensora_optim/
│   ├── tensora_data/
│   ├── tensora_train/
│   └── tensora_flutter/
├── native/
├── tools/
├── examples/
├── benchmarks/
├── docs/
└── tests/
```

## Package ownership

### `tensora`

Foundation package. Owns `Tensor`, `Shape`, `DType`, `Device`, native runtime loading, device discovery, core tensor operators, structured errors, low-level training/runtime compatibility APIs, and ONNX session APIs.

The package remains the only package allowed to bind directly to the native C ABI. Higher-level packages depend on its public Dart surface and must not bind native symbols independently. This rule is enforced by `tools/workspace_check.dart`.

### `tensora_nn`

Neural-network composition. Owns the declarative `Module` tree: deterministic named traversal, lazily materialized `Model.build()`, `Sequential`, native-backed `Linear`, activation modules, `StateDict`, and transactional device moves with rollback.

This is the package that carries Tensora's developer experience. Its composition semantics are Dart-native rather than a transliteration of another framework's object model.

### `tensora_optim`

Optimizers over explicit parameter collections. Owns `ParameterGroup` with per-group hyperparameter overrides, and `SGD`, `Adam`, and `AdamW`. It depends on `tensora` and keeps optimizer imports separate from the tensor foundation.

### `tensora_data`

Pure-Dart data contracts. Owns deterministic indexed datasets, immutable batches, and deterministic batching. It must not depend on neural-network or optimizer packages.

### `tensora_train`

Training orchestration contracts. Depends on `tensora`, `tensora_nn`, and `tensora_optim`. Owns immutable metric/history types and later trainer/checkpoint orchestration. It must not own tensor kernels.

### `tensora_flutter`

Flutter lifecycle and application integration. It is intentionally kept outside the Dart-only Pub workspace because it depends on the Flutter SDK. It depends on `tensora` and must not become a second native runtime implementation.

## Non-package directories

### `native/`

C++ runtime, C ABI, compute kernels, backend dispatch, native ownership, LibTorch integration, and ONNX Runtime integration.

`native/src/kernels/` holds the CPU compute kernels. Microkernels are compiled once per instruction set and selected at run time, so one binary stays correct across hosts with different SIMD support.

### `tools/`

Repository and release utilities. `workspace_check.dart` is the executable form of this document: it validates workspace membership, the allowed dependency edges, absence of cycles, and the rule that only `tensora` binds `dart:ffi`.

### `examples/`

Runnable end-user examples. Examples must use public package imports only.

### `benchmarks/`

Repeatable performance baselines with recorded environment metadata.

### `tests/`

Cross-package acceptance tests. Package-local tests stay next to their package; this directory verifies dependency boundaries and public composition across packages. It runs without a native runtime, so it covers composition contracts rather than native execution.

## Dependency direction

Allowed dependency graph:

```text
tensora
├── tensora_nn
│   └── tensora_optim
├── tensora_data
└── tensora_flutter

tensora_train -> tensora, tensora_nn, tensora_optim
```

Rules:

1. `tensora` depends on no other Tensora Dart package.
2. No package imports another package's `lib/src/` implementation.
3. Only `tensora` binds the C ABI.
4. Flutter is not a dependency of Dart-only packages.
5. Cross-package cycles are release-blocking.

## Pub workspace policy

The Dart-only packages use one Pub workspace and one shared dependency resolution. Paths are enumerated explicitly rather than using a glob so the repository keeps its Dart 3.7 minimum compatibility. `tensora_flutter` is resolved and tested separately with the Flutter SDK.

All Dart workspace members keep `environment.sdk: '>=3.7.0 <4.0.0'` and `resolution: workspace`.

## Reliability gates

A monorepo change is not release-ready until all of the following hold on one exact commit:

- root workspace resolution succeeds on the minimum supported Dart SDK and current stable Dart;
- every Dart-only package formats and analyzes cleanly with strict casts, inference, and raw types;
- every package-local test suite passes;
- cross-package dependency and composition tests pass;
- existing native, Dart FFI, training, inference, sanitizer, fuzz, soak, and coverage gates remain green;
- Flutter package formatting, analysis, and tests pass on stable Flutter;
- no dependency cycle or forbidden `lib/src/` cross-package import exists;
- generated build directories and package-local workspace lockfiles are not committed.

## Scope discipline

Creating a package boundary does not imply that roadmap features are implemented. Each package README must distinguish implemented contracts from future scope. Empty feature claims, silent fallbacks, and placeholder public methods are not permitted.
