# Tensora Monorepo Architecture

## Goal

Tensora is organized as a layered monorepo so tensor/runtime infrastructure, neural-network APIs, optimization, data, training orchestration, edge inference, Flutter integration, modality packages, native backends, compiler work, examples, benchmarks, fuzzing, and cross-package acceptance tests can evolve independently without circular dependencies.

## Repository layout

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

## Package ownership

### `tensora`

Foundation package. Owns `Tensor`, `Shape`, `DType`, `Device`, native runtime loading, device discovery, core tensor operators, structured errors, low-level training/runtime compatibility APIs, and ONNX compatibility APIs that already exist in the current development line.

The package remains the only package allowed to bind directly to the native C ABI. Higher-level packages depend on its public Dart surface and must not bind native symbols independently.

### `tensora_nn`

Canonical neural-network entrypoint. Owns the public neural-network namespace and exposes implemented module/layer/loss APIs. During the current development series it provides a compatibility-preserving facade over the already implemented native-backed `Module`, `Linear`, and `Losses` types in `tensora`; implementation ownership can move behind this boundary later without changing application imports.

### `tensora_optim`

Canonical optimizer entrypoint. Exposes implemented `Optimizer`, `SGD`, `Adam`, and `AdamW` APIs. It depends on `tensora` and keeps optimizer imports separate from the tensor foundation.

### `tensora_data`

Pure-Dart data contracts. Owns deterministic indexed datasets, immutable batches, and deterministic batching. It must not depend on neural-network or optimizer packages.

### `tensora_train`

Training orchestration contracts. Depends on `tensora`, `tensora_nn`, and `tensora_optim`. Owns immutable metric/history types and later trainer/checkpoint orchestration. It must not own tensor kernels.

### `tensora_edge`

Canonical portable-inference entrypoint. Exposes implemented ONNX Runtime session/provider APIs and later owns model packaging/runtime policies. It depends only on `tensora`.

### `tensora_flutter`

Flutter lifecycle and application integration. It is intentionally kept outside the Dart-only Pub workspace because it depends on the Flutter SDK. It depends on `tensora` and must not become a second native runtime implementation.

### `tensora_vision`

Vision-domain contracts built on `tensora`. Initial scope is validated tensor/image layout metadata; camera pipelines and preprocessing kernels remain future work until backed by real implementation and tests.

### `tensora_audio`

Audio-domain contracts built on `tensora`. Initial scope is validated audio tensor metadata; decoding, resampling, and spectral kernels are future work.

### `tensora_text`

Text-domain contracts. Owns tokenizer interfaces and immutable token sequences. It may depend on `tensora` for tensor-valued outputs but must not depend on transformers.

### `tensora_transformers`

Transformer/LLM-domain configuration and model-facing contracts. It depends on `tensora`, `tensora_nn`, and `tensora_text`; it must not pull Flutter or edge deployment policy into the model layer.

## Non-package directories

### `native/`

C++ runtime, C ABI, backend dispatch, native ownership, LibTorch integration, ONNX Runtime integration, and platform-specific accelerator bridges.

### `compiler/`

Graph/IR/compiler work. No compiler capability is claimed until executable graph capture/IR tests exist. The directory documents and contains compiler-specific code only when that code becomes real.

### `tools/`

Repository and release utilities. The first utility validates workspace structure and dependency direction.

### `examples/`

Runnable end-user examples. Examples must use public package imports only.

### `benchmarks/`

Repeatable performance baselines with recorded environment metadata.

### `fuzz/`

Fuzz targets and corpus metadata. Native fuzzing remains connected to the public C ABI and sanitizer builds.

### `tests/`

Cross-package acceptance tests. Package-local tests stay next to their package; this directory verifies dependency boundaries and public composition across packages.

## Dependency direction

Allowed high-level dependency graph:

```text
tensora
├── tensora_nn
│   ├── tensora_optim
│   └── tensora_train
├── tensora_data
├── tensora_edge
├── tensora_flutter
├── tensora_vision
├── tensora_audio
└── tensora_text
    └── tensora_transformers

# additional edges
tensora_train -> tensora_optim
tensora_train -> tensora_nn
tensora_transformers -> tensora_nn
```

Rules:

1. `tensora` depends on no other Tensora Dart package.
2. No package imports another package's `lib/src/` implementation.
3. Domain packages do not bind the C ABI directly.
4. Flutter is not a dependency of Dart-only packages.
5. `tensora_transformers` may depend on text and neural-network contracts; text never depends on transformers.
6. Cross-package cycles are release-blocking.

## Pub workspace policy

The Dart-only packages use one Pub workspace and one shared dependency resolution. Paths are enumerated explicitly rather than using a glob so the repository keeps its Dart 3.7 minimum compatibility. `tensora_flutter` is resolved and tested separately with the Flutter SDK.

All Dart workspace members keep `environment.sdk: '>=3.7.0 <4.0.0'` and `resolution: workspace`.

## API compatibility policy

The current development line already exposes training and ONNX APIs from `package:tensora/tensora.dart`. This architecture does not delete those APIs in the monorepo transition. New applications should prefer the canonical domain packages, while compatibility exports remain until a separately versioned removal decision is made.

This avoids a repository-layout change becoming an application-breaking runtime change.

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

Creating the package boundary does not imply that all roadmap features are implemented. Each package README must distinguish implemented contracts from future scope. Empty feature claims, silent fallbacks, and placeholder public methods are not permitted.
