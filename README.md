# Tensora

**Tensora** is a native machine-learning framework and runtime for **Dart** with a long-term Flutter deployment target.

The project is pre-1.0. The current implementation is already more than a CPU tensor prototype: it contains a native tensor core, optional LibTorch-backed training, optional ONNX Runtime inference, explicit accelerator devices/providers, deterministic native ownership, and multi-platform CI.

> Dart-first ML APIs. Native execution. Explicit ownership and device behavior.

## Current implementation

### Tensor runtime

- `Tensor`, immutable `Shape`, `DType.float32`;
- devices: CPU, CUDA, MPS, XPU, and HIP/ROCm identities;
- `fromList`, `zeros`, `ones`, `full`;
- reshape, 2D transpose, equal-shape add/multiply, sum, and 2D matmul;
- explicit `Tensor.to(Device)` transfer;
- optional `device:` on tensor factories;
- `TensoraRuntime.availableDevices` and `preferredDevice` diagnostics;
- explicit native-to-Dart extraction through `toList()`;
- deterministic `dispose()` with finalizers only as a safety net.

Tensor factories still default to **CPU**. Selecting `preferredDevice` is opt-in; Tensora does not silently move ordinary CPU code onto an accelerator.

### Training

A LibTorch-enabled build provides:

- reverse-mode autograd;
- gradients and `backward()`;
- ReLU, sigmoid, and tanh;
- MSE and cross-entropy losses;
- `Linear` modules;
- train/eval mode;
- parameter and buffer views;
- SGD, Adam, and AdamW;
- checkpoint save/load;
- CPU and accelerator module/tensor transfer.

### ONNX inference

An ONNX Runtime-enabled build provides:

- reusable `OnnxSession` handles;
- model input/output metadata;
- named float32 tensor inputs and outputs;
- profiling;
- explicit provider preference;
- selected-provider diagnostics;
- deterministic session disposal;
- structured model/runtime failures.

Explicit provider requests never silently fall back to CPU. `OnnxExecutionProvider.auto` is the opt-in deterministic fallback policy.

## Validation status

Support claims are intentionally narrower than implementation breadth.

| Surface | Current evidence |
| --- | --- |
| Core CPU runtime | GitHub-hosted Linux, macOS, and Windows native + Dart FFI validation |
| LibTorch CPU training | GitHub-hosted Linux, macOS, and Windows native + Dart training validation |
| Apple MPS training | Real Apple Silicon MPS execution in hosted macOS CI, including forward/backward/optimizer and lifecycle checks |
| ONNX CPU inference | GitHub-hosted Linux and Windows native + Dart inference validation |
| Apple CoreML inference | Real CoreML provider execution in hosted Apple Silicon CI |
| NVIDIA CUDA | Implementation and manual hardware qualification workflow; physical qualification still required |
| Intel XPU | Implementation and manual hardware qualification workflow; physical qualification still required |
| AMD HIP/ROCm | Implementation and manual hardware qualification workflow; physical qualification still required |
| DirectML / OpenVINO / MIGraphX | Provider integration paths exist; hardware/provider qualification is still required before support is promoted |

See [Compatibility](docs/COMPATIBILITY.md) for the normative matrix.

## Device example

```dart
import 'package:tensora/tensora.dart';

void main() {
  final device = TensoraRuntime.preferredDevice;

  final a = Tensor.fromList(
    [1.0, 2.0, 3.0, 4.0],
    shape: Shape([2, 2]),
    device: device,
  );
  final b = Tensor.fromList(
    [5.0, 6.0, 7.0, 8.0],
    shape: Shape([2, 2]),
    device: device,
  );

  final result = a.matmul(b);
  print('${result.device}: ${result.toList()}');

  result.dispose();
  b.dispose();
  a.dispose();
}
```

When `device` is an accelerator, host-created values are staged in CPU memory and transferred explicitly by the runtime. This is not a zero-copy import contract.

## Training example

```dart
import 'package:tensora/tensora.dart';

void main() {
  final device = TensoraRuntime.preferredDevice;
  final x = Tensor.fromList([-1, 0, 1, 2], shape: Shape([4, 1]), device: device);
  final y = Tensor.fromList([-1, 1, 3, 5], shape: Shape([4, 1]), device: device);

  final model = Linear(1, 1)..to(device);
  final optimizer = SGD(model, learningRate: 0.1);

  for (var step = 0; step < 100; step++) {
    optimizer.zeroGrad();
    final prediction = model(x);
    final loss = Losses.mse(prediction, y);
    loss.backward();
    optimizer.step();
    loss.dispose();
    prediction.dispose();
  }

  optimizer.dispose();
  model.dispose();
  y.dispose();
  x.dispose();
}
```

A training-enabled native library is required for this example.

## ONNX example

```dart
final session = OnnxSession(
  'model.onnx',
  provider: OnnxExecutionProvider.auto,
);

final input = Tensor.fromList([1, 2, 3, 4], shape: Shape([2, 2]));
final outputs = session.run({'X': input});

for (final tensor in outputs.values) {
  print(tensor.toList());
  tensor.dispose();
}
input.dispose();
session.dispose();
```

The current portable ONNX binding materializes input data on the host before handing it to ONNX Runtime. An accelerated execution provider therefore does **not** imply zero-copy GPU input binding.

## Native architecture

```text
Dart application
      ↓
Tensora public API
      ↓
Dart FFI
      ↓
C ABI v4
      ↓
Tensora native core
      ↓
CPU / LibTorch / ONNX Runtime backends
```

Public Dart types do not expose C++ implementation types. Native objects are represented through typed opaque handles.

Optional native dependencies remain build-time choices:

```text
TENSORA_WITH_TORCH=OFF       # dependency-light core build
TENSORA_WITH_ONNXRUNTIME=OFF # dependency-light core build
```

## Windows native dependencies

When `TENSORA_NATIVE_LIBRARY` points to an existing Windows DLL, Tensora loads it using a restricted dependency-search policy that prioritizes sidecar DLLs in the same directory as `tensora_native.dll` while retaining standard safe system directories.

Optional Windows distributions should therefore place required backend DLLs beside `tensora_native.dll`. CI validates this layout for LibTorch and ONNX Runtime rather than relying on an ambient PATH entry.

## Quality gates

The repository currently exercises:

- native Debug/Release Linux/macOS/Windows builds;
- warnings as errors;
- C11 ABI consumers;
- strict Dart formatting and analysis;
- minimum Dart 3.7 compatibility;
- Dart FFI Linux/macOS/Windows;
- ASan + UBSan;
- ThreadSanitizer;
- C ABI fuzzing;
- lifecycle stress/soak tests;
- training soak validation;
- ONNX concurrency and lifecycle checks;
- real Apple MPS training;
- real Apple CoreML inference;
- benchmark smoke and recorded environment metadata.

Hardware qualification for other GPU vendors is kept separate from ordinary hosted CI so an unavailable self-hosted runner cannot masquerade as either success or failure.

## Current limitations

- pre-1.0 API/ABI and source-built native runtime;
- only `float32` is a public tensor dtype today;
- no published native binary distribution contract yet;
- CUDA/XPU/HIP and several ONNX accelerator providers still require physical qualification before promotion;
- ONNX input binding is currently host-materialized rather than device zero-copy;
- Flutter/mobile runtime integration is not implemented yet;
- `.tmodel` is not implemented yet;
- no claim of universal deterministic floating-point equality across devices/providers.

## Build the dependency-light core

Prerequisites:

- Dart 3.7+;
- CMake 3.20+;
- C11 compiler;
- C++20 compiler.

Linux/macOS:

```bash
cmake -S native -B build/native \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORA_BUILD_TESTS=ON \
  -DTENSORA_BUILD_BENCHMARKS=ON
cmake --build build/native --config Release --parallel
ctest --test-dir build/native --build-config Release --output-on-failure
```

Set `TENSORA_NATIVE_LIBRARY` to the resulting shared library before running Dart integration code. Windows multi-config builds normally place the DLL under `build/native/Release/`.

See [Development](docs/DEVELOPMENT.md) and [Testing](docs/TESTING.md) for optional-backend and validation workflows.

## Engineering principles

1. Correctness before breadth.
2. Dart-first API, native numerical execution.
3. Backend-neutral public contracts.
4. Deterministic ownership for native resources.
5. Explicit data movement and observable fallback.
6. Real hardware evidence before hardware support claims.
7. Reproducible tests and benchmarks before performance claims.
8. Structured failures at every native boundary.
9. Dependency-light core builds remain first-class.
10. Production behavior matters more than demo breadth.

## Roadmap

The long-term roadmap covers model packaging, Flutter runtime integration, vision/audio/text pipelines, production edge features, broader training APIs, and compiler/runtime optimization. Multi-vendor accelerator work is a cross-cutting expansion of the training and inference layers; it does not redefine the `.tmodel` milestone numbering in [ROADMAP.md](ROADMAP.md).

## Contributing

Start with:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [Testing Strategy](docs/TESTING.md)
- [Compatibility Policy](docs/COMPATIBILITY.md)
- [Architecture](docs/ARCHITECTURE.md)
- [RFC Process](docs/RFC_PROCESS.md)
- [Security Policy](SECURITY.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
