# tensora

Core Dart API for the Tensora native machine-learning runtime.

The package is pre-1.0 and currently exposes native-backed float32 tensors, optional LibTorch training, optional ONNX Runtime inference, explicit device/provider selection, and deterministic native resource ownership.

## Requirements

- Dart 3.7+
- a Tensora native runtime built for the current desktop platform
- optional native backend dependencies when training or ONNX inference is enabled

During source development, point the Dart bridge at the native shared library:

```bash
export TENSORA_NATIVE_LIBRARY=/absolute/path/to/libtensora_native.so
```

Use `.dylib` on macOS and an explicit `tensora_native.dll` path on Windows.

## Tensor API

Implemented public tensor features include:

- `Tensor.fromList`
- `Tensor.zeros`
- `Tensor.ones`
- `Tensor.full`
- `Tensor.to(Device)`
- `reshape`
- 2D `transpose`
- equal-shape `add`
- equal-shape `multiply`
- `sum`
- 2D `matmul`
- `toList()`
- deterministic `dispose()`

Only `DType.float32` is currently public.

Devices are represented as:

- `Device.cpu`
- `Device.cuda(index)`
- `Device.mps`
- `Device.xpu(index)`
- `Device.hip(index)`

Tensor factories default to CPU. Passing `device:` stages host-created values in CPU memory and then transfers them to the requested device. This data movement is explicit in the API contract even though it is performed inside the factory implementation.

## Device discovery

```dart
final devices = TensoraRuntime.availableDevices;
final device = TensoraRuntime.preferredDevice;
```

`availableDevices` reports devices visible to the loaded runtime. `preferredDevice` chooses the first visible accelerator according to Tensora's deterministic preference order, otherwise CPU.

Using `preferredDevice` is opt-in. Ordinary tensor construction remains CPU by default.

## Training

A LibTorch-enabled native build adds:

- `requiresGrad` and gradients;
- `backward()`;
- ReLU, sigmoid, tanh;
- MSE and cross-entropy;
- `Linear`;
- train/eval mode;
- parameter and buffer views;
- module device transfer;
- SGD, Adam, AdamW;
- checkpoint save/load.

Example:

> These are the thin native handle owners. Most applications should use the
> composable `Module` API in `package:tensora_nn` and the optimizers in
> `package:tensora_optim`, which build on them.

```dart
final device = TensoraRuntime.preferredDevice;
final x = Tensor.fromList([-1, 0, 1, 2], shape: Shape([4, 1]), device: device);
final y = Tensor.fromList([-1, 1, 3, 5], shape: Shape([4, 1]), device: device);
final model = NativeLinear(1, 1)..to(device);
final optimizer = NativeSgd(model, learningRate: 0.1);

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
```

## ONNX inference

An ONNX Runtime-enabled build exposes `OnnxSession` and these provider preferences:

- `auto`
- CPU
- CUDA
- DirectML
- CoreML
- OpenVINO
- MIGraphX

Explicit provider requests do not silently fall back. Use `OnnxExecutionProvider.auto` when controlled platform-aware fallback is desired.

The current portable ONNX binding copies input tensor data to host memory before creating ONNX Runtime input values. An accelerated provider therefore does not imply zero-copy input binding.

## Ownership

Every native-backed wrapper owns one native handle reference. Call `dispose()` deterministically when the object is no longer needed.

- double-dispose is safe for public wrappers;
- use-after-dispose fails before native work;
- finalizers are only a backup for missed cleanup;
- tensor/module/optimizer/session handles are typed and cannot be substituted for one another;
- tensor wrappers are isolate-local.

## Windows dependency layout

When an explicit existing `tensora_native.dll` path is used, Tensora loads the DLL with a restricted dependency search that prioritizes sidecar libraries from the same directory.

Place optional backend DLLs beside `tensora_native.dll`. This applies to source-built Windows layouts using LibTorch or ONNX Runtime. Do not rely on an unrelated system DLL or an ambient PATH entry to provide a backend dependency.

## Current validation

- CPU core: Linux, macOS, Windows hosted CI
- LibTorch CPU training: Linux, macOS, Windows hosted CI
- Apple MPS training: real Apple Silicon hosted hardware CI
- ONNX CPU inference: Linux and Windows hosted CI
- CoreML inference: real Apple Silicon hosted hardware CI
- CUDA/XPU/HIP and other accelerator providers: implementation exists, physical qualification remains required before support promotion

The normative support matrix lives in `docs/COMPATIBILITY.md` in the repository root.

## Errors

Native status values are translated into typed Dart errors, including:

- `InvalidArgumentException`
- `InvalidShapeException`
- `OutOfMemoryException`
- `UnsupportedOperationException`
- `NativeRuntimeException`
- `ModelRuntimeException`
- `DisposedTensorException`

The Dart bridge validates native ABI version **4** before using the runtime.

## Not implemented yet

- additional public dtypes;
- Flutter/mobile runtime integration;
- `.tmodel` packaging/runtime;
- universal broadcasting/views;
- device-zero-copy ONNX input binding;
- a published native binary distribution contract.
