# tensora

The core Dart API for Tensora's native tensor runtime.

Milestone 1 supports native CPU-backed `float32` tensors and a compact eager operation set. The Dart object stores metadata plus an opaque native handle; numerical payloads live in Tensora-owned native memory.

## Requirements

- Dart 3.7+
- a built Tensora native runtime for the current desktop platform

During source development, point the Dart bridge at the native shared library:

```bash
export TENSORA_NATIVE_LIBRARY=/absolute/path/to/libtensora_native.so
```

Use `.dylib` on macOS and the generated `tensora_native.dll` on Windows.

## Example

```dart
import 'package:tensora/tensora.dart';

void main() {
  final a = Tensor.ones(Shape([2, 3]));
  final b = Tensor.full(Shape([3, 4]), 2.0);
  final c = a.matmul(b);

  print(c.shape);    // Shape([2, 4])
  print(c.toList()); // explicit native -> Dart copy

  c.dispose();
  b.dispose();
  a.dispose();
}
```

## Implemented operations

- `Tensor.fromList`
- `Tensor.zeros`
- `Tensor.ones`
- `Tensor.full`
- `reshape`
- 2D `transpose`
- equal-shape `add`
- equal-shape `multiply`
- `sum`
- 2D `matmul`
- explicit `toList`
- deterministic `dispose`

Only `DType.float32` and `Device.cpu` are implemented in Milestone 1.

## Ownership

Every `Tensor` wrapper owns one native handle reference. Call `dispose()` when the tensor is no longer needed. Calling `dispose()` twice is safe; using a successfully disposed tensor throws `DisposedTensorException` before native memory is touched.

A Dart finalizer exists only as a fallback for missed cleanup. Applications should not use garbage collection as the resource-lifetime strategy.

Tensor wrappers are isolate-local in Milestone 1. They are intentionally unsendable through Dart isolate ports. Recreate/import tensor state inside the destination isolate instead.

## Data movement

`Tensor.fromList` performs one explicit host-to-native import. `toList()` performs an explicit native-to-Dart extraction. `reshape`, `transpose`, elementwise operations, reductions, and `matmul` operate on native tensors without routing bulk numerical payloads through Dart.

Milestone 1 `reshape` and `transpose` return independent contiguous tensors rather than views.

## Errors

Native status values are mapped into typed Dart exceptions including:

- `InvalidArgumentException`
- `InvalidShapeException`
- `OutOfMemoryException`
- `UnsupportedOperationException`
- `NativeRuntimeException`
- `DisposedTensorException`

The bridge validates the native ABI version before creating tensors.

## Not implemented yet

CUDA, autograd, neural-network modules, ONNX, Flutter runtime integration, `.tmodel`, additional dtypes, broadcasting, tensor views, and cross-isolate tensor sharing are intentionally outside Milestone 1.
