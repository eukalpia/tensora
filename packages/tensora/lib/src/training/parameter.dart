import '../device/device.dart';
import '../dtype/dtype.dart';
import '../errors/tensora_exception.dart';
import '../shape/shape.dart';
import '../tensor/tensor.dart';

/// Safe retained reference to a trainable native-backed tensor.
///
/// This foundation type is intentionally independent from `tensora_nn` so
/// optimizers and neural-network composition can both depend only on `tensora`.
final class Parameter {
  /// Creates a parameter reference that owns [tensor].
  ///
  /// Framework code normally obtains these from native module parameter views.
  Parameter.fromTensor(Tensor tensor, {int? identity})
    : _tensor = tensor,
      identity = identity ?? identityHashCode(tensor);

  final Tensor _tensor;
  bool _disposed = false;

  /// Opaque stable identity used for de-duplication.
  ///
  /// The NN V2 native-identity ABI replaces the wrapper identity fallback while
  /// preserving this public contract.
  final int identity;

  Shape get shape {
    _ensureLive('shape');
    return _tensor.shape;
  }

  DType get dtype {
    _ensureLive('dtype');
    return _tensor.dtype;
  }

  Device get device {
    _ensureLive('device');
    return _tensor.device;
  }

  bool get requiresGrad {
    _ensureLive('requiresGrad');
    return _tensor.requiresGrad;
  }

  bool get isDisposed => _disposed;

  Tensor grad() {
    _ensureLive('grad');
    return _tensor.grad();
  }

  /// Returns an independently owned detached native state snapshot.
  Tensor snapshot() {
    _ensureLive('snapshot');
    return _tensor.withRequiresGrad(false);
  }

  /// @nodoc
  Tensor get tensorForRuntime {
    _ensureLive('runtime');
    return _tensor;
  }

  void dispose() {
    if (_disposed) return;
    _tensor.dispose();
    _disposed = true;
  }

  void _ensureLive(String operation) {
    if (_disposed) {
      throw DisposedTensorException(
        'Parameter has already been disposed.',
        operation: 'parameter.$operation',
      );
    }
  }
}
