import '../device/device.dart';
import '../dtype/dtype.dart';
import '../errors/tensora_exception.dart';
import '../native/native_training_runtime.dart';
import '../shape/shape.dart';
import '../tensor/native_adoption.dart';
import '../tensor/tensor.dart';

/// Safe retained reference to a trainable native-backed tensor.
///
/// This foundation type is intentionally independent from `tensora_nn` so
/// optimizers and neural-network composition can both depend only on `tensora`.
final class Parameter {
  /// Creates a parameter reference that owns [tensor].
  factory Parameter.fromTensor(Tensor tensor) {
    final handle = tensor.nativeHandleForRuntime(nativeTensorAdoptionToken);
    final identity = NativeTrainingRuntime.instance.tensorIdentity(handle);
    return Parameter._(tensor, identity);
  }

  Parameter._(this._tensor, this.identity);

  Tensor _tensor;
  bool _disposed = false;

  /// Opaque stable identity used for de-duplication across retained wrappers.
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

  /// Freezes this parameter in-place while preserving its native identity.
  void freeze() => _setRequiresGrad(false);

  /// Unfreezes this parameter in-place while preserving its native identity.
  void unfreeze() => _setRequiresGrad(true);

  Tensor grad() {
    _ensureLive('grad');
    return _tensor.grad();
  }

  /// Returns an independently owned detached native state snapshot.
  Tensor snapshot() {
    _ensureLive('snapshot');
    final handle = _tensor.nativeHandleForRuntime(nativeTensorAdoptionToken);
    final cloned = NativeTrainingRuntime.instance.cloneDetached(handle);
    return Tensor.adoptNativeHandleForRuntime(
      cloned,
      nativeTensorAdoptionToken,
    );
  }

  /// @nodoc
  Tensor get tensorForRuntime {
    _ensureLive('runtime');
    return _tensor;
  }

  /// Replaces the retained Tensor view after an in-place native module move.
  ///
  /// The replacement must resolve to the same opaque parameter identity.
  /// Ownership of [replacement] transfers to this Parameter on success.
  /// @nodoc
  void internalReplaceTensor(Tensor replacement) {
    _ensureLive('replaceTensor');
    final replacementHandle = replacement.nativeHandleForRuntime(
      nativeTensorAdoptionToken,
    );
    final replacementIdentity = NativeTrainingRuntime.instance.tensorIdentity(
      replacementHandle,
    );
    if (replacementIdentity != identity) {
      replacement.dispose();
      throw NativeRuntimeException(
        'Native module move changed parameter identity.',
        operation: 'parameter.replaceTensor',
      );
    }

    final previous = _tensor;
    _tensor = replacement;
    previous.dispose();
  }

  void dispose() {
    if (_disposed) return;
    _tensor.dispose();
    _disposed = true;
  }

  void _setRequiresGrad(bool value) {
    _ensureLive(value ? 'unfreeze' : 'freeze');
    final handle = _tensor.nativeHandleForRuntime(nativeTensorAdoptionToken);
    NativeTrainingRuntime.instance.setRequiresGrad(handle, value);
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
