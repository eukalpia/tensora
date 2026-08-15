import 'package:tensora/tensora.dart' as core;

/// A trainable native-backed tensor registered in a [Module] tree.
final class Parameter {
  /// @nodoc
  Parameter.fromTensor(core.Tensor tensor, {int? identity})
    : _tensor = tensor,
      identity = identity ?? identityHashCode(tensor);

  core.Tensor _tensor;
  bool _disposed = false;

  /// Stable parameter identity used for traversal and optimizer de-duplication.
  ///
  /// NN V2 upgrades this to native Tensor identity when the corresponding ABI
  /// lands. Callers must treat the value as opaque.
  final int identity;

  core.Shape get shape {
    _ensureLive('shape');
    return _tensor.shape;
  }

  core.DType get dtype {
    _ensureLive('dtype');
    return _tensor.dtype;
  }

  core.Device get device {
    _ensureLive('device');
    return _tensor.device;
  }

  bool get requiresGrad {
    _ensureLive('requiresGrad');
    return _tensor.requiresGrad;
  }

  bool get isDisposed => _disposed;

  /// Returns an independently owned snapshot of the current gradient.
  core.Tensor grad() {
    _ensureLive('grad');
    return _tensor.grad();
  }

  /// Creates an independently owned detached native snapshot.
  core.Tensor snapshot() {
    _ensureLive('snapshot');
    return _tensor.withRequiresGrad(false);
  }

  /// @nodoc
  core.Tensor get tensorForRuntime {
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
      throw core.DisposedTensorException(
        'Parameter has already been disposed.',
        operation: 'parameter.$operation',
      );
    }
  }
}

/// A registered non-trainable tensor state value.
final class Buffer {
  /// @nodoc
  Buffer.fromTensor(
    core.Tensor tensor, {
    this.persistent = true,
    int? identity,
  }) : _tensor = tensor,
       identity = identity ?? identityHashCode(tensor);

  final core.Tensor _tensor;
  bool _disposed = false;

  final int identity;
  final bool persistent;

  core.Shape get shape {
    _ensureLive('shape');
    return _tensor.shape;
  }

  core.DType get dtype {
    _ensureLive('dtype');
    return _tensor.dtype;
  }

  core.Device get device {
    _ensureLive('device');
    return _tensor.device;
  }

  bool get isDisposed => _disposed;

  core.Tensor snapshot() {
    _ensureLive('snapshot');
    return _tensor.withRequiresGrad(false);
  }

  /// @nodoc
  core.Tensor get tensorForRuntime {
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
      throw core.DisposedTensorException(
        'Buffer has already been disposed.',
        operation: 'buffer.$operation',
      );
    }
  }
}

final class NamedParameter {
  const NamedParameter(this.name, this.parameter);

  final String name;
  final Parameter parameter;
}

final class NamedBuffer {
  const NamedBuffer(this.name, this.buffer);

  final String name;
  final Buffer buffer;
}
