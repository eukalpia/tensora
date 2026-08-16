import 'package:tensora/tensora.dart' as core;

/// Canonical NN parameter type, implemented in the Tensora foundation so both
/// `tensora_nn` and `tensora_optim` can share it without a package cycle.
typedef Parameter = core.Parameter;

/// A registered non-trainable tensor state value.
final class Buffer {
  factory Buffer.fromTensor(core.Tensor tensor, {bool persistent = true}) {
    return Buffer._(
      tensor,
      persistent: persistent,
      identity: core.NativeTensorState.identity(tensor),
    );
  }

  Buffer._(this._tensor, {required this.persistent, required this.identity});

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
    return core.NativeTensorState.cloneDetached(_tensor);
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

  final Parameter parameter;
  final String name;
}

final class NamedBuffer {
  const NamedBuffer(this.name, this.buffer);

  final String name;
  final Buffer buffer;
}
