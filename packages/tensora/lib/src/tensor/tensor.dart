import '../device/device.dart';
import '../dtype/dtype.dart';
import '../errors/tensora_exception.dart';
import '../native/native_runtime.dart';
import '../native/native_training_runtime.dart';
import '../shape/shape.dart';
import 'native_adoption.dart';

final Finalizer<int> _tensorFinalizer = Finalizer<int>((handle) {
  NativeRuntime.instance.releaseFromFinalizer(handle);
});

/// An immutable native-backed tensor.
///
/// Tensor wrappers are isolate-local. Create or reconstruct a Tensor inside the
/// isolate that will own and use it rather than sending the native handle
/// wrapper through an isolate port.
@pragma('vm:isolate-unsendable')
final class Tensor {
  Tensor._(
    this._handle, {
    required this.shape,
    required this.dtype,
    required this.device,
  }) {
    _tensorFinalizer.attach(this, _handle, detach: this);
  }

  /// Copies host values once into Tensora-owned native storage.
  factory Tensor.fromList(
    List<num> values, {
    required Shape shape,
    DType dtype = DType.float32,
    Device device = Device.cpu,
  }) {
    _validateCreation(dtype: dtype, device: device, operation: 'fromList');
    if (values.length != shape.numel) {
      throw InvalidShapeException(
        'Input contains ${values.length} values, but $shape requires '
        '${shape.numel}.',
        operation: 'tensor.fromList',
      );
    }

    final handle = NativeRuntime.instance.createFromList(values, shape);
    return _adopt(handle);
  }

  /// Creates a native float32 tensor initialized to zero.
  factory Tensor.zeros(
    Shape shape, {
    DType dtype = DType.float32,
    Device device = Device.cpu,
  }) => Tensor.full(shape, 0, dtype: dtype, device: device);

  /// Creates a native float32 tensor initialized to one.
  factory Tensor.ones(
    Shape shape, {
    DType dtype = DType.float32,
    Device device = Device.cpu,
  }) => Tensor.full(shape, 1, dtype: dtype, device: device);

  /// Creates a native float32 tensor filled with [value].
  factory Tensor.full(
    Shape shape,
    num value, {
    DType dtype = DType.float32,
    Device device = Device.cpu,
  }) {
    _validateCreation(dtype: dtype, device: device, operation: 'full');
    final handle = NativeRuntime.instance.full(shape, value.toDouble());
    return _adopt(handle);
  }

  int _handle;
  bool _disposed = false;

  /// Immutable dimensions read from the native tensor at creation time.
  final Shape shape;

  /// Numerical data type.
  final DType dtype;

  /// Native execution/storage device.
  final Device device;

  /// Number of tensor elements.
  int get numel => shape.numel;

  /// Whether deterministic native release has completed.
  bool get isDisposed => _disposed;

  /// Whether this Tensor currently participates as an autograd value.
  bool get requiresGrad {
    _ensureLive('requiresGrad');
    return NativeTrainingRuntime.instance.requiresGrad(_handle);
  }

  /// Returns an independent tensor on [target].
  Tensor to(Device target) {
    _ensureLive('to');
    return _adopt(NativeRuntime.instance.toDevice(_handle, target));
  }

  /// Returns a detached leaf tensor with the requested autograd state.
  Tensor withRequiresGrad([bool value = true]) {
    _ensureLive('withRequiresGrad');
    return _adopt(
      NativeTrainingRuntime.instance.withRequiresGrad(_handle, value),
    );
  }

  /// Returns an independent contiguous tensor with [newShape].
  Tensor reshape(Shape newShape) {
    _ensureLive('reshape');
    final handle = NativeRuntime.instance.reshape(_handle, newShape);
    return _adopt(handle);
  }

  /// Returns the 2D transpose as an independent contiguous tensor.
  Tensor transpose() {
    _ensureLive('transpose');
    return _adopt(NativeRuntime.instance.transpose2D(_handle));
  }

  /// Elementwise addition. The current core contract requires equal shapes.
  Tensor add(Tensor other) {
    _ensureLive('add');
    other._ensureLive('add');
    return _adopt(NativeRuntime.instance.add(_handle, other._handle));
  }

  /// Elementwise multiplication. The current core contract requires equal shapes.
  Tensor multiply(Tensor other) {
    _ensureLive('multiply');
    other._ensureLive('multiply');
    return _adopt(NativeRuntime.instance.multiply(_handle, other._handle));
  }

  /// Reduces all elements into a rank-zero scalar Tensor.
  Tensor sum() {
    _ensureLive('sum');
    return _adopt(NativeRuntime.instance.sum(_handle));
  }

  /// Matrix multiplication for rank-2 float32 tensors.
  Tensor matmul(Tensor other) {
    _ensureLive('matmul');
    other._ensureLive('matmul');
    return _adopt(NativeRuntime.instance.matmul(_handle, other._handle));
  }

  /// Applies ReLU through the native training backend.
  Tensor relu() {
    _ensureLive('relu');
    return _adopt(NativeTrainingRuntime.instance.relu(_handle));
  }

  /// Applies sigmoid through the native training backend.
  Tensor sigmoid() {
    _ensureLive('sigmoid');
    return _adopt(NativeTrainingRuntime.instance.sigmoid(_handle));
  }

  /// Applies tanh through the native training backend.
  Tensor tanh() {
    _ensureLive('tanh');
    return _adopt(NativeTrainingRuntime.instance.tanh(_handle));
  }

  /// Runs reverse-mode autograd from this scalar loss Tensor.
  void backward() {
    _ensureLive('backward');
    NativeTrainingRuntime.instance.backward(_handle);
  }

  /// Returns a snapshot of this Tensor's accumulated gradient.
  Tensor grad() {
    _ensureLive('grad');
    return _adopt(NativeTrainingRuntime.instance.grad(_handle));
  }

  /// Explicitly copies all native float32 values into Dart memory.
  List<double> toList() {
    _ensureLive('toList');
    return NativeRuntime.instance.copyToHost(_handle, numel);
  }

  /// Deterministically releases this Tensor's native reference.
  void dispose() {
    if (_disposed) return;

    NativeRuntime.instance.release(_handle);
    _tensorFinalizer.detach(this);
    _handle = 0;
    _disposed = true;
  }

  /// @nodoc
  static Tensor adoptNativeHandleForRuntime(int handle, Object capability) {
    if (!identical(capability, nativeTensorAdoptionToken)) {
      throw StateError('Native tensor adoption capability is invalid.');
    }
    return _adopt(handle);
  }

  /// @nodoc
  int nativeHandleForRuntime(Object capability) {
    if (!identical(capability, nativeTensorAdoptionToken)) {
      throw StateError('Native tensor access capability is invalid.');
    }
    _ensureLive('nativeHandle');
    return _handle;
  }

  static Tensor _adopt(int handle) {
    final runtime = NativeRuntime.instance;
    try {
      final shape = runtime.shape(handle);
      final dtype = runtime.dtype(handle);
      final device = runtime.device(handle);
      final nativeNumel = runtime.numel(handle);
      if (nativeNumel != shape.numel) {
        throw NativeRuntimeException(
          'Native tensor metadata is inconsistent: shape numel ${shape.numel}, '
          'reported numel $nativeNumel.',
          operation: 'tensor.adopt',
        );
      }
      return Tensor._(handle, shape: shape, dtype: dtype, device: device);
    } catch (_) {
      runtime.releaseFromFinalizer(handle);
      rethrow;
    }
  }

  static void _validateCreation({
    required DType dtype,
    required Device device,
    required String operation,
  }) {
    if (dtype != DType.float32) {
      throw UnsupportedOperationException(
        'Tensor creation currently supports only DType.float32.',
        operation: 'tensor.$operation',
      );
    }
    if (device != Device.cpu) {
      throw UnsupportedOperationException(
        'Create host tensors on Device.cpu and transfer them explicitly.',
        operation: 'tensor.$operation',
      );
    }
  }

  void _ensureLive(String operation) {
    if (_disposed) {
      throw DisposedTensorException(
        'Tensor has already been disposed.',
        operation: 'tensor.$operation',
      );
    }
  }

  @override
  String toString() {
    final state = _disposed ? ', disposed' : '';
    return 'Tensor(shape: $shape, dtype: $dtype, device: $device$state)';
  }
}
