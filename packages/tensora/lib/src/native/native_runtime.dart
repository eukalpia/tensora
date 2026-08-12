import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

import '../device/device.dart';
import '../dtype/dtype.dart';
import '../errors/tensora_exception.dart';
import '../shape/shape.dart';
import 'native_bindings.dart';

final class NativeRuntime {
  NativeRuntime._(this._bindings, this.libraryPath) {
    final version = _bindings.abiVersion();
    if (version != expectedAbiVersion) {
      throw NativeRuntimeException(
        'Incompatible Tensora native ABI: expected $expectedAbiVersion, '
        'loaded $version from $libraryPath.',
        operation: 'runtime.load',
      );
    }
  }

  static const int expectedAbiVersion = 1;

  static NativeRuntime? _instance;

  static NativeRuntime get instance => _instance ??= _load();

  final NativeBindings _bindings;
  final String libraryPath;

  static NativeRuntime _load() {
    final path = _resolveLibraryPath();
    try {
      return NativeRuntime._(
        NativeBindings(DynamicLibrary.open(path)),
        path,
      );
    } on TensoraException {
      rethrow;
    } catch (error) {
      throw NativeRuntimeException(
        'Unable to load Tensora native runtime at "$path": $error',
        operation: 'runtime.load',
      );
    }
  }

  static String _resolveLibraryPath() {
    final override = Platform.environment['TENSORA_NATIVE_LIBRARY'];
    if (override != null && override.trim().isNotEmpty) {
      return override;
    }

    if (Platform.isLinux) return 'libtensora_native.so';
    if (Platform.isMacOS) return 'libtensora_native.dylib';
    if (Platform.isWindows) return 'tensora_native.dll';

    throw UnsupportedOperationException(
      'Milestone 1 native runtime discovery supports Linux, macOS, and Windows.',
      operation: 'runtime.load',
    );
  }

  int createFromList(List<num> values, Shape shape) {
    final data = calloc<Float>(values.length);
    try {
      for (var index = 0; index < values.length; index++) {
        data[index] = values[index].toDouble();
      }
      return _withDimensions(shape, (dims, rank) {
        return _newHandle(
          'tensor.fromList',
          (out) => _bindings.tensorFromF32(data, dims, rank, out),
        );
      });
    } finally {
      calloc.free(data);
    }
  }

  int full(Shape shape, double value) {
    return _withDimensions(shape, (dims, rank) {
      return _newHandle(
        'tensor.full',
        (out) => _bindings.tensorFullF32(dims, rank, value, out),
      );
    });
  }

  int reshape(int handle, Shape shape) {
    return _withDimensions(shape, (dims, rank) {
      return _newHandle(
        'tensor.reshape',
        (out) => _bindings.tensorReshape(handle, dims, rank, out),
      );
    });
  }

  int transpose2D(int handle) => _newHandle(
        'tensor.transpose',
        (out) => _bindings.tensorTranspose2D(handle, out),
      );

  int add(int left, int right) => _newHandle(
        'tensor.add',
        (out) => _bindings.tensorAdd(left, right, out),
      );

  int multiply(int left, int right) => _newHandle(
        'tensor.multiply',
        (out) => _bindings.tensorMultiply(left, right, out),
      );

  int sum(int handle) => _newHandle(
        'tensor.sum',
        (out) => _bindings.tensorSum(handle, out),
      );

  int matmul(int left, int right) => _newHandle(
        'tensor.matmul',
        (out) => _bindings.tensorMatmul(left, right, out),
      );

  Shape shape(int handle) {
    final rankPointer = calloc<Size>();
    try {
      _check(_bindings.tensorRank(handle, rankPointer), 'tensor.rank');
      final rank = rankPointer.value;
      if (rank > Shape.maxRank) {
        throw NativeRuntimeException(
          'Native runtime returned rank $rank above ${Shape.maxRank}.',
          operation: 'tensor.shape',
        );
      }

      if (rank == 0) {
        final returnedRank = calloc<Size>();
        try {
          _check(
            _bindings.tensorShape(
              handle,
              nullptr.cast<Int64>(),
              0,
              returnedRank,
            ),
            'tensor.shape',
          );
          if (returnedRank.value != 0) {
            throw NativeRuntimeException(
              'Native tensor rank changed while reading metadata.',
              operation: 'tensor.shape',
            );
          }
          return Shape(const []);
        } finally {
          calloc.free(returnedRank);
        }
      }

      final dims = calloc<Int64>(rank);
      final returnedRank = calloc<Size>();
      try {
        _check(
          _bindings.tensorShape(handle, dims, rank, returnedRank),
          'tensor.shape',
        );
        if (returnedRank.value != rank) {
          throw NativeRuntimeException(
            'Native tensor rank changed while reading metadata.',
            operation: 'tensor.shape',
          );
        }
        final values = List<int>.generate(rank, (index) => dims[index]);
        try {
          return Shape(values);
        } on ArgumentError catch (error) {
          throw NativeRuntimeException(
            'Native runtime returned invalid shape metadata: $error',
            operation: 'tensor.shape',
          );
        }
      } finally {
        calloc.free(returnedRank);
        calloc.free(dims);
      }
    } finally {
      calloc.free(rankPointer);
    }
  }

  DType dtype(int handle) {
    final value = calloc<Uint32>();
    try {
      _check(_bindings.tensorDType(handle, value), 'tensor.dtype');
      return switch (value.value) {
        1 => DType.float32,
        final code => throw NativeRuntimeException(
            'Native runtime returned unknown dtype code $code.',
            operation: 'tensor.dtype',
          ),
      };
    } finally {
      calloc.free(value);
    }
  }

  Device device(int handle) {
    final value = calloc<Uint32>();
    try {
      _check(_bindings.tensorDevice(handle, value), 'tensor.device');
      return switch (value.value) {
        1 => Device.cpu,
        final code => throw NativeRuntimeException(
            'Native runtime returned unknown device code $code.',
            operation: 'tensor.device',
          ),
      };
    } finally {
      calloc.free(value);
    }
  }

  int numel(int handle) {
    final value = calloc<Uint64>();
    try {
      _check(_bindings.tensorNumel(handle, value), 'tensor.numel');
      return value.value;
    } finally {
      calloc.free(value);
    }
  }

  List<double> copyToHost(int handle, int numel) {
    final values = calloc<Float>(numel);
    final written = calloc<Size>();
    try {
      _check(
        _bindings.tensorCopyToHostF32(handle, values, numel, written),
        'tensor.toList',
      );
      if (written.value != numel) {
        throw NativeRuntimeException(
          'Native copy wrote ${written.value} elements; expected $numel.',
          operation: 'tensor.toList',
        );
      }
      return List<double>.generate(numel, (index) => values[index]);
    } finally {
      calloc.free(written);
      calloc.free(values);
    }
  }

  void retain(int handle) {
    _check(_bindings.tensorRetain(handle), 'tensor.retain');
  }

  void release(int handle) {
    _check(_bindings.tensorRelease(handle), 'tensor.dispose');
  }

  /// Finalizer-only best effort. It intentionally does not throw from a GC hook.
  void releaseFromFinalizer(int handle) {
    _bindings.tensorRelease(handle);
  }

  void noop() {
    _check(_bindings.noop(), 'runtime.noop');
  }

  int liveTensorCount() {
    final value = calloc<Uint64>();
    try {
      _check(
        _bindings.runtimeLiveTensorCount(value),
        'runtime.liveTensorCount',
      );
      return value.value;
    } finally {
      calloc.free(value);
    }
  }

  int liveStorageBytes() {
    final value = calloc<Uint64>();
    try {
      _check(
        _bindings.runtimeLiveStorageBytes(value),
        'runtime.liveStorageBytes',
      );
      return value.value;
    } finally {
      calloc.free(value);
    }
  }

  int _newHandle(
    String operation,
    int Function(Pointer<Uint64> out) call,
  ) {
    final output = calloc<Uint64>();
    try {
      _check(call(output), operation);
      final handle = output.value;
      if (handle == 0) {
        throw NativeRuntimeException(
          'Native runtime returned a null tensor handle on success.',
          operation: operation,
        );
      }
      return handle;
    } finally {
      calloc.free(output);
    }
  }

  T _withDimensions<T>(
    Shape shape,
    T Function(Pointer<Int64> dims, int rank) operation,
  ) {
    if (shape.rank == 0) {
      return operation(nullptr.cast<Int64>(), 0);
    }

    final dims = calloc<Int64>(shape.rank);
    try {
      for (var index = 0; index < shape.rank; index++) {
        dims[index] = shape.dimensions[index];
      }
      return operation(dims, shape.rank);
    } finally {
      calloc.free(dims);
    }
  }

  void _check(int status, String operation) {
    if (status == 0) return;

    final errorPointer = _bindings.lastErrorMessage();
    final message = errorPointer.address == 0
        ? 'Native runtime returned status $status without a diagnostic.'
        : errorPointer.toDartString();

    switch (status) {
      case 1:
        throw InvalidArgumentException(message, operation: operation);
      case 2:
        throw InvalidShapeException(message, operation: operation);
      case 3:
        throw OutOfMemoryException(message, operation: operation);
      case 4:
        throw UnsupportedOperationException(message, operation: operation);
      case 5:
        throw NativeRuntimeException(message, operation: operation);
      case 6:
        throw NativeRuntimeException(message, operation: operation);
      default:
        throw NativeRuntimeException(
          'Unknown native status $status: $message',
          operation: operation,
        );
    }
  }
}
