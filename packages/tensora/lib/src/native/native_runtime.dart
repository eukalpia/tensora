import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

import '../device/device.dart';
import '../dtype/dtype.dart';
import '../errors/tensora_exception.dart';
import '../shape/shape.dart';
import 'native_bindings.dart';
import 'native_device_codec.dart';
import 'native_library_path.dart';
import 'native_windows_loader_state.dart';

typedef _LoadLibraryExWNative =
    Pointer<Void> Function(
      Pointer<Utf16> fileName,
      Pointer<Void> file,
      Uint32 flags,
    );
typedef _LoadLibraryExWDart =
    Pointer<Void> Function(
      Pointer<Utf16> fileName,
      Pointer<Void> file,
      int flags,
    );
typedef _GetLastErrorNative = Uint32 Function();
typedef _GetLastErrorDart = int Function();

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

  static const int expectedAbiVersion = 4;
  static const int _loadLibrarySearchDllLoadDir = 0x00000100;
  static const int _loadLibrarySearchDefaultDirs = 0x00001000;

  static NativeRuntime? _instance;

  static NativeRuntime get instance => _instance ??= _load();

  final NativeBindings _bindings;
  final String libraryPath;

  static NativeRuntime _load() {
    final path = _resolveLibraryPath();
    try {
      return NativeRuntime._(NativeBindings(_openLibrary(path)), path);
    } on TensoraException {
      rethrow;
    } catch (error) {
      throw NativeRuntimeException(
        'Unable to load Tensora native runtime at "$path": $error',
        operation: 'runtime.load',
      );
    }
  }

  // coverage:ignore-start
  // This branch is exercised end-to-end by Windows FFI/training/inference CI.
  // Linux High Assurance cannot execute Win32 loader APIs.
  static DynamicLibrary _openLibrary(String path) {
    if (!Platform.isWindows) {
      return DynamicLibrary.open(path);
    }

    final file = File(path);
    if (!file.existsSync()) {
      return DynamicLibrary.open(path);
    }

    final absolutePath = file.absolute.path;
    _preloadWindowsLibrary(absolutePath);
    return DynamicLibrary.open(absolutePath);
  }

  static void _preloadWindowsLibrary(String absolutePath) {
    final kernel32 = DynamicLibrary.open('kernel32.dll');
    final loadLibraryEx = kernel32
        .lookupFunction<_LoadLibraryExWNative, _LoadLibraryExWDart>(
          'LoadLibraryExW',
        );
    final getLastError = kernel32
        .lookupFunction<_GetLastErrorNative, _GetLastErrorDart>('GetLastError');

    final nativePath = absolutePath.toNativeUtf16();
    try {
      final handle = loadLibraryEx(
        nativePath,
        nullptr,
        _loadLibrarySearchDllLoadDir | _loadLibrarySearchDefaultDirs,
      );
      if (handle.address == 0) {
        final errorCode = getLastError();
        throw NativeRuntimeException(
          'Windows could not securely load Tensora and its sidecar '
          'dependencies from "$absolutePath" (Win32 error $errorCode).',
          operation: 'runtime.load',
        );
      }
      windowsPreloadedModules.add(handle);
    } finally {
      calloc.free(nativePath);
    }
  }
  // coverage:ignore-end

  static String _resolveLibraryPath() => resolveNativeLibraryPath(
    environment: Platform.environment,
    operatingSystem: Platform.operatingSystem,
  );

  int createFromList(List<num> values, Shape shape) {
    final data = calloc<Float>(values.length);
    try {
      for (var index = 0; index < values.length; index++) {
        data[index] = values[index].toDouble();
      }
      return _withDimensions(shape, (dims, rank) {
        return _newHandle(
          'tensor.fromList',
          (out) =>
              _bindings.tensorFromF32(data, values.length, dims, rank, out),
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

  int toDevice(int handle, Device device) => _newHandle(
    'tensor.to',
    (out) => _bindings.tensorToDevice(
      handle,
      nativeDeviceCode(device),
      device.index,
      out,
    ),
  );

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

  int add(int left, int right) =>
      _newHandle('tensor.add', (out) => _bindings.tensorAdd(left, right, out));

  int multiply(int left, int right) => _newHandle(
    'tensor.multiply',
    (out) => _bindings.tensorMultiply(left, right, out),
  );

  int sum(int handle) =>
      _newHandle('tensor.sum', (out) => _bindings.tensorSum(handle, out));

  int matmul(int left, int right) => _newHandle(
    'tensor.matmul',
    (out) => _bindings.tensorMatmul(left, right, out),
  );

  List<int> shape(int handle) {
    final rankPointer = calloc<Uint32>();
    try {
      _check('tensor.shape.rank', _bindings.tensorRank(handle, rankPointer));
      final rank = rankPointer.value;
      if (rank > Shape.maxRank) {
        throw NativeRuntimeException(
          'Native tensor rank $rank exceeds the supported maximum '
          '${Shape.maxRank}.',
          operation: 'tensor.shape',
        );
      }
      if (rank == 0) return const <int>[];

      final dims = calloc<Int64>(rank);
      try {
        final actualRankPointer = calloc<Uint32>();
        try {
          _check(
            'tensor.shape',
            _bindings.tensorShape(handle, dims, rank, actualRankPointer),
          );
          final actualRank = actualRankPointer.value;
          if (actualRank != rank) {
            throw NativeRuntimeException(
              'Native tensor rank changed during metadata query: '
              '$rank -> $actualRank.',
              operation: 'tensor.shape',
            );
          }
          return List<int>.generate(rank, (index) => dims[index]);
        } finally {
          calloc.free(actualRankPointer);
        }
      } finally {
        calloc.free(dims);
      }
    } finally {
      calloc.free(rankPointer);
    }
  }

  int numel(int handle) {
    final pointer = calloc<Uint64>();
    try {
      _check('tensor.numel', _bindings.tensorNumel(handle, pointer));
      return pointer.value;
    } finally {
      calloc.free(pointer);
    }
  }

  DType dtype(int handle) {
    final pointer = calloc<Uint32>();
    try {
      _check('tensor.dtype', _bindings.tensorDtype(handle, pointer));
      final value = pointer.value;
      for (final dtype in DType.values) {
        if (dtype.nativeCode == value) return dtype;
      }
      throw NativeRuntimeException(
        'Native tensor returned an unknown dtype code $value.',
        operation: 'tensor.dtype',
      );
    } finally {
      calloc.free(pointer);
    }
  }

  Device device(int handle) {
    final kind = calloc<Uint32>();
    final index = calloc<Int32>();
    try {
      _check('tensor.device', _bindings.tensorDevice(handle, kind, index));
      return decodeNativeDevice(kind.value, index.value);
    } finally {
      calloc.free(kind);
      calloc.free(index);
    }
  }

  int deviceCount(Device device) {
    final pointer = calloc<Uint32>();
    try {
      _check(
        'runtime.deviceCount',
        _bindings.deviceCount(nativeDeviceCode(device), pointer),
      );
      return pointer.value;
    } finally {
      calloc.free(pointer);
    }
  }

  List<Device> get availableDevices {
    final devices = <Device>[Device.cpu];
    for (final kind in const <Device>[
      Device.cuda(0),
      Device.mps,
      Device.xpu(0),
      Device.hip(0),
    ]) {
      final count = deviceCount(kind);
      for (var index = 0; index < count; index++) {
        devices.add(switch (kind.kind) {
          DeviceKind.cpu => Device.cpu,
          DeviceKind.cuda => Device.cuda(index),
          DeviceKind.mps => Device.mps,
          DeviceKind.xpu => Device.xpu(index),
          DeviceKind.hip => Device.hip(index),
        });
      }
    }
    return List<Device>.unmodifiable(devices);
  }

  Device get preferredDevice {
    final devices = availableDevices;
    for (final kind in const <DeviceKind>[
      DeviceKind.cuda,
      DeviceKind.mps,
      DeviceKind.xpu,
      DeviceKind.hip,
    ]) {
      for (final device in devices) {
        if (device.kind == kind) return device;
      }
    }
    return Device.cpu;
  }

  List<double> copyToHost(int handle, int length) {
    final values = calloc<Float>(length);
    final written = calloc<Uint64>();
    try {
      _check(
        'tensor.toList',
        _bindings.tensorCopyToF32(handle, values, length, written),
      );
      if (written.value != length) {
        throw NativeRuntimeException(
          'Native tensor copy wrote ${written.value} values, expected $length.',
          operation: 'tensor.toList',
        );
      }
      return List<double>.generate(length, (index) => values[index]);
    } finally {
      calloc.free(written);
      calloc.free(values);
    }
  }

  void retain(int handle) => _check('tensor.retain', _bindings.retain(handle));

  void release(int handle) => _check('tensor.release', _bindings.release(handle));

  void releaseFromFinalizer(int handle) {
    _bindings.release(handle);
  }

  int get liveTensors {
    final pointer = calloc<Uint64>();
    try {
      _check('runtime.liveTensors', _bindings.liveTensors(pointer));
      return pointer.value;
    } finally {
      calloc.free(pointer);
    }
  }

  int get liveStorageBytes {
    final pointer = calloc<Uint64>();
    try {
      _check('runtime.liveStorageBytes', _bindings.liveStorageBytes(pointer));
      return pointer.value;
    } finally {
      calloc.free(pointer);
    }
  }

  T _withDimensions<T>(
    Shape shape,
    T Function(Pointer<Int64> dims, int rank) operation,
  ) {
    final rank = shape.rank;
    if (rank == 0) {
      return operation(nullptr, 0);
    }
    final dims = calloc<Int64>(rank);
    try {
      for (var index = 0; index < rank; index++) {
        dims[index] = shape.dimensions[index];
      }
      return operation(dims, rank);
    } finally {
      calloc.free(dims);
    }
  }

  int _newHandle(
    String operation,
    int Function(Pointer<Uint64> out) nativeCall,
  ) {
    final out = calloc<Uint64>();
    try {
      final status = nativeCall(out);
      if (status != NativeStatus.ok) {
        _throwStatus(operation, status);
      }
      final handle = out.value;
      if (handle == 0) {
        throw NativeRuntimeException(
          'Native operation succeeded without returning a tensor handle.',
          operation: operation,
        );
      }
      return handle;
    } finally {
      calloc.free(out);
    }
  }

  void _check(String operation, int status) {
    if (status == NativeStatus.ok) return;
    _throwStatus(operation, status);
  }

  Never _throwStatus(String operation, int status) {
    final messagePointer = _bindings.lastError();
    final message = messagePointer == nullptr
        ? 'Native Tensora operation failed without a diagnostic.'
        : messagePointer.toDartString();
    switch (status) {
      case NativeStatus.invalidArgument:
        throw InvalidArgumentException(message, operation: operation);
      case NativeStatus.invalidShape:
        throw InvalidShapeException(message, operation: operation);
      case NativeStatus.invalidHandle:
        throw DisposedTensorException(message, operation: operation);
      case NativeStatus.unsupported:
        throw UnsupportedOperationException(message, operation: operation);
      case NativeStatus.outOfMemory:
        throw OutOfMemoryException(message, operation: operation);
      case NativeStatus.internal:
        throw NativeRuntimeException(message, operation: operation);
      default:
        throw NativeRuntimeException(
          'Unknown native status $status: $message',
          operation: operation,
        );
    }
  }
}
