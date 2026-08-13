import 'dart:ffi';

import 'package:ffi/ffi.dart';

typedef _AbiVersionNative = Uint32 Function();
typedef _AbiVersionDart = int Function();

typedef _LastErrorNative = Pointer<Utf8> Function();
typedef _LastErrorDart = Pointer<Utf8> Function();

typedef _NoopNative = Int32 Function();
typedef _NoopDart = int Function();

typedef _TensorFromF32Native =
    Int32 Function(Pointer<Float>, Size, Pointer<Int64>, Size, Pointer<Uint64>);
typedef _TensorFromF32Dart =
    int Function(Pointer<Float>, int, Pointer<Int64>, int, Pointer<Uint64>);

typedef _TensorFullF32Native =
    Int32 Function(Pointer<Int64>, Size, Float, Pointer<Uint64>);
typedef _TensorFullF32Dart =
    int Function(Pointer<Int64>, int, double, Pointer<Uint64>);

typedef _TensorRankNative = Int32 Function(Uint64, Pointer<Size>);
typedef _TensorRankDart = int Function(int, Pointer<Size>);

typedef _TensorShapeNative =
    Int32 Function(Uint64, Pointer<Int64>, Size, Pointer<Size>);
typedef _TensorShapeDart =
    int Function(int, Pointer<Int64>, int, Pointer<Size>);

typedef _TensorUint32MetadataNative = Int32 Function(Uint64, Pointer<Uint32>);
typedef _TensorUint32MetadataDart = int Function(int, Pointer<Uint32>);

typedef _TensorInt32MetadataNative = Int32 Function(Uint64, Pointer<Int32>);
typedef _TensorInt32MetadataDart = int Function(int, Pointer<Int32>);

typedef _TensorUint64MetadataNative = Int32 Function(Uint64, Pointer<Uint64>);
typedef _TensorUint64MetadataDart = int Function(int, Pointer<Uint64>);

typedef _TensorReshapeNative =
    Int32 Function(Uint64, Pointer<Int64>, Size, Pointer<Uint64>);
typedef _TensorReshapeDart =
    int Function(int, Pointer<Int64>, int, Pointer<Uint64>);

typedef _TensorToDeviceNative =
    Int32 Function(Uint64, Uint32, Int32, Pointer<Uint64>);
typedef _TensorToDeviceDart = int Function(int, int, int, Pointer<Uint64>);

typedef _TensorUnaryNative = Int32 Function(Uint64, Pointer<Uint64>);
typedef _TensorUnaryDart = int Function(int, Pointer<Uint64>);

typedef _TensorBinaryNative = Int32 Function(Uint64, Uint64, Pointer<Uint64>);
typedef _TensorBinaryDart = int Function(int, int, Pointer<Uint64>);

typedef _TensorCopyToHostNative =
    Int32 Function(Uint64, Pointer<Float>, Size, Pointer<Size>);
typedef _TensorCopyToHostDart =
    int Function(int, Pointer<Float>, int, Pointer<Size>);

typedef _TensorLifetimeNative = Int32 Function(Uint64);
typedef _TensorLifetimeDart = int Function(int);

typedef _RuntimeCounterNative = Int32 Function(Pointer<Uint64>);
typedef _RuntimeCounterDart = int Function(Pointer<Uint64>);

typedef _RuntimeUint32Native = Int32 Function(Pointer<Uint32>);
typedef _RuntimeUint32Dart = int Function(Pointer<Uint32>);

typedef _RuntimeDeviceCountNative = Int32 Function(Uint32, Pointer<Uint32>);
typedef _RuntimeDeviceCountDart = int Function(int, Pointer<Uint32>);

final class NativeBindings {
  NativeBindings(DynamicLibrary library)
    : abiVersion = library.lookupFunction<_AbiVersionNative, _AbiVersionDart>(
        'ts_abi_version',
      ),
      lastErrorMessage = library
          .lookupFunction<_LastErrorNative, _LastErrorDart>(
            'ts_last_error_message',
          ),
      noop = library.lookupFunction<_NoopNative, _NoopDart>('ts_noop'),
      runtimeDeviceCount = library
          .lookupFunction<_RuntimeDeviceCountNative, _RuntimeDeviceCountDart>(
            'ts_runtime_device_count',
          ),
      runtimeCudaDeviceCount = library
          .lookupFunction<_RuntimeUint32Native, _RuntimeUint32Dart>(
            'ts_runtime_cuda_device_count',
          ),
      tensorFromF32 = library
          .lookupFunction<_TensorFromF32Native, _TensorFromF32Dart>(
            'ts_tensor_from_f32',
          ),
      tensorFullF32 = library
          .lookupFunction<_TensorFullF32Native, _TensorFullF32Dart>(
            'ts_tensor_full_f32',
          ),
      tensorRank = library.lookupFunction<_TensorRankNative, _TensorRankDart>(
        'ts_tensor_rank',
      ),
      tensorShape = library
          .lookupFunction<_TensorShapeNative, _TensorShapeDart>(
            'ts_tensor_shape',
          ),
      tensorDType = library.lookupFunction<
        _TensorUint32MetadataNative,
        _TensorUint32MetadataDart
      >('ts_tensor_dtype'),
      tensorDevice = library.lookupFunction<
        _TensorUint32MetadataNative,
        _TensorUint32MetadataDart
      >('ts_tensor_device'),
      tensorDeviceIndex = library
          .lookupFunction<_TensorInt32MetadataNative, _TensorInt32MetadataDart>(
            'ts_tensor_device_index',
          ),
      tensorNumel = library.lookupFunction<
        _TensorUint64MetadataNative,
        _TensorUint64MetadataDart
      >('ts_tensor_numel'),
      tensorToDevice = library
          .lookupFunction<_TensorToDeviceNative, _TensorToDeviceDart>(
            'ts_tensor_to_device',
          ),
      tensorReshape = library
          .lookupFunction<_TensorReshapeNative, _TensorReshapeDart>(
            'ts_tensor_reshape',
          ),
      tensorTranspose2D = library
          .lookupFunction<_TensorUnaryNative, _TensorUnaryDart>(
            'ts_tensor_transpose2d',
          ),
      tensorAdd = library
          .lookupFunction<_TensorBinaryNative, _TensorBinaryDart>(
            'ts_tensor_add',
          ),
      tensorMultiply = library
          .lookupFunction<_TensorBinaryNative, _TensorBinaryDart>(
            'ts_tensor_multiply',
          ),
      tensorSum = library.lookupFunction<_TensorUnaryNative, _TensorUnaryDart>(
        'ts_tensor_sum',
      ),
      tensorMatmul = library
          .lookupFunction<_TensorBinaryNative, _TensorBinaryDart>(
            'ts_tensor_matmul',
          ),
      tensorCopyToHostF32 = library
          .lookupFunction<_TensorCopyToHostNative, _TensorCopyToHostDart>(
            'ts_tensor_copy_to_host_f32',
          ),
      tensorRetain = library
          .lookupFunction<_TensorLifetimeNative, _TensorLifetimeDart>(
            'ts_tensor_retain',
          ),
      tensorRelease = library
          .lookupFunction<_TensorLifetimeNative, _TensorLifetimeDart>(
            'ts_tensor_release',
          ),
      runtimeLiveTensorCount = library
          .lookupFunction<_RuntimeCounterNative, _RuntimeCounterDart>(
            'ts_runtime_live_tensor_count',
          ),
      runtimeLiveStorageBytes = library
          .lookupFunction<_RuntimeCounterNative, _RuntimeCounterDart>(
            'ts_runtime_live_storage_bytes',
          );

  final _AbiVersionDart abiVersion;
  final _LastErrorDart lastErrorMessage;
  final _NoopDart noop;
  final _RuntimeDeviceCountDart runtimeDeviceCount;
  final _RuntimeUint32Dart runtimeCudaDeviceCount;
  final _TensorFromF32Dart tensorFromF32;
  final _TensorFullF32Dart tensorFullF32;
  final _TensorRankDart tensorRank;
  final _TensorShapeDart tensorShape;
  final _TensorUint32MetadataDart tensorDType;
  final _TensorUint32MetadataDart tensorDevice;
  final _TensorInt32MetadataDart tensorDeviceIndex;
  final _TensorUint64MetadataDart tensorNumel;
  final _TensorToDeviceDart tensorToDevice;
  final _TensorReshapeDart tensorReshape;
  final _TensorUnaryDart tensorTranspose2D;
  final _TensorBinaryDart tensorAdd;
  final _TensorBinaryDart tensorMultiply;
  final _TensorUnaryDart tensorSum;
  final _TensorBinaryDart tensorMatmul;
  final _TensorCopyToHostDart tensorCopyToHostF32;
  final _TensorLifetimeDart tensorRetain;
  final _TensorLifetimeDart tensorRelease;
  final _RuntimeCounterDart runtimeLiveTensorCount;
  final _RuntimeCounterDart runtimeLiveStorageBytes;
}