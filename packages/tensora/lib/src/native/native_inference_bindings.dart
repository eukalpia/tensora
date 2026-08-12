import 'dart:ffi';

import 'package:ffi/ffi.dart';

typedef _LastErrorNative = Pointer<Utf8> Function();
typedef _LastErrorDart = Pointer<Utf8> Function();
typedef _ByteOutputNative = Int32 Function(Pointer<Uint8>);
typedef _ByteOutputDart = int Function(Pointer<Uint8>);
typedef _SizeOutputNative = Int32 Function(Pointer<Size>);
typedef _SizeOutputDart = int Function(Pointer<Size>);
typedef _Uint64OutputNative = Int32 Function(Pointer<Uint64>);
typedef _Uint64OutputDart = int Function(Pointer<Uint64>);

typedef _ProviderNameNative =
    Int32 Function(Size, Pointer<Utf8>, Size, Pointer<Size>);
typedef _ProviderNameDart =
    int Function(int, Pointer<Utf8>, int, Pointer<Size>);

typedef _SessionCreateNative =
    Int32 Function(Pointer<Utf8>, Uint8, Pointer<Utf8>, Pointer<Uint64>);
typedef _SessionCreateDart =
    int Function(Pointer<Utf8>, int, Pointer<Utf8>, Pointer<Uint64>);

typedef _SessionCountNative = Int32 Function(Uint64, Pointer<Size>);
typedef _SessionCountDart = int Function(int, Pointer<Size>);
typedef _SessionNameNative =
    Int32 Function(Uint64, Size, Pointer<Utf8>, Size, Pointer<Size>);
typedef _SessionNameDart =
    int Function(int, int, Pointer<Utf8>, int, Pointer<Size>);

typedef _SessionRunNative =
    Int32 Function(
      Uint64,
      Pointer<Pointer<Utf8>>,
      Pointer<Uint64>,
      Size,
      Pointer<Pointer<Utf8>>,
      Size,
      Pointer<Uint64>,
      Size,
      Pointer<Size>,
    );
typedef _SessionRunDart =
    int Function(
      int,
      Pointer<Pointer<Utf8>>,
      Pointer<Uint64>,
      int,
      Pointer<Pointer<Utf8>>,
      int,
      Pointer<Uint64>,
      int,
      Pointer<Size>,
    );

typedef _SessionEndProfilingNative =
    Int32 Function(Uint64, Pointer<Utf8>, Size, Pointer<Size>);
typedef _SessionEndProfilingDart =
    int Function(int, Pointer<Utf8>, int, Pointer<Size>);
typedef _SessionReleaseNative = Int32 Function(Uint64);
typedef _SessionReleaseDart = int Function(int);

final class NativeInferenceBindings {
  NativeInferenceBindings(DynamicLibrary library)
    : lastErrorMessage = library
          .lookupFunction<_LastErrorNative, _LastErrorDart>(
            'ts_last_error_message',
          ),
      available = library.lookupFunction<_ByteOutputNative, _ByteOutputDart>(
        'ts_onnx_available',
      ),
      providerCount = library
          .lookupFunction<_SizeOutputNative, _SizeOutputDart>(
            'ts_onnx_provider_count',
          ),
      providerName = library
          .lookupFunction<_ProviderNameNative, _ProviderNameDart>(
            'ts_onnx_provider_name',
          ),
      sessionCreate = library
          .lookupFunction<_SessionCreateNative, _SessionCreateDart>(
            'ts_onnx_session_create',
          ),
      sessionInputCount = library
          .lookupFunction<_SessionCountNative, _SessionCountDart>(
            'ts_onnx_session_input_count',
          ),
      sessionOutputCount = library
          .lookupFunction<_SessionCountNative, _SessionCountDart>(
            'ts_onnx_session_output_count',
          ),
      sessionInputName = library
          .lookupFunction<_SessionNameNative, _SessionNameDart>(
            'ts_onnx_session_input_name',
          ),
      sessionOutputName = library
          .lookupFunction<_SessionNameNative, _SessionNameDart>(
            'ts_onnx_session_output_name',
          ),
      sessionRun = library.lookupFunction<_SessionRunNative, _SessionRunDart>(
        'ts_onnx_session_run',
      ),
      sessionEndProfiling = library
          .lookupFunction<_SessionEndProfilingNative, _SessionEndProfilingDart>(
            'ts_onnx_session_end_profiling',
          ),
      sessionRelease = library
          .lookupFunction<_SessionReleaseNative, _SessionReleaseDart>(
            'ts_onnx_session_release',
          ),
      liveSessionCount = library
          .lookupFunction<_Uint64OutputNative, _Uint64OutputDart>(
            'ts_runtime_live_onnx_session_count',
          );

  final _LastErrorDart lastErrorMessage;
  final _ByteOutputDart available;
  final _SizeOutputDart providerCount;
  final _ProviderNameDart providerName;
  final _SessionCreateDart sessionCreate;
  final _SessionCountDart sessionInputCount;
  final _SessionCountDart sessionOutputCount;
  final _SessionNameDart sessionInputName;
  final _SessionNameDart sessionOutputName;
  final _SessionRunDart sessionRun;
  final _SessionEndProfilingDart sessionEndProfiling;
  final _SessionReleaseDart sessionRelease;
  final _Uint64OutputDart liveSessionCount;
}
