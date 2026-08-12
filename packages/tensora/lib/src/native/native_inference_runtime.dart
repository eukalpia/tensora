import 'dart:ffi';

import 'package:ffi/ffi.dart';

import '../errors/tensora_exception.dart';
import 'native_inference_bindings.dart';
import 'native_runtime.dart';

final class NativeInferenceRuntime {
  NativeInferenceRuntime._()
    : _bindings = NativeInferenceBindings(
        DynamicLibrary.open(NativeRuntime.instance.libraryPath),
      );

  static NativeInferenceRuntime? _instance;
  static NativeInferenceRuntime get instance =>
      _instance ??= NativeInferenceRuntime._();

  final NativeInferenceBindings _bindings;

  bool available() {
    final value = calloc<Uint8>();
    try {
      _check(_bindings.available(value), 'onnx.available');
      return value.value != 0;
    } finally {
      calloc.free(value);
    }
  }

  List<String> providers() {
    final count = calloc<Size>();
    try {
      _check(_bindings.providerCount(count), 'onnx.providers');
      return List<String>.generate(
        count.value,
        (index) => _providerName(index),
        growable: false,
      );
    } finally {
      calloc.free(count);
    }
  }

  int createSession(
    String modelPath, {
    required bool enableProfiling,
    String? profilingPrefix,
  }) {
    if (modelPath.trim().isEmpty) {
      throw ArgumentError.value(modelPath, 'modelPath', 'must not be empty');
    }
    final path = modelPath.toNativeUtf8();
    final prefix = profilingPrefix?.toNativeUtf8();
    final output = calloc<Uint64>();
    try {
      _check(
        _bindings.sessionCreate(
          path,
          enableProfiling ? 1 : 0,
          prefix ?? nullptr.cast<Utf8>(),
          output,
        ),
        'onnx.session.create',
      );
      if (output.value == 0) {
        throw NativeRuntimeException(
          'Native runtime returned a null ONNX session handle on success.',
          operation: 'onnx.session.create',
        );
      }
      return output.value;
    } finally {
      if (prefix != null) calloc.free(prefix);
      calloc.free(output);
      calloc.free(path);
    }
  }

  List<String> inputNames(int session) => _sessionNames(
    session,
    input: true,
  );

  List<String> outputNames(int session) => _sessionNames(
    session,
    input: false,
  );

  List<int> run(
    int session, {
    required List<String> inputNames,
    required List<int> inputHandles,
    required List<String> outputNames,
  }) {
    if (inputNames.length != inputHandles.length) {
      throw ArgumentError('inputNames and inputHandles must have equal length');
    }
    if (outputNames.isEmpty) {
      throw ArgumentError.value(outputNames, 'outputNames', 'must not be empty');
    }

    final nativeInputNames = inputNames.isEmpty
        ? nullptr.cast<Pointer<Utf8>>()
        : calloc<Pointer<Utf8>>(inputNames.length);
    final nativeInputHandles = inputHandles.isEmpty
        ? nullptr.cast<Uint64>()
        : calloc<Uint64>(inputHandles.length);
    final nativeOutputNames = calloc<Pointer<Utf8>>(outputNames.length);
    final nativeOutputs = calloc<Uint64>(outputNames.length);
    final written = calloc<Size>();
    final allocatedNames = <Pointer<Utf8>>[];

    try {
      for (var index = 0; index < inputNames.length; index++) {
        final name = inputNames[index].toNativeUtf8();
        allocatedNames.add(name);
        nativeInputNames[index] = name;
        nativeInputHandles[index] = inputHandles[index];
      }
      for (var index = 0; index < outputNames.length; index++) {
        final name = outputNames[index].toNativeUtf8();
        allocatedNames.add(name);
        nativeOutputNames[index] = name;
      }

      _check(
        _bindings.sessionRun(
          session,
          nativeInputNames,
          nativeInputHandles,
          inputNames.length,
          nativeOutputNames,
          outputNames.length,
          nativeOutputs,
          outputNames.length,
          written,
        ),
        'onnx.session.run',
      );
      if (written.value != outputNames.length) {
        for (var index = 0; index < written.value; index++) {
          NativeRuntime.instance.releaseFromFinalizer(nativeOutputs[index]);
        }
        throw NativeRuntimeException(
          'Native runtime returned ${written.value} ONNX outputs; expected '
          '${outputNames.length}.',
          operation: 'onnx.session.run',
        );
      }
      return List<int>.generate(
        written.value,
        (index) => nativeOutputs[index],
        growable: false,
      );
    } finally {
      for (final name in allocatedNames) {
        calloc.free(name);
      }
      calloc.free(written);
      calloc.free(nativeOutputs);
      calloc.free(nativeOutputNames);
      if (inputHandles.isNotEmpty) calloc.free(nativeInputHandles);
      if (inputNames.isNotEmpty) calloc.free(nativeInputNames);
    }
  }

  String endProfiling(int session) {
    const capacity = 65536;
    final buffer = calloc<Uint8>(capacity).cast<Utf8>();
    final required = calloc<Size>();
    try {
      _check(
        _bindings.sessionEndProfiling(session, buffer, capacity, required),
        'onnx.session.endProfiling',
      );
      if (required.value == 0 || required.value > capacity) {
        throw NativeRuntimeException(
          'Native runtime returned an invalid profiling path size '
          '${required.value}.',
          operation: 'onnx.session.endProfiling',
        );
      }
      return buffer.toDartString();
    } finally {
      calloc.free(required);
      calloc.free(buffer);
    }
  }

  void release(int session) {
    _check(_bindings.sessionRelease(session), 'onnx.session.dispose');
  }

  void releaseFromFinalizer(int session) {
    _bindings.sessionRelease(session);
  }

  int liveSessionCount() {
    final value = calloc<Uint64>();
    try {
      _check(_bindings.liveSessionCount(value), 'onnx.liveSessionCount');
      return value.value;
    } finally {
      calloc.free(value);
    }
  }

  List<String> _sessionNames(int session, {required bool input}) {
    final count = calloc<Size>();
    try {
      _check(
        input
            ? _bindings.sessionInputCount(session, count)
            : _bindings.sessionOutputCount(session, count),
        input ? 'onnx.session.inputCount' : 'onnx.session.outputCount',
      );
      return List<String>.generate(
        count.value,
        (index) => _sessionName(session, index, input: input),
        growable: false,
      );
    } finally {
      calloc.free(count);
    }
  }

  String _providerName(int index) {
    final required = calloc<Size>();
    try {
      _check(
        _bindings.providerName(index, nullptr.cast<Utf8>(), 0, required),
        'onnx.providerName',
      );
      return _readSizedUtf8(
        required.value,
        (buffer, capacity, secondRequired) => _bindings.providerName(
          index,
          buffer,
          capacity,
          secondRequired,
        ),
        'onnx.providerName',
      );
    } finally {
      calloc.free(required);
    }
  }

  String _sessionName(int session, int index, {required bool input}) {
    final required = calloc<Size>();
    final operation = input
        ? 'onnx.session.inputName'
        : 'onnx.session.outputName';
    try {
      _check(
        input
            ? _bindings.sessionInputName(
                session,
                index,
                nullptr.cast<Utf8>(),
                0,
                required,
              )
            : _bindings.sessionOutputName(
                session,
                index,
                nullptr.cast<Utf8>(),
                0,
                required,
              ),
        operation,
      );
      return _readSizedUtf8(
        required.value,
        (buffer, capacity, secondRequired) => input
            ? _bindings.sessionInputName(
                session,
                index,
                buffer,
                capacity,
                secondRequired,
              )
            : _bindings.sessionOutputName(
                session,
                index,
                buffer,
                capacity,
                secondRequired,
              ),
        operation,
      );
    } finally {
      calloc.free(required);
    }
  }

  String _readSizedUtf8(
    int required,
    int Function(Pointer<Utf8>, int, Pointer<Size>) read,
    String operation,
  ) {
    if (required <= 0) {
      throw NativeRuntimeException(
        'Native runtime returned an invalid UTF-8 buffer size $required.',
        operation: operation,
      );
    }
    final buffer = calloc<Uint8>(required).cast<Utf8>();
    final secondRequired = calloc<Size>();
    try {
      _check(read(buffer, required, secondRequired), operation);
      if (secondRequired.value != required) {
        throw NativeRuntimeException(
          'Native UTF-8 size changed from $required to '
          '${secondRequired.value}.',
          operation: operation,
        );
      }
      return buffer.toDartString();
    } finally {
      calloc.free(secondRequired);
      calloc.free(buffer);
    }
  }

  void _check(int status, String operation) {
    if (status == 0) return;
    final pointer = _bindings.lastErrorMessage();
    final message = pointer.address == 0
        ? 'Native runtime returned status $status without a diagnostic.'
        : pointer.toDartString();
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
      case 6:
        throw NativeRuntimeException(message, operation: operation);
      case 7:
        throw ModelRuntimeException(message, operation: operation);
      default:
        throw NativeRuntimeException(
          'Unknown native status $status: $message',
          operation: operation,
        );
    }
  }
}
