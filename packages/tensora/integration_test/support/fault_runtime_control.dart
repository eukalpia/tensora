import 'dart:ffi';
import 'dart:io';

typedef _SetModeNative = Void Function(Int32 mode);
typedef _SetModeDart = void Function(int mode);
typedef _CounterNative = Uint64 Function();
typedef _CounterDart = int Function();
typedef _ResetNative = Void Function();
typedef _ResetDart = void Function();

abstract final class FaultMode {
  static const normal = 0;
  static const abiMismatch = 1;

  static const invalidArgument = 10;
  static const invalidShapeStatus = 11;
  static const outOfMemory = 12;
  static const unsupported = 13;
  static const invalidHandle = 14;
  static const internalError = 15;
  static const modelError = 16;
  static const unknownStatus = 17;
  static const nullDiagnostic = 18;

  static const nullTensorHandle = 20;
  static const rankAboveLimit = 21;
  static const scalarRankChanged = 22;
  static const rankChanged = 23;
  static const invalidShapeMetadata = 24;
  static const unknownDType = 25;
  static const invalidCpuIndex = 26;
  static const invalidCudaIndex = 27;
  static const invalidMpsIndex = 28;
  static const invalidXpuIndex = 29;
  static const invalidHipIndex = 30;
  static const unknownDevice = 31;
  static const inconsistentNumel = 32;
  static const validCudaTensor = 33;

  static const nullTrainingHandle = 40;
  static const acceleratorCounts = 41;
  static const moduleBuffer = 42;
  static const moduleAdoptionFailure = 43;

  static const nullSessionHandle = 50;
  static const providerMismatch = 51;
  static const noOutputs = 52;
  static const invalidUtf8Size = 53;
  static const changedUtf8Size = 54;
  static const outputCountMismatch = 55;
  static const outputAdoptionFailure = 56;

  static const dtypeFloat16 = 60;
  static const dtypeBFloat16 = 61;
  static const dtypeFloat64 = 62;
  static const dtypeInt8 = 63;
  static const dtypeUint8 = 64;
  static const dtypeInt16 = 65;
  static const dtypeInt32 = 66;
  static const dtypeInt64 = 67;
  static const dtypeBool = 68;
}

final class FaultRuntimeControl {
  FaultRuntimeControl._(DynamicLibrary library)
    : _setMode = library.lookupFunction<_SetModeNative, _SetModeDart>(
        'ts_test_set_mode',
      ),
      _releaseCount = library.lookupFunction<_CounterNative, _CounterDart>(
        'ts_test_release_count',
      ),
      _resetCounters = library.lookupFunction<_ResetNative, _ResetDart>(
        'ts_test_reset_counters',
      );

  factory FaultRuntimeControl.fromEnvironment() {
    final path = Platform.environment['TENSORA_FAULT_LIBRARY'];
    if (path == null || path.trim().isEmpty) {
      throw StateError('TENSORA_FAULT_LIBRARY must identify the test runtime.');
    }
    return FaultRuntimeControl._(DynamicLibrary.open(path));
  }

  final _SetModeDart _setMode;
  final _CounterDart _releaseCount;
  final _ResetDart _resetCounters;

  void setMode(int mode) => _setMode(mode);
  int get releaseCount => _releaseCount();
  void resetCounters() => _resetCounters();
}
