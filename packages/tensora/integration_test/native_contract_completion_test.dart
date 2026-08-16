import 'dart:ffi';
import 'dart:io';

import 'package:tensora/src/inference/finalizer_release.dart';
import 'package:tensora/src/inference/output_adoption.dart';
import 'package:tensora/src/native/native_device_codec.dart';
import 'package:tensora/src/native/native_inference_runtime.dart';
import 'package:tensora/src/native/native_library_path.dart';
import 'package:tensora/src/native/native_runtime.dart';
import 'package:tensora/src/native/native_training_runtime.dart';
import 'package:tensora/src/native/native_windows_loader_state.dart';
import 'package:tensora/src/tensor/finalizer_release.dart';
import 'package:tensora/src/training/finalizer_release.dart';
import 'package:tensora/src/training/module_tensor_collection.dart';
import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

typedef _VoidNative = Void Function();
typedef _VoidDart = void Function();
typedef _SetUint8Native = Void Function(Uint8);
typedef _SetUint8Dart = void Function(int);
typedef _SetInt32Native = Void Function(Int32);
typedef _SetInt32Dart = void Function(int);
typedef _SetDeviceNative = Void Function(Uint32, Int32);
typedef _SetDeviceDart = void Function(int, int);
typedef _CounterNative = Uint64 Function();
typedef _CounterDart = int Function();

void main() {
  final fixturePath = Platform.environment['TENSORA_NATIVE_LIBRARY'];
  if (fixturePath == null || fixturePath.isEmpty) {
    fail('TENSORA_NATIVE_LIBRARY must point at the contract fixture.');
  }

  final fixture = DynamicLibrary.open(fixturePath);
  final reset = fixture.lookupFunction<_VoidNative, _VoidDart>('ts_test_reset');
  final setForcedStatus = fixture
      .lookupFunction<_SetInt32Native, _SetInt32Dart>(
        'ts_test_set_forced_status',
      );
  final setNullDiagnostic = fixture
      .lookupFunction<_SetUint8Native, _SetUint8Dart>(
        'ts_test_set_null_diagnostic',
      );
  final setDevice = fixture.lookupFunction<_SetDeviceNative, _SetDeviceDart>(
    'ts_test_set_device',
  );
  final setTrainingMode = fixture
      .lookupFunction<_SetInt32Native, _SetInt32Dart>(
        'ts_test_set_training_mode',
      );
  final tensorReleaseCount = fixture
      .lookupFunction<_CounterNative, _CounterDart>(
        'ts_test_tensor_release_count',
      );
  final moduleReleaseCount = fixture
      .lookupFunction<_CounterNative, _CounterDart>(
        'ts_test_module_release_count',
      );
  final optimizerReleaseCount = fixture
      .lookupFunction<_CounterNative, _CounterDart>(
        'ts_test_optimizer_release_count',
      );
  final sessionReleaseCount = fixture
      .lookupFunction<_CounterNative, _CounterDart>(
        'ts_test_session_release_count',
      );

  final runtime = NativeRuntime.instance;
  final training = NativeTrainingRuntime.instance;
  final inference = NativeInferenceRuntime.instance;

  setUp(reset);

  test(
    'native library-name resolution is deterministic for every platform',
    () {
      expect(
        nativeLibraryNameForOperatingSystem('linux'),
        'libtensora_native.so',
      );
      expect(
        nativeLibraryNameForOperatingSystem('macos'),
        'libtensora_native.dylib',
      );
      expect(
        nativeLibraryNameForOperatingSystem('windows'),
        'tensora_native.dll',
      );
      expect(
        () => nativeLibraryNameForOperatingSystem('plan9'),
        throwsA(isA<UnsupportedOperationException>()),
      );
      expect(windowsPreloadedModules, isEmpty);
    },
  );

  test(
    'native device codec covers all device families and rejects unknown names',
    () {
      final cases = <(Device, int)>[
        (Device.cpu, 1),
        (Device.cuda(7), 2),
        (Device.mps, 3),
        (Device.xpu(2), 4),
        (Device.hip(3), 5),
      ];
      for (final (device, code) in cases) {
        expect(nativeDeviceCode(device), code);
      }
      expect(nativeDeviceCodeForName('cuda:19'), 2);
      expect(
        () => nativeDeviceCodeForName('unknown:0'),
        throwsA(isA<UnsupportedError>()),
      );
    },
  );

  test('accelerator tensor creation releases its CPU staging handle', () {
    setDevice(2, 1);
    final tensor = Tensor.full(Shape([1]), 3, device: Device.cuda(1));
    expect(tensor.device, Device.cuda(1));
    expect(tensorReleaseCount(), 1);
    tensor.dispose();
    expect(tensorReleaseCount(), 2);
  });

  test('training status mapping synthesizes a missing native diagnostic', () {
    setNullDiagnostic(1);
    setForcedStatus(99);
    expect(
      () => training.manualSeed(1),
      throwsA(
        isA<NativeRuntimeException>().having(
          (error) => error.message,
          'message',
          contains('without a diagnostic'),
        ),
      ),
    );
  });

  test(
    'module tensor readers and collector own returned handles exactly once',
    () {
      final module = training.createLinear(1, 1, true);
      try {
        final parameterHandle = moduleParameterHandleReader(module)(0);
        runtime.release(parameterHandle);

        final bufferHandle = moduleBufferHandleReader(module)(0);
        runtime.release(bufferHandle);

        final handles = <int>[
          runtime.full(Shape([1]), 1),
          runtime.full(Shape([1]), 2),
        ];
        final tensors = collectModuleTensors(2, (index) => handles[index]);
        expect(tensors, hasLength(2));
        for (final tensor in tensors) {
          tensor.dispose();
        }
      } finally {
        training.moduleRelease(module);
      }
    },
  );

  test('finalizer release callbacks are deterministic and non-throwing', () {
    final tensor = runtime.full(Shape([1]), 1);
    releaseTensorHandleFromFinalizer(tensor);
    expect(tensorReleaseCount(), 1);

    final module = training.createLinear(1, 1, true);
    final optimizer = training.createSgd(
      module,
      learningRate: 0.1,
      momentum: 0,
      weightDecay: 0,
    );
    releaseOptimizerHandleFromFinalizer(optimizer);
    releaseModuleHandleFromFinalizer(module);
    expect(optimizerReleaseCount(), 1);
    expect(moduleReleaseCount(), 1);

    final session = inference.createSession(
      'fixture.onnx',
      enableProfiling: false,
      profilingPrefix: null,
    );
    releaseOnnxSessionHandleFromFinalizer(session);
    expect(sessionReleaseCount(), 1);
  });

  test('ONNX output adoption releases outputs that were never attempted', () {
    final module = training.createLinear(1, 1, true);
    try {
      setTrainingMode(1);
      final first = training.moduleParameterAt(module, 0);
      final malformed = training.moduleParameterAt(module, 1);
      final unattempted = runtime.full(Shape([1]), 3);

      expect(
        () => adoptOnnxOutputHandles(<int>[first, malformed, unattempted]),
        throwsA(isA<NativeRuntimeException>()),
      );
      expect(
        tensorReleaseCount(),
        3,
        reason: 'adopted, failed, and unattempted outputs each release once',
      );
    } finally {
      training.moduleRelease(module);
    }
  });
}
