import 'dart:io';

import 'package:tensora/src/native/native_inference_runtime.dart';
import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  final runtime = NativeInferenceRuntime.instance;
  final invalidHandleError = isA<NativeRuntimeException>();
  final modelPath = Platform.environment['TENSORA_ONNX_TEST_MODEL'];

  test('inference runtime validates request structure before native calls', () {
    expect(
      () => runtime.createSession(
        '',
        enableProfiling: false,
        profilingPrefix: null,
      ),
      throwsArgumentError,
    );
    expect(modelPath, isNotNull);
    expect(
      () => runtime.createSession(
        modelPath!,
        providerName: '   ',
        enableProfiling: false,
        profilingPrefix: null,
      ),
      throwsArgumentError,
    );
    expect(
      () => runtime.run(
        0,
        inputNames: const ['X'],
        inputHandles: const [],
        outputNames: const ['Y'],
      ),
      throwsArgumentError,
    );
    expect(
      () => runtime.run(
        0,
        inputNames: const [],
        inputHandles: const [],
        outputNames: const [],
      ),
      throwsArgumentError,
    );
  });

  test('invalid session handles fail deterministically across operations', () {
    expect(() => runtime.inputNames(0), throwsA(invalidHandleError));
    expect(() => runtime.outputNames(0), throwsA(invalidHandleError));
    expect(
      () => runtime.run(
        0,
        inputNames: const [],
        inputHandles: const [],
        outputNames: const ['Y'],
      ),
      throwsA(invalidHandleError),
    );
    expect(() => runtime.endProfiling(0), throwsA(invalidHandleError));
    expect(() => runtime.release(0), throwsA(invalidHandleError));
  });

  test('inference diagnostics remain available after rejected operations', () {
    expect(runtime.available(), isTrue);
    expect(runtime.providers(), isNotEmpty);
    expect(runtime.liveSessionCount(), greaterThanOrEqualTo(0));
  });

  test('finalizer release path returns the session counter to baseline', () {
    expect(modelPath, isNotNull);
    final baseline = runtime.liveSessionCount();
    final session = runtime.createSession(
      modelPath!,
      enableProfiling: false,
      profilingPrefix: null,
    );

    expect(runtime.liveSessionCount(), baseline + 1);
    runtime.releaseFromFinalizer(session);
    expect(runtime.liveSessionCount(), baseline);
  });
}
