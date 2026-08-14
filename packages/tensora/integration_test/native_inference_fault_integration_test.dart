import 'package:tensora/src/native/finalizer_callbacks.dart';
import 'package:tensora/src/native/native_inference_runtime.dart';
import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

import 'support/fault_runtime_control.dart';

void main() {
  final control = FaultRuntimeControl.fromEnvironment();
  control.setMode(FaultMode.normal);
  final runtime = NativeInferenceRuntime.instance;

  tearDown(() => control.setMode(FaultMode.normal));

  test('inference status mapping preserves typed failures', () {
    final cases = <(int, Matcher)>[
      (FaultMode.invalidShapeStatus, isA<InvalidShapeException>()),
      (FaultMode.outOfMemory, isA<OutOfMemoryException>()),
      (
        FaultMode.unknownStatus,
        isA<NativeRuntimeException>().having(
          (error) => error.message,
          'message',
          contains('Unknown native status 99'),
        ),
      ),
    ];

    for (final entry in cases) {
      control.setMode(entry.$1);
      expect(() => runtime.available(), throwsA(entry.$2));
    }

    control.setMode(FaultMode.nullDiagnostic);
    expect(
      () => runtime.available(),
      throwsA(
        isA<NativeRuntimeException>().having(
          (error) => error.message,
          'message',
          contains('without a diagnostic'),
        ),
      ),
    );
  });

  test('session finalizer callback delegates to native release', () {
    control.resetCounters();
    releaseOnnxSessionFromFinalizer(91);
    expect(control.releaseCount, 1);
  });

  test('successful session creation must return a non-zero handle', () {
    control.setMode(FaultMode.nullSessionHandle);
    expect(
      () => runtime.createSession(
        'model.onnx',
        enableProfiling: false,
        profilingPrefix: null,
      ),
      throwsA(
        isA<NativeRuntimeException>()
            .having(
              (error) => error.operation,
              'operation',
              'onnx.session.create',
            )
            .having(
              (error) => error.message,
              'message',
              contains('null ONNX session handle'),
            ),
      ),
    );
  });

  test('UTF-8 metadata size contracts cannot change between calls', () {
    control.setMode(FaultMode.invalidUtf8Size);
    expect(
      runtime.providers,
      throwsA(
        isA<NativeRuntimeException>().having(
          (error) => error.message,
          'message',
          contains('invalid UTF-8 buffer size'),
        ),
      ),
    );

    control.setMode(FaultMode.changedUtf8Size);
    expect(
      runtime.providers,
      throwsA(
        isA<NativeRuntimeException>().having(
          (error) => error.message,
          'message',
          contains('size changed'),
        ),
      ),
    );
  });

  test(
    'output count mismatch releases outputs already returned by native code',
    () {
      control.resetCounters();
      control.setMode(FaultMode.outputCountMismatch);

      expect(
        () => runtime.run(
          1,
          inputNames: const [],
          inputHandles: const [],
          outputNames: const ['Y0', 'Y1'],
        ),
        throwsA(
          isA<NativeRuntimeException>()
              .having(
                (error) => error.operation,
                'operation',
                'onnx.session.run',
              )
              .having(
                (error) => error.message,
                'message',
                contains('returned 1 ONNX outputs; expected 2'),
              ),
        ),
      );
      expect(control.releaseCount, 1);
    },
  );

  test(
    'explicit provider mismatch is rejected without leaking the session',
    () {
      control.resetCounters();
      control.setMode(FaultMode.providerMismatch);

      expect(
        () => OnnxSession('model.onnx', provider: OnnxExecutionProvider.cuda),
        throwsA(
          isA<NativeRuntimeException>()
              .having(
                (error) => error.operation,
                'operation',
                'onnx.session.create',
              )
              .having(
                (error) => error.message,
                'message',
                contains('selected CPUExecutionProvider'),
              ),
        ),
      );
      expect(control.releaseCount, 1);
    },
  );

  test('sessions without outputs are rejected transactionally', () {
    control.resetCounters();
    control.setMode(FaultMode.noOutputs);

    expect(
      () => OnnxSession('model.onnx'),
      throwsA(
        isA<ModelRuntimeException>()
            .having(
              (error) => error.operation,
              'operation',
              'onnx.session.create',
            )
            .having(
              (error) => error.message,
              'message',
              'ONNX model exposes no outputs.',
            ),
      ),
    );
    expect(control.releaseCount, 1);
  });

  test('partial output adoption disposes and releases every native tensor', () {
    control.resetCounters();
    control.setMode(FaultMode.outputAdoptionFailure);
    final input = Tensor.ones(Shape([1]));
    final session = OnnxSession('model.onnx');
    try {
      expect(
        () => session.run({'X': input}),
        throwsA(
          isA<NativeRuntimeException>().having(
            (error) => error.operation,
            'operation',
            'tensor.dtype',
          ),
        ),
      );
      expect(control.releaseCount, 3);
    } finally {
      session.dispose();
      input.dispose();
    }
  });
}
