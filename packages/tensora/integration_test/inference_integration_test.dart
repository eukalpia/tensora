import 'dart:io';

import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void expectValues(
  List<double> actual,
  List<double> expected, {
  double tolerance = 1e-5,
}) {
  expect(actual, hasLength(expected.length));
  for (var index = 0; index < expected.length; index++) {
    expect(actual[index], closeTo(expected[index], tolerance));
  }
}

void main() {
  final modelPath = Platform.environment['TENSORA_ONNX_TEST_MODEL'];

  test('ONNX runtime reports its portable CPU provider', () {
    expect(OnnxRuntime.available, isTrue);
    expect(OnnxRuntime.providers, contains('CPUExecutionProvider'));
  });

  test('unknown native provider names are rejected explicitly', () {
    expect(
      () => OnnxExecutionProvider.fromRuntimeName(
        'DefinitelyUnknownExecutionProvider',
      ),
      throwsA(isA<NativeRuntimeException>()),
    );
  });

  test('auto and explicit CPU sessions report the selected provider', () {
    final automatic = OnnxSession(modelPath!);
    final explicit = OnnxSession(
      modelPath,
      provider: OnnxExecutionProvider.cpu,
    );
    addTearDown(automatic.dispose);
    addTearDown(explicit.dispose);

    expect(automatic.requestedProvider, OnnxExecutionProvider.auto);
    expect(automatic.selectedProvider, OnnxExecutionProvider.cpu);
    expect(explicit.requestedProvider, OnnxExecutionProvider.cpu);
    expect(explicit.selectedProvider, OnnxExecutionProvider.cpu);
  });

  test('explicit unavailable provider never falls back to CPU', () {
    if (!OnnxRuntime.providers.contains(
      OnnxExecutionProvider.cuda.runtimeName,
    )) {
      expect(
        () => OnnxSession(modelPath!, provider: OnnxExecutionProvider.cuda),
        throwsA(isA<UnsupportedOperationException>()),
      );
    }
  });

  test('missing model maps into ModelRuntimeException', () {
    expect(
      () => OnnxSession('/definitely/missing/tensora-model.onnx'),
      throwsA(isA<ModelRuntimeException>()),
    );
  });

  test('session exposes immutable model metadata and reference output', () {
    expect(modelPath, isNotNull);
    final baseline = OnnxRuntime.liveSessionCount;
    final session = OnnxSession(modelPath!);
    addTearDown(session.dispose);

    expect(session.inputNames, ['X']);
    expect(session.outputNames, ['Y']);
    expect(() => session.inputNames.add('bad'), throwsUnsupportedError);
    expect(() => session.outputNames.add('bad'), throwsUnsupportedError);

    final input = Tensor.fromList([1, 2, 3, 4], shape: Shape([2, 2]));
    addTearDown(input.dispose);
    final outputs = session.run({'X': input});
    addTearDown(() {
      for (final tensor in outputs.values) {
        tensor.dispose();
      }
    });

    expect(outputs.keys, ['Y']);
    expect(outputs['Y']!.shape, Shape([2, 2]));
    expect(outputs['Y']!.device, Device.cpu);
    expectValues(outputs['Y']!.toList(), [3, 5, 7, 11]);

    session.dispose();
    expect(OnnxRuntime.liveSessionCount, baseline);
  });

  test('explicit output selection returns the requested result', () {
    final session = OnnxSession(modelPath!);
    final input = Tensor.fromList([1, 2, 3, 4], shape: Shape([2, 2]));
    addTearDown(session.dispose);
    addTearDown(input.dispose);

    final outputs = session.run({'X': input}, outputs: const ['Y']);
    final output = outputs['Y']!;
    addTearDown(output.dispose);

    expect(outputs.keys, ['Y']);
    expectValues(output.toList(), [3, 5, 7, 11]);
  });

  test('session validates named input and output contracts', () {
    final session = OnnxSession(modelPath!);
    final input = Tensor.ones(Shape([2, 2]));
    addTearDown(session.dispose);
    addTearDown(input.dispose);

    expect(
      () => session.run({'wrong': input}),
      throwsA(
        isA<InvalidArgumentException>()
            .having((error) => error.operation, 'operation', 'onnx.session.run')
            .having(
              (error) => error.message,
              'message',
              'Unknown ONNX input "wrong".',
            ),
      ),
    );
    expect(
      () => session.run({}),
      throwsA(
        isA<InvalidArgumentException>()
            .having((error) => error.operation, 'operation', 'onnx.session.run')
            .having(
              (error) => error.message,
              'message',
              'Missing ONNX input "X".',
            ),
      ),
    );
    expect(
      () => session.run({'X': input}, outputs: const []),
      throwsA(isA<InvalidArgumentException>()),
    );
    expect(
      () => session.run({'X': input}, outputs: const ['wrong']),
      throwsA(isA<InvalidArgumentException>()),
    );
    expect(
      () => session.run({'X': input}, outputs: const ['Y', 'Y']),
      throwsA(isA<InvalidArgumentException>()),
    );
  });

  test(
    'native runtime rejects incompatible input shape without leaking output',
    () {
      final session = OnnxSession(modelPath!);
      final wrongShape = Tensor.ones(Shape([1, 4]));
      addTearDown(session.dispose);
      addTearDown(wrongShape.dispose);

      expect(
        () => session.run({'X': wrongShape}),
        throwsA(isA<TensoraException>()),
      );
    },
  );

  test('reusable session remains stable across 1000 Dart inference calls', () {
    final baselineSessions = OnnxRuntime.liveSessionCount;
    final session = OnnxSession(modelPath!);
    final input = Tensor.fromList([1, 2, 3, 4], shape: Shape([2, 2]));

    for (var iteration = 0; iteration < 1000; iteration++) {
      final output = session.run({'X': input})['Y']!;
      expectValues(output.toList(), [3, 5, 7, 11]);
      output.dispose();
    }

    input.dispose();
    session.dispose();
    expect(OnnxRuntime.liveSessionCount, baselineSessions);
  });

  test('profiling writes a real profile and lifecycle is deterministic', () {
    final prefix =
        '${Directory.systemTemp.path}${Platform.pathSeparator}tensora-dart-ort';
    final session = OnnxSession(
      modelPath!,
      enableProfiling: true,
      profilingPrefix: prefix,
    );
    final input = Tensor.ones(Shape([2, 2]));
    addTearDown(input.dispose);
    addTearDown(session.dispose);

    final output = session.run({'X': input})['Y']!;
    output.dispose();

    final profilePath = session.endProfiling();
    final profile = File(profilePath);
    expect(profile.existsSync(), isTrue);
    profile.deleteSync();
    expect(session.endProfiling, throwsA(isA<InvalidArgumentException>()));

    session.dispose();
    session.dispose();
    expect(session.isDisposed, isTrue);
    expect(
      () => session.run({'X': input}),
      throwsA(isA<NativeRuntimeException>()),
    );
  });

  test('endProfiling rejects sessions created without profiling', () {
    final session = OnnxSession(modelPath!);
    addTearDown(session.dispose);
    expect(session.endProfiling, throwsA(isA<InvalidArgumentException>()));
  });
}
