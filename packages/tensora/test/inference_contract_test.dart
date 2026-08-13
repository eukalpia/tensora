import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  test('core build exposes explicit disabled ONNX diagnostics', () {
    expect(OnnxRuntime.available, isFalse);
    expect(OnnxRuntime.providers, isEmpty);
    expect(OnnxRuntime.liveSessionCount, 0);
  });

  test('ONNX provider values have stable runtime names', () {
    expect(OnnxExecutionProvider.auto.runtimeName, 'auto');
    expect(OnnxExecutionProvider.cpu.runtimeName, 'CPUExecutionProvider');
    expect(OnnxExecutionProvider.cuda.runtimeName, 'CUDAExecutionProvider');
    expect(
      OnnxExecutionProvider.directML.runtimeName,
      'DmlExecutionProvider',
    );
    expect(OnnxExecutionProvider.coreML.runtimeName, 'CoreMLExecutionProvider');
    expect(
      OnnxExecutionProvider.openVino.runtimeName,
      'OpenVINOExecutionProvider',
    );
    expect(
      OnnxExecutionProvider.miGraphX.runtimeName,
      'MIGraphXExecutionProvider',
    );
  });

  test('empty ONNX model path is rejected before native work', () {
    expect(() => OnnxSession(''), throwsArgumentError);
    expect(() => OnnxSession('   '), throwsArgumentError);
  });

  test('session creation fails explicitly when ONNX backend is disabled', () {
    expect(
      () => OnnxSession('/tmp/model.onnx'),
      throwsA(isA<UnsupportedOperationException>()),
    );
    expect(
      () => OnnxSession(
        '/tmp/model.onnx',
        provider: OnnxExecutionProvider.cuda,
      ),
      throwsA(isA<UnsupportedOperationException>()),
    );
    expect(OnnxRuntime.liveSessionCount, 0);
  });

  test('ModelRuntimeException preserves operation context', () {
    const error = ModelRuntimeException(
      'invalid model',
      operation: 'onnx.session.create',
    );
    expect(error.toString(), contains('onnx.session.create'));
    expect(error.toString(), contains('invalid model'));
  });
}
