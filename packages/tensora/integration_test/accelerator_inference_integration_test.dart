import 'dart:io';

import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

OnnxExecutionProvider _targetProvider() {
  final value = Platform.environment['TENSORA_TEST_ONNX_PROVIDER'];
  return switch (value) {
    'cuda' => OnnxExecutionProvider.cuda,
    'directml' => OnnxExecutionProvider.directML,
    'coreml' => OnnxExecutionProvider.coreML,
    'openvino' => OnnxExecutionProvider.openVino,
    'migraphx' => OnnxExecutionProvider.miGraphX,
    _ => throw StateError(
      'TENSORA_TEST_ONNX_PROVIDER must be one of: '
      'cuda, directml, coreml, openvino, migraphx.',
    ),
  };
}

void main() {
  test(
    'selected ONNX provider executes the reference model without fallback',
    () {
      final provider = _targetProvider();
      final modelPath = Platform.environment['TENSORA_ONNX_TEST_MODEL'];
      if (modelPath == null || modelPath.trim().isEmpty) {
        throw StateError(
          'TENSORA_ONNX_TEST_MODEL must point to the ONNX fixture.',
        );
      }

      expect(OnnxRuntime.available, isTrue);
      expect(
        OnnxRuntime.providers,
        contains(provider.runtimeName),
        reason: '${provider.runtimeName} must be present in the linked runtime',
      );

      final baselineSessions = OnnxRuntime.liveSessionCount;
      final session = OnnxSession(modelPath, provider: provider);
      final input = Tensor.fromList([1, 2, 3, 4], shape: Shape([2, 2]));

      try {
        expect(session.requestedProvider, provider);
        expect(
          session.selectedProvider,
          provider,
          reason: 'explicit accelerator provider must never fall back to CPU',
        );

        final outputs = session.run({'X': input});
        try {
          expect(outputs.keys, ['Y']);
          expect(outputs['Y']!.device, Device.cpu);
          expect(outputs['Y']!.shape, Shape([2, 2]));
          expect(outputs['Y']!.toList(), [3, 5, 7, 11]);
        } finally {
          for (final output in outputs.values) {
            output.dispose();
          }
        }
      } finally {
        input.dispose();
        session.dispose();
      }

      expect(OnnxRuntime.liveSessionCount, baselineSessions);
    },
  );
}
