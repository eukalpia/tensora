export 'package:tensora_nn/tensora_nn.dart' show Module;

import 'package:tensora_text/tensora_text.dart';

final class GenerationConfig {
  GenerationConfig({
    this.maxNewTokens = 128,
    this.temperature = 1.0,
    this.topK = 0,
    this.topP = 1.0,
    Iterable<int> stopTokenIds = const <int>[],
  }) : stopTokenIds = Set<int>.unmodifiable(stopTokenIds) {
    if (maxNewTokens <= 0) {
      throw ArgumentError.value(maxNewTokens, 'maxNewTokens', 'must be positive');
    }
    if (!temperature.isFinite || temperature < 0) {
      throw ArgumentError.value(
        temperature,
        'temperature',
        'must be finite and non-negative',
      );
    }
    if (topK < 0) {
      throw ArgumentError.value(topK, 'topK', 'must be non-negative');
    }
    if (!topP.isFinite || topP <= 0 || topP > 1) {
      throw ArgumentError.value(topP, 'topP', 'must be finite in (0, 1]');
    }
    for (final token in this.stopTokenIds) {
      if (token < 0) {
        throw ArgumentError.value(
          token,
          'stopTokenIds',
          'token ids must be non-negative',
        );
      }
    }
  }

  final int maxNewTokens;
  final double temperature;
  final int topK;
  final double topP;
  final Set<int> stopTokenIds;

  bool get greedy => temperature == 0;
}

final class GenerationRequest {
  GenerationRequest({required this.prompt, GenerationConfig? config})
    : config = config ?? GenerationConfig();

  final TokenSequence prompt;
  final GenerationConfig config;
}
