import 'package:tensora_text/tensora_text.dart';
import 'package:tensora_transformers/tensora_transformers.dart';
import 'package:test/test.dart';

void main() {
  test('generation configuration is validated and immutable', () {
    final config = GenerationConfig(
      maxNewTokens: 32,
      temperature: 0,
      topK: 20,
      topP: 0.9,
      stopTokenIds: const <int>[2, 3],
    );

    expect(config.greedy, isTrue);
    expect(config.stopTokenIds, <int>{2, 3});
    expect(() => config.stopTokenIds.add(4), throwsUnsupportedError);
  });

  test('invalid generation parameters are rejected', () {
    expect(() => GenerationConfig(maxNewTokens: 0), throwsArgumentError);
    expect(() => GenerationConfig(temperature: -1), throwsArgumentError);
    expect(() => GenerationConfig(topK: -1), throwsArgumentError);
    expect(() => GenerationConfig(topP: 0), throwsArgumentError);
    expect(
      () => GenerationConfig(stopTokenIds: const <int>[-1]),
      throwsArgumentError,
    );
  });

  test('generation request keeps prompt and explicit policy', () {
    final prompt = TokenSequence(const <int>[10, 11]);
    final config = GenerationConfig(maxNewTokens: 4);
    final request = GenerationRequest(prompt: prompt, config: config);

    expect(request.prompt, prompt);
    expect(request.config, same(config));
  });
}
