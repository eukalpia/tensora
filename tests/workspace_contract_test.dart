import 'package:tensora/tensora.dart';
import 'package:tensora_audio/tensora_audio.dart';
import 'package:tensora_data/tensora_data.dart';
import 'package:tensora_text/tensora_text.dart';
import 'package:tensora_transformers/tensora_transformers.dart';
import 'package:tensora_vision/tensora_vision.dart';
import 'package:test/test.dart';

void main() {
  test('public domain packages compose', () {
    final data = ListDataset<TokenSequence>(<TokenSequence>[
      TokenSequence(const <int>[1, 2]),
      TokenSequence(const <int>[3]),
    ]);
    final prompt =
        DataLoader<TokenSequence>(
          data,
          batchSize: 2,
        ).batches().first.values.first;
    final request = GenerationRequest(
      prompt: prompt,
      config: GenerationConfig(maxNewTokens: 8, temperature: 0),
    );
    final image = ImageTensorSpec(width: 32, height: 16, channels: 3);
    final audio = AudioTensorSpec(sampleRate: 8000, channels: 1, frames: 4000);

    expect(request.config.greedy, isTrue);
    expect(image.shape, Shape(<int>[3, 16, 32]));
    expect(audio.shape, Shape(<int>[1, 4000]));
    expect(audio.duration, const Duration(milliseconds: 500));
  });
}
