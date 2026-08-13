import 'package:tensora/tensora.dart';
import 'package:tensora_audio/tensora_audio.dart';
import 'package:test/test.dart';

void main() {
  test('audio metadata derives shape and duration', () {
    final spec = AudioTensorSpec(sampleRate: 16000, channels: 1, frames: 16000);
    expect(spec.shape, Shape(<int>[1, 16000]));
    expect(spec.duration, const Duration(seconds: 1));
  });

  test('invalid audio metadata is rejected', () {
    expect(
      () => AudioTensorSpec(sampleRate: 0, channels: 1, frames: 1),
      throwsArgumentError,
    );
    expect(
      () => AudioTensorSpec(sampleRate: 16000, channels: 0, frames: 1),
      throwsArgumentError,
    );
    expect(
      () => AudioTensorSpec(sampleRate: 16000, channels: 1, frames: 0),
      throwsArgumentError,
    );
  });
}
