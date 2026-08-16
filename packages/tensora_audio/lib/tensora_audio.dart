import 'package:tensora/tensora.dart' show Shape;

final class AudioTensorSpec {
  AudioTensorSpec({
    required this.sampleRate,
    required this.channels,
    required this.frames,
  }) {
    if (sampleRate <= 0) {
      throw ArgumentError.value(sampleRate, 'sampleRate', 'must be positive');
    }
    if (channels <= 0) {
      throw ArgumentError.value(channels, 'channels', 'must be positive');
    }
    if (frames <= 0) {
      throw ArgumentError.value(frames, 'frames', 'must be positive');
    }
  }

  final int sampleRate;
  final int channels;
  final int frames;

  Shape get shape => Shape(<int>[channels, frames]);

  Duration get duration => Duration(
    microseconds:
        (frames * Duration.microsecondsPerSecond / sampleRate).round(),
  );
}
