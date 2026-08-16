import 'package:tensora/tensora.dart' show Shape;

enum ImageLayout { chw, hwc, nchw, nhwc }

final class ImageTensorSpec {
  ImageTensorSpec({
    required this.width,
    required this.height,
    required this.channels,
    this.layout = ImageLayout.chw,
    this.batchSize = 1,
  }) {
    if (width <= 0) {
      throw ArgumentError.value(width, 'width', 'must be positive');
    }
    if (height <= 0) {
      throw ArgumentError.value(height, 'height', 'must be positive');
    }
    if (channels <= 0) {
      throw ArgumentError.value(channels, 'channels', 'must be positive');
    }
    if (batchSize <= 0) {
      throw ArgumentError.value(batchSize, 'batchSize', 'must be positive');
    }
    if ((layout == ImageLayout.chw || layout == ImageLayout.hwc) &&
        batchSize != 1) {
      throw ArgumentError.value(
        batchSize,
        'batchSize',
        'must be 1 for an unbatched image layout',
      );
    }
  }

  final int width;
  final int height;
  final int channels;
  final ImageLayout layout;
  final int batchSize;

  Shape get shape => switch (layout) {
    ImageLayout.chw => Shape(<int>[channels, height, width]),
    ImageLayout.hwc => Shape(<int>[height, width, channels]),
    ImageLayout.nchw => Shape(<int>[batchSize, channels, height, width]),
    ImageLayout.nhwc => Shape(<int>[batchSize, height, width, channels]),
  };
}
