import 'package:tensora/tensora.dart';
import 'package:tensora_vision/tensora_vision.dart';
import 'package:test/test.dart';

void main() {
  test('image shape contract', () {
    final spec = ImageTensorSpec(width: 32, height: 16, channels: 3);
    expect(spec.shape, Shape(<int>[3, 16, 32]));
  });

  test('invalid image metadata is rejected', () {
    expect(
      () => ImageTensorSpec(width: 0, height: 16, channels: 3),
      throwsArgumentError,
    );
  });
}
