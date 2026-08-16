import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  test('liveTensorCount tracks explicit Tensor ownership', () {
    final baseline = TensoraRuntime.liveTensorCount;
    final tensor = Tensor.ones(Shape(<int>[2]));

    expect(TensoraRuntime.liveTensorCount, baseline + 1);
    tensor.dispose();
    expect(TensoraRuntime.liveTensorCount, baseline);
  });
}
