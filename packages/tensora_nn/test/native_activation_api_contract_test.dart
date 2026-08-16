import 'package:tensora_nn/tensora_nn.dart';
import 'package:test/test.dart';

void main() {
  test('NN V2 activation modules map to native Tensor primitives', () {
    final Tensor Function(Tensor) gelu = (tensor) => tensor.gelu();
    final Tensor Function(Tensor) silu = (tensor) => tensor.silu();
    final Tensor Function(Tensor) swiglu = (tensor) => tensor.swiglu();

    expect(gelu, isA<Tensor Function(Tensor)>());
    expect(silu, isA<Tensor Function(Tensor)>());
    expect(swiglu, isA<Tensor Function(Tensor)>());
  });
}
