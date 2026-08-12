import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  test('Tensora exposes the implemented dtype', () {
    expect(DType.values, [DType.float32]);
    expect(DType.float32.toString(), 'DType.float32');
  });

  test('CPU device remains stable', () {
    expect(Device.cpu.name, 'cpu');
    expect(Device.cpu.index, 0);
    expect(Device.cpu.isCpu, isTrue);
    expect(Device.cpu.isCuda, isFalse);
    expect(Device.cpu.toString(), 'Device.cpu');
  });

  test('CUDA devices carry a validated index and value semantics', () {
    final cuda0 = Device.cuda(0);
    final anotherCuda0 = Device.cuda(0);
    final cuda1 = Device.cuda(1);

    expect(cuda0.name, 'cuda:0');
    expect(cuda0.index, 0);
    expect(cuda0.isCpu, isFalse);
    expect(cuda0.isCuda, isTrue);
    expect(cuda0, anotherCuda0);
    expect(cuda0.hashCode, anotherCuda0.hashCode);
    expect(cuda0, isNot(cuda1));
    expect(cuda0.toString(), 'Device.cuda(0)');
  });

  test('CUDA device index cannot be negative', () {
    expect(() => Device.cuda(-1), throwsArgumentError);
  });
}
