import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  test('Milestone 1 exposes only the implemented dtype', () {
    expect(DType.values, [DType.float32]);
    expect(DType.float32.toString(), 'DType.float32');
  });

  test('Milestone 1 exposes the CPU device', () {
    expect(Device.cpu.name, 'cpu');
    expect(Device.cpu.toString(), 'Device.cpu');
  });
}
