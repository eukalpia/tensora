import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  test('Tensora exposes the stable dtype descriptor table', () {
    expect(DType.values, hasLength(10));
    expect(DType.float32.toString(), 'DType.float32');
  });

  test('CPU device remains stable', () {
    expect(Device.cpu.name, 'cpu');
    expect(Device.cpu.index, 0);
    expect(Device.cpu.isCpu, isTrue);
    expect(Device.cpu.isCuda, isFalse);
    expect(Device.cpu.isMps, isFalse);
    expect(Device.cpu.isXpu, isFalse);
    expect(Device.cpu.isHip, isFalse);
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
    expect(cuda0.isMps, isFalse);
    expect(cuda0.isXpu, isFalse);
    expect(cuda0.isHip, isFalse);
    expect(cuda0, anotherCuda0);
    expect(cuda0.hashCode, anotherCuda0.hashCode);
    expect(cuda0, isNot(cuda1));
    expect(cuda0.toString(), 'Device.cuda(0)');
  });

  test('MPS identifies the macOS Metal training device', () {
    expect(Device.mps.name, 'mps');
    expect(Device.mps.index, 0);
    expect(Device.mps.isCpu, isFalse);
    expect(Device.mps.isCuda, isFalse);
    expect(Device.mps.isMps, isTrue);
    expect(Device.mps.isXpu, isFalse);
    expect(Device.mps.isHip, isFalse);
    expect(Device.mps.toString(), 'Device.mps');
  });

  test('XPU devices carry a validated index and value semantics', () {
    final xpu0 = Device.xpu(0);
    final anotherXpu0 = Device.xpu(0);
    final xpu1 = Device.xpu(1);

    expect(xpu0.name, 'xpu:0');
    expect(xpu0.index, 0);
    expect(xpu0.isCpu, isFalse);
    expect(xpu0.isCuda, isFalse);
    expect(xpu0.isMps, isFalse);
    expect(xpu0.isXpu, isTrue);
    expect(xpu0.isHip, isFalse);
    expect(xpu0, anotherXpu0);
    expect(xpu0.hashCode, anotherXpu0.hashCode);
    expect(xpu0, isNot(xpu1));
    expect(xpu0.toString(), 'Device.xpu(0)');
  });

  test('HIP devices carry a validated index and value semantics', () {
    final hip0 = Device.hip(0);
    final anotherHip0 = Device.hip(0);
    final hip1 = Device.hip(1);

    expect(hip0.name, 'hip:0');
    expect(hip0.index, 0);
    expect(hip0.isCpu, isFalse);
    expect(hip0.isCuda, isFalse);
    expect(hip0.isMps, isFalse);
    expect(hip0.isXpu, isFalse);
    expect(hip0.isHip, isTrue);
    expect(hip0, anotherHip0);
    expect(hip0.hashCode, anotherHip0.hashCode);
    expect(hip0, isNot(hip1));
    expect(hip0.toString(), 'Device.hip(0)');
  });

  test('indexed accelerator device indices cannot be negative', () {
    expect(() => Device.cuda(-1), throwsArgumentError);
    expect(() => Device.xpu(-1), throwsArgumentError);
    expect(() => Device.hip(-1), throwsArgumentError);
  });
}
