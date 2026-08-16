import 'package:tensora/tensora.dart' as core;
import 'package:tensora_nn/tensora_nn.dart';
import 'package:test/test.dart';

final class _BaseProbe extends Module {
  @override
  core.Tensor forward(core.Tensor input) => input;
}

void main() {
  test('base Module exposes stateless defaults without hidden ownership', () {
    final probe = _BaseProbe();

    expect(probe.isDisposed, isFalse);
    expect(probe.isMaterialized, isTrue);
    expect(probe.children, isEmpty);
    expect(probe.toString(), '_BaseProbe');
    expect(probe.internalMoveDevice, isNull);
    probe.internalPreflightMove(core.Device.cpu);
    probe.internalOnMove(core.Device.cpu);
    probe.internalOnTrainingModeChanged(true);
    probe.internalOnDispose();

    probe.dispose();
  });

  test('Sequential rejects a disposed child before ownership can attach', () {
    final disposed = Identity()..dispose();
    expect(
      () => Sequential(children: <Module>[disposed]),
      throwsArgumentError,
    );
  });
}
