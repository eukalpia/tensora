import 'package:tensora/src/native/native_runtime.dart';
import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

import 'support/fault_runtime_control.dart';

void main() {
  test('ABI mismatch is rejected before runtime use', () {
    final control = FaultRuntimeControl.fromEnvironment();
    control.setMode(FaultMode.abiMismatch);

    expect(
      () => NativeRuntime.instance,
      throwsA(
        isA<NativeRuntimeException>()
            .having((error) => error.operation, 'operation', 'runtime.load')
            .having(
              (error) => error.message,
              'message',
              contains('Incompatible Tensora native ABI'),
            ),
      ),
    );
  });
}
