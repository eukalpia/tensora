import 'package:tensora/src/errors/tensora_exception.dart';
import 'package:tensora/src/native/native_runtime.dart';
import 'package:test/test.dart';

void main() {
  test('missing native override becomes a structured runtime load failure', () {
    expect(
      () => NativeRuntime.instance,
      throwsA(
        isA<NativeRuntimeException>().having(
          (error) => error.operation,
          'operation',
          'runtime.load',
        ),
      ),
    );
  });
}
