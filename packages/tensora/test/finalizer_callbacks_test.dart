import 'package:tensora/src/native/finalizer_callbacks.dart';
import 'package:test/test.dart';

void main() {
  test('native finalizer callbacks have stable callable signatures', () {
    expect(releaseTensorFromFinalizer, isA<void Function(int)>());
    expect(releaseModuleFromFinalizer, isA<void Function(int)>());
    expect(releaseOptimizerFromFinalizer, isA<void Function(int)>());
    expect(releaseOnnxSessionFromFinalizer, isA<void Function(int)>());
  });
}
