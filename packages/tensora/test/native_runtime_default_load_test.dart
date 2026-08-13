import 'package:tensora/src/native/native_runtime.dart';
import 'package:test/test.dart';

void main() {
  test('default Linux native discovery loads the canonical library name', () {
    final runtime = NativeRuntime.instance;

    expect(runtime.libraryPath, 'libtensora_native.so');
    runtime.noop();
  });
}
