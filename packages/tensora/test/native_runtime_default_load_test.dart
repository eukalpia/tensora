import 'dart:io';

import 'package:tensora/src/native/native_runtime.dart';
import 'package:test/test.dart';

void main() {
  final override = Platform.environment['TENSORA_NATIVE_LIBRARY'];

  test(
    'default Linux native discovery loads the canonical library name',
    () {
      final runtime = NativeRuntime.instance;
      expect(runtime.libraryPath, 'libtensora_native.so');
      runtime.noop();
    },
    skip:
        override != null && override.trim().isNotEmpty
            ? 'requires default native library discovery'
            : false,
  );
}
