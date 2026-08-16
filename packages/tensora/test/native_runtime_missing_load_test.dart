import 'dart:io';

import 'package:tensora/src/errors/tensora_exception.dart';
import 'package:tensora/src/native/native_runtime.dart';
import 'package:test/test.dart';

void main() {
  final override = Platform.environment['TENSORA_NATIVE_LIBRARY'] ?? '';

  test(
    'missing native override becomes a structured runtime load failure',
    () {
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
    },
    skip:
        override.contains('definitely-missing')
            ? false
            : 'requires an intentionally missing native runtime override',
  );
}
