import 'package:tensora/src/native/native_library_path.dart';
import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  test('native library defaults are deterministic by operating system', () {
    expect(
      resolveNativeLibraryPath(
        environment: const <String, String>{},
        operatingSystem: 'linux',
      ),
      'libtensora_native.so',
    );
    expect(
      resolveNativeLibraryPath(
        environment: const <String, String>{},
        operatingSystem: 'macos',
      ),
      'libtensora_native.dylib',
    );
    expect(
      resolveNativeLibraryPath(
        environment: const <String, String>{},
        operatingSystem: 'windows',
      ),
      'tensora_native.dll',
    );
  });

  test('an explicit non-empty native library path wins unchanged', () {
    expect(
      resolveNativeLibraryPath(
        environment: const <String, String>{
          'TENSORA_NATIVE_LIBRARY': ' /tmp/runtime.so ',
        },
        operatingSystem: 'linux',
      ),
      ' /tmp/runtime.so ',
    );
  });

  test('blank native library overrides fall back to platform default', () {
    expect(
      resolveNativeLibraryPath(
        environment: const <String, String>{
          'TENSORA_NATIVE_LIBRARY': '   ',
        },
        operatingSystem: 'linux',
      ),
      'libtensora_native.so',
    );
  });

  test('unsupported operating systems fail with a typed diagnostic', () {
    expect(
      () => resolveNativeLibraryPath(
        environment: const <String, String>{},
        operatingSystem: 'android',
      ),
      throwsA(
        isA<UnsupportedOperationException>()
            .having((error) => error.operation, 'operation', 'runtime.load')
            .having(
              (error) => error.message,
              'message',
              contains('Linux, macOS, and Windows'),
            ),
      ),
    );
  });
}
