import 'package:tensora/src/native/native_library_resolver.dart';
import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  test('native library defaults are deterministic by operating system', () {
    expect(NativeLibraryResolver.defaultName('linux'), 'libtensora_native.so');
    expect(
      NativeLibraryResolver.defaultName('macos'),
      'libtensora_native.dylib',
    );
    expect(NativeLibraryResolver.defaultName('windows'), 'tensora_native.dll');
  });

  test('an explicit non-empty native library path wins', () {
    expect(
      NativeLibraryResolver.resolve(
        environment: const {'TENSORA_NATIVE_LIBRARY': ' /tmp/runtime.so '},
        operatingSystem: 'linux',
      ),
      ' /tmp/runtime.so ',
    );
  });

  test('blank overrides fall back to the platform default', () {
    expect(
      NativeLibraryResolver.resolve(
        environment: const {'TENSORA_NATIVE_LIBRARY': '   '},
        operatingSystem: 'linux',
      ),
      'libtensora_native.so',
    );
  });

  test('unsupported operating systems fail with a typed diagnostic', () {
    expect(
      () => NativeLibraryResolver.defaultName('android'),
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
