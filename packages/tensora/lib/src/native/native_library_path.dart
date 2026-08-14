import '../errors/tensora_exception.dart';

/// Returns the platform-default Tensora native library filename.
String nativeLibraryNameForOperatingSystem(String operatingSystem) =>
    switch (operatingSystem) {
      'linux' => 'libtensora_native.so',
      'macos' => 'libtensora_native.dylib',
      'windows' => 'tensora_native.dll',
      _ =>
        throw UnsupportedOperationException(
          'Native runtime discovery supports Linux, macOS, and Windows.',
          operation: 'runtime.load',
        ),
    };
