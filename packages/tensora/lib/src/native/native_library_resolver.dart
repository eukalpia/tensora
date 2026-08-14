import '../errors/tensora_exception.dart';

/// Resolves the native runtime path without embedding platform policy in FFI.
abstract final class NativeLibraryResolver {
  /// Returns an explicit override or the default for [operatingSystem].
  static String resolve({
    required Map<String, String> environment,
    required String operatingSystem,
  }) {
    final override = environment['TENSORA_NATIVE_LIBRARY'];
    if (override != null && override.trim().isNotEmpty) return override;
    return defaultName(operatingSystem);
  }

  /// Returns the canonical dynamic-library name for a desktop platform.
  static String defaultName(String operatingSystem) =>
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
}
