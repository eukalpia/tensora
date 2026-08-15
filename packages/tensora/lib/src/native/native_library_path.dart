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

/// Resolves an explicit native runtime override or the platform default.
///
/// Whitespace-only overrides are treated as unset. A non-empty override is
/// returned byte-for-byte so callers retain control over intentional spaces in
/// a filesystem path.
String resolveNativeLibraryPath({
  required Map<String, String> environment,
  required String operatingSystem,
}) {
  final override = environment['TENSORA_NATIVE_LIBRARY'];
  if (override != null && override.trim().isNotEmpty) {
    return override;
  }
  return nativeLibraryNameForOperatingSystem(operatingSystem);
}
