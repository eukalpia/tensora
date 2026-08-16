import 'dart:ffi';

/// Win32 module references retained for the process lifetime after secure
/// dependency preloading. This state is internal to native library discovery.
final List<Pointer<Void>> windowsPreloadedModules = <Pointer<Void>>[];
