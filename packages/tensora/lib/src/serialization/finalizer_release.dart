import 'dart:io';

/// Best-effort file-handle close used by the Dart finalizer safety net.
///
/// A finalizer callback must never throw, and the operating system may already
/// have reclaimed the handle by the time the callback runs, so the failure is
/// deliberately swallowed here instead of being surfaced.
void closeSafetensorsHandleFromFinalizer(RandomAccessFile handle) {
  try {
    handle.closeSync();
  } on FileSystemException {
    return;
  }
}
