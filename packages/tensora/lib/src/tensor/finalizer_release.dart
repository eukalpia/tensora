import '../native/native_runtime.dart';

/// Best-effort tensor release used by the Dart finalizer safety net.
void releaseTensorHandleFromFinalizer(int handle) {
  NativeRuntime.instance.releaseFromFinalizer(handle);
}
