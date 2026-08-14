import '../native/native_training_runtime.dart';

/// Best-effort module release used by the Dart finalizer safety net.
void releaseModuleHandleFromFinalizer(int handle) {
  NativeTrainingRuntime.instance.moduleReleaseFromFinalizer(handle);
}

/// Best-effort optimizer release used by the Dart finalizer safety net.
void releaseOptimizerHandleFromFinalizer(int handle) {
  NativeTrainingRuntime.instance.optimizerReleaseFromFinalizer(handle);
}
