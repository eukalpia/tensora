import '../native/native_training_runtime.dart';

/// Best-effort module release used by the Dart finalizer safety net.
void releaseModuleHandleFromFinalizer(int handle) {
  NativeTrainingRuntime.instance.moduleReleaseFromFinalizer(handle);
}

/// Best-effort legacy optimizer release used by the Dart finalizer safety net.
void releaseOptimizerHandleFromFinalizer(int handle) {
  NativeTrainingRuntime.instance.optimizerReleaseFromFinalizer(handle);
}

/// Best-effort parameter optimizer release used by the Dart finalizer safety net.
void releaseParameterOptimizerHandleFromFinalizer(int handle) {
  NativeTrainingRuntime.instance.parameterOptimizerReleaseFromFinalizer(handle);
}
