import 'native_inference_runtime.dart';
import 'native_runtime.dart';
import 'native_training_runtime.dart';

/// Best-effort tensor release used only by a Dart finalizer.
void releaseTensorFromFinalizer(int handle) {
  NativeRuntime.instance.releaseFromFinalizer(handle);
}

/// Best-effort module release used only by a Dart finalizer.
void releaseModuleFromFinalizer(int handle) {
  NativeTrainingRuntime.instance.moduleReleaseFromFinalizer(handle);
}

/// Best-effort optimizer release used only by a Dart finalizer.
void releaseOptimizerFromFinalizer(int handle) {
  NativeTrainingRuntime.instance.optimizerReleaseFromFinalizer(handle);
}

/// Best-effort ONNX session release used only by a Dart finalizer.
void releaseOnnxSessionFromFinalizer(int handle) {
  NativeInferenceRuntime.instance.releaseFromFinalizer(handle);
}
