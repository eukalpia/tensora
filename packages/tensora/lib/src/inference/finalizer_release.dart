import '../native/native_inference_runtime.dart';

/// Best-effort ONNX session release used by the Dart finalizer safety net.
void releaseOnnxSessionHandleFromFinalizer(int handle) {
  NativeInferenceRuntime.instance.releaseFromFinalizer(handle);
}
