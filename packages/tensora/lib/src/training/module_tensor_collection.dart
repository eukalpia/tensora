import '../native/native_training_runtime.dart';
import '../tensor/native_adoption.dart';
import '../tensor/tensor.dart';

/// Returns a native parameter-handle reader bound to [module].
int Function(int) moduleParameterHandleReader(int module) =>
    (index) => NativeTrainingRuntime.instance.moduleParameterAt(module, index);

/// Returns a native buffer-handle reader bound to [module].
int Function(int) moduleBufferHandleReader(int module) =>
    (index) => NativeTrainingRuntime.instance.moduleBufferAt(module, index);

/// Adopts a native module tensor collection transactionally.
List<Tensor> collectModuleTensors(
  int count,
  int Function(int index) getHandle,
) {
  final tensors = <Tensor>[];
  try {
    for (var index = 0; index < count; index++) {
      tensors.add(
        Tensor.adoptNativeHandleForRuntime(
          getHandle(index),
          nativeTensorAdoptionToken,
        ),
      );
    }
    return tensors;
  } catch (_) {
    for (final tensor in tensors) {
      tensor.dispose();
    }
    rethrow;
  }
}
