import '../native/native_runtime.dart';
import '../tensor/native_adoption.dart';
import '../tensor/tensor.dart';

/// Adopts native ONNX output handles transactionally.
///
/// Tensor adoption owns rollback for the handle currently being adopted. If a
/// later adoption fails, already-adopted tensors are disposed and only handles
/// that were never handed to Tensor adoption are released here.
List<Tensor> adoptOnnxOutputHandles(List<int> handles) {
  final adopted = <Tensor>[];
  try {
    for (final handle in handles) {
      adopted.add(
        Tensor.adoptNativeHandleForRuntime(handle, nativeTensorAdoptionToken),
      );
    }
    return adopted;
  } catch (_) {
    for (final tensor in adopted) {
      tensor.dispose();
    }
    for (var index = adopted.length + 1; index < handles.length; index++) {
      NativeRuntime.instance.releaseFromFinalizer(handles[index]);
    }
    rethrow;
  }
}
