import '../native/native_training_runtime.dart';
import '../tensor/native_adoption.dart';
import '../tensor/tensor.dart';

/// Safe foundation facade for opaque Tensor identity and state transactions.
abstract final class NativeTensorState {
  static int identity(Tensor tensor) {
    final handle = tensor.nativeHandleForRuntime(nativeTensorAdoptionToken);
    return NativeTrainingRuntime.instance.tensorIdentity(handle);
  }

  static Tensor cloneDetached(Tensor tensor) {
    final handle = tensor.nativeHandleForRuntime(nativeTensorAdoptionToken);
    final cloned = NativeTrainingRuntime.instance.cloneDetached(handle);
    return Tensor.adoptNativeHandleForRuntime(
      cloned,
      nativeTensorAdoptionToken,
    );
  }

  /// Assigns [sources] into [targets] as one native transaction.
  static void assignMany({
    required List<Tensor> targets,
    required List<Tensor> sources,
  }) {
    if (targets.length != sources.length) {
      throw ArgumentError(
        'State assignment requires equal target and source counts.',
      );
    }
    if (targets.isEmpty) return;

    final targetHandles = <int>[];
    final sourceHandles = <int>[];
    for (var index = 0; index < targets.length; index++) {
      targetHandles.add(
        targets[index].nativeHandleForRuntime(nativeTensorAdoptionToken),
      );
      sourceHandles.add(
        sources[index].nativeHandleForRuntime(nativeTensorAdoptionToken),
      );
    }
    NativeTrainingRuntime.instance.assignMany(targetHandles, sourceHandles);
  }
}
