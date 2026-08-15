import '../native/native_training_runtime.dart';
import '../tensor/native_adoption.dart';
import 'finalizer_release.dart';
import 'parameter.dart';

final Finalizer<int> _parameterOptimizerFinalizer = Finalizer<int>(
  releaseParameterOptimizerHandleFromFinalizer,
);

/// Low-level safe owner for an NN V2 optimizer over explicit [Parameter]s.
///
/// This type intentionally lives in `tensora` so `tensora_optim` can remain a
/// feature package that depends only on the Tensora foundation. Raw native
/// handles never cross the public package boundary.
@pragma('vm:isolate-unsendable')
final class NativeParameterOptimizer {
  NativeParameterOptimizer._(this._handle) {
    _parameterOptimizerFinalizer.attach(this, _handle, detach: this);
  }

  factory NativeParameterOptimizer.sgd({
    required Iterable<Parameter> parameters,
    required double learningRate,
    required double momentum,
    required double weightDecay,
  }) {
    final handles = _parameterHandles(parameters);
    final handle = NativeTrainingRuntime.instance.createParameterSgd(
      handles,
      learningRate: learningRate,
      momentum: momentum,
      weightDecay: weightDecay,
    );
    return NativeParameterOptimizer._(handle);
  }

  factory NativeParameterOptimizer.adam({
    required Iterable<Parameter> parameters,
    required double learningRate,
    required double beta1,
    required double beta2,
    required double epsilon,
    required double weightDecay,
    bool decoupled = false,
  }) {
    final handles = _parameterHandles(parameters);
    final handle = NativeTrainingRuntime.instance.createParameterAdam(
      handles,
      learningRate: learningRate,
      beta1: beta1,
      beta2: beta2,
      epsilon: epsilon,
      weightDecay: weightDecay,
      decoupled: decoupled,
    );
    return NativeParameterOptimizer._(handle);
  }

  int _handle;
  bool _disposed = false;

  bool get isDisposed => _disposed;

  void zeroGrad() {
    _ensureLive('zeroGrad');
    NativeTrainingRuntime.instance.parameterOptimizerZeroGrad(_handle);
  }

  void step() {
    _ensureLive('step');
    NativeTrainingRuntime.instance.parameterOptimizerStep(_handle);
  }

  void dispose() {
    if (_disposed) return;
    NativeTrainingRuntime.instance.parameterOptimizerRelease(_handle);
    _parameterOptimizerFinalizer.detach(this);
    _handle = 0;
    _disposed = true;
  }

  void _ensureLive(String operation) {
    if (_disposed) {
      throw StateError(
        'Parameter optimizer has already been disposed: $operation',
      );
    }
  }

  static List<int> _parameterHandles(Iterable<Parameter> parameters) {
    final handles = <int>[];
    for (final parameter in parameters) {
      if (parameter.isDisposed) {
        throw ArgumentError('Optimizer parameters must not be disposed.');
      }
      handles.add(
        parameter.tensorForRuntime.nativeHandleForRuntime(
          nativeTensorAdoptionToken,
        ),
      );
    }
    if (handles.isEmpty) {
      throw ArgumentError.value(handles, 'parameters', 'must not be empty');
    }
    return List<int>.unmodifiable(handles);
  }
}
