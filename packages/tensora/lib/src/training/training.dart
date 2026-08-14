import '../device/device.dart';
import '../errors/tensora_exception.dart';
import '../native/finalizer_callbacks.dart';
import '../native/native_training_runtime.dart';
import '../tensor/native_adoption.dart';
import '../tensor/tensor.dart';

final Finalizer<int> _moduleFinalizer = Finalizer<int>(
  releaseModuleFromFinalizer,
);

final Finalizer<int> _optimizerFinalizer = Finalizer<int>(
  releaseOptimizerFromFinalizer,
);

/// Runtime-level training capabilities and diagnostics.
abstract final class TensoraRuntime {
  /// Whether the loaded native library contains the training backend.
  static bool get trainingAvailable =>
      NativeTrainingRuntime.instance.trainingAvailable();

  /// Number of visible devices for [device]'s accelerator kind.
  ///
  /// The index carried by [device] is ignored for counting. CPU always reports
  /// one. Accelerator kinds return zero when the loaded native runtime does not
  /// contain or cannot see that backend.
  static int deviceCount(Device device) =>
      NativeTrainingRuntime.instance.deviceCount(device);

  /// Every device visible through Tensora's tensor/training backend.
  ///
  /// Accelerators are ordered by backend kind and device index, followed by
  /// [Device.cpu]. The returned list is immutable and reflects the currently
  /// loaded native runtime.
  static List<Device> get availableDevices {
    final devices = <Device>[];

    final cudaCount = deviceCount(Device.cuda(0));
    for (var index = 0; index < cudaCount; index++) {
      devices.add(Device.cuda(index));
    }

    if (deviceCount(Device.mps) > 0) {
      devices.add(Device.mps);
    }

    final xpuCount = deviceCount(Device.xpu(0));
    for (var index = 0; index < xpuCount; index++) {
      devices.add(Device.xpu(index));
    }

    final hipCount = deviceCount(Device.hip(0));
    for (var index = 0; index < hipCount; index++) {
      devices.add(Device.hip(index));
    }

    devices.add(Device.cpu);
    return List<Device>.unmodifiable(devices);
  }

  /// Preferred tensor/training device for the currently loaded runtime.
  ///
  /// An available accelerator is preferred over CPU. Tensora never changes the
  /// default device of tensor factories implicitly; callers opt into this value
  /// explicitly when they want automatic accelerator selection.
  static Device get preferredDevice {
    final devices = availableDevices;
    for (final device in devices) {
      if (!device.isCpu) return device;
    }
    return Device.cpu;
  }

  /// Number of visible NVIDIA CUDA devices reported by the training backend.
  static int get cudaDeviceCount => deviceCount(Device.cuda(0));

  /// Sets the native training random seed.
  static void manualSeed(int seed) =>
      NativeTrainingRuntime.instance.manualSeed(seed);

  /// Number of live native Module handles owned through the ABI.
  static int get liveModuleCount =>
      NativeTrainingRuntime.instance.liveModuleCount();

  /// Number of live native Optimizer handles owned through the ABI.
  static int get liveOptimizerCount =>
      NativeTrainingRuntime.instance.liveOptimizerCount();
}

/// Base class for native-backed neural-network modules.
@pragma('vm:isolate-unsendable')
abstract base class Module {
  Module._(this._handle) {
    _moduleFinalizer.attach(this, _handle, detach: this);
  }

  int _handle;
  bool _disposed = false;

  /// Whether this wrapper has deterministically released its native handle.
  bool get isDisposed => _disposed;

  /// Executes the native forward pass.
  Tensor call(Tensor input) {
    _ensureLive('forward');
    final inputHandle = input.nativeHandleForRuntime(nativeTensorAdoptionToken);
    final output = NativeTrainingRuntime.instance.moduleForward(
      _handle,
      inputHandle,
    );
    return Tensor.adoptNativeHandleForRuntime(
      output,
      nativeTensorAdoptionToken,
    );
  }

  /// Puts the module into training mode.
  void train() {
    _ensureLive('train');
    NativeTrainingRuntime.instance.moduleSetTraining(_handle, true);
  }

  /// Puts the module into evaluation mode.
  void eval() {
    _ensureLive('eval');
    NativeTrainingRuntime.instance.moduleSetTraining(_handle, false);
  }

  /// Moves module parameters and buffers to [device].
  void to(Device device) {
    _ensureLive('to');
    NativeTrainingRuntime.instance.moduleToDevice(_handle, device);
  }

  /// Returns native parameter views as independently owned Tensor handles.
  ///
  /// Callers must dispose each returned Tensor when finished with it.
  List<Tensor> parameters() {
    _ensureLive('parameters');
    final runtime = NativeTrainingRuntime.instance;
    final count = runtime.moduleParameterCount(_handle);
    return _collectTensors(
      count,
      (index) => runtime.moduleParameterAt(_handle, index),
    );
  }

  /// Returns native buffer views as independently owned Tensor handles.
  ///
  /// Callers must dispose each returned Tensor when finished with it.
  List<Tensor> buffers() {
    _ensureLive('buffers');
    final runtime = NativeTrainingRuntime.instance;
    final count = runtime.moduleBufferCount(_handle);
    return _collectTensors(
      count,
      (index) => runtime.moduleBufferAt(_handle, index),
    );
  }

  /// Saves this module's native state to [path].
  void save(String path) {
    _ensureLive('save');
    NativeTrainingRuntime.instance.moduleSave(_handle, path);
  }

  /// Loads native state from a checkpoint previously saved for this module.
  void load(String path) {
    _ensureLive('load');
    NativeTrainingRuntime.instance.moduleLoad(_handle, path);
  }

  /// Deterministically releases the native module handle.
  void dispose() {
    if (_disposed) return;
    NativeTrainingRuntime.instance.moduleRelease(_handle);
    _moduleFinalizer.detach(this);
    _handle = 0;
    _disposed = true;
  }

  List<Tensor> _collectTensors(int count, int Function(int index) getHandle) {
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

  void _ensureLive(String operation) {
    if (_disposed) {
      throw DisposedTensorException(
        'Module has already been disposed.',
        operation: 'module.$operation',
      );
    }
  }
}

/// A native fully connected affine layer.
final class Linear extends Module {
  Linear._(int handle, this.inFeatures, this.outFeatures, this.bias)
    : super._(handle);

  /// Creates a Linear layer with shape `[inFeatures, outFeatures]`.
  factory Linear(int inFeatures, int outFeatures, {bool bias = true}) {
    final handle = NativeTrainingRuntime.instance.createLinear(
      inFeatures,
      outFeatures,
      bias,
    );
    return Linear._(handle, inFeatures, outFeatures, bias);
  }

  /// Number of input features.
  final int inFeatures;

  /// Number of output features.
  final int outFeatures;

  /// Whether this layer owns a bias parameter.
  final bool bias;
}

/// Base class for native-backed optimizers.
@pragma('vm:isolate-unsendable')
abstract base class Optimizer {
  Optimizer._(this._handle) {
    _optimizerFinalizer.attach(this, _handle, detach: this);
  }

  int _handle;
  bool _disposed = false;

  /// Whether this wrapper has deterministically released its native handle.
  bool get isDisposed => _disposed;

  /// Clears accumulated gradients for all optimized parameters.
  void zeroGrad() {
    _ensureLive('zeroGrad');
    NativeTrainingRuntime.instance.optimizerZeroGrad(_handle);
  }

  /// Applies one optimizer update.
  void step() {
    _ensureLive('step');
    NativeTrainingRuntime.instance.optimizerStep(_handle);
  }

  /// Deterministically releases the native optimizer handle.
  void dispose() {
    if (_disposed) return;
    NativeTrainingRuntime.instance.optimizerRelease(_handle);
    _optimizerFinalizer.detach(this);
    _handle = 0;
    _disposed = true;
  }

  void _ensureLive(String operation) {
    if (_disposed) {
      throw NativeRuntimeException(
        'Optimizer has already been disposed.',
        operation: 'optimizer.$operation',
      );
    }
  }
}

/// Stochastic gradient descent.
final class SGD extends Optimizer {
  SGD._(int handle) : super._(handle);

  factory SGD(
    Module module, {
    double learningRate = 0.01,
    double momentum = 0,
    double weightDecay = 0,
  }) {
    module._ensureLive('optimizer');
    _validatePositiveFinite(learningRate, 'learningRate');
    _validateNonNegativeFinite(momentum, 'momentum');
    _validateNonNegativeFinite(weightDecay, 'weightDecay');
    final handle = NativeTrainingRuntime.instance.createSgd(
      module._handle,
      learningRate: learningRate,
      momentum: momentum,
      weightDecay: weightDecay,
    );
    return SGD._(handle);
  }
}

/// Adam optimizer.
final class Adam extends Optimizer {
  Adam._(int handle) : super._(handle);

  factory Adam(
    Module module, {
    double learningRate = 0.001,
    double beta1 = 0.9,
    double beta2 = 0.999,
    double epsilon = 1e-8,
    double weightDecay = 0,
  }) => Adam._(
    _createAdam(
      module,
      learningRate: learningRate,
      beta1: beta1,
      beta2: beta2,
      epsilon: epsilon,
      weightDecay: weightDecay,
      decoupled: false,
    ),
  );
}

/// AdamW optimizer with decoupled weight decay.
final class AdamW extends Optimizer {
  AdamW._(int handle) : super._(handle);

  factory AdamW(
    Module module, {
    double learningRate = 0.001,
    double beta1 = 0.9,
    double beta2 = 0.999,
    double epsilon = 1e-8,
    double weightDecay = 0.01,
  }) => AdamW._(
    _createAdam(
      module,
      learningRate: learningRate,
      beta1: beta1,
      beta2: beta2,
      epsilon: epsilon,
      weightDecay: weightDecay,
      decoupled: true,
    ),
  );
}

/// Native training losses.
abstract final class Losses {
  /// Mean squared error over equal-shaped tensors.
  static Tensor mse(Tensor prediction, Tensor target) {
    final predictionHandle = prediction.nativeHandleForRuntime(
      nativeTensorAdoptionToken,
    );
    final targetHandle = target.nativeHandleForRuntime(
      nativeTensorAdoptionToken,
    );
    final handle = NativeTrainingRuntime.instance.mseLoss(
      predictionHandle,
      targetHandle,
    );
    return Tensor.adoptNativeHandleForRuntime(
      handle,
      nativeTensorAdoptionToken,
    );
  }

  /// Cross entropy for rank-2 logits and equal-shaped one-hot float32 targets.
  static Tensor crossEntropy(Tensor logits, Tensor oneHotTarget) {
    final logitsHandle = logits.nativeHandleForRuntime(
      nativeTensorAdoptionToken,
    );
    final targetHandle = oneHotTarget.nativeHandleForRuntime(
      nativeTensorAdoptionToken,
    );
    final handle = NativeTrainingRuntime.instance.crossEntropyLoss(
      logitsHandle,
      targetHandle,
    );
    return Tensor.adoptNativeHandleForRuntime(
      handle,
      nativeTensorAdoptionToken,
    );
  }
}

int _createAdam(
  Module module, {
  required double learningRate,
  required double beta1,
  required double beta2,
  required double epsilon,
  required double weightDecay,
  required bool decoupled,
}) {
  module._ensureLive('optimizer');
  _validatePositiveFinite(learningRate, 'learningRate');
  _validateBeta(beta1, 'beta1');
  _validateBeta(beta2, 'beta2');
  _validatePositiveFinite(epsilon, 'epsilon');
  _validateNonNegativeFinite(weightDecay, 'weightDecay');
  return NativeTrainingRuntime.instance.createAdam(
    module._handle,
    learningRate: learningRate,
    beta1: beta1,
    beta2: beta2,
    epsilon: epsilon,
    weightDecay: weightDecay,
    decoupled: decoupled,
  );
}

void _validatePositiveFinite(double value, String name) {
  if (!value.isFinite || value <= 0) {
    throw ArgumentError.value(value, name, 'must be finite and positive');
  }
}

void _validateNonNegativeFinite(double value, String name) {
  if (!value.isFinite || value < 0) {
    throw ArgumentError.value(value, name, 'must be finite and non-negative');
  }
}

void _validateBeta(double value, String name) {
  if (!value.isFinite || value < 0 || value >= 1) {
    throw ArgumentError.value(value, name, 'must be finite in [0, 1)');
  }
}
