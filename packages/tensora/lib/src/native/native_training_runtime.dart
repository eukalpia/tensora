import 'dart:ffi';

import 'package:ffi/ffi.dart';

import '../device/device.dart';
import '../errors/tensora_exception.dart';
import 'native_device_codec.dart';
import 'native_runtime.dart';
import 'native_training_bindings.dart';

final class NativeTrainingRuntime {
  NativeTrainingRuntime._()
    : _bindings = NativeTrainingBindings(
        DynamicLibrary.open(NativeRuntime.instance.libraryPath),
      );

  static NativeTrainingRuntime? _instance;
  static NativeTrainingRuntime get instance =>
      _instance ??= NativeTrainingRuntime._();

  final NativeTrainingBindings _bindings;

  bool trainingAvailable() {
    final value = calloc<Uint8>();
    try {
      _check(_bindings.trainingAvailable(value), 'training.available');
      return value.value != 0;
    } finally {
      calloc.free(value);
    }
  }

  int deviceCount(Device device) => NativeRuntime.instance.deviceCount(device);

  int cudaDeviceCount() => deviceCount(Device.cuda(0));

  void manualSeed(int seed) {
    if (seed < 0) {
      throw ArgumentError.value(seed, 'seed', 'must be non-negative');
    }
    _check(_bindings.manualSeed(seed), 'training.manualSeed');
  }

  int withRequiresGrad(int tensor, bool requiresGrad) => _newTensorHandle(
    'tensor.withRequiresGrad',
    (out) =>
        _bindings.tensorWithRequiresGrad(tensor, requiresGrad ? 1 : 0, out),
  );

  bool requiresGrad(int tensor) {
    final value = calloc<Uint8>();
    try {
      _check(
        _bindings.tensorRequiresGrad(tensor, value),
        'tensor.requiresGrad',
      );
      return value.value != 0;
    } finally {
      calloc.free(value);
    }
  }

  void backward(int tensor) {
    _check(_bindings.tensorBackward(tensor), 'tensor.backward');
  }

  int grad(int tensor) => _newTensorHandle(
    'tensor.grad',
    (out) => _bindings.tensorGrad(tensor, out),
  );

  int relu(int tensor) => _newTensorHandle(
    'tensor.relu',
    (out) => _bindings.tensorRelu(tensor, out),
  );

  int sigmoid(int tensor) => _newTensorHandle(
    'tensor.sigmoid',
    (out) => _bindings.tensorSigmoid(tensor, out),
  );

  int tanh(int tensor) => _newTensorHandle(
    'tensor.tanh',
    (out) => _bindings.tensorTanh(tensor, out),
  );

  int gelu(int tensor) => _newTensorHandle(
    'tensor.gelu',
    (out) => _bindings.tensorGelu(tensor, out),
  );

  int silu(int tensor) => _newTensorHandle(
    'tensor.silu',
    (out) => _bindings.tensorSilu(tensor, out),
  );

  int swiglu(int tensor) => _newTensorHandle(
    'tensor.swiglu',
    (out) => _bindings.tensorSwiGlu(tensor, out),
  );

  int mseLoss(int prediction, int target) => _newTensorHandle(
    'loss.mse',
    (out) => _bindings.mseLoss(prediction, target, out),
  );

  int crossEntropyLoss(int logits, int target) => _newTensorHandle(
    'loss.crossEntropy',
    (out) => _bindings.crossEntropyLoss(logits, target, out),
  );

  int createLinear(int inFeatures, int outFeatures, bool bias) {
    if (inFeatures <= 0) {
      throw ArgumentError.value(inFeatures, 'inFeatures', 'must be positive');
    }
    if (outFeatures <= 0) {
      throw ArgumentError.value(outFeatures, 'outFeatures', 'must be positive');
    }
    return _newObjectHandle(
      'linear.create',
      (out) =>
          _bindings.linearCreate(inFeatures, outFeatures, bias ? 1 : 0, out),
    );
  }

  int moduleForward(int module, int input) => _newTensorHandle(
    'module.forward',
    (out) => _bindings.moduleForward(module, input, out),
  );

  void moduleSetTraining(int module, bool training) {
    _check(
      _bindings.moduleSetTraining(module, training ? 1 : 0),
      'module.setTraining',
    );
  }

  void moduleToDevice(int module, Device device) {
    _check(
      _bindings.moduleToDevice(module, nativeDeviceCode(device), device.index),
      'module.to',
    );
  }

  int moduleParameterCount(int module) => _moduleCount(
    module,
    'module.parameterCount',
    _bindings.moduleParameterCount,
  );

  int moduleBufferCount(int module) =>
      _moduleCount(module, 'module.bufferCount', _bindings.moduleBufferCount);

  int moduleParameterAt(int module, int index) {
    if (index < 0) {
      throw ArgumentError.value(index, 'index', 'must be non-negative');
    }
    return _newTensorHandle(
      'module.parameterAt',
      (out) => _bindings.moduleParameterAt(module, index, out),
    );
  }

  int moduleBufferAt(int module, int index) {
    if (index < 0) {
      throw ArgumentError.value(index, 'index', 'must be non-negative');
    }
    return _newTensorHandle(
      'module.bufferAt',
      (out) => _bindings.moduleBufferAt(module, index, out),
    );
  }

  void moduleSave(int module, String path) => _withPath(
    path,
    'module.save',
    (nativePath) => _bindings.moduleSave(module, nativePath),
  );

  void moduleLoad(int module, String path) => _withPath(
    path,
    'module.load',
    (nativePath) => _bindings.moduleLoad(module, nativePath),
  );

  void moduleRelease(int module) {
    _check(_bindings.moduleRelease(module), 'module.dispose');
  }

  void moduleReleaseFromFinalizer(int module) {
    _bindings.moduleRelease(module);
  }

  int createSgd(
    int module, {
    required double learningRate,
    required double momentum,
    required double weightDecay,
  }) => _newObjectHandle(
    'optimizer.sgd',
    (out) =>
        _bindings.sgdCreate(module, learningRate, momentum, weightDecay, out),
  );

  int createAdam(
    int module, {
    required double learningRate,
    required double beta1,
    required double beta2,
    required double epsilon,
    required double weightDecay,
    required bool decoupled,
  }) => _newObjectHandle(
    decoupled ? 'optimizer.adamW' : 'optimizer.adam',
    (out) => (decoupled ? _bindings.adamWCreate : _bindings.adamCreate)(
      module,
      learningRate,
      beta1,
      beta2,
      epsilon,
      weightDecay,
      out,
    ),
  );

  int createParameterSgd(
    List<int> parameterHandles, {
    required double learningRate,
    required double momentum,
    required double weightDecay,
  }) => _withTensorHandles(
    parameterHandles,
    'optimizer.parameterSgd',
    (handles, count, out) => _bindings.parameterSgdCreate(
      handles,
      count,
      learningRate,
      momentum,
      weightDecay,
      out,
    ),
  );

  int createParameterAdam(
    List<int> parameterHandles, {
    required double learningRate,
    required double beta1,
    required double beta2,
    required double epsilon,
    required double weightDecay,
    required bool decoupled,
  }) => _withTensorHandles(
    parameterHandles,
    decoupled ? 'optimizer.parameterAdamW' : 'optimizer.parameterAdam',
    (handles, count, out) =>
        (decoupled
            ? _bindings.parameterAdamWCreate
            : _bindings.parameterAdamCreate)(
          handles,
          count,
          learningRate,
          beta1,
          beta2,
          epsilon,
          weightDecay,
          out,
        ),
  );

  void optimizerZeroGrad(int optimizer) {
    _check(_bindings.optimizerZeroGrad(optimizer), 'optimizer.zeroGrad');
  }

  void optimizerStep(int optimizer) {
    _check(_bindings.optimizerStep(optimizer), 'optimizer.step');
  }

  void optimizerRelease(int optimizer) {
    _check(_bindings.optimizerRelease(optimizer), 'optimizer.dispose');
  }

  void optimizerReleaseFromFinalizer(int optimizer) {
    _bindings.optimizerRelease(optimizer);
  }

  void parameterOptimizerZeroGrad(int optimizer) {
    _check(
      _bindings.parameterOptimizerZeroGrad(optimizer),
      'optimizer.parameterZeroGrad',
    );
  }

  void parameterOptimizerStep(int optimizer) {
    _check(
      _bindings.parameterOptimizerStep(optimizer),
      'optimizer.parameterStep',
    );
  }

  void parameterOptimizerRelease(int optimizer) {
    _check(
      _bindings.parameterOptimizerRelease(optimizer),
      'optimizer.parameterDispose',
    );
  }

  void parameterOptimizerReleaseFromFinalizer(int optimizer) {
    _bindings.parameterOptimizerRelease(optimizer);
  }

  int liveModuleCount() =>
      _uint64Counter('runtime.liveModuleCount', _bindings.liveModuleCount);

  int liveOptimizerCount() => _uint64Counter(
    'runtime.liveOptimizerCount',
    _bindings.liveOptimizerCount,
  );

  int _moduleCount(
    int module,
    String operation,
    int Function(int, Pointer<Size>) call,
  ) {
    final value = calloc<Size>();
    try {
      _check(call(module, value), operation);
      return value.value;
    } finally {
      calloc.free(value);
    }
  }

  int _uint64Counter(String operation, int Function(Pointer<Uint64>) call) {
    final value = calloc<Uint64>();
    try {
      _check(call(value), operation);
      return value.value;
    } finally {
      calloc.free(value);
    }
  }

  int _withTensorHandles(
    List<int> handles,
    String operation,
    int Function(Pointer<Uint64>, int, Pointer<Uint64>) call,
  ) {
    if (handles.isEmpty) {
      throw ArgumentError.value(handles, 'handles', 'must not be empty');
    }
    final nativeHandles = calloc<Uint64>(handles.length);
    try {
      for (var index = 0; index < handles.length; index++) {
        nativeHandles[index] = handles[index];
      }
      return _newObjectHandle(
        operation,
        (out) => call(nativeHandles, handles.length, out),
      );
    } finally {
      calloc.free(nativeHandles);
    }
  }

  int _newTensorHandle(String operation, int Function(Pointer<Uint64>) call) =>
      _newObjectHandle(operation, call);

  int _newObjectHandle(String operation, int Function(Pointer<Uint64>) call) {
    final output = calloc<Uint64>();
    try {
      _check(call(output), operation);
      if (output.value == 0) {
        throw NativeRuntimeException(
          'Native runtime returned a null handle on success.',
          operation: operation,
        );
      }
      return output.value;
    } finally {
      calloc.free(output);
    }
  }

  void _withPath(
    String path,
    String operation,
    int Function(Pointer<Utf8>) call,
  ) {
    if (path.trim().isEmpty) {
      throw ArgumentError.value(path, 'path', 'must not be empty');
    }
    final nativePath = path.toNativeUtf8();
    try {
      _check(call(nativePath), operation);
    } finally {
      calloc.free(nativePath);
    }
  }

  void _check(int status, String operation) {
    if (status == 0) return;

    final pointer = _bindings.lastErrorMessage();
    final message =
        pointer.address == 0
            ? 'Native runtime returned status $status without a diagnostic.'
            : pointer.toDartString();

    switch (status) {
      case 1:
        throw InvalidArgumentException(message, operation: operation);
      case 2:
        throw InvalidShapeException(message, operation: operation);
      case 3:
        throw OutOfMemoryException(message, operation: operation);
      case 4:
        throw UnsupportedOperationException(message, operation: operation);
      case 5:
      case 6:
        throw NativeRuntimeException(message, operation: operation);
      default:
        throw NativeRuntimeException(
          'Unknown native status $status: $message',
          operation: operation,
        );
    }
  }
}
