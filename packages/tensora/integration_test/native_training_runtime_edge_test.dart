import 'package:tensora/src/native/native_training_runtime.dart';
import 'package:tensora/src/tensor/native_adoption.dart';
import 'package:tensora/src/training/finalizer_release.dart';
import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

int nativeHandle(Tensor tensor) =>
    tensor.nativeHandleForRuntime(nativeTensorAdoptionToken);

Parameter makeParameter(double value) {
  final source = Tensor.fromList(<num>[value], shape: Shape(<int>[1]));
  final leaf = source.withRequiresGrad();
  source.dispose();
  return Parameter.fromTensor(leaf);
}

void main() {
  final runtime = NativeTrainingRuntime.instance;
  final invalidHandleError = isA<NativeRuntimeException>();

  test('training runtime validates arguments before native calls', () {
    expect(() => runtime.manualSeed(-1), throwsArgumentError);
    expect(() => runtime.createLinear(0, 1, true), throwsArgumentError);
    expect(() => runtime.createLinear(1, 0, true), throwsArgumentError);
    expect(() => runtime.moduleParameterAt(0, -1), throwsArgumentError);
    expect(() => runtime.moduleBufferAt(0, -1), throwsArgumentError);
    expect(() => runtime.moduleSave(0, ''), throwsArgumentError);
    expect(() => runtime.moduleLoad(0, '  '), throwsArgumentError);
    expect(
      () => runtime.assignMany(<int>[1], const <int>[]),
      throwsArgumentError,
    );
    expect(
      () => runtime.createParameterSgd(
        const <int>[],
        learningRate: 0.01,
        momentum: 0,
        weightDecay: 0,
      ),
      throwsArgumentError,
    );
    expect(
      () => runtime.createParameterAdam(
        const <int>[],
        learningRate: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weightDecay: 0,
        decoupled: false,
      ),
      throwsArgumentError,
    );
    runtime.assignMany(const <int>[], const <int>[]);
  });

  test('training wrapper maps every public accelerator device code', () {
    expect(runtime.cudaDeviceCount(), greaterThanOrEqualTo(0));

    final module = runtime.createLinear(1, 1, true);
    try {
      for (final device in <Device>[
        Device.cuda(0),
        Device.mps,
        Device.xpu(0),
        Device.hip(0),
      ]) {
        final count = runtime.deviceCount(device);
        if (count == 0) {
          expect(
            () => runtime.moduleToDevice(module, device),
            throwsA(isA<UnsupportedOperationException>()),
            reason: '$device is unavailable on this validation host',
          );
        }
      }
    } finally {
      runtime.moduleRelease(module);
    }
  });

  test('invalid module handles fail deterministically across operations', () {
    expect(() => runtime.moduleForward(0, 0), throwsA(invalidHandleError));
    expect(
      () => runtime.moduleSetTraining(0, true),
      throwsA(invalidHandleError),
    );
    expect(
      () => runtime.moduleToDevice(0, Device.cpu),
      throwsA(invalidHandleError),
    );
    expect(() => runtime.moduleParameterCount(0), throwsA(invalidHandleError));
    expect(() => runtime.moduleBufferCount(0), throwsA(invalidHandleError));
    expect(() => runtime.moduleParameterAt(0, 0), throwsA(invalidHandleError));
    expect(() => runtime.moduleBufferAt(0, 0), throwsA(invalidHandleError));
    expect(() => runtime.moduleRelease(0), throwsA(invalidHandleError));
  });

  test('invalid optimizer handles fail without corrupting diagnostics', () {
    expect(
      () =>
          runtime.createSgd(0, learningRate: 0.01, momentum: 0, weightDecay: 0),
      throwsA(invalidHandleError),
    );
    expect(
      () => runtime.createAdam(
        0,
        learningRate: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weightDecay: 0,
        decoupled: false,
      ),
      throwsA(invalidHandleError),
    );
    expect(() => runtime.optimizerZeroGrad(0), throwsA(invalidHandleError));
    expect(() => runtime.optimizerStep(0), throwsA(invalidHandleError));
    expect(() => runtime.optimizerRelease(0), throwsA(invalidHandleError));
    expect(
      () => runtime.createParameterSgd(
        const <int>[0],
        learningRate: 0.01,
        momentum: 0,
        weightDecay: 0,
      ),
      throwsA(invalidHandleError),
    );
    expect(
      () => runtime.createParameterAdam(
        const <int>[0],
        learningRate: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weightDecay: 0,
        decoupled: true,
      ),
      throwsA(invalidHandleError),
    );
    expect(
      () => runtime.parameterOptimizerZeroGrad(0),
      throwsA(invalidHandleError),
    );
    expect(
      () => runtime.parameterOptimizerStep(0),
      throwsA(invalidHandleError),
    );
    expect(
      () => runtime.parameterOptimizerRelease(0),
      throwsA(invalidHandleError),
    );
    expect(() => runtime.tensorIdentity(0), throwsA(invalidHandleError));
    expect(() => runtime.cloneDetached(0), throwsA(invalidHandleError));
    expect(() => runtime.setRequiresGrad(0, false), throwsA(invalidHandleError));

    expect(runtime.trainingAvailable(), isTrue);
    expect(runtime.liveModuleCount(), greaterThanOrEqualTo(0));
    expect(runtime.liveOptimizerCount(), greaterThanOrEqualTo(0));
  });

  test('NN V2 tensor state facade is native, detached, and transactional', () {
    final target = Tensor.fromList(<num>[1, 2], shape: Shape(<int>[2]));
    final source = Tensor.fromList(<num>[7, 8], shape: Shape(<int>[2]));
    addTearDown(target.dispose);
    addTearDown(source.dispose);

    final identity = NativeTensorState.identity(target);
    expect(identity, greaterThan(0));
    expect(runtime.tensorIdentity(nativeHandle(target)), identity);

    final clone = NativeTensorState.cloneDetached(target);
    addTearDown(clone.dispose);
    expect(clone.requiresGrad, isFalse);
    expect(clone.toList(), <double>[1, 2]);
    expect(NativeTensorState.identity(clone), isNot(identity));

    NativeTensorState.assignMany(
      targets: <Tensor>[target],
      sources: <Tensor>[source],
    );
    expect(target.toList(), <double>[7, 8]);

    NativeTensorState.assignMany(targets: const <Tensor>[], sources: const <Tensor>[]);
    expect(
      () => NativeTensorState.assignMany(
        targets: <Tensor>[target],
        sources: const <Tensor>[],
      ),
      throwsArgumentError,
    );

    final wrongShape = Tensor.ones(Shape(<int>[1]));
    addTearDown(wrongShape.dispose);
    expect(
      () => NativeTensorState.assignMany(
        targets: <Tensor>[target],
        sources: <Tensor>[wrongShape],
      ),
      throwsA(isA<InvalidShapeException>()),
    );
  });

  test('raw parameter optimizer wrappers own and release native state', () {
    final first = makeParameter(1);
    final second = makeParameter(2);
    addTearDown(first.dispose);
    addTearDown(second.dispose);
    final handles = <int>[
      nativeHandle(first.tensorForRuntime),
      nativeHandle(second.tensorForRuntime),
    ];

    final firstLoss = first.tensorForRuntime.sum();
    final secondLoss = second.tensorForRuntime.sum();
    firstLoss.backward();
    secondLoss.backward();
    firstLoss.dispose();
    secondLoss.dispose();

    final sgd = runtime.createParameterSgd(
      handles,
      learningRate: 0.05,
      momentum: 0.1,
      weightDecay: 0.01,
    );
    runtime.parameterOptimizerStep(sgd);
    runtime.parameterOptimizerZeroGrad(sgd);
    runtime.parameterOptimizerRelease(sgd);

    for (final decoupled in <bool>[false, true]) {
      final adam = runtime.createParameterAdam(
        handles,
        learningRate: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weightDecay: 0.01,
        decoupled: decoupled,
      );
      runtime.parameterOptimizerZeroGrad(adam);
      runtime.parameterOptimizerStep(adam);
      runtime.parameterOptimizerRelease(adam);
    }

    final finalizerOwned = runtime.createParameterSgd(
      handles,
      learningRate: 0.01,
      momentum: 0,
      weightDecay: 0,
    );
    releaseParameterOptimizerHandleFromFinalizer(finalizerOwned);
  });

  test('NativeParameterOptimizer validates ownership and lifecycle', () {
    final first = makeParameter(1);
    final second = makeParameter(2);
    addTearDown(first.dispose);
    addTearDown(second.dispose);

    expect(
      () => NativeParameterOptimizer.sgd(
        parameters: const <Parameter>[],
        learningRate: 0.01,
        momentum: 0,
        weightDecay: 0,
      ),
      throwsArgumentError,
    );

    final sgd = NativeParameterOptimizer.sgd(
      parameters: <Parameter>[first, second],
      learningRate: 0.01,
      momentum: 0,
      weightDecay: 0,
    );
    expect(sgd.isDisposed, isFalse);
    sgd.zeroGrad();
    sgd.step();
    sgd.dispose();
    sgd.dispose();
    expect(sgd.isDisposed, isTrue);
    expect(sgd.zeroGrad, throwsStateError);
    expect(sgd.step, throwsStateError);

    final adam = NativeParameterOptimizer.adam(
      parameters: <Parameter>[first, second],
      learningRate: 0.001,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
      weightDecay: 0,
    );
    adam.zeroGrad();
    adam.step();
    adam.dispose();

    final adamW = NativeParameterOptimizer.adam(
      parameters: <Parameter>[first, second],
      learningRate: 0.001,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
      weightDecay: 0.01,
      decoupled: true,
    );
    adamW.dispose();

    final disposed = makeParameter(3)..dispose();
    expect(
      () => NativeParameterOptimizer.sgd(
        parameters: <Parameter>[disposed],
        learningRate: 0.01,
        momentum: 0,
        weightDecay: 0,
      ),
      throwsArgumentError,
    );
  });

  test('Parameter covers state, replacement, freeze, and disposal contracts', () {
    final module = Linear(1, 1);
    addTearDown(module.dispose);

    final initialViews = module.parameters();
    final parameter = Parameter.fromTensor(initialViews.first);
    for (final view in initialViews.skip(1)) {
      view.dispose();
    }
    addTearDown(parameter.dispose);

    expect(parameter.shape, Shape(<int>[1, 1]));
    expect(parameter.dtype, DType.float32);
    expect(parameter.device, Device.cpu);
    expect(parameter.requiresGrad, isTrue);
    expect(parameter.isDisposed, isFalse);

    final identity = parameter.identity;
    parameter.freeze();
    expect(parameter.requiresGrad, isFalse);
    parameter.unfreeze();
    expect(parameter.requiresGrad, isTrue);
    expect(parameter.identity, identity);

    final loss = parameter.tensorForRuntime.sum();
    loss.backward();
    loss.dispose();
    final gradient = parameter.grad();
    expect(gradient.toList(), <double>[1]);
    gradient.dispose();

    final snapshot = parameter.snapshot();
    expect(snapshot.toList(), parameter.tensorForRuntime.toList());
    expect(snapshot.requiresGrad, isFalse);
    snapshot.dispose();

    final refreshed = module.parameters();
    final replacement = refreshed.first;
    for (final view in refreshed.skip(1)) {
      view.dispose();
    }
    parameter.internalReplaceTensor(replacement);
    expect(parameter.identity, identity);

    final mismatched = Tensor.ones(Shape(<int>[1]));
    expect(
      () => parameter.internalReplaceTensor(mismatched),
      throwsA(isA<NativeRuntimeException>()),
    );
    expect(mismatched.isDisposed, isTrue);
    expect(parameter.identity, identity);

    parameter.dispose();
    parameter.dispose();
    expect(parameter.isDisposed, isTrue);
    expect(() => parameter.shape, throwsA(isA<DisposedTensorException>()));
    expect(() => parameter.dtype, throwsA(isA<DisposedTensorException>()));
    expect(() => parameter.device, throwsA(isA<DisposedTensorException>()));
    expect(
      () => parameter.requiresGrad,
      throwsA(isA<DisposedTensorException>()),
    );
    expect(parameter.grad, throwsA(isA<DisposedTensorException>()));
    expect(parameter.snapshot, throwsA(isA<DisposedTensorException>()));
    expect(
      () => parameter.tensorForRuntime,
      throwsA(isA<DisposedTensorException>()),
    );
    expect(parameter.freeze, throwsA(isA<DisposedTensorException>()));
    expect(parameter.unfreeze, throwsA(isA<DisposedTensorException>()));

    final unusableReplacement = Tensor.ones(Shape(<int>[1, 1]));
    addTearDown(unusableReplacement.dispose);
    expect(
      () => parameter.internalReplaceTensor(unusableReplacement),
      throwsA(isA<DisposedTensorException>()),
    );
  });

  test(
    'finalizer release paths return module and optimizer counters to baseline',
    () {
      final baselineModules = runtime.liveModuleCount();
      final baselineOptimizers = runtime.liveOptimizerCount();
      final module = runtime.createLinear(1, 1, true);
      final optimizer = runtime.createSgd(
        module,
        learningRate: 0.01,
        momentum: 0,
        weightDecay: 0,
      );

      expect(runtime.liveModuleCount(), baselineModules + 1);
      expect(runtime.liveOptimizerCount(), baselineOptimizers + 1);

      runtime.optimizerReleaseFromFinalizer(optimizer);
      runtime.moduleReleaseFromFinalizer(module);

      expect(runtime.liveOptimizerCount(), baselineOptimizers);
      expect(runtime.liveModuleCount(), baselineModules);
    },
  );
}
