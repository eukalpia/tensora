import 'dart:io';

import 'package:tensora_train/tensora_train.dart';
import 'package:test/test.dart';

final class _TinyMLP extends Model {
  int buildCalls = 0;

  @override
  Module build() {
    buildCalls += 1;
    return Sequential(
      children: <Module>[
        Linear(inFeatures: 1, outFeatures: 4),
        GELU(),
        Linear(inFeatures: 4, outFeatures: 1),
      ],
    );
  }
}

final class _MoveProbe extends Module {
  _MoveProbe({required Device initialDevice, this.failWhenMovingTo})
    : _device = initialDevice;

  Device _device;
  final Device? failWhenMovingTo;
  final List<Device> history = <Device>[];

  Device get currentDevice => _device;

  @override
  Device get internalMoveDevice => _device;

  @override
  Tensor forward(Tensor input) => input;

  @override
  void internalOnMove(Device device) {
    _device = device;
    history.add(device);
    if (device == failWhenMovingTo) {
      throw StateError('synthetic move failure');
    }
  }
}

void expectValues(
  List<double> actual,
  List<double> expected, {
  double tolerance = 1e-5,
}) {
  expect(actual, hasLength(expected.length));
  for (var index = 0; index < expected.length; index++) {
    expect(actual[index], closeTo(expected[index], tolerance));
  }
}

void main() {
  final nativeLibrary = Platform.environment['TENSORA_NATIVE_LIBRARY'];
  final integrationSkip = nativeLibrary == null || nativeLibrary.isEmpty
      ? 'requires the NN V2 native integration runtime'
      : false;

  test(
    'declarative MLP trains, freezes, moves, snapshots, and restores',
    () {
      TensoraRuntime.manualSeed(20260816);
      final startModules = TensoraRuntime.liveModuleCount;
      final startOptimizers = TensoraRuntime.liveOptimizerCount;

      final x = Tensor.fromList(<num>[-1, 0, 1, 2], shape: Shape(<int>[4, 1]));
      final y = Tensor.fromList(<num>[-1, 1, 3, 5], shape: Shape(<int>[4, 1]));
      final model = _TinyMLP();
      final criterion = MSELoss();

      expect(model.isMaterialized, isFalse);
      expect(model.buildCalls, 0);
      final parameters = model.parameters;
      expect(model.isMaterialized, isTrue);
      expect(model.buildCalls, 1);
      expect(parameters, hasLength(4));
      final identitiesBeforeMove = parameters
          .map((parameter) => parameter.identity)
          .toList(growable: false);

      model.train();
      model.to(Device.cpu);
      expect(model.buildCalls, 1);
      expect(
        model.parameters.map((parameter) => parameter.identity),
        orderedEquals(identitiesBeforeMove),
      );
      expect(model.parameters.every((parameter) => parameter.device == Device.cpu), isTrue);

      final optimizer = AdamW(
        parameters: model.parameters,
        learningRate: 0.03,
        weightDecay: 0,
      );

      double? initialLoss;
      var finalLoss = double.nan;
      for (var step = 0; step < 250; step++) {
        optimizer.zeroGrad();
        final prediction = model(x);
        final loss = criterion(prediction, y);
        final value = loss.toList().single;
        initialLoss ??= value;
        finalLoss = value;
        loss.backward();
        optimizer.step();
        loss.dispose();
        prediction.dispose();
      }

      expect(initialLoss, isNotNull);
      expect(initialLoss!.isFinite, isTrue);
      expect(finalLoss, lessThan(initialLoss!));
      expect(finalLoss, lessThan(0.05));
      expect(model.buildCalls, 1);

      optimizer.dispose();

      final frozen = model.parameters.first;
      final frozenIdentity = frozen.identity;
      final beforeFreezeStep = frozen.snapshot();
      frozen.freeze();
      expect(frozen.requiresGrad, isFalse);
      expect(frozen.identity, frozenIdentity);

      final frozenOptimizer = AdamW(
        parameters: model.parameters,
        learningRate: 0.01,
        weightDecay: 0,
      );
      frozenOptimizer.zeroGrad();
      final frozenPrediction = model(x);
      final frozenLoss = criterion(frozenPrediction, y);
      frozenLoss.backward();
      frozenOptimizer.step();
      final afterFreezeStep = frozen.snapshot();
      expectValues(afterFreezeStep.toList(), beforeFreezeStep.toList());
      frozenLoss.dispose();
      frozenPrediction.dispose();
      beforeFreezeStep.dispose();
      afterFreezeStep.dispose();
      frozenOptimizer.dispose();

      frozen.unfreeze();
      expect(frozen.requiresGrad, isTrue);
      expect(frozen.identity, frozenIdentity);

      model.eval();
      final state = model.stateDict();
      final savedOutputTensor = model(x);
      final savedOutput = savedOutputTensor.toList();
      savedOutputTensor.dispose();

      final perturbOptimizer = SGD(
        parameters: model.parameters,
        learningRate: 0.01,
      );
      for (var step = 0; step < 8; step++) {
        perturbOptimizer.zeroGrad();
        final prediction = model(x);
        final loss = criterion(prediction, y);
        loss.backward();
        perturbOptimizer.step();
        loss.dispose();
        prediction.dispose();
      }
      perturbOptimizer.dispose();

      final loadResult = model.loadStateDict(state);
      expect(loadResult.isSuccess, isTrue);
      final restoredOutput = model(x);
      expectValues(restoredOutput.toList(), savedOutput, tolerance: 2e-5);
      restoredOutput.dispose();

      final tree = model.toTreeString();
      expect(tree, contains('_TinyMLP'));
      expect(tree, contains('Sequential'));
      expect(tree, contains('Linear(inFeatures: 1, outFeatures: 4'));
      expect(tree, contains('GELU()'));
      expect(tree, contains('Linear(inFeatures: 4, outFeatures: 1'));
      expect(tree, isNot(contains('handle')));

      state.dispose();
      model.dispose();
      x.dispose();
      y.dispose();

      expect(TensoraRuntime.liveOptimizerCount, startOptimizers);
      expect(TensoraRuntime.liveModuleCount, startModules);
    },
    skip: integrationSkip,
  );

  test(
    'transactional device move rolls back every attempted leaf',
    () {
      final first = _MoveProbe(initialDevice: Device.cuda(0));
      final second = _MoveProbe(
        initialDevice: Device.cuda(0),
        failWhenMovingTo: Device.cpu,
      );
      final model = Sequential(children: <Module>[first, second]);

      expect(() => model.to(Device.cpu), throwsStateError);
      expect(first.currentDevice, Device.cuda(0));
      expect(second.currentDevice, Device.cuda(0));
      expect(first.history, orderedEquals(<Device>[Device.cpu, Device.cuda(0)]));
      expect(second.history, orderedEquals(<Device>[Device.cpu, Device.cuda(0)]));

      model.dispose();
    },
    skip: integrationSkip,
  );
}
