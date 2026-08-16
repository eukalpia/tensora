import 'package:tensora/tensora.dart';
import 'package:tensora_nn/tensora_nn.dart' as nn;
import 'package:tensora_optim/tensora_optim.dart' as optim;
import 'package:test/test.dart';

final class ToyMlp extends nn.Model {
  ToyMlp({this.hiddenFeatures = 4});

  final int hiddenFeatures;
  int buildCount = 0;

  @override
  nn.Module build() {
    buildCount += 1;
    return nn.Sequential(
      children: <nn.Module>[
        nn.Linear(inFeatures: 1, outFeatures: hiddenFeatures),
        nn.GELU(),
        nn.Linear(inFeatures: hiddenFeatures, outFeatures: 1),
      ],
    );
  }
}

final class TrackingMoveLeaf extends nn.Module {
  TrackingMoveLeaf({this.failFirstMove = false});

  final bool failFirstMove;
  Device _device = Device.cpu;
  bool _failed = false;
  int moveCalls = 0;

  Device get currentDevice => _device;

  @override
  Device get internalMoveDevice => _device;

  @override
  void internalOnMove(Device device) {
    moveCalls += 1;
    _device = device;
    if (failFirstMove && !_failed) {
      _failed = true;
      throw StateError('synthetic late move failure');
    }
  }

  @override
  Tensor forward(Tensor input) => input;
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
  test('declarative MLP trains, restores state, and builds exactly once', () {
    TensoraRuntime.manualSeed(20260816);
    final liveModulesBefore = TensoraRuntime.liveModuleCount;
    final liveOptimizersBefore = TensoraRuntime.liveOptimizerCount;

    final input = Tensor.fromList([-2, -1, 1, 2], shape: Shape([4, 1]));
    final target = Tensor.fromList([-4, -2, 2, 4], shape: Shape([4, 1]));
    final model = ToyMlp();
    addTearDown(input.dispose);
    addTearDown(target.dispose);
    addTearDown(model.dispose);

    expect(model.isMaterialized, isFalse);
    final parameters = model.parameters;
    expect(model.isMaterialized, isTrue);
    expect(model.buildCount, 1);
    expect(parameters, hasLength(4));
    expect(model.namedParameters.map((entry) => entry.name), <String>[
      '0.weight',
      '0.bias',
      '2.weight',
      '2.bias',
    ]);

    final tree = model.toTreeString();
    expect(tree, contains('ToyMlp'));
    expect(
      tree,
      contains('0: Linear(inFeatures: 1, outFeatures: 4, bias: true)'),
    );
    expect(tree, contains('1: GELU()'));
    expect(
      tree,
      contains('2: Linear(inFeatures: 4, outFeatures: 1, bias: true)'),
    );
    expect(tree, isNot(contains('handle')));

    final snapshot = model.stateDict();
    addTearDown(snapshot.dispose);
    expect(snapshot.length, 4);

    final beforeTensor = model(input);
    final before = beforeTensor.toList();
    beforeTensor.dispose();

    const lossFunction = nn.MSELoss();
    final optimizer = optim.Adam(parameters: parameters, learningRate: 0.03);
    addTearDown(optimizer.dispose);

    var initialLoss = double.nan;
    var finalLoss = double.nan;
    for (var step = 0; step < 300; step++) {
      optimizer.zeroGrad();
      final prediction = model(input);
      final loss = lossFunction(prediction, target);
      finalLoss = loss.toList().single;
      if (step == 0) initialLoss = finalLoss;
      loss.backward();
      optimizer.step();
      loss.dispose();
      prediction.dispose();
    }

    expect(initialLoss.isFinite, isTrue);
    expect(finalLoss.isFinite, isTrue);
    expect(finalLoss, lessThan(initialLoss * 0.1));
    expect(finalLoss, lessThan(0.05));
    expect(model.buildCount, 1);

    final trainedTensor = model(input);
    final trained = trainedTensor.toList();
    trainedTensor.dispose();
    expect(trained, isNot(equals(before)));

    final loadResult = model.loadStateDict(snapshot);
    expect(loadResult.isSuccess, isTrue);
    expect(loadResult.missingKeys, isEmpty);
    expect(loadResult.unexpectedKeys, isEmpty);

    final restoredTensor = model(input);
    final restored = restoredTensor.toList();
    restoredTensor.dispose();
    expectValues(restored, before, tolerance: 2e-5);
    expect(model.buildCount, 1);

    optimizer.dispose();
    snapshot.dispose();
    model.dispose();
    expect(TensoraRuntime.liveOptimizerCount, liveOptimizersBefore);
    expect(TensoraRuntime.liveModuleCount, liveModulesBefore);
  });

  test('parameter freeze and device refresh preserve opaque identity', () {
    final layer = nn.Linear(inFeatures: 2, outFeatures: 2);
    addTearDown(layer.dispose);
    final parameters = layer.parameters;
    expect(parameters, hasLength(2));
    final identities =
        parameters.map((parameter) => parameter.identity).toList();

    final weight = parameters.first;
    expect(weight.requiresGrad, isTrue);
    weight.freeze();
    expect(weight.requiresGrad, isFalse);
    expect(weight.identity, identities.first);
    weight.unfreeze();
    expect(weight.requiresGrad, isTrue);
    expect(weight.identity, identities.first);

    layer.to(Device.cpu);
    expect(layer.parameters.map((parameter) => parameter.identity), identities);
    expect(
      layer.parameters.every((parameter) => parameter.device == Device.cpu),
      isTrue,
    );
  });

  test('module move rolls back every attempted leaf after a late failure', () {
    final first = TrackingMoveLeaf();
    final failing = TrackingMoveLeaf(failFirstMove: true);
    final tree = nn.Sequential(children: <nn.Module>[first, failing]);
    addTearDown(tree.dispose);

    expect(() => tree.to(Device.cpu), throwsA(isA<StateError>()));
    expect(first.currentDevice, Device.cpu);
    expect(failing.currentDevice, Device.cpu);
    expect(first.moveCalls, 2, reason: 'first leaf must be rolled back');
    expect(
      failing.moveCalls,
      2,
      reason: 'failing leaf must also be rolled back',
    );
  });
}
