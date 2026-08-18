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
  TrackingMoveLeaf({required Device initialDevice, this.failFirstMove = false})
    : _device = initialDevice;

  final bool failFirstMove;
  Device _device;
  bool _failed = false;
  int moveCalls = 0;
  final List<Device> history = <Device>[];

  Device get currentDevice => _device;

  @override
  Device get internalMoveDevice => _device;

  @override
  void internalOnMove(Device device) {
    moveCalls += 1;
    _device = device;
    history.add(device);
    if (failFirstMove && !_failed) {
      _failed = true;
      throw StateError('synthetic late move failure');
    }
  }

  @override
  Tensor forward(Tensor input) => input;
}

final class RollbackFailLeaf extends nn.Module {
  RollbackFailLeaf({required Device initialDevice}) : _device = initialDevice;

  Device _device;
  bool _forwardFailureSeen = false;

  @override
  Device get internalMoveDevice => _device;

  @override
  void internalOnMove(Device device) {
    if (device == Device.cpu && !_forwardFailureSeen) {
      _device = device;
      _forwardFailureSeen = true;
      throw StateError('synthetic move failure');
    }
    if (_forwardFailureSeen) {
      throw StateError('synthetic rollback failure');
    }
    _device = device;
  }

  @override
  Tensor forward(Tensor input) => input;
}

final class ThrowingForward extends nn.Module {
  @override
  Tensor forward(Tensor input) => throw StateError('synthetic forward failure');
}

final class BufferHolder extends nn.Module {
  BufferHolder(this.buffer)
    : _buffers = <nn.NamedBuffer>[nn.NamedBuffer('running', buffer)];

  final nn.Buffer buffer;
  final List<nn.NamedBuffer> _buffers;

  @override
  List<nn.NamedBuffer> get internalRegisteredBuffers => _buffers;

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
    final before = beforeTensor.toList<double>();
    beforeTensor.dispose();

    const lossFunction = nn.MSELoss();
    expect(lossFunction.toString(), 'MSELoss()');
    final optimizer = optim.Adam(parameters: parameters, learningRate: 0.03);
    addTearDown(optimizer.dispose);

    var initialLoss = double.nan;
    var finalLoss = double.nan;
    for (var step = 0; step < 300; step++) {
      optimizer.zeroGrad();
      final prediction = model(input);
      final loss = lossFunction(prediction, target);
      finalLoss = loss.toList<double>().single;
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
    final trained = trainedTensor.toList<double>();
    trainedTensor.dispose();
    expect(trained, isNot(equals(before)));

    final loadResult = model.loadStateDict(snapshot);
    expect(loadResult.isSuccess, isTrue);
    expect(loadResult.missingKeys, isEmpty);
    expect(loadResult.unexpectedKeys, isEmpty);

    final restoredTensor = model(input);
    final restored = restoredTensor.toList<double>();
    restoredTensor.dispose();
    expectValues(restored, before, tolerance: 2e-5);
    expect(model.buildCount, 1);

    optimizer.dispose();
    snapshot.dispose();
    model.dispose();
    expect(TensoraRuntime.liveOptimizerCount, liveOptimizersBefore);
    expect(TensoraRuntime.liveModuleCount, liveModulesBefore);
  });

  test('all activation modules and both loss objects execute native paths', () {
    final input = Tensor.fromList([-1, 0, 1], shape: Shape([3]));
    addTearDown(input.dispose);
    final modules = <nn.Module>[
      nn.ReLU(),
      nn.Sigmoid(),
      nn.Tanh(),
      nn.GELU(),
      nn.SiLU(),
    ];
    addTearDown(() {
      for (final module in modules) {
        module.dispose();
      }
    });

    for (final module in modules) {
      final output = module(input);
      expect(output.shape, Shape([3]));
      output.dispose();
    }

    final swigluInput = Tensor.fromList([1, -1, 2, 3], shape: Shape([1, 4]));
    final swiglu = nn.SwiGLU();
    addTearDown(swigluInput.dispose);
    addTearDown(swiglu.dispose);
    final gated = swiglu(swigluInput);
    addTearDown(gated.dispose);
    expect(gated.shape, Shape([1, 2]));

    final prediction = Tensor.fromList([1, 2], shape: Shape([2]));
    final target = Tensor.fromList([0, 0], shape: Shape([2]));
    addTearDown(prediction.dispose);
    addTearDown(target.dispose);
    const mse = nn.MSELoss();
    final mseValue = mse(prediction, target);
    addTearDown(mseValue.dispose);
    expect(mseValue.toList().single, closeTo(2.5, 1e-6));

    final logits = Tensor.fromList([2, 1], shape: Shape([1, 2]));
    final oneHot = Tensor.fromList([1, 0], shape: Shape([1, 2]));
    addTearDown(logits.dispose);
    addTearDown(oneHot.dispose);
    const crossEntropy = nn.CrossEntropyLoss();
    expect(crossEntropy.toString(), 'CrossEntropyLoss()');
    final ceValue = crossEntropy(logits, oneHot);
    addTearDown(ceValue.dispose);
    expect(ceValue.toList().single, closeTo(0.31326166, 1e-5));
  });

  test('Sequential releases intermediates when a later child throws', () {
    final input = Tensor.fromList([-1, 2], shape: Shape([2]));
    addTearDown(input.dispose);
    final baseline = TensoraRuntime.liveTensorCount;
    final sequence = nn.Sequential(
      children: <nn.Module>[nn.ReLU(), ThrowingForward()],
    );
    addTearDown(sequence.dispose);

    expect(() => sequence(input), throwsStateError);
    expect(TensoraRuntime.liveTensorCount, baseline);

    final identitySequence = nn.Sequential(
      children: <nn.Module>[nn.Identity(), nn.ReLU()],
    );
    addTearDown(identitySequence.dispose);
    final output = identitySequence(input);
    addTearDown(output.dispose);
    expectValues(output.toList(), <double>[0, 2]);
  });

  test('Buffer exposes native metadata, snapshots, and disposed guards', () {
    final tensor = Tensor.fromList([3, 4], shape: Shape([2]));
    final buffer = nn.Buffer.fromTensor(tensor);
    expect(buffer.persistent, isTrue);
    expect(buffer.shape, Shape([2]));
    expect(buffer.dtype, DType.float32);
    expect(buffer.device, Device.cpu);
    expect(buffer.identity, greaterThan(0));
    expect(buffer.tensorForRuntime, same(tensor));

    final snapshot = buffer.snapshot();
    expectValues(snapshot.toList(), <double>[3, 4]);
    expect(NativeTensorState.identity(snapshot), isNot(buffer.identity));
    snapshot.dispose();

    buffer.dispose();
    buffer.dispose();
    expect(buffer.isDisposed, isTrue);
    expect(() => buffer.shape, throwsA(isA<DisposedTensorException>()));
    expect(() => buffer.dtype, throwsA(isA<DisposedTensorException>()));
    expect(() => buffer.device, throwsA(isA<DisposedTensorException>()));
    expect(() => buffer.snapshot(), throwsA(isA<DisposedTensorException>()));
    expect(
      () => buffer.tensorForRuntime,
      throwsA(isA<DisposedTensorException>()),
    );
  });

  test(
    'persistent buffers round-trip through StateDict; transient ones do not',
    () {
      final persistentTensor = Tensor.fromList([5], shape: Shape([1]));
      final persistent = nn.Buffer.fromTensor(persistentTensor);
      final holder = BufferHolder(persistent);
      addTearDown(holder.dispose);

      expect(holder.buffers, <nn.Buffer>[persistent]);
      expect(holder.namedBuffers.single.name, 'running');
      final state = holder.stateDict();
      addTearDown(state.dispose);
      expect(state.length, 1);
      expect(state.isEmpty, isFalse);
      expect(state.keys, <String>['running']);
      expect(state['running'], isNotNull);
      expect(state.entries.keys, <String>['running']);
      expect(
        () => state.entries['other'] = state['running']!,
        throwsUnsupportedError,
      );

      final replacement = Tensor.fromList([9], shape: Shape([1]));
      addTearDown(replacement.dispose);
      NativeTensorState.assignMany(
        targets: <Tensor>[persistent.tensorForRuntime],
        sources: <Tensor>[replacement],
      );
      expectValues(persistent.tensorForRuntime.toList(), <double>[9]);
      expect(holder.loadStateDict(state).isSuccess, isTrue);
      expectValues(persistent.tensorForRuntime.toList(), <double>[5]);

      final transientTensor = Tensor.fromList([7], shape: Shape([1]));
      final transient = nn.Buffer.fromTensor(
        transientTensor,
        persistent: false,
      );
      final transientHolder = BufferHolder(transient);
      addTearDown(transientHolder.dispose);
      final transientState = transientHolder.stateDict();
      addTearDown(transientState.dispose);
      expect(transientState.isEmpty, isTrue);
    },
  );

  test('StateDict strictness and metadata checks are deterministic', () {
    final layer = nn.Linear(inFeatures: 1, outFeatures: 1);
    addTearDown(layer.dispose);

    final extra = Tensor.fromList([1], shape: Shape([1]));
    final extraState = nn.StateDict.fromOwned(<String, Tensor>{'extra': extra});
    addTearDown(extraState.dispose);
    expect(
      () => layer.loadStateDict(extraState),
      throwsA(isA<InvalidArgumentException>()),
    );
    final nonStrict = layer.loadStateDict(extraState, strict: false);
    expect(nonStrict.missingKeys, <String>['bias', 'weight']);
    expect(nonStrict.unexpectedKeys, <String>['extra']);
    expect(nonStrict.isSuccess, isFalse);

    final wrongWeight = Tensor.fromList([1, 2], shape: Shape([2]));
    final wrongState = nn.StateDict.fromOwned(<String, Tensor>{
      'weight': wrongWeight,
    });
    addTearDown(wrongState.dispose);
    expect(
      () => layer.loadStateDict(wrongState, strict: false),
      throwsA(isA<InvalidArgumentException>()),
    );

    extraState.dispose();
    expect(extraState.isDisposed, isTrue);
    expect(() => extraState.length, throwsStateError);
    expect(() => extraState.keys, throwsStateError);
    expect(() => extraState.entries, throwsStateError);
    expect(() => extraState['extra'], throwsStateError);
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

    layer.eval();
    expect(layer.isTraining, isFalse);
    layer.train();
    expect(layer.isTraining, isTrue);
    layer.to(Device.cpu);
    expect(layer.parameters.map((parameter) => parameter.identity), identities);
    expect(
      layer.parameters.every((parameter) => parameter.device == Device.cpu),
      isTrue,
    );
  });

  test('unavailable explicit accelerator requests fail before mutation', () {
    final module = nn.Identity();
    addTearDown(module.dispose);
    if (TensoraRuntime.deviceCount(Device.cuda(0)) == 0) {
      expect(
        () => module.to(Device.cuda(0)),
        throwsA(isA<UnsupportedOperationException>()),
      );
    }
  });

  test('module move rolls back every attempted leaf after a late failure', () {
    final original = Device.cuda(0);
    final first = TrackingMoveLeaf(initialDevice: original);
    final failing = TrackingMoveLeaf(
      initialDevice: original,
      failFirstMove: true,
    );
    final tree = nn.Sequential(children: <nn.Module>[first, failing]);
    addTearDown(tree.dispose);

    expect(() => tree.to(Device.cpu), throwsA(isA<StateError>()));
    expect(first.currentDevice, original);
    expect(failing.currentDevice, original);
    expect(first.moveCalls, 2, reason: 'first leaf must be rolled back');
    expect(
      failing.moveCalls,
      2,
      reason: 'failing leaf must also be rolled back',
    );
    expect(first.history, <Device>[Device.cpu, original]);
    expect(failing.history, <Device>[Device.cpu, original]);
  });

  test('rollback failures surface an explicit indeterminate-state error', () {
    final first = TrackingMoveLeaf(initialDevice: Device.cuda(0));
    final failing = RollbackFailLeaf(initialDevice: Device.cuda(0));
    final tree = nn.Sequential(children: <nn.Module>[first, failing]);
    addTearDown(tree.dispose);

    expect(() => tree.to(Device.cpu), throwsA(isA<NativeRuntimeException>()));
  });
}
