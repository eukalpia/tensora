import 'dart:io';

import 'package:tensora/tensora.dart' as core;
import 'package:tensora_nn/tensora_nn.dart';
import 'package:test/test.dart';

final class _BareModule extends Module {
  _BareModule({
    List<NamedModule> children = const <NamedModule>[],
    List<NamedParameter> parameters = const <NamedParameter>[],
    List<NamedBuffer> buffers = const <NamedBuffer>[],
  }) : _children = children,
       _parameters = parameters,
       _buffers = buffers;

  final List<NamedModule> _children;
  final List<NamedParameter> _parameters;
  final List<NamedBuffer> _buffers;
  int trainingChanges = 0;
  int disposeCalls = 0;

  @override
  List<NamedModule> get internalRegisteredChildren => _children;

  @override
  List<NamedParameter> get internalRegisteredParameters => _parameters;

  @override
  List<NamedBuffer> get internalRegisteredBuffers => _buffers;

  @override
  core.Tensor forward(core.Tensor input) => input;

  @override
  void internalOnTrainingModeChanged(bool training) {
    trainingChanges += 1;
  }

  @override
  void internalOnDispose() {
    disposeCalls += 1;
  }
}

final class _CaptureRelu extends Module {
  core.Tensor? captured;

  @override
  core.Tensor forward(core.Tensor input) {
    captured = input.relu();
    return captured!;
  }
}

final class _ThrowForward extends Module {
  @override
  core.Tensor forward(core.Tensor input) {
    throw StateError('synthetic forward failure');
  }
}

final class _RecursiveModel extends Model {
  @override
  Module build() {
    parameters;
    return Identity();
  }
}

final class _SelfModel extends Model {
  @override
  Module build() => this;
}

final class _FixedModel extends Model {
  _FixedModel(this.candidate);

  final Module candidate;

  @override
  Module build() => candidate;
}

final class _CycleCarrier extends Module {
  _CycleCarrier(this.child);

  final Module child;

  @override
  List<NamedModule> get internalRegisteredChildren => <NamedModule>[
    NamedModule('cycle', child),
  ];

  @override
  core.Tensor forward(core.Tensor input) => input;
}

final class _CycleModel extends Model {
  @override
  Module build() => _CycleCarrier(this);
}

final class _MoveLeaf extends Module {
  _MoveLeaf({
    required this.original,
    this.failPreflight = false,
    this.failFirstMove = false,
    this.failRollback = false,
  }) : device = original;

  final core.Device original;
  final bool failPreflight;
  final bool failFirstMove;
  final bool failRollback;
  core.Device device;
  bool _moveFailed = false;
  int moveCalls = 0;

  @override
  core.Device get internalMoveDevice => device;

  @override
  void internalPreflightMove(core.Device target) {
    if (failPreflight) throw StateError('synthetic preflight failure');
  }

  @override
  void internalOnMove(core.Device target) {
    moveCalls += 1;
    device = target;
    if (failFirstMove && !_moveFailed) {
      _moveFailed = true;
      throw StateError('synthetic move failure');
    }
    if (failRollback && _moveFailed && target == original) {
      throw StateError('synthetic rollback failure');
    }
  }

  @override
  core.Tensor forward(core.Tensor input) => input;
}

Parameter _parameter(double value) {
  final source = core.Tensor.fromList(<num>[value], shape: core.Shape(<int>[1]));
  final leaf = source.withRequiresGrad();
  source.dispose();
  return Parameter.fromTensor(leaf);
}

Buffer _buffer(double value, {bool persistent = true}) {
  return Buffer.fromTensor(
    core.Tensor.fromList(<num>[value], shape: core.Shape(<int>[1])),
    persistent: persistent,
  );
}

void _expectValues(
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
  final nativeSkip = nativeLibrary == null || nativeLibrary.isEmpty
      ? 'requires a Tensora native integration runtime'
      : false;

  test(
    'all activation and loss modules execute native primitives and diagnostics',
    () {
      final input = core.Tensor.fromList(
        <num>[-1, 0, 1],
        shape: core.Shape(<int>[3]),
      );
      addTearDown(input.dispose);

      final activations = <(Module, List<double>)>[
        (ReLU(), <double>[0, 0, 1]),
        (Sigmoid(), <double>[0.26894143, 0.5, 0.7310586]),
        (Tanh(), <double>[-0.7615942, 0, 0.7615942]),
        (GELU(), <double>[-0.15865526, 0, 0.8413447]),
        (SiLU(), <double>[-0.26894143, 0, 0.7310586]),
      ];
      for (final (activation, expected) in activations) {
        final output = activation(input);
        _expectValues(output.toList(), expected, tolerance: 3e-5);
        expect(activation.toString(), endsWith('()'));
        output.dispose();
        activation.dispose();
      }

      final gatedInput = core.Tensor.fromList(
        <num>[1, -1, 2, 3],
        shape: core.Shape(<int>[1, 4]),
      );
      final swiglu = SwiGLU();
      final gated = swiglu(gatedInput);
      expect(gated.shape, core.Shape(<int>[1, 2]));
      expect(swiglu.toString(), 'SwiGLU()');
      gated.dispose();
      gatedInput.dispose();
      swiglu.dispose();

      const mse = MSELoss();
      const crossEntropy = CrossEntropyLoss();
      final prediction = core.Tensor.fromList(
        <num>[1, 3],
        shape: core.Shape(<int>[1, 2]),
      );
      final target = core.Tensor.fromList(
        <num>[2, 2],
        shape: core.Shape(<int>[1, 2]),
      );
      final mseValue = mse(prediction, target);
      expect(mse.toString(), 'MSELoss()');
      expect(mseValue.toList().single, closeTo(1, 1e-6));
      mseValue.dispose();
      prediction.dispose();
      target.dispose();

      final logits = core.Tensor.fromList(
        <num>[2, 1],
        shape: core.Shape(<int>[1, 2]),
      );
      final oneHot = core.Tensor.fromList(
        <num>[1, 0],
        shape: core.Shape(<int>[1, 2]),
      );
      final ceValue = crossEntropy(logits, oneHot);
      expect(crossEntropy.toString(), 'CrossEntropyLoss()');
      expect(ceValue.toList().single, closeTo(0.31326166, 1e-5));
      ceValue.dispose();
      logits.dispose();
      oneHot.dispose();
    },
    skip: nativeSkip,
  );

  test(
    'Sequential validates disposed children and releases intermediates',
    () {
      final disposed = Identity()..dispose();
      expect(
        () => Sequential(children: <Module>[disposed]),
        throwsArgumentError,
      );

      final input = core.Tensor.fromList(
        <num>[-1, 1],
        shape: core.Shape(<int>[2]),
      );
      final capture = _CaptureRelu();
      final normal = Sequential(children: <Module>[capture, Sigmoid()]);
      final output = normal(input);
      expect(capture.captured!.isDisposed, isTrue);
      output.dispose();
      normal.dispose();
      expect(() => normal.children, throwsA(isA<core.DisposedTensorException>()));

      final failingCapture = _CaptureRelu();
      final failing = Sequential(
        children: <Module>[failingCapture, _ThrowForward()],
      );
      expect(() => failing(input), throwsStateError);
      expect(failingCapture.captured!.isDisposed, isTrue);
      failing.dispose();
      input.dispose();
    },
    skip: nativeSkip,
  );

  test(
    'Linear validates dimensions and covers bias, train, eval, forward, and move',
    () {
      expect(
        () => Linear(inFeatures: 0, outFeatures: 1),
        throwsArgumentError,
      );
      expect(
        () => Linear(inFeatures: 1, outFeatures: 0),
        throwsArgumentError,
      );

      final layer = Linear(inFeatures: 2, outFeatures: 1, bias: false);
      final input = core.Tensor.fromList(
        <num>[1, 2],
        shape: core.Shape(<int>[1, 2]),
      );
      final output = layer(input);
      expect(output.shape, core.Shape(<int>[1, 1]));
      expect(layer.parameters, hasLength(1));
      expect(layer.toString(), contains('bias: false'));
      layer.eval();
      expect(layer.isTraining, isFalse);
      layer.train();
      expect(layer.isTraining, isTrue);
      final identities = layer.parameters
          .map((parameter) => parameter.identity)
          .toList();
      layer.to(core.Device.cpu);
      expect(
        layer.parameters.map((parameter) => parameter.identity),
        identities,
      );
      output.dispose();
      input.dispose();
      layer.dispose();
    },
    skip: nativeSkip,
  );

  test('Model rejects recursive, self, disposed, owned, and cyclic builds', () {
    final recursive = _RecursiveModel();
    expect(
      () => recursive.parameters,
      throwsA(isA<core.InvalidArgumentException>()),
    );
    expect(recursive.isMaterialized, isFalse);
    recursive.dispose();

    final self = _SelfModel();
    expect(
      () => self.parameters,
      throwsA(isA<core.InvalidArgumentException>()),
    );
    self.dispose();

    final disposedCandidate = Identity()..dispose();
    final disposedModel = _FixedModel(disposedCandidate);
    expect(
      () => disposedModel.parameters,
      throwsA(isA<core.InvalidArgumentException>()),
    );
    disposedModel.dispose();

    final ownedCandidate = Identity();
    final owner = Sequential(children: <Module>[ownedCandidate]);
    final ownedModel = _FixedModel(ownedCandidate);
    expect(
      () => ownedModel.parameters,
      throwsA(isA<core.InvalidArgumentException>()),
    );
    ownedModel.dispose();
    owner.dispose();

    final cycle = _CycleModel();
    expect(
      () => cycle.parameters,
      throwsA(isA<core.InvalidArgumentException>()),
    );
    cycle.dispose();
  });

  test('Module ownership, traversal, default hooks, and disposal are deterministic', () {
    final child = _BareModule();
    final root = _BareModule(
      children: <NamedModule>[
        NamedModule('first', child),
        NamedModule('again', child),
      ],
    );

    expect(root.children, <Module>[child, child]);
    expect(root.modules, <Module>[root, child]);
    expect(root.namedModules.map((entry) => entry.name), <String>['', 'first']);
    expect(root.parameters, isEmpty);
    expect(root.namedParameters, isEmpty);
    expect(root.buffers, isEmpty);
    expect(root.namedBuffers, isEmpty);
    expect(root.isTraining, isTrue);
    root.eval();
    expect(root.isTraining, isFalse);
    expect(child.isTraining, isFalse);
    root.train();
    expect(root.isTraining, isTrue);
    expect(child.isTraining, isTrue);
    expect(child.trainingChanges, 2);
    expect(root.toString(), '_BareModule');

    root.internalOnMove(core.Device.cpu);
    root.internalPreflightMove(core.Device.cpu);
    expect(root.internalMoveDevice, isNull);
    root.internalDetachOwner(child);
    expect(root.internalContainsModule(root, <Module>{}), isTrue);
    expect(root.internalContainsModule(child, <Module>{root}), isFalse);

    final selfOwner = _BareModule();
    expect(
      () => selfOwner.internalAttachOwner(selfOwner),
      throwsArgumentError,
    );
    selfOwner.dispose();

    final ownerA = _BareModule();
    final ownerB = _BareModule();
    final owned = _BareModule();
    owned.internalAttachOwner(ownerA);
    expect(owned.internalOwner, ownerA);
    expect(() => owned.internalAttachOwner(ownerB), throwsArgumentError);
    owned.internalDetachOwner(ownerB);
    expect(owned.internalOwner, ownerA);
    owned.internalDetachOwner(ownerA);
    expect(owned.internalOwner, isNull);
    owned.dispose();
    ownerA.dispose();
    ownerB.dispose();

    root.dispose();
    root.dispose();
    expect(root.disposeCalls, 1);
    expect(child.isDisposed, isTrue);
    expect(() => root.children, throwsA(isA<core.DisposedTensorException>()));
    expect(
      () => root.namedParameters,
      throwsA(isA<core.DisposedTensorException>()),
    );
    expect(
      () => root.namedBuffers,
      throwsA(isA<core.DisposedTensorException>()),
    );
    expect(
      () => root.namedModules,
      throwsA(isA<core.DisposedTensorException>()),
    );
    expect(root.train, throwsA(isA<core.DisposedTensorException>()));
    expect(root.eval, throwsA(isA<core.DisposedTensorException>()));
    expect(root.stateDict, throwsA(isA<core.DisposedTensorException>()));
    expect(root.toTreeString, throwsA(isA<core.DisposedTensorException>()));
  });

  test(
    'Module move preflight, original rethrow, and rollback failure are explicit',
    () {
      final unavailable = _BareModule();
      expect(
        () => unavailable.to(core.Device.cuda(0)),
        throwsA(isA<core.UnsupportedOperationException>()),
      );
      unavailable.dispose();

      final preflightLeaf = _MoveLeaf(
        original: core.Device.cuda(0),
        failPreflight: true,
      );
      final preflightTree = Sequential(children: <Module>[preflightLeaf]);
      expect(() => preflightTree.to(core.Device.cpu), throwsStateError);
      expect(preflightLeaf.moveCalls, 0);
      preflightTree.dispose();

      final rollbackFailure = _MoveLeaf(
        original: core.Device.cuda(0),
        failFirstMove: true,
        failRollback: true,
      );
      final rollbackTree = Sequential(children: <Module>[rollbackFailure]);
      expect(
        () => rollbackTree.to(core.Device.cpu),
        throwsA(
          isA<core.NativeRuntimeException>().having(
            (error) => error.message,
            'message',
            contains('rollback also failed'),
          ),
        ),
      );
      expect(rollbackFailure.moveCalls, 2);
      rollbackTree.dispose();
    },
    skip: nativeSkip,
  );

  test(
    'Buffer, parameter traversal, StateDict, strict loading, and disposal cover all state contracts',
    () {
      final parameter = _parameter(2);
      final persistent = _buffer(3);
      final transient = _buffer(4, persistent: false);
      final module = _BareModule(
        parameters: <NamedParameter>[
          NamedParameter('weight', parameter),
          NamedParameter('shared', parameter),
        ],
        buffers: <NamedBuffer>[
          NamedBuffer('running', persistent),
          NamedBuffer('runningAgain', persistent),
          NamedBuffer('transient', transient),
        ],
      );

      expect(module.parameters, <Parameter>[parameter]);
      expect(module.namedParameters.single.name, 'weight');
      expect(module.buffers, <Buffer>[persistent, transient]);
      expect(module.namedBuffers.map((entry) => entry.name), <String>[
        'running',
        'transient',
      ]);

      expect(persistent.shape, core.Shape(<int>[1]));
      expect(persistent.dtype, core.DType.float32);
      expect(persistent.device, core.Device.cpu);
      expect(persistent.persistent, isTrue);
      expect(persistent.isDisposed, isFalse);
      expect(persistent.tensorForRuntime.toList(), <double>[3]);
      final bufferSnapshot = persistent.snapshot();
      expect(bufferSnapshot.toList(), <double>[3]);
      bufferSnapshot.dispose();

      final state = module.stateDict();
      expect(state.length, 2);
      expect(state.isEmpty, isFalse);
      expect(state.keys.toSet(), <String>{'weight', 'running'});
      expect(state.entries.length, 2);
      expect(state['missing'], isNull);
      expect(state.isDisposed, isFalse);

      final parameterSource = core.Tensor.fromList(
        <num>[9],
        shape: core.Shape(<int>[1]),
      );
      final bufferSource = core.Tensor.fromList(
        <num>[8],
        shape: core.Shape(<int>[1]),
      );
      final replacement = StateDict.fromOwned(<String, core.Tensor>{
        'weight': parameterSource,
        'running': bufferSource,
      });
      final restored = module.loadStateDict(replacement);
      expect(restored.isSuccess, isTrue);
      expect(parameter.tensorForRuntime.toList(), <double>[9]);
      expect(persistent.tensorForRuntime.toList(), <double>[8]);
      replacement.dispose();

      final unexpectedTensor = core.Tensor.ones(core.Shape(<int>[1]));
      final incompatible = StateDict.fromOwned(<String, core.Tensor>{
        'weight': core.Tensor.ones(core.Shape(<int>[1])),
        'unexpected': unexpectedTensor,
      });
      expect(
        () => module.loadStateDict(incompatible),
        throwsA(isA<core.InvalidArgumentException>()),
      );
      final nonStrict = module.loadStateDict(incompatible, strict: false);
      expect(nonStrict.isSuccess, isFalse);
      expect(nonStrict.missingKeys, <String>['running']);
      expect(nonStrict.unexpectedKeys, <String>['unexpected']);
      incompatible.dispose();

      final wrongShape = StateDict.fromOwned(<String, core.Tensor>{
        'weight': core.Tensor.ones(core.Shape(<int>[2])),
        'running': core.Tensor.ones(core.Shape(<int>[1])),
      });
      expect(
        () => module.loadStateDict(wrongShape),
        throwsA(isA<core.InvalidArgumentException>()),
      );
      wrongShape.dispose();

      expect(
        () => state.entries['illegal'] = core.Tensor.ones(core.Shape(<int>[1])),
        throwsUnsupportedError,
      );
      state.dispose();
      state.dispose();
      expect(state.isDisposed, isTrue);
      expect(() => state.length, throwsStateError);
      expect(() => state.keys, throwsStateError);
      expect(() => state.entries, throwsStateError);
      expect(() => state['weight'], throwsStateError);

      module.dispose();
      expect(parameter.isDisposed, isTrue);
      expect(persistent.isDisposed, isTrue);
      expect(transient.isDisposed, isTrue);
      expect(() => persistent.shape, throwsA(isA<core.DisposedTensorException>()));
      expect(() => persistent.dtype, throwsA(isA<core.DisposedTensorException>()));
      expect(() => persistent.device, throwsA(isA<core.DisposedTensorException>()));
      expect(persistent.snapshot, throwsA(isA<core.DisposedTensorException>()));
      expect(
        () => persistent.tensorForRuntime,
        throwsA(isA<core.DisposedTensorException>()),
      );
      persistent.dispose();
    },
    skip: nativeSkip,
  );

  test(
    'state snapshot failure releases prior snapshots and rethrows',
    () {
      final first = _parameter(1);
      final disposed = _parameter(2)..dispose();
      final module = _BareModule(
        parameters: <NamedParameter>[
          NamedParameter('first', first),
          NamedParameter('disposed', disposed),
        ],
      );
      expect(
        module.stateDict,
        throwsA(isA<core.DisposedTensorException>()),
      );
      module.dispose();
    },
    skip: nativeSkip,
  );
}
