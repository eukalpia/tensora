import 'package:tensora/tensora.dart' as core;
import 'package:tensora_nn/tensora_nn.dart';
import 'package:test/test.dart';

final class TrainingProbe extends Module {
  final List<bool> changes = <bool>[];

  @override
  void internalOnTrainingModeChanged(bool training) => changes.add(training);

  @override
  core.Tensor forward(core.Tensor input) => input;
}

final class RecursiveBuildModel extends Model {
  @override
  Module build() {
    parameters;
    return Identity();
  }
}

final class SelfReturningModel extends Model {
  @override
  Module build() => this;
}

final class CandidateModel extends Model {
  CandidateModel(this.candidate);

  final Module candidate;

  @override
  Module build() => candidate;
}

final class CycleProbe extends Module {
  @override
  bool internalContainsModule(Module target, Set<Module> visited) => true;

  @override
  core.Tensor forward(core.Tensor input) => input;
}

void expectDisposed(void Function() body) {
  expect(body, throwsA(isA<core.DisposedTensorException>()));
}

void main() {
  test('activation and loss diagnostics expose Flutter-style labels', () {
    final modules = <Module>[
      ReLU(),
      Sigmoid(),
      Tanh(),
      GELU(),
      SiLU(),
      SwiGLU(),
      Identity(),
    ];
    expect(
      modules.map((module) => module.toString()),
      <String>[
        'ReLU()',
        'Sigmoid()',
        'Tanh()',
        'GELU()',
        'SiLU()',
        'SwiGLU()',
        'Identity()',
      ],
    );
    expect(const MSELoss().toString(), 'MSELoss()');
    expect(const CrossEntropyLoss().toString(), 'CrossEntropyLoss()');
    for (final module in modules) {
      module.dispose();
    }
  });

  test('Linear rejects invalid dimensions before native materialization', () {
    expect(
      () => Linear(inFeatures: 0, outFeatures: 1),
      throwsArgumentError,
    );
    expect(
      () => Linear(inFeatures: 1, outFeatures: 0),
      throwsArgumentError,
    );
  });

  test('Model detects recursive, self, disposed, owned, and cyclic builds', () {
    final recursive = RecursiveBuildModel();
    expect(
      () => recursive.parameters,
      throwsA(isA<core.InvalidArgumentException>()),
    );
    expect(recursive.isMaterialized, isFalse);
    recursive.dispose();

    final self = SelfReturningModel();
    expect(
      () => self.modules,
      throwsA(isA<core.InvalidArgumentException>()),
    );
    expect(self.isMaterialized, isFalse);
    self.dispose();

    final disposedCandidate = Identity()..dispose();
    final disposed = CandidateModel(disposedCandidate);
    expect(
      () => disposed.namedModules,
      throwsA(isA<core.InvalidArgumentException>()),
    );
    disposed.dispose();

    final ownedCandidate = Identity();
    final owner = Sequential(children: <Module>[ownedCandidate]);
    final owned = CandidateModel(ownedCandidate);
    expect(
      () => owned.children,
      throwsA(isA<core.InvalidArgumentException>()),
    );
    owned.dispose();
    owner.dispose();

    final cyclic = CandidateModel(CycleProbe());
    expect(
      () => cyclic.parameters,
      throwsA(isA<core.InvalidArgumentException>()),
    );
    cyclic.dispose();
  });

  test('Sequential ownership is atomic when a later attachment fails', () {
    final first = Identity();
    final alreadyOwned = Identity();
    final originalOwner = Sequential(children: <Module>[alreadyOwned]);

    expect(
      () => Sequential(children: <Module>[first, alreadyOwned]),
      throwsArgumentError,
    );
    expect(first.internalOwner, isNull);
    expect(alreadyOwned.internalOwner, same(originalOwner));

    first.dispose();
    originalOwner.dispose();
  });

  test('module ownership helpers reject self and ambiguous parents', () {
    final module = Identity();
    final owner = Identity();
    final otherOwner = Identity();

    expect(() => module.internalAttachOwner(module), throwsArgumentError);
    module.internalAttachOwner(owner);
    expect(module.internalOwner, same(owner));
    expect(() => module.internalAttachOwner(otherOwner), throwsArgumentError);

    module.internalDetachOwner(otherOwner);
    expect(module.internalOwner, same(owner));
    module.internalDetachOwner(owner);
    expect(module.internalOwner, isNull);

    module.dispose();
    owner.dispose();
    otherOwner.dispose();
  });

  test('train and eval propagate deterministically through the tree', () {
    final first = TrainingProbe();
    final second = TrainingProbe();
    final tree = Sequential(children: <Module>[first, second]);

    tree.eval();
    expect(tree.isTraining, isFalse);
    expect(first.isTraining, isFalse);
    expect(second.isTraining, isFalse);
    expect(first.changes, <bool>[false]);
    expect(second.changes, <bool>[false]);

    tree.train();
    expect(tree.isTraining, isTrue);
    expect(first.isTraining, isTrue);
    expect(second.isTraining, isTrue);
    expect(first.changes, <bool>[false, true]);
    expect(second.changes, <bool>[false, true]);

    tree.dispose();
  });

  test('disposed modules reject every stateful public traversal', () {
    final tree = Sequential(children: <Module>[Identity(), ReLU()]);
    tree.dispose();
    tree.dispose();

    expectDisposed(() => tree.children);
    expectDisposed(() => tree.namedParameters);
    expectDisposed(() => tree.namedBuffers);
    expectDisposed(() => tree.namedModules);
    expectDisposed(tree.stateDict);
    expectDisposed(tree.toTreeString);
    expectDisposed(tree.train);
    expectDisposed(tree.eval);
    expectDisposed(() => tree.to(core.Device.cpu));
  });

  test('empty StateDict exposes immutable views and disposed guards', () {
    final state = StateDict.fromOwned(<String, core.Tensor>{});
    expect(state.length, 0);
    expect(state.isEmpty, isTrue);
    expect(state.keys, isEmpty);
    expect(state.entries, isEmpty);
    expect(state['missing'], isNull);
    expect(state.isDisposed, isFalse);

    state.dispose();
    state.dispose();
    expect(state.isDisposed, isTrue);
    expect(() => state.length, throwsStateError);
    expect(() => state.keys, throwsStateError);
    expect(() => state.entries, throwsStateError);
    expect(() => state['missing'], throwsStateError);
  });

  test('StateLoadResult freezes key lists and reports compatibility', () {
    final success = StateLoadResult();
    expect(success.isSuccess, isTrue);

    final missing = <String>['weight'];
    final unexpected = <String>['extra'];
    final failure = StateLoadResult(
      missingKeys: missing,
      unexpectedKeys: unexpected,
    );
    missing.add('later');
    unexpected.add('later');
    expect(failure.isSuccess, isFalse);
    expect(failure.missingKeys, <String>['weight']);
    expect(failure.unexpectedKeys, <String>['extra']);
    expect(() => failure.missingKeys.add('x'), throwsUnsupportedError);
  });
}
