// Importing the foundation and the neural-network package together must work.
// It did not before the native handle owners were renamed, because both
// packages exported a type called Module.
import 'package:tensora/tensora.dart';
import 'package:tensora_data/tensora_data.dart';
import 'package:tensora_train/tensora_train.dart';
import 'package:test/test.dart';

/// Cross-package acceptance tests.
///
/// These run without a native runtime, so they cover the contracts that compose
/// across package boundaries rather than native execution, which each package
/// validates against a real compiled library in its own suite.
void main() {
  test('data batching drives deterministic training history', () {
    final dataset = ListDataset<int>(List<int>.generate(7, (index) => index));
    final loader = DataLoader<int>(dataset, batchSize: 3);

    var history = TrainingHistory();
    var step = 0;
    for (final batch in loader.batches()) {
      final mean = batch.values.reduce((a, b) => a + b) / batch.length;
      history = history.add(MetricPoint(name: 'loss', step: step, value: mean));
      step += 1;
    }

    expect(step, 3);
    expect(history.forMetric('loss').map((point) => point.value), <double>[
      1.0,
      4.0,
      6.0,
    ]);
    expect(history.forMetric('accuracy'), isEmpty);
  });

  test('dropLast discards the short trailing batch', () {
    final dataset = ListDataset<int>(List<int>.generate(7, (index) => index));
    final batches =
        DataLoader<int>(dataset, batchSize: 3, dropLast: true).batches();

    expect(batches.map((batch) => batch.length), <int>[3, 3]);
  });

  test('module composition exposes deterministic names across packages', () {
    final model = Sequential(children: <Module>[Identity(), Identity()]);

    expect(model.namedModules.map((entry) => entry.name), <String>[
      '',
      '0',
      '1',
    ]);
    expect(model.parameters, isEmpty);
    expect(model.isTraining, isTrue);

    model.eval();
    expect(model.children.every((child) => !child.isTraining), isTrue);

    expect(model.toTreeString(), contains('Sequential'));
    expect(model.toTreeString(), contains('Identity()'));

    model.dispose();
    expect(model.isDisposed, isTrue);
  });

  test('a module may not be owned by two parents', () {
    final shared = Identity();
    Sequential(children: <Module>[shared]);

    expect(
      () => Sequential(children: <Module>[shared]),
      throwsA(isA<ArgumentError>()),
    );
  });

  test('foundation value types are shared by every package', () {
    expect(Shape(<int>[2, 3]).numel, 6);
    expect(Shape(<int>[2, 3]), Shape(<int>[2, 3]));
    expect(Device.cuda(1), isNot(Device.cuda(0)));
    expect(DType.float32.byteWidth, 4);
  });

  test('the composable and native module hierarchies coexist by name', () {
    // Naming regression guard. Both packages are imported unprefixed above, so
    // this only compiles while the two hierarchies keep distinct names.
    expect(Identity(), isA<Module>());
    expect(NativeLinear, isNot(Linear));
    expect(<Type>[NativeModule, NativeOptimizer, NativeSgd], hasLength(3));
  });
}
