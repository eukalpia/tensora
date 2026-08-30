import 'package:tensora/tensora.dart' as core;
import 'package:tensora_nn/tensora_nn.dart';
import 'package:test/test.dart';

/// Builds a deterministic rank-2 input.
core.Tensor _input(int batch, int features) => core.Tensor.fromList(
  List<double>.generate(batch * features, (index) => (index % 7) * 0.25 - 0.75),
  shape: core.Shape(<int>[batch, features]),
);

void main() {
  test('rejects a rank that is not smaller than the projection', () {
    expect(
      () => LoRALinear(inFeatures: 4, outFeatures: 3, rank: 0),
      throwsA(isA<ArgumentError>()),
    );
    expect(
      () => LoRALinear(inFeatures: 4, outFeatures: 3, rank: 4),
      throwsA(isA<ArgumentError>()),
    );
  });

  test('freezes the base projection and trains only the factors', () {
    final adapter = LoRALinear(
      inFeatures: 6,
      outFeatures: 4,
      rank: 2,
      seed: 11,
    );
    addTearDown(adapter.dispose);

    final named = <String, Parameter>{
      for (final entry in adapter.namedParameters) entry.name: entry.parameter,
    };

    expect(named.keys, containsAll(<String>['base.weight', 'base.bias']));
    expect(named['base.weight']!.requiresGrad, isFalse);
    expect(named['base.bias']!.requiresGrad, isFalse);

    final trainable = adapter.parameters.where((p) => p.requiresGrad).toList();
    expect(trainable, hasLength(2));

    // The whole point: far fewer trainable values than the projection holds.
    final trainableValues = trainable.fold<int>(
      0,
      (sum, p) => sum + p.shape.numel,
    );
    expect(trainableValues, 2 * 6 + 2 * 4);
    expect(trainableValues, lessThan(6 * 4));
  });

  test('a fresh adapter is exactly the base projection', () {
    final adapter = LoRALinear(inFeatures: 5, outFeatures: 3, rank: 2, seed: 5);
    addTearDown(adapter.dispose);

    final x = _input(4, 5);
    addTearDown(x.dispose);

    final adapted = adapter(x);
    addTearDown(adapted.dispose);
    final plain = adapter.base(x);
    addTearDown(plain.dispose);

    // B starts at zero, so the low-rank path contributes nothing yet.
    final a = adapted.toList();
    final b = plain.toList();
    expect(a, hasLength(b.length));
    for (var index = 0; index < a.length; index++) {
      expect(a[index], closeTo(b[index], 1e-6));
    }
  });

  test('gradients reach the factors and leave the base untouched', () {
    final adapter = LoRALinear(inFeatures: 4, outFeatures: 3, rank: 2, seed: 3);
    addTearDown(adapter.dispose);

    final x = _input(2, 4);
    addTearDown(x.dispose);
    final target = core.Tensor.zeros(core.Shape(<int>[2, 3]));
    addTearDown(target.dispose);

    final named = <String, Parameter>{
      for (final entry in adapter.namedParameters) entry.name: entry.parameter,
    };

    // Give B a non-zero value so the path carries gradient on the first step.
    final seeded = core.Tensor.full(named['loraB.weight']!.shape, 0.1);
    core.NativeTensorState.assignMany(
      targets: <core.Tensor>[named['loraB.weight']!.tensorForRuntime],
      sources: <core.Tensor>[seeded],
    );
    seeded.dispose();

    final prediction = adapter(x);
    final loss = core.Losses.mse(prediction, target);
    loss.backward();

    for (final name in <String>['loraA.weight', 'loraB.weight']) {
      final grad = named[name]!.grad();
      expect(
        grad.toList().any((value) => value != 0),
        isTrue,
        reason: '$name received no gradient',
      );
      grad.dispose();
    }

    // A frozen parameter accumulates nothing, so asking for its gradient is an
    // error rather than a silently zero tensor.
    expect(
      () => named['base.weight']!.grad(),
      throwsA(isA<core.TensoraException>()),
    );

    loss.dispose();
    prediction.dispose();
  });

  test('merging folds the update in without changing the output', () {
    final adapter = LoRALinear(
      inFeatures: 5,
      outFeatures: 4,
      rank: 2,
      alpha: 4,
      seed: 21,
    );
    addTearDown(adapter.dispose);

    final named = <String, Parameter>{
      for (final entry in adapter.namedParameters) entry.name: entry.parameter,
    };
    final seeded = core.Tensor.full(named['loraB.weight']!.shape, 0.05);
    core.NativeTensorState.assignMany(
      targets: <core.Tensor>[named['loraB.weight']!.tensorForRuntime],
      sources: <core.Tensor>[seeded],
    );
    seeded.dispose();

    final x = _input(3, 5);
    addTearDown(x.dispose);

    final before = adapter(x);
    final beforeValues = before.toList();
    before.dispose();

    final baseIdentity = named['base.weight']!.identity;
    adapter.mergeIntoBase();

    // Identity must survive the merge, otherwise an optimizer built earlier
    // would be pointing at a parameter that no longer exists.
    expect(named['base.weight']!.identity, baseIdentity);

    final after = adapter(x);
    final afterValues = after.toList();
    after.dispose();

    for (var index = 0; index < beforeValues.length; index++) {
      expect(afterValues[index], closeTo(beforeValues[index], 1e-5));
    }

    // Merging twice must not apply the update twice.
    adapter.mergeIntoBase();
    final again = adapter(x);
    final againValues = again.toList();
    again.dispose();
    for (var index = 0; index < beforeValues.length; index++) {
      expect(againValues[index], closeTo(beforeValues[index], 1e-5));
    }
  });

  test('reports its shape in the module tree', () {
    final adapter = LoRALinear(inFeatures: 8, outFeatures: 8, rank: 4);
    addTearDown(adapter.dispose);

    expect(adapter.scaling, closeTo(1 / 4, 1e-12));
    expect(adapter.toTreeString(), contains('LoRALinear'));
    expect(adapter.toTreeString(), contains('base:'));
    expect(adapter.namedModules.map((entry) => entry.name), <String>[
      '',
      'base',
      'loraA',
      'loraB',
    ]);
  });
}
