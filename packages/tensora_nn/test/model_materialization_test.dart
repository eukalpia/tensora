import 'package:tensora_nn/tensora_nn.dart';
import 'package:test/test.dart';

final class CountingModel extends Model {
  int buildCount = 0;

  @override
  Module build() {
    buildCount += 1;
    return Sequential(children: <Module>[Identity(), ReLU()]);
  }
}

final class FlakyModel extends Model {
  int buildCount = 0;

  @override
  Module build() {
    buildCount += 1;
    if (buildCount == 1) {
      throw StateError('first build fails');
    }
    return Identity();
  }
}

void main() {
  test('Model.build materializes lazily exactly once after success', () {
    final model = CountingModel();
    expect(model.buildCount, 0);

    expect(model.parameters, isEmpty);
    expect(model.buildCount, 1);
    expect(model.modules.length, 4); // model + sequential + identity + relu

    expect(model.parameters, isEmpty);
    expect(model.toTreeString(), contains('CountingModel'));
    expect(model.buildCount, 1);

    model.dispose();
  });

  test('failed materialization is atomic and can be retried', () {
    final model = FlakyModel();

    expect(() => model.parameters.toList(), throwsStateError);
    expect(model.buildCount, 1);
    expect(model.isMaterialized, isFalse);

    expect(model.parameters, isEmpty);
    expect(model.buildCount, 2);
    expect(model.isMaterialized, isTrue);

    model.dispose();
  });

  test('Sequential defensively freezes child order', () {
    final children = <Module>[Identity()];
    final sequential = Sequential(children: children);
    children.add(ReLU());

    expect(sequential.children, hasLength(1));
    expect(() => sequential.children.add(ReLU()), throwsUnsupportedError);

    sequential.dispose();
  });

  test('same module instance cannot have ambiguous ownership', () {
    final child = Identity();
    expect(
      () => Sequential(children: <Module>[child, child]),
      throwsArgumentError,
    );
    child.dispose();
  });
}
