import 'package:tensora_nn/tensora_nn.dart';
import 'package:test/test.dart';

void main() {
  test('empty declarative model has deterministic empty StateDict', () {
    final model = Sequential(children: <Module>[Identity(), GELU()]);
    final state = model.stateDict();

    expect(state.keys, isEmpty);
    expect(state.length, 0);
    final result = model.loadStateDict(state);
    expect(result.isSuccess, isTrue);
    expect(result.missingKeys, isEmpty);
    expect(result.unexpectedKeys, isEmpty);

    state.dispose();
    model.dispose();
  });

  test('tree diagnostics are stable and Flutter-like', () {
    final model = Sequential(
      children: <Module>[
        Identity(),
        GELU(),
        Sequential(children: <Module>[ReLU(), SiLU()]),
      ],
    );

    final tree = model.toTreeString();
    expect(tree, contains('Sequential'));
    expect(tree, contains('0: Identity()'));
    expect(tree, contains('1: GELU()'));
    expect(tree, contains('2: Sequential'));
    expect(tree, contains('0: ReLU()'));
    expect(tree, contains('1: SiLU()'));
    expect(tree, isNot(contains('handle')));

    model.dispose();
  });
}
