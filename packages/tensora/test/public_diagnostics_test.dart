import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  test('Tensor string representation reflects lifecycle state', () {
    final tensor = Tensor.ones(Shape([2, 2]));

    expect(tensor.toString(), contains('Shape([2, 2])'));
    expect(tensor.toString(), isNot(contains('disposed')));

    tensor.dispose();

    expect(tensor.toString(), contains('disposed'));
  });

  test('typed exceptions include operation context when available', () {
    const withOperation = InvalidShapeException(
      'bad shape',
      operation: 'tensor.matmul',
    );
    const withoutOperation = InvalidShapeException('bad shape');

    expect(withOperation.toString(), contains('tensor.matmul'));
    expect(withOperation.toString(), endsWith('bad shape'));
    expect(withoutOperation.toString(), isNot(contains('(')));
    expect(withoutOperation.toString(), endsWith('bad shape'));
  });
}
