import 'package:tensora_edge/tensora_edge.dart';
import 'package:test/test.dart';

void main() {
  test('edge namespace exposes explicit provider choices', () {
    expect(
      OnnxExecutionProvider.values,
      containsAll(<OnnxExecutionProvider>[
        OnnxExecutionProvider.auto,
        OnnxExecutionProvider.cpu,
        OnnxExecutionProvider.cuda,
        OnnxExecutionProvider.directML,
        OnnxExecutionProvider.coreML,
      ]),
    );
  });
}
