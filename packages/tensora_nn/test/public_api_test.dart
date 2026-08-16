import 'package:tensora_nn/tensora_nn.dart';
import 'package:test/test.dart';

void main() {
  test('public namespace compiles', () {
    final List<Module> modules = <Module>[];
    final List<Linear> linears = <Linear>[];
    expect(modules, isEmpty);
    expect(linears, isEmpty);
  });
}
