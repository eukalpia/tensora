import 'package:tensora_nn/tensora_nn.dart';
import 'package:test/test.dart';

void main() {
  test('StateLoadResult owns immutable deterministic key snapshots', () {
    final missing = <String>['b', 'a'];
    final unexpected = <String>['x'];
    final result = StateLoadResult(
      missingKeys: missing,
      unexpectedKeys: unexpected,
    );

    missing.add('later');
    unexpected.clear();
    expect(result.missingKeys, <String>['b', 'a']);
    expect(result.unexpectedKeys, <String>['x']);
    expect(result.isSuccess, isFalse);
    expect(() => result.missingKeys.add('nope'), throwsUnsupportedError);
    expect(() => result.unexpectedKeys.clear(), throwsUnsupportedError);
  });
}
