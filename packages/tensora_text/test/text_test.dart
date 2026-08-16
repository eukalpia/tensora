import 'package:tensora_text/tensora_text.dart';
import 'package:test/test.dart';

void main() {
  test('token sequences are immutable value objects', () {
    final source = <int>[1, 2, 3];
    final sequence = TokenSequence(source);
    source[0] = 9;

    expect(sequence.tokens, <int>[1, 2, 3]);
    expect(sequence.length, 3);
    expect(sequence, TokenSequence(<int>[1, 2, 3]));
    expect(() => sequence.tokens.add(4), throwsUnsupportedError);
    expect(() => TokenSequence(<int>[-1]), throwsArgumentError);
  });

  test('tokenizer interface is a public extension point', () {
    final List<Tokenizer> implementations = <Tokenizer>[];
    expect(implementations, isEmpty);
  });
}
