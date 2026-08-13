import 'package:tensora_text/tensora_text.dart';
import 'package:test/test.dart';

final class _CodeUnitTokenizer implements Tokenizer {
  @override
  TokenSequence encode(String text) => TokenSequence(text.codeUnits);

  @override
  String decode(TokenSequence sequence) => String.fromCharCodes(sequence.tokens);
}

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

  test('tokenizer interface supports deterministic round trips', () {
    final tokenizer = _CodeUnitTokenizer();
    final encoded = tokenizer.encode('Tensora');
    expect(tokenizer.decode(encoded), 'Tensora');
  });
}
