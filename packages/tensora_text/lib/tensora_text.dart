abstract interface class Tokenizer {
  TokenSequence encode(String text);
  String decode(TokenSequence sequence);
}

final class TokenSequence {
  TokenSequence(Iterable<int> tokens)
    : tokens = List<int>.unmodifiable(tokens) {
    for (final token in this.tokens) {
      if (token < 0) {
        throw ArgumentError.value(
          token,
          'tokens',
          'token ids must be non-negative',
        );
      }
    }
  }

  final List<int> tokens;

  int get length => tokens.length;
  bool get isEmpty => tokens.isEmpty;

  @override
  bool operator ==(Object other) {
    if (other is! TokenSequence || other.tokens.length != tokens.length) {
      return false;
    }
    for (var index = 0; index < tokens.length; index++) {
      if (tokens[index] != other.tokens[index]) return false;
    }
    return true;
  }

  @override
  int get hashCode => Object.hashAll(tokens);
}
