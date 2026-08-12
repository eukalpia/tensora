import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  group('Shape', () {
    test('is immutable and defensively copies input dimensions', () {
      final source = <int>[2, 3, 4];
      final shape = Shape(source);
      source[0] = 99;

      expect(shape.dimensions, [2, 3, 4]);
      expect(() => shape.dimensions[0] = 7, throwsUnsupportedError);
    });

    test('reports rank and element count', () {
      final shape = Shape([32, 3, 224, 224]);

      expect(shape.rank, 4);
      expect(shape.numel, 4816896);
      expect(shape.toString(), 'Shape([32, 3, 224, 224])');
    });

    test('rank-zero shape represents one scalar element', () {
      final scalar = Shape(const []);

      expect(scalar.rank, 0);
      expect(scalar.numel, 1);
    });

    test('has value equality and stable hashing', () {
      final first = Shape([2, 3]);
      final second = Shape([2, 3]);
      final different = Shape([3, 2]);

      expect(first, second);
      expect(first.hashCode, second.hashCode);
      expect(first, isNot(different));
    });

    test('rejects zero and negative dimensions', () {
      expect(() => Shape([2, 0]), throwsArgumentError);
      expect(() => Shape([2, -1]), throwsArgumentError);
    });

    test('rejects rank above the native contract', () {
      expect(
        () => Shape(List<int>.filled(Shape.maxRank + 1, 1)),
        throwsArgumentError,
      );
    });

    test('rejects element-count overflow before native allocation', () {
      expect(() => Shape([Shape.maxNumel, 2]), throwsArgumentError);
    });
  });
}
