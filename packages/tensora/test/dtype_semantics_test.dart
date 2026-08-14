import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  test('dtype metadata is complete and stable', () {
    final expected = <DType, (int, bool, bool, bool, int)>{
      DType.float16: (2, true, false, false, 2),
      DType.bfloat16: (2, true, false, false, 3),
      DType.float32: (4, true, false, false, 1),
      DType.float64: (8, true, false, false, 4),
      DType.int8: (1, false, true, false, 5),
      DType.uint8: (1, false, true, false, 6),
      DType.int16: (2, false, true, false, 7),
      DType.int32: (4, false, true, false, 8),
      DType.int64: (8, false, true, false, 9),
      DType.boolean: (1, false, false, true, 10),
    };

    expect(DType.values, expected.keys.toList());
    for (final entry in expected.entries) {
      final dtype = entry.key;
      final metadata = entry.value;
      expect(dtype.byteWidth, metadata.$1, reason: '$dtype byte width');
      expect(dtype.isFloatingPoint, metadata.$2, reason: '$dtype floating');
      expect(dtype.isInteger, metadata.$3, reason: '$dtype integer');
      expect(dtype.isBoolean, metadata.$4, reason: '$dtype boolean');
      expect(dtype.nativeCode, metadata.$5, reason: '$dtype ABI code');
      expect(dtype.supportsGradients, dtype.isFloatingPoint);
      expect(dtype.supportsArithmetic, isNot(dtype.isBoolean));
      expect(dtype.toString(), 'DType.${dtype.name}');
    }
  });

  test('reduction accumulators preserve numerical range', () {
    expect(DType.float16.reductionAccumulator, DType.float32);
    expect(DType.bfloat16.reductionAccumulator, DType.float32);
    expect(DType.float32.reductionAccumulator, DType.float32);
    expect(DType.float64.reductionAccumulator, DType.float64);
    expect(DType.int8.reductionAccumulator, DType.int64);
    expect(DType.uint8.reductionAccumulator, DType.int64);
    expect(DType.int16.reductionAccumulator, DType.int64);
    expect(DType.int32.reductionAccumulator, DType.int64);
    expect(DType.int64.reductionAccumulator, DType.int64);
    expect(DType.boolean.reductionAccumulator, DType.int64);
  });

  test('promotion is symmetric and handles mixed numerical families', () {
    for (final left in DType.values) {
      for (final right in DType.values) {
        expect(
          DType.promote(left, right),
          DType.promote(right, left),
          reason: '$left and $right',
        );
      }
    }

    expect(DType.promote(DType.boolean, DType.boolean), DType.boolean);
    expect(DType.promote(DType.boolean, DType.int8), DType.int8);
    expect(DType.promote(DType.uint8, DType.int8), DType.int16);
    expect(DType.promote(DType.uint8, DType.int16), DType.int16);
    expect(DType.promote(DType.int16, DType.int32), DType.int32);
    expect(DType.promote(DType.int32, DType.int64), DType.int64);
    expect(DType.promote(DType.float16, DType.bfloat16), DType.float32);
    expect(DType.promote(DType.float16, DType.int64), DType.float16);
    expect(DType.promote(DType.bfloat16, DType.int64), DType.bfloat16);
    expect(DType.promote(DType.float32, DType.float16), DType.float32);
    expect(DType.promote(DType.float64, DType.float32), DType.float64);
  });

  test('only implemented native storage can be allocated today', () {
    expect(DType.float32.nativeStorageImplemented, isTrue);
    for (final dtype in DType.values.where((value) => value != DType.float32)) {
      expect(dtype.nativeStorageImplemented, isFalse, reason: '$dtype');
      expect(
        () => Tensor.zeros(Shape([1]), dtype: dtype),
        throwsA(
          isA<UnsupportedOperationException>()
              .having((error) => error.operation, 'operation', 'tensor.full')
              .having(
                (error) => error.message,
                'message',
                contains('only DType.float32'),
              ),
        ),
      );
    }
  });
}
