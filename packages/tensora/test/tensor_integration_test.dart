import 'package:tensora/src/native/native_runtime.dart';
import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void expectValues(
  List<double> actual,
  List<double> expected, {
  double tolerance = 1e-5,
}) {
  expect(actual, hasLength(expected.length));
  for (var index = 0; index < expected.length; index++) {
    expect(actual[index], closeTo(expected[index], tolerance));
  }
}

void main() {
  group('Tensor FFI integration', () {
    test('fromList owns native storage and reports metadata', () {
      final source = <num>[1, 2, 3, 4];
      final tensor = Tensor.fromList(source, shape: Shape([2, 2]));
      addTearDown(tensor.dispose);
      source[0] = 99;

      expect(tensor.shape, Shape([2, 2]));
      expect(tensor.dtype, DType.float32);
      expect(tensor.device, Device.cpu);
      expect(tensor.numel, 4);
      expectValues(tensor.toList(), [1, 2, 3, 4]);
    });

    test('zeros, ones, and full execute through native storage', () {
      final zeros = Tensor.zeros(Shape([2, 3]));
      final ones = Tensor.ones(Shape([2, 3]));
      final full = Tensor.full(Shape([2, 3]), 2.5);
      addTearDown(zeros.dispose);
      addTearDown(ones.dispose);
      addTearDown(full.dispose);

      expectValues(zeros.toList(), [0, 0, 0, 0, 0, 0]);
      expectValues(ones.toList(), [1, 1, 1, 1, 1, 1]);
      expectValues(full.toList(), [2.5, 2.5, 2.5, 2.5, 2.5, 2.5]);
    });

    test('CPU transfer returns an independent tensor with stable metadata', () {
      final input = Tensor.fromList([1, 2, 3, 4], shape: Shape([2, 2]));
      final copied = input.to(Device.cpu);
      addTearDown(input.dispose);
      addTearDown(copied.dispose);

      expect(copied.device, Device.cpu);
      expect(copied.shape, input.shape);
      expectValues(copied.toList(), [1, 2, 3, 4]);

      input.dispose();
      expectValues(copied.toList(), [1, 2, 3, 4]);
    });

    test('core runtime reports no CUDA devices and rejects CUDA transfer', () {
      expect(NativeRuntime.instance.cudaDeviceCount(), 0);

      final input = Tensor.ones(Shape([2, 2]));
      addTearDown(input.dispose);
      expect(
        () => input.to(Device.cuda(0)),
        throwsA(isA<UnsupportedOperationException>()),
      );
    });

    test(
      'creation requires explicit CPU host import before device transfer',
      () {
        expect(
          () => Tensor.ones(Shape([1]), device: Device.cuda(0)),
          throwsA(isA<UnsupportedOperationException>()),
        );
      },
    );

    test('reshape preserves values and element count', () {
      final input = Tensor.fromList([1, 2, 3, 4, 5, 6], shape: Shape([2, 3]));
      final reshaped = input.reshape(Shape([3, 2]));
      addTearDown(input.dispose);
      addTearDown(reshaped.dispose);

      expect(reshaped.shape, Shape([3, 2]));
      expectValues(reshaped.toList(), [1, 2, 3, 4, 5, 6]);
    });

    test('2D transpose matches the mathematical reference', () {
      final input = Tensor.fromList([1, 2, 3, 4, 5, 6], shape: Shape([2, 3]));
      final transposed = input.transpose();
      addTearDown(input.dispose);
      addTearDown(transposed.dispose);

      expect(transposed.shape, Shape([3, 2]));
      expectValues(transposed.toList(), [1, 4, 2, 5, 3, 6]);
    });

    test('add and multiply require equal shapes and compute natively', () {
      final a = Tensor.fromList([1, 2, 3, 4], shape: Shape([2, 2]));
      final b = Tensor.fromList([5, 6, 7, 8], shape: Shape([2, 2]));
      final add = a.add(b);
      final multiply = a.multiply(b);
      addTearDown(a.dispose);
      addTearDown(b.dispose);
      addTearDown(add.dispose);
      addTearDown(multiply.dispose);

      expectValues(add.toList(), [6, 8, 10, 12]);
      expectValues(multiply.toList(), [5, 12, 21, 32]);
    });

    test('sum returns a rank-zero scalar Tensor', () {
      final input = Tensor.fromList([1, 2, 3, 4], shape: Shape([2, 2]));
      final sum = input.sum();
      addTearDown(input.dispose);
      addTearDown(sum.dispose);

      expect(sum.shape, Shape(const []));
      expectValues(sum.toList(), [10]);
    });

    test('matmul matches the Milestone 1 reference example', () {
      final a = Tensor.fromList([1, 2, 3, 4], shape: Shape([2, 2]));
      final b = Tensor.fromList([5, 6, 7, 8], shape: Shape([2, 2]));
      final result = a.matmul(b);
      addTearDown(a.dispose);
      addTearDown(b.dispose);
      addTearDown(result.dispose);

      expect(result.shape, Shape([2, 2]));
      expectValues(result.toList(), [19, 22, 43, 50]);
    });

    test('invalid reshape maps native status into InvalidShapeException', () {
      final input = Tensor.fromList([1, 2, 3, 4], shape: Shape([2, 2]));
      addTearDown(input.dispose);

      expect(
        () => input.reshape(Shape([3, 1])),
        throwsA(isA<InvalidShapeException>()),
      );
    });

    test('invalid matmul maps native status into InvalidShapeException', () {
      final left = Tensor.ones(Shape([2, 3]));
      final right = Tensor.ones(Shape([2, 4]));
      addTearDown(left.dispose);
      addTearDown(right.dispose);

      expect(() => left.matmul(right), throwsA(isA<InvalidShapeException>()));
    });

    test('fromList rejects host payload with the wrong length', () {
      expect(
        () => Tensor.fromList([1, 2, 3], shape: Shape([2, 2])),
        throwsA(isA<InvalidShapeException>()),
      );
    });

    test(
      'dispose is deterministic, double-dispose is safe, use-after fails',
      () {
        final tensor = Tensor.ones(Shape([2, 2]));

        tensor.dispose();
        tensor.dispose();

        expect(tensor.isDisposed, isTrue);
        expect(tensor.toList, throwsA(isA<DisposedTensorException>()));
        expect(tensor.sum, throwsA(isA<DisposedTensorException>()));
        expect(
          () => tensor.to(Device.cpu),
          throwsA(isA<DisposedTensorException>()),
        );
      },
    );

    test(
      'invalid native handles are converted into structured Dart errors',
      () {
        expect(
          () => NativeRuntime.instance.numel(0x7ffffffffffffffe),
          throwsA(isA<NativeRuntimeException>()),
        );
      },
    );
  });

  group('Tensor property invariants', () {
    test('transpose twice restores deterministic matrices', () {
      for (var rows = 1; rows <= 4; rows++) {
        for (var columns = 1; columns <= 4; columns++) {
          final values = List<num>.generate(
            rows * columns,
            (index) => (index * 3 - 7) / 5,
          );
          final input = Tensor.fromList(values, shape: Shape([rows, columns]));
          final transposed = input.transpose();
          final restored = transposed.transpose();

          expect(restored.shape, input.shape);
          expectValues(
            restored.toList(),
            values.map((value) => value.toDouble()).toList(),
          );

          restored.dispose();
          transposed.dispose();
          input.dispose();
        }
      }
    });

    test('adding zeros and multiplying ones preserve values', () {
      for (var length = 1; length <= 32; length++) {
        final values = List<num>.generate(length, (index) => index - 11.5);
        final input = Tensor.fromList(values, shape: Shape([length]));
        final zeros = Tensor.zeros(Shape([length]));
        final ones = Tensor.ones(Shape([length]));
        final added = input.add(zeros);
        final multiplied = input.multiply(ones);

        final expected = values.map((value) => value.toDouble()).toList();
        expectValues(added.toList(), expected);
        expectValues(multiplied.toList(), expected);

        multiplied.dispose();
        added.dispose();
        ones.dispose();
        zeros.dispose();
        input.dispose();
      }
    });
  });

  test(
    'repeated Dart FFI lifecycle returns native counters to baseline',
    () {
      final runtime = NativeRuntime.instance;
      final startCount = runtime.liveTensorCount();
      final startBytes = runtime.liveStorageBytes();

      for (var index = 0; index < 1000; index++) {
        final a = Tensor.fromList([1, 2, 3, 4], shape: Shape([2, 2]));
        final b = Tensor.fromList([5, 6, 7, 8], shape: Shape([2, 2]));
        final result = a.matmul(b);
        result.dispose();
        b.dispose();
        a.dispose();
      }

      expect(runtime.liveTensorCount(), startCount);
      expect(runtime.liveStorageBytes(), startBytes);
    },
    tags: 'lifecycle',
  );
}
