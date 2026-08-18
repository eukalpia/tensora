import 'dart:typed_data';

import 'package:tensora/src/native/native_runtime.dart';
import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  group('typed Tensor FFI integration', () {
    test('all ten public dtypes round-trip through native storage', () {
      final tensors = <Tensor>[];
      addTearDown(() {
        for (final tensor in tensors.reversed) {
          tensor.dispose();
        }
      });

      Tensor track(Tensor tensor) {
        tensors.add(tensor);
        return tensor;
      }

      final float16 = track(
        Tensor.fromList(
          <Object>[1.5, -2, 0.25],
          shape: Shape([3]),
          dtype: DType.float16,
        ),
      );
      expect(float16.toList<double>(), <double>[1.5, -2, 0.25]);
      expect(float16.toTypedData(), isA<Uint16List>());

      final bfloat16 = track(
        Tensor.fromList(
          <Object>[1.5, -2, 0.25],
          shape: Shape([3]),
          dtype: DType.bfloat16,
        ),
      );
      expect(bfloat16.toList<double>(), <double>[1.5, -2, 0.25]);
      expect(bfloat16.toTypedData(), isA<Uint16List>());

      final float32 = track(
        Tensor.fromList(<Object>[1.5, -2, 0.25], shape: Shape([3])),
      );
      expect(float32.toList<double>(), <double>[1.5, -2, 0.25]);
      expect(float32.toTypedData(), isA<Float32List>());

      final float64 = track(
        Tensor.fromList(
          <Object>[1.5, -2, 0.25],
          shape: Shape([3]),
          dtype: DType.float64,
        ),
      );
      expect(float64.toList<double>(), <double>[1.5, -2, 0.25]);
      expect(float64.toTypedData(), isA<Float64List>());

      final int8 = track(
        Tensor.fromList(
          <Object>[-128, 0, 127],
          shape: Shape([3]),
          dtype: DType.int8,
        ),
      );
      expect(int8.toList<int>(), <int>[-128, 0, 127]);
      expect(int8.toTypedData(), isA<Int8List>());

      final uint8 = track(
        Tensor.fromList(
          <Object>[0, 128, 255],
          shape: Shape([3]),
          dtype: DType.uint8,
        ),
      );
      expect(uint8.toList<int>(), <int>[0, 128, 255]);
      expect(uint8.toTypedData(), isA<Uint8List>());

      final int16 = track(
        Tensor.fromList(
          <Object>[-32768, 0, 32767],
          shape: Shape([3]),
          dtype: DType.int16,
        ),
      );
      expect(int16.toList<int>(), <int>[-32768, 0, 32767]);
      expect(int16.toTypedData(), isA<Int16List>());

      final int32 = track(
        Tensor.fromList(
          <Object>[-2147483648, 0, 2147483647],
          shape: Shape([3]),
          dtype: DType.int32,
        ),
      );
      expect(int32.toList<int>(), <int>[-2147483648, 0, 2147483647]);
      expect(int32.toTypedData(), isA<Int32List>());

      final int64 = track(
        Tensor.fromList(
          <Object>[-9007199254740991, 0, 9007199254740991],
          shape: Shape([3]),
          dtype: DType.int64,
        ),
      );
      expect(int64.toList<int>(), <int>[
        -9007199254740991,
        0,
        9007199254740991,
      ]);
      expect(int64.toTypedData(), isA<Int64List>());

      final boolean = track(
        Tensor.fromList(
          <Object>[false, true, true],
          shape: Shape([3]),
          dtype: DType.boolean,
        ),
      );
      expect(boolean.toList<bool>(), <bool>[false, true, true]);
      expect(boolean.toTypedData(), isA<Uint8List>());
    });

    test('typed full, zeros, and ones preserve dtype and values', () {
      final full = Tensor.full(Shape([3]), -7, dtype: DType.int16);
      final zeros = Tensor.zeros(Shape([3]), dtype: DType.int64);
      final ones = Tensor.ones(Shape([3]), dtype: DType.boolean);
      final falseValues = Tensor.full(Shape([3]), false, dtype: DType.boolean);
      addTearDown(full.dispose);
      addTearDown(zeros.dispose);
      addTearDown(ones.dispose);
      addTearDown(falseValues.dispose);

      expect(full.dtype, DType.int16);
      expect(full.toList<int>(), <int>[-7, -7, -7]);
      expect(zeros.toList<int>(), <int>[0, 0, 0]);
      expect(ones.toList<bool>(), <bool>[true, true, true]);
      expect(falseValues.toList<bool>(), <bool>[false, false, false]);
    });

    test('cast and CPU transfer materialize logical view order', () {
      final source = Tensor.fromList(
        <Object>[1, 2, 3, 4, 5, 6],
        shape: Shape([2, 3]),
        dtype: DType.int32,
      );
      final transposed = source.transpose();
      final cast = transposed.cast(DType.float64);
      final copied = transposed.to(Device.cpu);
      addTearDown(source.dispose);
      addTearDown(transposed.dispose);
      addTearDown(cast.dispose);
      addTearDown(copied.dispose);

      expect(cast.shape, Shape([3, 2]));
      expect(cast.dtype, DType.float64);
      expect(cast.toList<double>(), <double>[1, 4, 2, 5, 3, 6]);
      expect(copied.toList<int>(), <int>[1, 4, 2, 5, 3, 6]);

      source.dispose();
      transposed.dispose();
      expect(cast.toList<double>(), <double>[1, 4, 2, 5, 3, 6]);
      expect(copied.toList<int>(), <int>[1, 4, 2, 5, 3, 6]);
    });

    test('typed host validation rejects mismatched values and ranges', () {
      expect(
        () => Tensor.fromList(
          <Object>[128],
          shape: Shape([1]),
          dtype: DType.int8,
        ),
        throwsA(isA<InvalidArgumentException>()),
      );
      expect(
        () => Tensor.fromList(
          <Object>[1.0],
          shape: Shape([1]),
          dtype: DType.int32,
        ),
        throwsA(isA<InvalidArgumentException>()),
      );
      expect(
        () => Tensor.fromList(
          <Object>[1],
          shape: Shape([1]),
          dtype: DType.boolean,
        ),
        throwsA(isA<InvalidArgumentException>()),
      );
      expect(
        () => Tensor.full(Shape([1]), true, dtype: DType.float32),
        throwsA(isA<InvalidArgumentException>()),
      );
    });

    test('toList validates its requested Dart element type', () {
      final tensor = Tensor.ones(Shape([1]), dtype: DType.int32);
      addTearDown(tensor.dispose);

      expect(
        tensor.toList<double>,
        throwsA(
          isA<InvalidArgumentException>().having(
            (error) => error.operation,
            'operation',
            'tensor.toList',
          ),
        ),
      );
    });

    test('integer and boolean tensors cannot enter autograd', () {
      final integer = Tensor.ones(Shape([1]), dtype: DType.int32);
      final boolean = Tensor.ones(Shape([1]), dtype: DType.boolean);
      addTearDown(integer.dispose);
      addTearDown(boolean.dispose);

      expect(
        integer.withRequiresGrad,
        throwsA(isA<UnsupportedOperationException>()),
      );
      expect(
        boolean.withRequiresGrad,
        throwsA(isA<UnsupportedOperationException>()),
      );
    });

    test('typed lifetimes return native counters to their baseline', () {
      final runtime = NativeRuntime.instance;
      final baselineTensors = runtime.liveTensorCount();
      final baselineStorage = runtime.liveStorageBytes();

      final source = Tensor.fromList(
        <Object>[1, 2, 3, 4],
        shape: Shape([2, 2]),
        dtype: DType.int64,
      );
      final cast = source.cast(DType.float64);
      final copy = source.to(Device.cpu);
      copy.dispose();
      cast.dispose();
      source.dispose();

      expect(runtime.liveTensorCount(), baselineTensors);
      expect(runtime.liveStorageBytes(), baselineStorage);
    });
  });
}
