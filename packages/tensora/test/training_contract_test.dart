import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  test('core build exposes explicit disabled training diagnostics', () {
    expect(TensoraRuntime.trainingAvailable, isFalse);
    expect(TensoraRuntime.cudaDeviceCount, 0);
    expect(TensoraRuntime.liveModuleCount, 0);
    expect(TensoraRuntime.liveOptimizerCount, 0);
    expect(
      () => TensoraRuntime.manualSeed(42),
      throwsA(isA<UnsupportedOperationException>()),
    );
    expect(() => TensoraRuntime.manualSeed(-1), throwsArgumentError);
  });

  test(
    'core tensors report no autograd and training transforms fail clearly',
    () {
      final tensor = Tensor.fromList([-1, 2], shape: Shape([2]));
      addTearDown(tensor.dispose);

      expect(tensor.requiresGrad, isFalse);
      expect(
        tensor.withRequiresGrad,
        throwsA(isA<UnsupportedOperationException>()),
      );
      expect(tensor.relu, throwsA(isA<UnsupportedOperationException>()));
      expect(tensor.sigmoid, throwsA(isA<UnsupportedOperationException>()));
      expect(tensor.tanh, throwsA(isA<UnsupportedOperationException>()));
      expect(tensor.backward, throwsA(isA<UnsupportedOperationException>()));
      expect(tensor.grad, throwsA(isA<UnsupportedOperationException>()));
    },
  );

  test(
    'disabled losses return structured unsupported errors without leaks',
    () {
      final a = Tensor.ones(Shape([2]));
      final b = Tensor.zeros(Shape([2]));
      addTearDown(a.dispose);
      addTearDown(b.dispose);

      expect(
        () => Losses.mse(a, b),
        throwsA(isA<UnsupportedOperationException>()),
      );
      expect(
        () => Losses.crossEntropy(a, b),
        throwsA(isA<UnsupportedOperationException>()),
      );
    },
  );

  test('module creation validates dimensions before native work', () {
    expect(() => Linear(0, 1), throwsArgumentError);
    expect(() => Linear(1, 0), throwsArgumentError);
    expect(() => Linear(1, 1), throwsA(isA<UnsupportedOperationException>()));
  });

  test('internal native adoption capability cannot be guessed', () {
    final tensor = Tensor.ones(Shape([1]));
    addTearDown(tensor.dispose);

    expect(() => tensor.nativeHandleForRuntime(Object()), throwsStateError);
    expect(
      () => Tensor.adoptNativeHandleForRuntime(1, Object()),
      throwsStateError,
    );
  });
}
