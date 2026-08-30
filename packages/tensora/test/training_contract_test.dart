import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  test('core build exposes native CPU training diagnostics', () {
    expect(TensoraRuntime.trainingAvailable, isTrue);
    expect(TensoraRuntime.cudaDeviceCount, 0);
    expect(TensoraRuntime.liveModuleCount, 0);
    expect(TensoraRuntime.liveOptimizerCount, 0);
    TensoraRuntime.manualSeed(42);
    expect(() => TensoraRuntime.manualSeed(-1), throwsArgumentError);
  });

  test('core tensors expose native autograd and training transforms', () {
    final tensor = Tensor.fromList([-1, 2], shape: Shape([2]));
    final leaf = tensor.withRequiresGrad();
    final activated = leaf.relu();
    final loss = activated.sum();
    addTearDown(tensor.dispose);
    addTearDown(leaf.dispose);
    addTearDown(activated.dispose);
    addTearDown(loss.dispose);

    expect(tensor.requiresGrad, isFalse);
    expect(leaf.requiresGrad, isTrue);
    loss.backward();

    final gradient = leaf.grad();
    addTearDown(gradient.dispose);
    expect(gradient.toList(), [0, 1]);

    final sigmoid = tensor.sigmoid();
    final tanh = tensor.tanh();
    addTearDown(sigmoid.dispose);
    addTearDown(tanh.dispose);
    expect(sigmoid.toList().first, closeTo(0.26894143, 1e-5));
    expect(tanh.toList().last, closeTo(0.9640276, 1e-5));
  });

  test('core losses execute through the native training engine', () {
    final prediction = Tensor.ones(Shape([2]));
    final target = Tensor.zeros(Shape([2]));
    final mse = Losses.mse(prediction, target);
    final logits = Tensor.fromList([2, 1], shape: Shape([1, 2]));
    final oneHot = Tensor.fromList([1, 0], shape: Shape([1, 2]));
    final crossEntropy = Losses.crossEntropy(logits, oneHot);
    addTearDown(prediction.dispose);
    addTearDown(target.dispose);
    addTearDown(mse.dispose);
    addTearDown(logits.dispose);
    addTearDown(oneHot.dispose);
    addTearDown(crossEntropy.dispose);

    expect(mse.toList().single, closeTo(1, 1e-6));
    expect(crossEntropy.toList().single, closeTo(0.31326166, 1e-5));
  });

  test('module creation validates dimensions before native work', () {
    expect(() => NativeLinear(0, 1), throwsArgumentError);
    expect(() => NativeLinear(1, 0), throwsArgumentError);

    final model = NativeLinear(1, 1);
    expect(model.isDisposed, isFalse);
    model.dispose();
    expect(model.isDisposed, isTrue);
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
