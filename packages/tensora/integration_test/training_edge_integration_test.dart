import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  test('autograd disabled leaf rejects backward and missing gradient', () {
    final input = Tensor.fromList([1, 2], shape: Shape([2]));
    final detached = input.withRequiresGrad(false);
    addTearDown(input.dispose);
    addTearDown(detached.dispose);

    expect(detached.requiresGrad, isFalse);
    expect(detached.backward, throwsA(isA<InvalidShapeException>()));
    expect(detached.grad, throwsA(isA<InvalidArgumentException>()));
  });

  test('training CPU transfer is independent and CUDA absence is explicit', () {
    final input = Tensor.fromList([1, -2], shape: Shape([2]));
    final copied = input.to(Device.cpu);
    addTearDown(input.dispose);
    addTearDown(copied.dispose);

    input.dispose();
    expect(copied.device, Device.cpu);
    expect(copied.toList(), [1, -2]);

    if (TensoraRuntime.cudaDeviceCount == 0) {
      expect(
        () => copied.to(Device.cuda(0)),
        throwsA(isA<UnsupportedOperationException>()),
      );
    }
  });

  test('loss functions reject incompatible shapes', () {
    final prediction = Tensor.ones(Shape([2]));
    final target = Tensor.ones(Shape([3]));
    final logits1d = Tensor.ones(Shape([2]));
    final target1d = Tensor.ones(Shape([2]));
    final logits2d = Tensor.ones(Shape([1, 2]));
    final wrongTarget2d = Tensor.ones(Shape([1, 3]));
    addTearDown(prediction.dispose);
    addTearDown(target.dispose);
    addTearDown(logits1d.dispose);
    addTearDown(target1d.dispose);
    addTearDown(logits2d.dispose);
    addTearDown(wrongTarget2d.dispose);

    expect(
      () => Losses.mse(prediction, target),
      throwsA(isA<InvalidShapeException>()),
    );
    expect(
      () => Losses.crossEntropy(logits1d, target1d),
      throwsA(isA<InvalidShapeException>()),
    );
    expect(
      () => Losses.crossEntropy(logits2d, wrongTarget2d),
      throwsA(isA<InvalidShapeException>()),
    );
  });

  test('module and optimizer reject use after deterministic disposal', () {
    final input = Tensor.ones(Shape([1, 1]));
    final model = NativeLinear(1, 1);
    final optimizer = NativeSgd(model);
    addTearDown(input.dispose);

    optimizer.dispose();
    optimizer.dispose();
    expect(optimizer.isDisposed, isTrue);
    expect(optimizer.zeroGrad, throwsA(isA<NativeRuntimeException>()));
    expect(optimizer.step, throwsA(isA<NativeRuntimeException>()));

    model.dispose();
    model.dispose();
    expect(model.isDisposed, isTrue);
    expect(model.train, throwsA(isA<DisposedTensorException>()));
    expect(model.eval, throwsA(isA<DisposedTensorException>()));
    expect(() => model.to(Device.cpu), throwsA(isA<DisposedTensorException>()));
    expect(model.parameters, throwsA(isA<DisposedTensorException>()));
    expect(model.buffers, throwsA(isA<DisposedTensorException>()));
    expect(
      () => model.save('/tmp/tensora-unused.pt'),
      throwsA(isA<DisposedTensorException>()),
    );
    expect(
      () => model.load('/tmp/tensora-unused.pt'),
      throwsA(isA<DisposedTensorException>()),
    );
    expect(() => model(input), throwsA(isA<DisposedTensorException>()));
    expect(() => NativeSgd(model), throwsA(isA<DisposedTensorException>()));
  });

  test('module paths and optimizer hyperparameters validate eagerly', () {
    final model = NativeLinear(2, 1, bias: false);
    addTearDown(model.dispose);

    expect(() => model.save(''), throwsArgumentError);
    expect(() => model.load('  '), throwsArgumentError);
    expect(() => NativeSgd(model, learningRate: 0), throwsArgumentError);
    expect(() => NativeSgd(model, momentum: -1), throwsArgumentError);
    expect(() => NativeSgd(model, weightDecay: -1), throwsArgumentError);
    expect(() => NativeAdam(model, learningRate: 0), throwsArgumentError);
    expect(() => NativeAdam(model, beta1: -0.1), throwsArgumentError);
    expect(() => NativeAdam(model, beta2: 1), throwsArgumentError);
    expect(() => NativeAdam(model, epsilon: 0), throwsArgumentError);
    expect(() => NativeAdam(model, weightDecay: -1), throwsArgumentError);
    expect(() => NativeAdamW(model, beta1: double.nan), throwsArgumentError);
    expect(() => NativeLinear(0, 1), throwsArgumentError);
    expect(() => NativeLinear(1, 0), throwsArgumentError);
  });
}
