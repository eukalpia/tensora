import 'dart:io';

import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void expectValues(
  List<double> actual,
  List<double> expected, {
  double tolerance = 1e-4,
}) {
  expect(actual, hasLength(expected.length));
  for (var index = 0; index < expected.length; index++) {
    expect(actual[index], closeTo(expected[index], tolerance));
  }
}

void main() {
  test('training backend is available in the training integration build', () {
    expect(TensoraRuntime.trainingAvailable, isTrue);
    expect(TensoraRuntime.cudaDeviceCount, greaterThanOrEqualTo(0));
    TensoraRuntime.manualSeed(123456);
  });

  test('autograd produces the reference ReLU gradient', () {
    final input = Tensor.fromList([-1, 2], shape: Shape([2]));
    final leaf = input.withRequiresGrad();
    final activated = leaf.relu();
    final loss = activated.sum();
    addTearDown(input.dispose);
    addTearDown(leaf.dispose);
    addTearDown(activated.dispose);
    addTearDown(loss.dispose);

    expect(leaf.requiresGrad, isTrue);
    expect(input.requiresGrad, isFalse);
    loss.backward();

    final gradient = leaf.grad();
    addTearDown(gradient.dispose);
    expectValues(gradient.toList(), [0, 1]);
  });

  test('sigmoid and tanh match deterministic references', () {
    final input = Tensor.fromList([0, 1], shape: Shape([2]));
    final sigmoid = input.sigmoid();
    final tanh = input.tanh();
    addTearDown(input.dispose);
    addTearDown(sigmoid.dispose);
    addTearDown(tanh.dispose);

    expectValues(sigmoid.toList(), [0.5, 0.7310586], tolerance: 1e-5);
    expectValues(tanh.toList(), [0, 0.7615942], tolerance: 1e-5);
  });

  test('GELU and SiLU match exact deterministic references', () {
    final input = Tensor.fromList([-1, 0, 1], shape: Shape([3]));
    final gelu = input.gelu();
    final silu = input.silu();
    addTearDown(input.dispose);
    addTearDown(gelu.dispose);
    addTearDown(silu.dispose);

    expectValues(
      gelu.toList(),
      [-0.15865526, 0, 0.8413447],
      tolerance: 2e-5,
    );
    expectValues(
      silu.toList(),
      [-0.26894143, 0, 0.7310586],
      tolerance: 2e-5,
    );
  });

  test('GELU and SiLU autograd match analytical derivatives', () {
    final input = Tensor.fromList([-1, 0, 1], shape: Shape([3]));
    final geluLeaf = input.withRequiresGrad();
    final siluLeaf = input.withRequiresGrad();
    final gelu = geluLeaf.gelu();
    final silu = siluLeaf.silu();
    final geluLoss = gelu.sum();
    final siluLoss = silu.sum();
    addTearDown(input.dispose);
    addTearDown(geluLeaf.dispose);
    addTearDown(siluLeaf.dispose);
    addTearDown(gelu.dispose);
    addTearDown(silu.dispose);
    addTearDown(geluLoss.dispose);
    addTearDown(siluLoss.dispose);

    geluLoss.backward();
    siluLoss.backward();

    final geluGradient = geluLeaf.grad();
    final siluGradient = siluLeaf.grad();
    addTearDown(geluGradient.dispose);
    addTearDown(siluGradient.dispose);
    expectValues(
      geluGradient.toList(),
      [-0.08331547, 0.5, 1.0833155],
      tolerance: 3e-5,
    );
    expectValues(
      siluGradient.toList(),
      [0.07232949, 0.5, 0.9276705],
      tolerance: 3e-5,
    );
  });

  test('SwiGLU halves the final dimension and has exact backward', () {
    final input = Tensor.fromList([1, -1, 2, 3], shape: Shape([1, 4]));
    final leaf = input.withRequiresGrad();
    final output = leaf.swiglu();
    final loss = output.sum();
    addTearDown(input.dispose);
    addTearDown(leaf.dispose);
    addTearDown(output.dispose);
    addTearDown(loss.dispose);

    expect(output.shape, Shape([1, 2]));
    expectValues(output.toList(), [1.4621172, -0.80682427], tolerance: 3e-5);

    loss.backward();
    final gradient = leaf.grad();
    addTearDown(gradient.dispose);
    expectValues(
      gradient.toList(),
      [1.855341, 0.21698847, 0.7310586, -0.26894143],
      tolerance: 4e-5,
    );
  });

  test('SwiGLU rejects rank-zero and odd final dimensions', () {
    final scalar = Tensor.fromList([1], shape: Shape([]));
    final odd = Tensor.fromList([1, 2, 3], shape: Shape([1, 3]));
    addTearDown(scalar.dispose);
    addTearDown(odd.dispose);

    expect(scalar.swiglu, throwsA(isA<InvalidShapeException>()));
    expect(odd.swiglu, throwsA(isA<InvalidShapeException>()));
  });

  test('cross entropy matches the float32 one-hot reference', () {
    final logits = Tensor.fromList([2, 1], shape: Shape([1, 2]));
    final target = Tensor.fromList([1, 0], shape: Shape([1, 2]));
    final loss = Losses.crossEntropy(logits, target);
    addTearDown(logits.dispose);
    addTearDown(target.dispose);
    addTearDown(loss.dispose);

    expectValues(loss.toList(), [0.31326166], tolerance: 1e-5);
  });

  test('Linear trains y = 2x + 1 and checkpoint restores output', () {
    TensoraRuntime.manualSeed(42);
    final startModules = TensoraRuntime.liveModuleCount;
    final startOptimizers = TensoraRuntime.liveOptimizerCount;

    final x = Tensor.fromList([-1, 0, 1, 2], shape: Shape([4, 1]));
    final y = Tensor.fromList([-1, 1, 3, 5], shape: Shape([4, 1]));
    final model = Linear(1, 1);
    final optimizer = SGD(model, learningRate: 0.1);
    addTearDown(x.dispose);
    addTearDown(y.dispose);
    addTearDown(optimizer.dispose);
    addTearDown(model.dispose);

    model.train();
    model.to(Device.cpu);

    final parametersBefore = model.parameters();
    expect(parametersBefore, hasLength(2));
    final firstWeightBefore = parametersBefore.first.toList().first;
    for (final parameter in parametersBefore) {
      parameter.dispose();
    }
    expect(model.buffers(), isEmpty);

    var initialLoss = double.nan;
    var finalLoss = double.nan;
    for (var step = 0; step < 200; step++) {
      optimizer.zeroGrad();
      final prediction = model(x);
      final loss = Losses.mse(prediction, y);
      final value = loss.toList().single;
      if (step == 0) initialLoss = value;
      finalLoss = value;
      loss.backward();
      optimizer.step();
      loss.dispose();
      prediction.dispose();
    }

    expect(initialLoss.isFinite, isTrue);
    expect(finalLoss, lessThan(initialLoss));
    expect(finalLoss, lessThan(1e-3));

    final parametersAfter = model.parameters();
    final firstWeightAfter = parametersAfter.first.toList().first;
    expect(firstWeightAfter, isNot(closeTo(firstWeightBefore, 1e-6)));
    for (final parameter in parametersAfter) {
      parameter.dispose();
    }

    final checkpoint = File(
      '${Directory.systemTemp.path}${Platform.pathSeparator}'
      'tensora-dart-training-checkpoint.pt',
    );
    if (checkpoint.existsSync()) checkpoint.deleteSync();
    addTearDown(() {
      if (checkpoint.existsSync()) checkpoint.deleteSync();
    });

    model.eval();
    final savedOutputTensor = model(x);
    final savedOutput = savedOutputTensor.toList();
    savedOutputTensor.dispose();
    model.save(checkpoint.path);
    expect(checkpoint.existsSync(), isTrue);

    for (var step = 0; step < 10; step++) {
      optimizer.zeroGrad();
      final prediction = model(x);
      final loss = Losses.mse(prediction, y);
      loss.backward();
      optimizer.step();
      loss.dispose();
      prediction.dispose();
    }

    model.load(checkpoint.path);
    final restoredOutputTensor = model(x);
    expectValues(restoredOutputTensor.toList(), savedOutput, tolerance: 1e-5);
    restoredOutputTensor.dispose();

    optimizer.dispose();
    model.dispose();
    expect(TensoraRuntime.liveOptimizerCount, startOptimizers);
    expect(TensoraRuntime.liveModuleCount, startModules);
  });

  test('Adam and AdamW validate and own native optimizer handles', () {
    final model = Linear(2, 1, bias: false);
    final adam = Adam(model);
    final adamW = AdamW(model);
    addTearDown(model.dispose);
    addTearDown(adam.dispose);
    addTearDown(adamW.dispose);

    expect(adam.isDisposed, isFalse);
    expect(adamW.isDisposed, isFalse);
    adam.zeroGrad();
    adamW.zeroGrad();

    expect(() => SGD(model, learningRate: 0), throwsArgumentError);
    expect(() => Adam(model, beta1: 1), throwsArgumentError);
  });
}
