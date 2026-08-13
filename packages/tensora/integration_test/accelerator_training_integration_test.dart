import 'dart:io';

import 'package:tensora/src/native/native_runtime.dart';
import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

Device _targetDevice() {
  final value = Platform.environment['TENSORA_TEST_DEVICE'];
  return switch (value) {
    'cuda' => Device.cuda(0),
    'mps' => Device.mps,
    'xpu' => Device.xpu(0),
    'hip' => Device.hip(0),
    _ => throw StateError(
      'TENSORA_TEST_DEVICE must be one of: cuda, mps, xpu, hip.',
    ),
  };
}

void _expectValues(
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
  test('selected accelerator executes tensor ops and a real training step', () {
    final device = _targetDevice();
    final runtime = NativeRuntime.instance;
    final baselineTensors = runtime.liveTensorCount();
    final baselineStorage = runtime.liveStorageBytes();
    final baselineModules = TensoraRuntime.liveModuleCount;
    final baselineOptimizers = TensoraRuntime.liveOptimizerCount;

    expect(TensoraRuntime.trainingAvailable, isTrue);
    expect(
      TensoraRuntime.deviceCount(device),
      greaterThan(0),
      reason: '$device must be visible to the loaded training runtime',
    );

    Tensor? left;
    Tensor? right;
    Tensor? product;
    Tensor? input;
    Tensor? target;
    Linear? module;
    SGD? optimizer;

    try {
      TensoraRuntime.manualSeed(20260813);

      left = Tensor.fromList([1, 2, 3, 4], shape: Shape([2, 2])).to(device);
      right = Tensor.fromList([5, 6, 7, 8], shape: Shape([2, 2])).to(device);
      product = left.matmul(right);

      expect(left.device, device);
      expect(right.device, device);
      expect(product.device, device);
      _expectValues(product.toList(), [19, 22, 43, 50]);

      module = Linear(1, 1);
      module.to(device);
      module.train();

      final parameters = module.parameters();
      try {
        expect(parameters, isNotEmpty);
        for (final parameter in parameters) {
          expect(
            parameter.device,
            device,
            reason: 'module parameters must not fall back to CPU',
          );
        }
      } finally {
        for (final parameter in parameters) {
          parameter.dispose();
        }
      }

      optimizer = SGD(module, learningRate: 0.05);
      input = Tensor.fromList([-2, -1, 1, 2], shape: Shape([4, 1])).to(device);
      target = Tensor.fromList([-3, -1, 3, 5], shape: Shape([4, 1])).to(device);

      expect(input.device, device);
      expect(target.device, device);

      double? firstLoss;
      var lastLoss = double.infinity;
      for (var step = 0; step < 120; step++) {
        optimizer.zeroGrad();
        final prediction = module(input);
        final loss = Losses.mse(prediction, target);
        try {
          expect(
            prediction.device,
            device,
            reason: 'forward output must remain on the requested accelerator',
          );
          expect(
            loss.device,
            device,
            reason: 'loss must remain on the requested accelerator',
          );
          final value = loss.toList().single;
          expect(value.isFinite, isTrue);
          firstLoss ??= value;
          lastLoss = value;
          loss.backward();
          optimizer.step();
        } finally {
          loss.dispose();
          prediction.dispose();
        }
      }

      expect(firstLoss, isNotNull);
      expect(
        lastLoss,
        lessThan(firstLoss! * 0.25),
        reason: 'accelerator-backed optimization must materially reduce loss',
      );

      final finalPrediction = module(input);
      try {
        expect(finalPrediction.device, device);
        final values = finalPrediction.toList();
        expect(values.every((value) => value.isFinite), isTrue);
      } finally {
        finalPrediction.dispose();
      }
    } finally {
      optimizer?.dispose();
      module?.dispose();
      target?.dispose();
      input?.dispose();
      product?.dispose();
      right?.dispose();
      left?.dispose();
    }

    expect(TensoraRuntime.liveOptimizerCount, baselineOptimizers);
    expect(TensoraRuntime.liveModuleCount, baselineModules);
    expect(runtime.liveTensorCount(), baselineTensors);
    expect(runtime.liveStorageBytes(), baselineStorage);
  });
}
