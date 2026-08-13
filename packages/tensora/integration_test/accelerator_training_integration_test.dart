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
    _ =>
      throw StateError(
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
    expect(TensoraRuntime.availableDevices, contains(device));
    expect(TensoraRuntime.preferredDevice, isNot(Device.cpu));

    TensoraRuntime.manualSeed(20260813);

    final left = Tensor.fromList(
      [1, 2, 3, 4],
      shape: Shape([2, 2]),
      device: device,
    );
    final right = Tensor.fromList(
      [5, 6, 7, 8],
      shape: Shape([2, 2]),
      device: device,
    );
    final product = left.matmul(right);
    try {
      expect(left.device, device);
      expect(right.device, device);
      expect(product.device, device);
      _expectValues(product.toList(), [19, 22, 43, 50]);
    } finally {
      product.dispose();
      right.dispose();
      left.dispose();
    }
    expect(runtime.liveTensorCount(), baselineTensors);
    expect(runtime.liveStorageBytes(), baselineStorage);

    final module = Linear(1, 1);
    SGD? optimizer;
    Tensor? input;
    Tensor? target;
    try {
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
      expect(runtime.liveTensorCount(), baselineTensors);
      expect(
        runtime.liveStorageBytes(),
        baselineStorage,
        reason: 'disposed parameter views must release every Tensora wrapper',
      );

      optimizer = SGD(module, learningRate: 0.05);
      input = Tensor.fromList(
        [-2, -1, 1, 2],
        shape: Shape([4, 1]),
        device: device,
      );
      target = Tensor.fromList(
        [-3, -1, 3, 5],
        shape: Shape([4, 1]),
        device: device,
      );

      expect(input.device, device);
      expect(target.device, device);
      final persistentTrainingBytes =
          baselineStorage + (input.numel + target.numel) * 4;
      expect(runtime.liveTensorCount(), baselineTensors + 2);
      expect(runtime.liveStorageBytes(), persistentTrainingBytes);

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
        if (step == 0 || step == 119) {
          expect(runtime.liveTensorCount(), baselineTensors + 2);
          expect(
            runtime.liveStorageBytes(),
            persistentTrainingBytes,
            reason: 'training temporaries must not survive a completed step',
          );
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
      expect(runtime.liveTensorCount(), baselineTensors + 2);
      expect(
        runtime.liveStorageBytes(),
        persistentTrainingBytes,
        reason: 'disposing the final prediction must release its wrapper',
      );
    } finally {
      optimizer?.dispose();
      target?.dispose();
      input?.dispose();
      module.dispose();
    }

    expect(TensoraRuntime.liveOptimizerCount, baselineOptimizers);
    expect(TensoraRuntime.liveModuleCount, baselineModules);
    expect(runtime.liveTensorCount(), baselineTensors);
    expect(runtime.liveStorageBytes(), baselineStorage);
  });
}
