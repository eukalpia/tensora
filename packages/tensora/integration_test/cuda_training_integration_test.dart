import 'dart:io';

import 'package:tensora/src/native/native_runtime.dart';
import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  test('real CUDA training executes without CPU fallback', () {
    expect(TensoraRuntime.trainingAvailable, isTrue);
    expect(
      TensoraRuntime.cudaDeviceCount,
      greaterThan(0),
      reason: 'CUDA validation requires a real visible NVIDIA CUDA device',
    );

    final runtime = NativeRuntime.instance;
    final baselineTensors = runtime.liveTensorCount();
    final baselineStorage = runtime.liveStorageBytes();
    final baselineModules = TensoraRuntime.liveModuleCount;
    final baselineOptimizers = TensoraRuntime.liveOptimizerCount;
    final cuda = Device.cuda(0);

    TensoraRuntime.manualSeed(20260813);

    final xHost = Tensor.fromList([-1, 0, 1, 2], shape: Shape([4, 1]));
    final yHost = Tensor.fromList([-1, 1, 3, 5], shape: Shape([4, 1]));
    final x = xHost.to(cuda);
    final y = yHost.to(cuda);
    expect(x.device, cuda);
    expect(y.device, cuda);

    final model = NativeLinear(1, 1);
    model.to(cuda);
    final parameters = model.parameters();
    expect(parameters, isNotEmpty);
    for (final parameter in parameters) {
      expect(parameter.device, cuda);
      parameter.dispose();
    }

    final optimizer = NativeSgd(model, learningRate: 0.1);
    var initialLoss = double.nan;
    var finalLoss = double.nan;

    for (var step = 0; step < 200; step++) {
      optimizer.zeroGrad();
      final prediction = model(x);
      expect(prediction.device, cuda);
      final loss = Losses.mse(prediction, y);
      expect(loss.device, cuda);
      final value = loss.toList().single;
      expect(value.isFinite, isTrue);
      if (step == 0) initialLoss = value;
      finalLoss = value;
      loss.backward();
      optimizer.step();
      loss.dispose();
      prediction.dispose();
    }

    expect(finalLoss, lessThan(initialLoss));
    expect(finalLoss, lessThan(1e-3));

    final checkpoint = File(
      '${Directory.systemTemp.path}${Platform.pathSeparator}'
      'tensora-cuda-checkpoint.pt',
    );
    if (checkpoint.existsSync()) checkpoint.deleteSync();
    model.save(checkpoint.path);
    expect(checkpoint.existsSync(), isTrue);
    model.load(checkpoint.path);
    checkpoint.deleteSync();

    optimizer.dispose();
    model.dispose();
    y.dispose();
    x.dispose();
    yHost.dispose();
    xHost.dispose();

    expect(runtime.liveTensorCount(), baselineTensors);
    expect(runtime.liveStorageBytes(), baselineStorage);
    expect(TensoraRuntime.liveModuleCount, baselineModules);
    expect(TensoraRuntime.liveOptimizerCount, baselineOptimizers);
  });
}
