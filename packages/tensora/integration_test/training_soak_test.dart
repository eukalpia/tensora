import 'dart:io';

import 'package:tensora/src/native/native_runtime.dart';
import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  test('1000 training lifecycles return all native counters to baseline', () {
    final runtime = NativeRuntime.instance;
    final baselineTensors = runtime.liveTensorCount();
    final baselineStorage = runtime.liveStorageBytes();
    final baselineModules = TensoraRuntime.liveModuleCount;
    final baselineOptimizers = TensoraRuntime.liveOptimizerCount;
    final checkpoint = File(
      '${Directory.systemTemp.path}${Platform.pathSeparator}'
      'tensora-training-soak.pt',
    );
    if (checkpoint.existsSync()) checkpoint.deleteSync();
    addTearDown(() {
      if (checkpoint.existsSync()) checkpoint.deleteSync();
    });

    TensoraRuntime.manualSeed(20260813);

    for (var cycle = 0; cycle < 1000; cycle++) {
      final x = Tensor.fromList([-1, 0, 1, 2], shape: Shape([4, 1]));
      final y = Tensor.fromList([-1, 1, 3, 5], shape: Shape([4, 1]));
      final model = Linear(1, 1);
      final optimizer = SGD(model, learningRate: 0.05);

      optimizer.zeroGrad();
      final prediction = model(x);
      final loss = Losses.mse(prediction, y);
      final before = loss.toList().single;
      expect(before.isFinite, isTrue);
      loss.backward();
      optimizer.step();

      model.save(checkpoint.path);
      model.load(checkpoint.path);

      loss.dispose();
      prediction.dispose();
      optimizer.dispose();
      model.dispose();
      y.dispose();
      x.dispose();

      if ((cycle + 1) % 100 == 0) {
        expect(
          runtime.liveTensorCount(),
          baselineTensors,
          reason: 'tensor handles leaked by cycle ${cycle + 1}',
        );
        expect(
          runtime.liveStorageBytes(),
          baselineStorage,
          reason: 'tensor storage leaked by cycle ${cycle + 1}',
        );
        expect(
          TensoraRuntime.liveModuleCount,
          baselineModules,
          reason: 'module handles leaked by cycle ${cycle + 1}',
        );
        expect(
          TensoraRuntime.liveOptimizerCount,
          baselineOptimizers,
          reason: 'optimizer handles leaked by cycle ${cycle + 1}',
        );
      }
    }

    expect(runtime.liveTensorCount(), baselineTensors);
    expect(runtime.liveStorageBytes(), baselineStorage);
    expect(TensoraRuntime.liveModuleCount, baselineModules);
    expect(TensoraRuntime.liveOptimizerCount, baselineOptimizers);
  });
}
