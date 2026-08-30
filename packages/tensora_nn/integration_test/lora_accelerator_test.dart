// Proves the fine-tuning story end to end on whatever accelerator the loaded
// runtime exposes: a frozen base projection, trainable low-rank factors, an
// optimizer handed every parameter that still only updates the trainable ones,
// and a merge that preserves the result.
//
// Set TENSORA_TEST_DEVICE to require a specific accelerator. Without it the
// test runs wherever the runtime prefers, so it is useful on a CPU-only build
// as well.
import 'dart:io';

import 'package:tensora/tensora.dart' as core;
import 'package:tensora_nn/tensora_nn.dart';
import 'package:tensora_optim/tensora_optim.dart' as optim;
import 'package:test/test.dart';

core.Device _targetDevice() {
  final requested = Platform.environment['TENSORA_TEST_DEVICE'];
  if (requested == null || requested.isEmpty) {
    return core.TensoraRuntime.preferredDevice;
  }
  return switch (requested) {
    'cuda' => core.Device.cuda(0),
    'mps' => core.Device.mps,
    'xpu' => core.Device.xpu(0),
    'hip' => core.Device.hip(0),
    'cpu' => core.Device.cpu,
    _ =>
      throw StateError(
        'TENSORA_TEST_DEVICE must be one of: cpu, cuda, mps, xpu, hip.',
      ),
  };
}

void main() {
  test('LoRA fine-tunes on the selected device without touching the base', () {
    final device = _targetDevice();
    expect(
      core.TensoraRuntime.deviceCount(device),
      greaterThan(0),
      reason: 'requested device $device is not visible to the runtime',
    );

    core.TensoraRuntime.manualSeed(20260830);

    final adapter = LoRALinear(
      inFeatures: 8,
      outFeatures: 4,
      rank: 2,
      alpha: 4,
      seed: 7,
    );
    addTearDown(adapter.dispose);
    adapter.to(device);

    for (final entry in adapter.namedParameters) {
      expect(
        entry.parameter.device,
        device,
        reason: '${entry.name} did not move',
      );
    }

    // Record the frozen weights so we can prove training never altered them.
    final baseWeight =
        adapter.namedParameters
            .firstWhere((entry) => entry.name == 'base.weight')
            .parameter;
    final baseBefore = baseWeight.snapshot();
    addTearDown(baseBefore.dispose);
    final baseBeforeValues = baseBefore.toList();

    final x = core.Tensor.fromList(
      List<double>.generate(4 * 8, (i) => (i % 5) * 0.3 - 0.6),
      shape: core.Shape(<int>[4, 8]),
      device: device,
    );
    addTearDown(x.dispose);
    final y = core.Tensor.fromList(
      List<double>.generate(4 * 4, (i) => (i % 3) * 0.5 - 0.5),
      shape: core.Shape(<int>[4, 4]),
      device: device,
    );
    addTearDown(y.dispose);

    // Every parameter is handed over, including the frozen ones. The runtime
    // filters to what actually requires gradients, so callers do not have to.
    final optimizer = optim.Adam(
      parameters: adapter.parameters,
      learningRate: 0.05,
    );
    addTearDown(optimizer.dispose);

    var first = double.nan;
    var last = double.nan;
    for (var step = 0; step < 120; step++) {
      optimizer.zeroGrad();
      final prediction = adapter(x);
      final loss = core.Losses.mse(prediction, y);
      final value = loss.toList().single;
      expect(value.isFinite, isTrue);
      if (step == 0) first = value;
      last = value;
      loss.backward();
      optimizer.step();
      loss.dispose();
      prediction.dispose();
    }

    expect(last, lessThan(first), reason: 'the adapter did not learn');

    final baseAfterValues = baseWeight.snapshot();
    addTearDown(baseAfterValues.dispose);
    final after = baseAfterValues.toList();
    for (var index = 0; index < baseBeforeValues.length; index++) {
      expect(
        after[index],
        baseBeforeValues[index],
        reason: 'frozen base weight changed at $index',
      );
    }

    // Merging must not disturb the trained behavior.
    final beforeMerge = adapter(x).toList();
    adapter.mergeIntoBase();
    final afterMerge = adapter(x);
    addTearDown(afterMerge.dispose);
    final merged = afterMerge.toList();
    for (var index = 0; index < merged.length; index++) {
      expect(merged[index], closeTo(beforeMerge[index], 1e-4));
    }
  });
}
