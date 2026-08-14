import 'package:tensora/src/native/finalizer_callbacks.dart';
import 'package:tensora/src/native/native_training_runtime.dart';
import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

import 'support/fault_runtime_control.dart';

void main() {
  final control = FaultRuntimeControl.fromEnvironment();
  control.setMode(FaultMode.normal);
  final runtime = NativeTrainingRuntime.instance;

  tearDown(() => control.setMode(FaultMode.normal));

  test('training status mapping preserves native diagnostics', () {
    final cases = <(int, Matcher)>[
      (FaultMode.outOfMemory, isA<OutOfMemoryException>()),
      (FaultMode.internalError, isA<NativeRuntimeException>()),
      (
        FaultMode.unknownStatus,
        isA<NativeRuntimeException>().having(
          (error) => error.message,
          'message',
          contains('Unknown native status 99'),
        ),
      ),
    ];

    for (final entry in cases) {
      control.setMode(entry.$1);
      expect(() => runtime.trainingAvailable(), throwsA(entry.$2));
    }

    control.setMode(FaultMode.nullDiagnostic);
    expect(
      () => runtime.trainingAvailable(),
      throwsA(
        isA<NativeRuntimeException>().having(
          (error) => error.message,
          'message',
          contains('without a diagnostic'),
        ),
      ),
    );
  });

  test('training finalizer callbacks delegate to native release', () {
    control.resetCounters();
    releaseModuleFromFinalizer(81);
    releaseOptimizerFromFinalizer(82);
    expect(control.releaseCount, 2);
  });

  test('successful training creation must return a non-zero handle', () {
    control.setMode(FaultMode.nullTrainingHandle);
    expect(
      () => runtime.createLinear(1, 1, true),
      throwsA(
        isA<NativeRuntimeException>()
            .having((error) => error.operation, 'operation', 'linear.create')
            .having(
              (error) => error.message,
              'message',
              contains('null handle'),
            ),
      ),
    );
  });

  test('runtime enumerates every visible accelerator deterministically', () {
    control.setMode(FaultMode.acceleratorCounts);
    expect(TensoraRuntime.availableDevices, <Device>[
      Device.cuda(0),
      Device.cuda(1),
      Device.mps,
      Device.xpu(0),
      Device.xpu(1),
      Device.hip(0),
      Device.hip(1),
      Device.cpu,
    ]);
    expect(TensoraRuntime.preferredDevice, Device.cuda(0));
  });

  test('module buffers are adopted as independently owned tensors', () {
    control.setMode(FaultMode.moduleBuffer);
    final module = Linear(1, 1);
    try {
      final buffers = module.buffers();
      expect(buffers, hasLength(1));
      expect(buffers.single.shape, Shape([1]));
      buffers.single.dispose();
    } finally {
      module.dispose();
    }
  });

  test('partial parameter adoption releases every acquired handle', () {
    control.resetCounters();
    control.setMode(FaultMode.moduleAdoptionFailure);
    final module = Linear(1, 1);
    try {
      expect(
        module.parameters,
        throwsA(
          isA<NativeRuntimeException>().having(
            (error) => error.operation,
            'operation',
            'tensor.dtype',
          ),
        ),
      );
      expect(control.releaseCount, 2);
    } finally {
      module.dispose();
    }
  });
}
