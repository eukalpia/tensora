import 'package:tensora/src/native/finalizer_callbacks.dart';
import 'package:tensora/src/native/native_runtime.dart';
import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

import 'support/fault_runtime_control.dart';

void main() {
  final control = FaultRuntimeControl.fromEnvironment();
  control.setMode(FaultMode.normal);
  final runtime = NativeRuntime.instance;

  tearDown(() => control.setMode(FaultMode.normal));

  test('core status mapping preserves typed errors and diagnostics', () {
    final cases = <(int, Matcher)>[
      (FaultMode.invalidArgument, isA<InvalidArgumentException>()),
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
      expect(() => runtime.noop(), throwsA(entry.$2));
    }

    control.setMode(FaultMode.nullDiagnostic);
    expect(
      () => runtime.noop(),
      throwsA(
        isA<NativeRuntimeException>().having(
          (error) => error.message,
          'message',
          contains('without a diagnostic'),
        ),
      ),
    );
  });

  test('tensor finalizer callback delegates to native release', () {
    control.resetCounters();
    releaseTensorFromFinalizer(77);
    expect(control.releaseCount, 1);
  });

  test('successful tensor creation must return a non-zero handle', () {
    control.setMode(FaultMode.nullTensorHandle);
    expect(
      () => runtime.full(Shape([1]), 0),
      throwsA(
        isA<NativeRuntimeException>()
            .having((error) => error.operation, 'operation', 'tensor.full')
            .having(
              (error) => error.message,
              'message',
              contains('null tensor handle'),
            ),
      ),
    );
  });

  test('shape metadata violations are rejected explicitly', () {
    final cases = <(int, String)>[
      (FaultMode.rankAboveLimit, 'rank 65 above'),
      (FaultMode.scalarRankChanged, 'rank changed'),
      (FaultMode.rankChanged, 'rank changed'),
      (FaultMode.invalidShapeMetadata, 'invalid shape metadata'),
    ];

    for (final entry in cases) {
      control.setMode(entry.$1);
      expect(
        () => runtime.shape(1),
        throwsA(
          isA<NativeRuntimeException>()
              .having((error) => error.operation, 'operation', 'tensor.shape')
              .having((error) => error.message, 'message', contains(entry.$2)),
        ),
      );
    }
  });

  test('every stable native dtype code decodes at the ABI boundary', () {
    final cases = <(int, DType)>[
      (FaultMode.dtypeFloat16, DType.float16),
      (FaultMode.dtypeBFloat16, DType.bfloat16),
      (FaultMode.dtypeFloat64, DType.float64),
      (FaultMode.dtypeInt8, DType.int8),
      (FaultMode.dtypeUint8, DType.uint8),
      (FaultMode.dtypeInt16, DType.int16),
      (FaultMode.dtypeInt32, DType.int32),
      (FaultMode.dtypeInt64, DType.int64),
      (FaultMode.dtypeBool, DType.boolean),
    ];

    for (final entry in cases) {
      control.setMode(entry.$1);
      expect(runtime.dtype(1), entry.$2);
    }
  });

  test('unknown dtype metadata is rejected explicitly', () {
    control.setMode(FaultMode.unknownDType);
    expect(
      () => runtime.dtype(1),
      throwsA(
        isA<NativeRuntimeException>()
            .having((error) => error.operation, 'operation', 'tensor.dtype')
            .having(
              (error) => error.message,
              'message',
              contains('unknown dtype code 99'),
            ),
      ),
    );
  });

  test('invalid device metadata is rejected for every backend kind', () {
    final cases = <(int, String)>[
      (FaultMode.invalidCpuIndex, 'CPU tensor'),
      (FaultMode.invalidCudaIndex, 'CUDA tensor'),
      (FaultMode.invalidMpsIndex, 'MPS tensor'),
      (FaultMode.invalidXpuIndex, 'XPU tensor'),
      (FaultMode.invalidHipIndex, 'HIP tensor'),
      (FaultMode.unknownDevice, 'unknown device code 99'),
    ];

    for (final entry in cases) {
      control.setMode(entry.$1);
      expect(
        () => runtime.device(1),
        throwsA(
          isA<NativeRuntimeException>()
              .having((error) => error.operation, 'operation', 'tensor.device')
              .having((error) => error.message, 'message', contains(entry.$2)),
        ),
      );
    }
  });

  test('accelerator creation adopts the transferred tensor', () {
    control.setMode(FaultMode.validCudaTensor);
    final tensor = Tensor.ones(Shape([1]), device: Device.cuda(1));
    try {
      expect(tensor.device, Device.cuda(1));
      expect(tensor.shape, Shape([1]));
    } finally {
      tensor.dispose();
    }
  });

  test('inconsistent tensor numel rolls back the created handle', () {
    control.resetCounters();
    control.setMode(FaultMode.inconsistentNumel);

    expect(
      () => Tensor.ones(Shape([1])),
      throwsA(
        isA<NativeRuntimeException>()
            .having((error) => error.operation, 'operation', 'tensor.adopt')
            .having(
              (error) => error.message,
              'message',
              contains('metadata is inconsistent'),
            ),
      ),
    );
    expect(control.releaseCount, 1);
  });
}
