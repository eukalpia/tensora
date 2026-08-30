import 'dart:ffi';
import 'dart:io';

import 'package:tensora/src/native/native_inference_runtime.dart';
import 'package:tensora/src/native/native_runtime.dart';
import 'package:tensora/src/native/native_training_runtime.dart';
import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

typedef _VoidNative = Void Function();
typedef _VoidDart = void Function();
typedef _SetUint8Native = Void Function(Uint8);
typedef _SetUint8Dart = void Function(int);
typedef _SetUint32Native = Void Function(Uint32);
typedef _SetUint32Dart = void Function(int);
typedef _SetInt32Native = Void Function(Int32);
typedef _SetInt32Dart = void Function(int);
typedef _SetInt64Native = Void Function(Int64);
typedef _SetInt64Dart = void Function(int);
typedef _SetUint64Native = Void Function(Uint64);
typedef _SetUint64Dart = void Function(int);
typedef _SetSizeNative = Void Function(Size);
typedef _SetSizeDart = void Function(int);
typedef _SetDeviceNative = Void Function(Uint32, Int32);
typedef _SetDeviceDart = void Function(int, int);
typedef _CounterNative = Uint64 Function();
typedef _CounterDart = int Function();

void main() {
  final fixturePath = Platform.environment['TENSORA_NATIVE_LIBRARY'];
  if (fixturePath == null || fixturePath.isEmpty) {
    fail('TENSORA_NATIVE_LIBRARY must point at the contract fixture.');
  }

  final fixture = DynamicLibrary.open(fixturePath);
  final reset = fixture.lookupFunction<_VoidNative, _VoidDart>('ts_test_reset');
  final setAbiVersion = fixture
      .lookupFunction<_SetUint32Native, _SetUint32Dart>(
        'ts_test_set_abi_version',
      );
  final setForcedStatus = fixture
      .lookupFunction<_SetInt32Native, _SetInt32Dart>(
        'ts_test_set_forced_status',
      );
  final setNullDiagnostic = fixture
      .lookupFunction<_SetUint8Native, _SetUint8Dart>(
        'ts_test_set_null_diagnostic',
      );
  final setNullHandle = fixture.lookupFunction<_SetUint8Native, _SetUint8Dart>(
    'ts_test_set_null_handle',
  );
  final setRank = fixture.lookupFunction<_SetSizeNative, _SetSizeDart>(
    'ts_test_set_rank',
  );
  final setShapeReturnedRank = fixture
      .lookupFunction<_SetSizeNative, _SetSizeDart>(
        'ts_test_set_shape_returned_rank',
      );
  final setShapeDimension = fixture
      .lookupFunction<_SetInt64Native, _SetInt64Dart>(
        'ts_test_set_shape_dimension',
      );
  final setDType = fixture.lookupFunction<_SetUint32Native, _SetUint32Dart>(
    'ts_test_set_dtype',
  );
  final setDevice = fixture.lookupFunction<_SetDeviceNative, _SetDeviceDart>(
    'ts_test_set_device',
  );
  final setNumel = fixture.lookupFunction<_SetUint64Native, _SetUint64Dart>(
    'ts_test_set_numel',
  );
  final setCopyWritten = fixture.lookupFunction<_SetSizeNative, _SetSizeDart>(
    'ts_test_set_copy_written',
  );
  final setTrainingMode = fixture
      .lookupFunction<_SetInt32Native, _SetInt32Dart>(
        'ts_test_set_training_mode',
      );
  final setInferenceMode = fixture
      .lookupFunction<_SetInt32Native, _SetInt32Dart>(
        'ts_test_set_inference_mode',
      );
  final tensorReleaseCount = fixture
      .lookupFunction<_CounterNative, _CounterDart>(
        'ts_test_tensor_release_count',
      );
  final sessionReleaseCount = fixture
      .lookupFunction<_CounterNative, _CounterDart>(
        'ts_test_session_release_count',
      );

  late NativeRuntime runtime;
  late NativeTrainingRuntime training;
  late NativeInferenceRuntime inference;

  setUpAll(() {
    reset();
    setAbiVersion(NativeRuntime.expectedAbiVersion + 1);
    expect(
      () => NativeRuntime.instance,
      throwsA(
        isA<NativeRuntimeException>()
            .having((error) => error.operation, 'operation', 'runtime.load')
            .having(
              (error) => error.message,
              'message',
              contains('Incompatible Tensora native ABI'),
            ),
      ),
    );

    setAbiVersion(NativeRuntime.expectedAbiVersion);
    runtime = NativeRuntime.instance;
    training = NativeTrainingRuntime.instance;
    inference = NativeInferenceRuntime.instance;
  });

  setUp(reset);

  test('core status mapping preserves every structured failure class', () {
    final cases = <(int, Matcher)>[
      (1, isA<InvalidArgumentException>()),
      (2, isA<InvalidShapeException>()),
      (3, isA<OutOfMemoryException>()),
      (4, isA<UnsupportedOperationException>()),
      (5, isA<NativeRuntimeException>()),
      (6, isA<NativeRuntimeException>()),
      (99, isA<NativeRuntimeException>()),
    ];

    for (final (status, matcher) in cases) {
      setForcedStatus(status);
      expect(() => runtime.noop(), throwsA(matcher), reason: 'status $status');
    }
  });

  test(
    'core status mapping synthesizes a diagnostic when native returns null',
    () {
      setNullDiagnostic(1);
      setForcedStatus(99);
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
    },
  );

  test('successful native creation may never return a null tensor handle', () {
    setNullHandle(1);
    expect(
      () => runtime.full(Shape([1]), 1),
      throwsA(
        isA<NativeRuntimeException>().having(
          (error) => error.message,
          'message',
          contains('null tensor handle'),
        ),
      ),
    );
  });

  test('shape metadata rejects rank overflow and rank instability', () {
    final handle = runtime.full(Shape([1]), 1);
    try {
      setRank(Shape.maxRank + 1);
      expect(
        () => runtime.shape(handle),
        throwsA(isA<NativeRuntimeException>()),
      );

      reset();
      setRank(0);
      setShapeReturnedRank(1);
      expect(
        () => runtime.shape(handle),
        throwsA(isA<NativeRuntimeException>()),
      );

      reset();
      setRank(1);
      setShapeReturnedRank(2);
      expect(
        () => runtime.shape(handle),
        throwsA(isA<NativeRuntimeException>()),
      );
    } finally {
      reset();
      runtime.release(handle);
    }
  });

  test('shape and dtype metadata reject values outside the Dart contract', () {
    final handle = runtime.full(Shape([1]), 1);
    try {
      setShapeDimension(0);
      expect(
        () => runtime.shape(handle),
        throwsA(isA<NativeRuntimeException>()),
      );

      reset();
      setDType(999);
      expect(
        () => runtime.dtype(handle),
        throwsA(isA<NativeRuntimeException>()),
      );
    } finally {
      reset();
      runtime.release(handle);
    }
  });

  test('device metadata validates every device kind and invalid index', () {
    final handle = runtime.full(Shape([1]), 1);
    try {
      final valid = <(int, int, Device)>[
        (1, 0, Device.cpu),
        (2, 1, Device.cuda(1)),
        (3, 0, Device.mps),
        (4, 2, Device.xpu(2)),
        (5, 3, Device.hip(3)),
      ];
      for (final (kind, index, expected) in valid) {
        setDevice(kind, index);
        expect(runtime.device(handle), expected);
      }

      for (final (kind, index) in <(int, int)>[
        (1, 1),
        (2, -1),
        (3, 1),
        (4, -1),
        (5, -1),
        (999, 0),
      ]) {
        setDevice(kind, index);
        expect(
          () => runtime.device(handle),
          throwsA(isA<NativeRuntimeException>()),
          reason: 'device=$kind index=$index',
        );
      }
    } finally {
      reset();
      runtime.release(handle);
    }
  });

  test('host copy rejects a native written-count contract violation', () {
    final handle = runtime.full(Shape([1]), 1);
    try {
      setCopyWritten(0);
      expect(
        () => runtime.copyToHost(handle, 1),
        throwsA(isA<NativeRuntimeException>()),
      );
    } finally {
      reset();
      runtime.release(handle);
    }
  });

  test(
    'Tensor adoption rolls back a native handle on inconsistent metadata',
    () {
      setNumel(2);
      expect(
        () => Tensor.full(Shape([1]), 1),
        throwsA(isA<NativeRuntimeException>()),
      );
      expect(tensorReleaseCount(), 1);
    },
  );

  test('runtime-only Tensor capabilities reject ordinary objects', () {
    expect(
      () => Tensor.adoptNativeHandleForRuntime(123, Object()),
      throwsStateError,
    );

    final tensor = Tensor.full(Shape([1]), 1);
    try {
      expect(() => tensor.nativeHandleForRuntime(Object()), throwsStateError);
    } finally {
      tensor.dispose();
    }
  });

  test(
    'runtime device enumeration covers all visible accelerator families',
    () {
      expect(TensoraRuntime.availableDevices, <Device>[
        Device.cuda(0),
        Device.cuda(1),
        Device.mps,
        Device.xpu(0),
        Device.hip(0),
        Device.cpu,
      ]);
      expect(TensoraRuntime.preferredDevice, Device.cuda(0));
    },
  );

  test('training status mapping preserves every structured failure class', () {
    final cases = <(int, Matcher)>[
      (1, isA<InvalidArgumentException>()),
      (2, isA<InvalidShapeException>()),
      (3, isA<OutOfMemoryException>()),
      (4, isA<UnsupportedOperationException>()),
      (5, isA<NativeRuntimeException>()),
      (6, isA<NativeRuntimeException>()),
      (99, isA<NativeRuntimeException>()),
    ];
    for (final (status, matcher) in cases) {
      setForcedStatus(status);
      expect(
        () => training.manualSeed(1),
        throwsA(matcher),
        reason: 'status $status',
      );
    }
  });

  test(
    'training object creation rejects a null handle returned on success',
    () {
      setNullHandle(1);
      expect(
        () => training.createLinear(1, 1, true),
        throwsA(isA<NativeRuntimeException>()),
      );
    },
  );

  test('training device encoding covers CPU and every accelerator family', () {
    final module = training.createLinear(1, 1, true);
    try {
      for (final device in <Device>[
        Device.cpu,
        Device.cuda(0),
        Device.mps,
        Device.xpu(0),
        Device.hip(0),
      ]) {
        training.moduleToDevice(module, device);
      }
    } finally {
      training.moduleRelease(module);
    }
  });

  test(
    'Module parameter collection releases all partially adopted tensors',
    () {
      final module = NativeLinear(1, 1);
      try {
        setTrainingMode(1);
        expect(
          () => module.parameters(),
          throwsA(isA<NativeRuntimeException>()),
        );
        expect(tensorReleaseCount(), 2);
      } finally {
        module.dispose();
      }
    },
  );

  test('inference status mapping includes model and unknown failures', () {
    final cases = <(int, Matcher)>[
      (1, isA<InvalidArgumentException>()),
      (2, isA<InvalidShapeException>()),
      (3, isA<OutOfMemoryException>()),
      (4, isA<UnsupportedOperationException>()),
      (5, isA<NativeRuntimeException>()),
      (6, isA<NativeRuntimeException>()),
      (7, isA<ModelRuntimeException>()),
      (99, isA<NativeRuntimeException>()),
    ];
    for (final (status, matcher) in cases) {
      setForcedStatus(status);
      expect(
        () => inference.available(),
        throwsA(matcher),
        reason: 'status $status',
      );
    }
  });

  test('inference status mapping handles a missing native diagnostic', () {
    setNullDiagnostic(1);
    setForcedStatus(99);
    expect(
      () => inference.available(),
      throwsA(
        isA<NativeRuntimeException>().having(
          (error) => error.message,
          'message',
          contains('without a diagnostic'),
        ),
      ),
    );
  });

  test('ONNX session creation rejects a null handle returned on success', () {
    setNullHandle(1);
    expect(
      () => inference.createSession(
        'fixture.onnx',
        enableProfiling: false,
        profilingPrefix: null,
      ),
      throwsA(isA<NativeRuntimeException>()),
    );
  });

  test('two-pass UTF-8 metadata rejects zero and unstable required sizes', () {
    setInferenceMode(2);
    expect(() => inference.providers(), throwsA(isA<NativeRuntimeException>()));

    reset();
    setInferenceMode(3);
    expect(() => inference.providers(), throwsA(isA<NativeRuntimeException>()));
  });

  test('provider decoding rejects an unknown native provider', () {
    expect(
      () => OnnxExecutionProvider.fromRuntimeName('UnknownExecutionProvider'),
      throwsA(isA<NativeRuntimeException>()),
    );

    setInferenceMode(4);
    expect(
      () => OnnxSession('fixture.onnx'),
      throwsA(isA<NativeRuntimeException>()),
    );
    expect(sessionReleaseCount(), 1);
  });

  test('explicit ONNX provider cannot silently fall back to CPU', () {
    setInferenceMode(5);
    expect(
      () => OnnxSession('fixture.onnx', provider: OnnxExecutionProvider.cuda),
      throwsA(isA<NativeRuntimeException>()),
    );
    expect(sessionReleaseCount(), 1);
  });

  test('ONNX session creation rejects models without inputs or outputs', () {
    setInferenceMode(6);
    expect(
      () => OnnxSession('fixture.onnx'),
      throwsA(isA<ModelRuntimeException>()),
    );
    expect(sessionReleaseCount(), 1);

    reset();
    setInferenceMode(7);
    expect(
      () => OnnxSession('fixture.onnx'),
      throwsA(isA<ModelRuntimeException>()),
    );
    expect(sessionReleaseCount(), 1);
  });

  test('native inference run rolls back partially returned output handles', () {
    final session = inference.createSession(
      'fixture.onnx',
      enableProfiling: false,
      profilingPrefix: null,
    );
    try {
      setInferenceMode(8);
      expect(
        () => inference.run(
          session,
          inputNames: const [],
          inputHandles: const [],
          outputNames: const ['Y0', 'Y1'],
        ),
        throwsA(isA<NativeRuntimeException>()),
      );
      expect(tensorReleaseCount(), 1);
    } finally {
      inference.release(session);
    }
  });

  test('ONNX Tensor adoption releases each failed output exactly once', () {
    final input = Tensor.full(Shape([1]), 1);
    setInferenceMode(9);
    final session = OnnxSession('fixture.onnx');
    try {
      expect(
        () => session.run({'X': input}),
        throwsA(isA<NativeRuntimeException>()),
      );
      expect(
        tensorReleaseCount(),
        2,
        reason:
            'one adopted output and one failed output must each be released once',
      );
    } finally {
      session.dispose();
      input.dispose();
    }
  });
}
