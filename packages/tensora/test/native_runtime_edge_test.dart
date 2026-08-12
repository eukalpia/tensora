import 'package:tensora/src/native/native_runtime.dart';
import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  final runtime = NativeRuntime.instance;
  final invalidHandle = 0;
  final invalidHandleError = isA<NativeRuntimeException>();

  test('rank-zero native metadata and copy remain internally consistent', () {
    final handle = runtime.full(Shape(const []), 7.5);
    try {
      expect(runtime.shape(handle), Shape(const []));
      expect(runtime.dtype(handle), DType.float32);
      expect(runtime.device(handle), Device.cpu);
      expect(runtime.numel(handle), 1);
      expect(runtime.copyToHost(handle, 1), [7.5]);
    } finally {
      runtime.release(handle);
    }
  });

  test('zero tensor handles are rejected across the native wrapper surface', () {
    expect(
      () => runtime.toDevice(invalidHandle, Device.cpu),
      throwsA(invalidHandleError),
    );
    expect(
      () => runtime.reshape(invalidHandle, Shape([1])),
      throwsA(invalidHandleError),
    );
    expect(
      () => runtime.transpose2D(invalidHandle),
      throwsA(invalidHandleError),
    );
    expect(
      () => runtime.add(invalidHandle, invalidHandle),
      throwsA(invalidHandleError),
    );
    expect(
      () => runtime.multiply(invalidHandle, invalidHandle),
      throwsA(invalidHandleError),
    );
    expect(() => runtime.sum(invalidHandle), throwsA(invalidHandleError));
    expect(
      () => runtime.matmul(invalidHandle, invalidHandle),
      throwsA(invalidHandleError),
    );
    expect(() => runtime.shape(invalidHandle), throwsA(invalidHandleError));
    expect(() => runtime.dtype(invalidHandle), throwsA(invalidHandleError));
    expect(() => runtime.device(invalidHandle), throwsA(invalidHandleError));
    expect(() => runtime.numel(invalidHandle), throwsA(invalidHandleError));
    expect(
      () => runtime.copyToHost(invalidHandle, 0),
      throwsA(invalidHandleError),
    );
    expect(() => runtime.retain(invalidHandle), throwsA(invalidHandleError));
    expect(() => runtime.release(invalidHandle), throwsA(invalidHandleError));
  });

  test('runtime diagnostics remain callable after rejected handle operations', () {
    runtime.noop();
    expect(runtime.cudaDeviceCount(), greaterThanOrEqualTo(0));
    expect(runtime.liveTensorCount(), greaterThanOrEqualTo(0));
    expect(runtime.liveStorageBytes(), greaterThanOrEqualTo(0));
  });
}
