import 'package:tensora/src/native/native_runtime.dart';
import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  test(
    '2000 Dart FFI tensor lifecycles return native counters to baseline',
    () {
      final runtime = NativeRuntime.instance;
      final baselineTensors = runtime.liveTensorCount();
      final baselineStorage = runtime.liveStorageBytes();

      for (var cycle = 0; cycle < 2000; cycle++) {
        final a = Tensor.fromList([1, 2, 3, 4], shape: Shape([2, 2]));
        final b = Tensor.fromList([5, 6, 7, 8], shape: Shape([2, 2]));
        final result = a.matmul(b);
        final copied = result.to(Device.cpu);

        expect(copied.toList(), [19, 22, 43, 50]);

        copied.dispose();
        result.dispose();
        b.dispose();
        a.dispose();

        if ((cycle + 1) % 250 == 0) {
          expect(
            runtime.liveTensorCount(),
            baselineTensors,
            reason: 'tensor handle leak by cycle ${cycle + 1}',
          );
          expect(
            runtime.liveStorageBytes(),
            baselineStorage,
            reason: 'storage leak by cycle ${cycle + 1}',
          );
        }
      }

      expect(runtime.liveTensorCount(), baselineTensors);
      expect(runtime.liveStorageBytes(), baselineStorage);
    },
    tags: 'soak',
  );
}
