import 'dart:ffi';
import 'dart:io';

import 'package:tensora/tensora.dart' as core;
import 'package:tensora_nn/tensora_nn.dart';
import 'package:test/test.dart';

typedef _VoidNative = Void Function();
typedef _VoidDart = void Function();
typedef _SetIntNative = Void Function(Int32);
typedef _SetIntDart = void Function(int);

void main() {
  final fixturePath = Platform.environment['TENSORA_NN_CONTRACT_FIXTURE'];
  final fixtureSkip = fixturePath == null || fixturePath.isEmpty
      ? 'requires the adversarial NN V2 contract fixture'
      : false;

  test(
    'Linear rejects native parameter-count contract violations atomically',
    () {
      final library = DynamicLibrary.open(fixturePath!);
      final reset = library.lookupFunction<_VoidNative, _VoidDart>(
        'ts_test_reset',
      );
      final setTrainingMode = library.lookupFunction<_SetIntNative, _SetIntDart>(
        'ts_test_set_training_mode',
      );
      reset();
      setTrainingMode(0); // fixture exposes one parameter

      expect(
        () => Linear(inFeatures: 1, outFeatures: 1, bias: true),
        throwsA(
          isA<core.NativeRuntimeException>().having(
            (error) => error.operation,
            'operation',
            'nn.linear.create',
          ),
        ),
      );
      reset();
    },
    skip: fixtureSkip,
  );

  test(
    'Linear rejects parameter identity changes during move and reports rollback failure',
    () {
      final library = DynamicLibrary.open(fixturePath!);
      final reset = library.lookupFunction<_VoidNative, _VoidDart>(
        'ts_test_reset',
      );
      final setTrainingMode = library.lookupFunction<_SetIntNative, _SetIntDart>(
        'ts_test_set_training_mode',
      );
      reset();
      setTrainingMode(0);

      final layer = Linear(inFeatures: 1, outFeatures: 1, bias: false);
      addTearDown(layer.dispose);
      final identity = layer.parameters.single.identity;

      expect(
        () => layer.to(core.Device.cpu),
        throwsA(
          isA<core.NativeRuntimeException>().having(
            (error) => error.operation,
            'operation',
            'module.to',
          ),
        ),
      );
      expect(layer.parameters.single.identity, identity);
      reset();
    },
    skip: fixtureSkip,
  );
}
