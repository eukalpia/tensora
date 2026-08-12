import 'package:tensora/src/native/native_training_runtime.dart';
import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void main() {
  final runtime = NativeTrainingRuntime.instance;
  final invalidHandleError = isA<NativeRuntimeException>();

  test('training runtime validates arguments before native calls', () {
    expect(() => runtime.manualSeed(-1), throwsArgumentError);
    expect(() => runtime.createLinear(0, 1, true), throwsArgumentError);
    expect(() => runtime.createLinear(1, 0, true), throwsArgumentError);
    expect(() => runtime.moduleParameterAt(0, -1), throwsArgumentError);
    expect(() => runtime.moduleBufferAt(0, -1), throwsArgumentError);
    expect(() => runtime.moduleSave(0, ''), throwsArgumentError);
    expect(() => runtime.moduleLoad(0, '  '), throwsArgumentError);
  });

  test('invalid module handles fail deterministically across operations', () {
    expect(
      () => runtime.moduleForward(0, 0),
      throwsA(invalidHandleError),
    );
    expect(
      () => runtime.moduleSetTraining(0, true),
      throwsA(invalidHandleError),
    );
    expect(
      () => runtime.moduleToDevice(0, Device.cpu),
      throwsA(invalidHandleError),
    );
    expect(
      () => runtime.moduleParameterCount(0),
      throwsA(invalidHandleError),
    );
    expect(
      () => runtime.moduleBufferCount(0),
      throwsA(invalidHandleError),
    );
    expect(
      () => runtime.moduleParameterAt(0, 0),
      throwsA(invalidHandleError),
    );
    expect(
      () => runtime.moduleBufferAt(0, 0),
      throwsA(invalidHandleError),
    );
    expect(() => runtime.moduleRelease(0), throwsA(invalidHandleError));
  });

  test('invalid optimizer handles fail without corrupting diagnostics', () {
    expect(
      () => runtime.createSgd(
        0,
        learningRate: 0.01,
        momentum: 0,
        weightDecay: 0,
      ),
      throwsA(invalidHandleError),
    );
    expect(
      () => runtime.createAdam(
        0,
        learningRate: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weightDecay: 0,
        decoupled: false,
      ),
      throwsA(invalidHandleError),
    );
    expect(() => runtime.optimizerZeroGrad(0), throwsA(invalidHandleError));
    expect(() => runtime.optimizerStep(0), throwsA(invalidHandleError));
    expect(() => runtime.optimizerRelease(0), throwsA(invalidHandleError));

    expect(runtime.trainingAvailable(), isTrue);
    expect(runtime.liveModuleCount(), greaterThanOrEqualTo(0));
    expect(runtime.liveOptimizerCount(), greaterThanOrEqualTo(0));
  });
}
