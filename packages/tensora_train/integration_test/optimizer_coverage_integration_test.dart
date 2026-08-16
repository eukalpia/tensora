import 'package:tensora/tensora.dart';
import 'package:tensora_nn/tensora_nn.dart' as nn;
import 'package:tensora_optim/tensora_optim.dart' as optim;
import 'package:test/test.dart';

void main() {
  test(
    'SGD, Adam, AdamW, groups, lifecycle, and rollback are deterministic',
    () {
      final layer = nn.Linear(inFeatures: 1, outFeatures: 1);
      addTearDown(layer.dispose);
      final parameters = layer.parameters;
      expect(parameters, hasLength(2));
      final baselineOptimizers = TensoraRuntime.liveOptimizerCount;

      final sgd = optim.SGD(
        parameters: parameters,
        learningRate: 0.02,
        momentum: 0.1,
        weightDecay: 0.01,
      );
      expect(sgd.isDisposed, isFalse);
      expect(sgd.learningRate, 0.02);
      expect(sgd.momentum, 0.1);
      expect(sgd.weightDecay, 0.01);
      expect(sgd.groups, hasLength(1));
      sgd.zeroGrad();
      sgd.step();
      sgd.dispose();
      sgd.dispose();
      expect(sgd.isDisposed, isTrue);
      expect(sgd.zeroGrad, throwsStateError);
      expect(sgd.step, throwsStateError);

      final sgdGroup = optim.ParameterGroup(
        parameters: <Parameter>[parameters.first],
        learningRate: 0.03,
        momentum: 0.2,
        weightDecay: 0.04,
      );
      final groupedSgd = optim.SGD.groups(
        groups: <optim.ParameterGroup>[sgdGroup],
        learningRate: 0.01,
        momentum: 0,
        weightDecay: 0,
      );
      expect(groupedSgd.groups.single, same(sgdGroup));
      groupedSgd.dispose();

      final adamGroup = optim.ParameterGroup(
        parameters: parameters,
        learningRate: 0.004,
        beta1: 0.8,
        beta2: 0.95,
        epsilon: 1e-7,
        weightDecay: 0.02,
      );
      final groupedAdam = optim.Adam.groups(
        groups: <optim.ParameterGroup>[adamGroup],
        learningRate: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weightDecay: 0,
      );
      expect(groupedAdam.learningRate, 0.001);
      expect(groupedAdam.beta1, 0.9);
      expect(groupedAdam.beta2, 0.999);
      expect(groupedAdam.epsilon, 1e-8);
      expect(groupedAdam.weightDecay, 0);
      groupedAdam.zeroGrad();
      groupedAdam.step();
      groupedAdam.dispose();

      final adamW = optim.AdamW(
        parameters: parameters,
        learningRate: 0.002,
        beta1: 0.85,
        beta2: 0.97,
        epsilon: 1e-6,
        weightDecay: 0.03,
      );
      expect(adamW.learningRate, 0.002);
      expect(adamW.beta1, 0.85);
      expect(adamW.beta2, 0.97);
      expect(adamW.epsilon, 1e-6);
      expect(adamW.weightDecay, 0.03);
      adamW.dispose();

      final adamWGroup = optim.ParameterGroup(
        parameters: <Parameter>[parameters.last],
        learningRate: 0.006,
        beta1: 0.7,
        beta2: 0.9,
        epsilon: 1e-5,
        weightDecay: 0.05,
      );
      final groupedAdamW = optim.AdamW.groups(
        groups: <optim.ParameterGroup>[adamWGroup],
        learningRate: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weightDecay: 0.01,
      );
      expect(groupedAdamW.groups.single, same(adamWGroup));
      groupedAdamW.zeroGrad();
      groupedAdamW.step();
      groupedAdamW.dispose();

      final duplicateA = optim.ParameterGroup(
        parameters: <Parameter>[parameters.first],
      );
      final duplicateB = optim.ParameterGroup(
        parameters: <Parameter>[parameters.first],
      );
      expect(
        () => optim.SGD.groups(
          groups: <optim.ParameterGroup>[duplicateA, duplicateB],
        ),
        throwsArgumentError,
      );

      final rollbackBaseline = TensoraRuntime.liveOptimizerCount;
      parameters.last.freeze();
      final trainableGroup = optim.ParameterGroup(
        parameters: <Parameter>[parameters.first],
      );
      final frozenGroup = optim.ParameterGroup(
        parameters: <Parameter>[parameters.last],
      );
      expect(
        () => optim.SGD.groups(
          groups: <optim.ParameterGroup>[trainableGroup, frozenGroup],
        ),
        throwsA(isA<InvalidArgumentException>()),
      );
      expect(TensoraRuntime.liveOptimizerCount, rollbackBaseline);
      parameters.last.unfreeze();

      expect(TensoraRuntime.liveOptimizerCount, baselineOptimizers);
    },
  );

  test(
    'optimizer and ParameterGroup validators reject every invalid hyperparameter',
    () {
      expect(
        () => optim.SGD.groups(
          groups: const <optim.ParameterGroup>[],
          learningRate: 0,
        ),
        throwsArgumentError,
      );
      expect(
        () => optim.SGD.groups(
          groups: const <optim.ParameterGroup>[],
          momentum: -1,
        ),
        throwsArgumentError,
      );
      expect(
        () => optim.SGD.groups(
          groups: const <optim.ParameterGroup>[],
          weightDecay: double.nan,
        ),
        throwsArgumentError,
      );
      expect(
        () => optim.Adam.groups(
          groups: const <optim.ParameterGroup>[],
          learningRate: double.infinity,
        ),
        throwsArgumentError,
      );
      expect(
        () => optim.Adam.groups(
          groups: const <optim.ParameterGroup>[],
          beta1: 1,
        ),
        throwsArgumentError,
      );
      expect(
        () => optim.Adam.groups(
          groups: const <optim.ParameterGroup>[],
          beta2: -0.1,
        ),
        throwsArgumentError,
      );
      expect(
        () => optim.Adam.groups(
          groups: const <optim.ParameterGroup>[],
          epsilon: 0,
        ),
        throwsArgumentError,
      );
      expect(
        () => optim.AdamW.groups(
          groups: const <optim.ParameterGroup>[],
          weightDecay: -0.1,
        ),
        throwsArgumentError,
      );

      final layer = nn.Linear(inFeatures: 1, outFeatures: 1, bias: false);
      addTearDown(layer.dispose);
      final parameter = layer.parameters.single;

      expect(
        () => optim.ParameterGroup(
          parameters: <Parameter>[parameter, parameter],
        ),
        throwsArgumentError,
      );
      expect(
        () => optim.ParameterGroup(
          parameters: <Parameter>[parameter],
          learningRate: 0,
        ),
        throwsArgumentError,
      );
      expect(
        () => optim.ParameterGroup(
          parameters: <Parameter>[parameter],
          momentum: -1,
        ),
        throwsArgumentError,
      );
      expect(
        () => optim.ParameterGroup(
          parameters: <Parameter>[parameter],
          beta1: 1,
        ),
        throwsArgumentError,
      );
      expect(
        () => optim.ParameterGroup(
          parameters: <Parameter>[parameter],
          beta2: double.nan,
        ),
        throwsArgumentError,
      );
      expect(
        () => optim.ParameterGroup(
          parameters: <Parameter>[parameter],
          epsilon: 0,
        ),
        throwsArgumentError,
      );
      expect(
        () => optim.ParameterGroup(
          parameters: <Parameter>[parameter],
          weightDecay: -1,
        ),
        throwsArgumentError,
      );

      parameter.dispose();
      expect(
        () => optim.ParameterGroup(parameters: <Parameter>[parameter]),
        throwsArgumentError,
      );
    },
  );
}
