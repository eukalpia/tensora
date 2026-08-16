import 'package:tensora/tensora.dart' show NativeParameterOptimizer, Parameter;

import 'parameter_group.dart';

/// Base class for parameter-oriented NN V2 optimizers.
abstract base class Optimizer {
  Optimizer(this.groups, List<NativeParameterOptimizer> nativeOptimizers)
    : _nativeOptimizers = nativeOptimizers;

  final List<ParameterGroup> groups;
  final List<NativeParameterOptimizer> _nativeOptimizers;
  bool _disposed = false;

  bool get isDisposed => _disposed;

  void zeroGrad() {
    _ensureLive('zeroGrad');
    for (final optimizer in _nativeOptimizers) {
      optimizer.zeroGrad();
    }
  }

  void step() {
    _ensureLive('step');
    for (final optimizer in _nativeOptimizers) {
      optimizer.step();
    }
  }

  void dispose() {
    if (_disposed) return;
    for (final optimizer in _nativeOptimizers.reversed) {
      optimizer.dispose();
    }
    _disposed = true;
  }

  void _ensureLive(String operation) {
    if (_disposed) {
      throw StateError('Optimizer has already been disposed: $operation');
    }
  }
}

final class SGD extends Optimizer {
  factory SGD({
    required Iterable<Parameter> parameters,
    double learningRate = 0.01,
    double momentum = 0,
    double weightDecay = 0,
  }) => SGD.groups(
    groups: <ParameterGroup>[
      ParameterGroup(
        parameters: parameters,
        learningRate: learningRate,
        momentum: momentum,
        weightDecay: weightDecay,
      ),
    ],
    learningRate: learningRate,
    momentum: momentum,
    weightDecay: weightDecay,
  );

  factory SGD.groups({
    required Iterable<ParameterGroup> groups,
    double learningRate = 0.01,
    double momentum = 0,
    double weightDecay = 0,
  }) {
    _positive(learningRate, 'learningRate');
    _nonNegative(momentum, 'momentum');
    _nonNegative(weightDecay, 'weightDecay');
    final validated = _validatedGroups(groups);
    final native = _createNativeGroups(
      validated,
      (group) => NativeParameterOptimizer.sgd(
        parameters: group.parameters,
        learningRate: group.learningRate ?? learningRate,
        momentum: group.momentum ?? momentum,
        weightDecay: group.weightDecay ?? weightDecay,
      ),
    );
    return SGD._(
      validated,
      native,
      learningRate: learningRate,
      momentum: momentum,
      weightDecay: weightDecay,
    );
  }

  SGD._(
    super.groups,
    super.nativeOptimizers, {
    required this.learningRate,
    required this.momentum,
    required this.weightDecay,
  });

  final double learningRate;
  final double momentum;
  final double weightDecay;
}

final class Adam extends Optimizer {
  factory Adam({
    required Iterable<Parameter> parameters,
    double learningRate = 0.001,
    double beta1 = 0.9,
    double beta2 = 0.999,
    double epsilon = 1e-8,
    double weightDecay = 0,
  }) => Adam.groups(
    groups: <ParameterGroup>[
      ParameterGroup(
        parameters: parameters,
        learningRate: learningRate,
        beta1: beta1,
        beta2: beta2,
        epsilon: epsilon,
        weightDecay: weightDecay,
      ),
    ],
    learningRate: learningRate,
    beta1: beta1,
    beta2: beta2,
    epsilon: epsilon,
    weightDecay: weightDecay,
  );

  factory Adam.groups({
    required Iterable<ParameterGroup> groups,
    double learningRate = 0.001,
    double beta1 = 0.9,
    double beta2 = 0.999,
    double epsilon = 1e-8,
    double weightDecay = 0,
  }) {
    _validateAdamDefaults(learningRate, beta1, beta2, epsilon, weightDecay);
    final validated = _validatedGroups(groups);
    final native = _createNativeGroups(
      validated,
      (group) => NativeParameterOptimizer.adam(
        parameters: group.parameters,
        learningRate: group.learningRate ?? learningRate,
        beta1: group.beta1 ?? beta1,
        beta2: group.beta2 ?? beta2,
        epsilon: group.epsilon ?? epsilon,
        weightDecay: group.weightDecay ?? weightDecay,
      ),
    );
    return Adam._(
      validated,
      native,
      learningRate: learningRate,
      beta1: beta1,
      beta2: beta2,
      epsilon: epsilon,
      weightDecay: weightDecay,
    );
  }

  Adam._(
    super.groups,
    super.nativeOptimizers, {
    required this.learningRate,
    required this.beta1,
    required this.beta2,
    required this.epsilon,
    required this.weightDecay,
  });

  final double learningRate;
  final double beta1;
  final double beta2;
  final double epsilon;
  final double weightDecay;
}

final class AdamW extends Optimizer {
  factory AdamW({
    required Iterable<Parameter> parameters,
    double learningRate = 0.001,
    double beta1 = 0.9,
    double beta2 = 0.999,
    double epsilon = 1e-8,
    double weightDecay = 0.01,
  }) => AdamW.groups(
    groups: <ParameterGroup>[
      ParameterGroup(
        parameters: parameters,
        learningRate: learningRate,
        beta1: beta1,
        beta2: beta2,
        epsilon: epsilon,
        weightDecay: weightDecay,
      ),
    ],
    learningRate: learningRate,
    beta1: beta1,
    beta2: beta2,
    epsilon: epsilon,
    weightDecay: weightDecay,
  );

  factory AdamW.groups({
    required Iterable<ParameterGroup> groups,
    double learningRate = 0.001,
    double beta1 = 0.9,
    double beta2 = 0.999,
    double epsilon = 1e-8,
    double weightDecay = 0.01,
  }) {
    _validateAdamDefaults(learningRate, beta1, beta2, epsilon, weightDecay);
    final validated = _validatedGroups(groups);
    final native = _createNativeGroups(
      validated,
      (group) => NativeParameterOptimizer.adam(
        parameters: group.parameters,
        learningRate: group.learningRate ?? learningRate,
        beta1: group.beta1 ?? beta1,
        beta2: group.beta2 ?? beta2,
        epsilon: group.epsilon ?? epsilon,
        weightDecay: group.weightDecay ?? weightDecay,
        decoupled: true,
      ),
    );
    return AdamW._(
      validated,
      native,
      learningRate: learningRate,
      beta1: beta1,
      beta2: beta2,
      epsilon: epsilon,
      weightDecay: weightDecay,
    );
  }

  AdamW._(
    super.groups,
    super.nativeOptimizers, {
    required this.learningRate,
    required this.beta1,
    required this.beta2,
    required this.epsilon,
    required this.weightDecay,
  });

  final double learningRate;
  final double beta1;
  final double beta2;
  final double epsilon;
  final double weightDecay;
}

List<ParameterGroup> _validatedGroups(Iterable<ParameterGroup> groups) {
  final copied = List<ParameterGroup>.unmodifiable(groups);
  if (copied.isEmpty) {
    throw ArgumentError.value(copied, 'groups', 'must not be empty');
  }
  final identities = <int>{};
  for (final group in copied) {
    for (final parameter in group.parameters) {
      if (!identities.add(parameter.identity)) {
        throw ArgumentError(
          'A parameter may belong to only one optimizer group.',
        );
      }
    }
  }
  return copied;
}

List<NativeParameterOptimizer> _createNativeGroups(
  List<ParameterGroup> groups,
  NativeParameterOptimizer Function(ParameterGroup group) create,
) {
  final created = <NativeParameterOptimizer>[];
  try {
    for (final group in groups) {
      created.add(create(group));
    }
    return List<NativeParameterOptimizer>.unmodifiable(created);
  } catch (_) {
    for (final optimizer in created.reversed) {
      optimizer.dispose();
    }
    rethrow;
  }
}

void _validateAdamDefaults(
  double learningRate,
  double beta1,
  double beta2,
  double epsilon,
  double weightDecay,
) {
  _positive(learningRate, 'learningRate');
  _beta(beta1, 'beta1');
  _beta(beta2, 'beta2');
  _positive(epsilon, 'epsilon');
  _nonNegative(weightDecay, 'weightDecay');
}

void _positive(double value, String name) {
  if (!value.isFinite || value <= 0) {
    throw ArgumentError.value(value, name, 'must be finite and positive');
  }
}

void _nonNegative(double value, String name) {
  if (!value.isFinite || value < 0) {
    throw ArgumentError.value(value, name, 'must be finite and non-negative');
  }
}

void _beta(double value, String name) {
  if (!value.isFinite || value < 0 || value >= 1) {
    throw ArgumentError.value(value, name, 'must be finite in [0, 1)');
  }
}
