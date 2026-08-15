import 'package:tensora_nn/tensora_nn.dart';

/// One immutable optimizer parameter group with optional hyperparameter overrides.
final class ParameterGroup {
  factory ParameterGroup({
    required Iterable<Parameter> parameters,
    double? learningRate,
    double? momentum,
    double? beta1,
    double? beta2,
    double? epsilon,
    double? weightDecay,
  }) {
    final copied = List<Parameter>.unmodifiable(parameters);
    if (copied.isEmpty) {
      throw ArgumentError.value(copied, 'parameters', 'must not be empty');
    }
    final seen = <int>{};
    for (final parameter in copied) {
      if (parameter.isDisposed) {
        throw ArgumentError('Optimizer parameters must not be disposed.');
      }
      if (!seen.add(parameter.identity)) {
        throw ArgumentError('ParameterGroup contains a duplicate parameter.');
      }
    }
    _validatePositive(learningRate, 'learningRate');
    _validateNonNegative(momentum, 'momentum');
    _validateBeta(beta1, 'beta1');
    _validateBeta(beta2, 'beta2');
    _validatePositive(epsilon, 'epsilon');
    _validateNonNegative(weightDecay, 'weightDecay');
    return ParameterGroup._(
      copied,
      learningRate: learningRate,
      momentum: momentum,
      beta1: beta1,
      beta2: beta2,
      epsilon: epsilon,
      weightDecay: weightDecay,
    );
  }

  ParameterGroup._(
    this.parameters, {
    this.learningRate,
    this.momentum,
    this.beta1,
    this.beta2,
    this.epsilon,
    this.weightDecay,
  });

  final List<Parameter> parameters;
  final double? learningRate;
  final double? momentum;
  final double? beta1;
  final double? beta2;
  final double? epsilon;
  final double? weightDecay;
}

void _validatePositive(double? value, String name) {
  if (value != null && (!value.isFinite || value <= 0)) {
    throw ArgumentError.value(value, name, 'must be finite and positive');
  }
}

void _validateNonNegative(double? value, String name) {
  if (value != null && (!value.isFinite || value < 0)) {
    throw ArgumentError.value(value, name, 'must be finite and non-negative');
  }
}

void _validateBeta(double? value, String name) {
  if (value != null && (!value.isFinite || value < 0 || value >= 1)) {
    throw ArgumentError.value(value, name, 'must be finite in [0, 1)');
  }
}
