import 'package:tensora/tensora.dart' as core;

import 'module.dart';

final class ReLU extends Module {
  ReLU();

  @override
  core.Tensor forward(core.Tensor input) => input.relu();

  @override
  String get internalDiagnosticLabel => 'ReLU()';
}

final class Sigmoid extends Module {
  Sigmoid();

  @override
  core.Tensor forward(core.Tensor input) => input.sigmoid();

  @override
  String get internalDiagnosticLabel => 'Sigmoid()';
}

final class Tanh extends Module {
  Tanh();

  @override
  core.Tensor forward(core.Tensor input) => input.tanh();

  @override
  String get internalDiagnosticLabel => 'Tanh()';
}

/// Exact Gaussian Error Linear Unit.
final class GELU extends Module {
  GELU();

  @override
  core.Tensor forward(core.Tensor input) => input.gelu();

  @override
  String get internalDiagnosticLabel => 'GELU()';
}

/// Sigmoid Linear Unit (`x * sigmoid(x)`).
final class SiLU extends Module {
  SiLU();

  @override
  core.Tensor forward(core.Tensor input) => input.silu();

  @override
  String get internalDiagnosticLabel => 'SiLU()';
}

/// SwiGLU over equal halves of the final input dimension.
final class SwiGLU extends Module {
  SwiGLU();

  @override
  core.Tensor forward(core.Tensor input) => input.swiglu();

  @override
  String get internalDiagnosticLabel => 'SwiGLU()';
}
