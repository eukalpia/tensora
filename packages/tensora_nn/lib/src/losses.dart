import 'package:tensora/tensora.dart' as core;

/// Object-style mean-squared-error loss.
final class MSELoss {
  const MSELoss();

  core.Tensor call(core.Tensor prediction, core.Tensor target) =>
      core.Losses.mse(prediction, target);

  @override
  String toString() => 'MSELoss()';
}

/// Object-style cross-entropy loss using the current one-hot float32 contract.
final class CrossEntropyLoss {
  const CrossEntropyLoss();

  core.Tensor call(core.Tensor logits, core.Tensor oneHotTarget) =>
      core.Losses.crossEntropy(logits, oneHotTarget);

  @override
  String toString() => 'CrossEntropyLoss()';
}
