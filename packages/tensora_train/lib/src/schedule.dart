import 'dart:math' as math;

/// A learning-rate schedule: a pure function from a step counter to a rate.
///
/// A schedule is arithmetic over step counts. It never touches a Tensor, a
/// native handle, or an optimizer, so it stays exhaustively testable without a
/// native runtime and a training curve can be inspected, plotted, or diffed
/// before a single step runs.
///
/// **Nothing in Tensora installs a schedule into an optimizer yet.** Every
/// `tensora_optim` optimizer fixes `learningRate` at construction and hands it
/// to the native runtime there; ABI v5 exposes no way to change the rate of a
/// live optimizer. This type therefore has no `attach`, `apply`, or `step`
/// method taking an optimizer: an API that silently failed to change the
/// rate would be worse than no API. See the package README of `tensora_optim`
/// and `TS_ABI_VERSION` in `native/include/tensora.h` for the missing
/// primitive.
///
/// Rebuilding an optimizer per step to fake a rate change is not a workaround —
/// it discards momentum and Adam moment estimates, which corrupts training
/// silently.
abstract base class LearningRateSchedule {
  const LearningRateSchedule();

  /// Learning rate for the zero-based [step].
  ///
  /// The step axis is whatever the caller counts with it. Schedules do not
  /// distinguish optimizer steps from epochs; pass one or the other
  /// consistently.
  double call(int step) {
    if (step < 0) {
      throw ArgumentError.value(step, 'step', 'must be non-negative');
    }
    return rateAt(step);
  }

  /// Subclasses implement the schedule arithmetic for an already validated
  /// non-negative [step].
  double rateAt(int step);

  /// Rates for steps `0` through `count - 1`.
  ///
  /// Materializing the curve up front is how a caller checks a schedule against
  /// intent without running training.
  List<double> sample(int count) {
    if (count < 0) {
      throw ArgumentError.value(count, 'count', 'must be non-negative');
    }
    return List<double>.unmodifiable(
      List<double>.generate(count, (step) => call(step)),
    );
  }
}

/// A rate that never changes.
final class ConstantSchedule extends LearningRateSchedule {
  ConstantSchedule(this.learningRate) {
    _positive(learningRate, 'learningRate');
  }

  final double learningRate;

  @override
  double rateAt(int step) => learningRate;

  @override
  String toString() => 'ConstantSchedule($learningRate)';
}

/// Multiplies the rate by [gamma] once every [stepSize] steps.
///
/// The decay is a floor division of the step counter, so the rate is a
/// staircase: it is constant inside a period and drops on the boundary.
final class StepDecaySchedule extends LearningRateSchedule {
  StepDecaySchedule({
    required this.baseLearningRate,
    required this.stepSize,
    required this.gamma,
  }) {
    _positive(baseLearningRate, 'baseLearningRate');
    if (stepSize <= 0) {
      throw ArgumentError.value(stepSize, 'stepSize', 'must be positive');
    }
    _positive(gamma, 'gamma');
  }

  final double baseLearningRate;
  final int stepSize;

  /// Per-period multiplier. Values below one decay; one holds the base rate.
  final double gamma;

  @override
  double rateAt(int step) =>
      baseLearningRate * math.pow(gamma, step ~/ stepSize).toDouble();

  @override
  String toString() =>
      'StepDecaySchedule(base: $baseLearningRate, stepSize: $stepSize, '
      'gamma: $gamma)';
}

/// Multiplies the rate by [gamma] on every step.
final class ExponentialDecaySchedule extends LearningRateSchedule {
  ExponentialDecaySchedule({
    required this.baseLearningRate,
    required this.gamma,
  }) {
    _positive(baseLearningRate, 'baseLearningRate');
    _positive(gamma, 'gamma');
  }

  final double baseLearningRate;

  /// Per-step multiplier. Values below one decay; one holds the base rate.
  final double gamma;

  @override
  double rateAt(int step) =>
      baseLearningRate * math.pow(gamma, step).toDouble();

  @override
  String toString() =>
      'ExponentialDecaySchedule(base: $baseLearningRate, gamma: $gamma)';
}

/// Anneals from [baseLearningRate] to [minLearningRate] along a half cosine.
///
/// The rate reaches [minLearningRate] exactly at [totalSteps] and is clamped
/// there afterwards, so overrunning the planned budget cannot push the rate
/// back up or below the floor.
final class CosineAnnealingSchedule extends LearningRateSchedule {
  CosineAnnealingSchedule({
    required this.baseLearningRate,
    required this.totalSteps,
    this.minLearningRate = 0,
  }) {
    _positive(baseLearningRate, 'baseLearningRate');
    if (totalSteps <= 0) {
      throw ArgumentError.value(totalSteps, 'totalSteps', 'must be positive');
    }
    if (!minLearningRate.isFinite || minLearningRate < 0) {
      throw ArgumentError.value(
        minLearningRate,
        'minLearningRate',
        'must be finite and non-negative',
      );
    }
    if (minLearningRate > baseLearningRate) {
      throw ArgumentError.value(
        minLearningRate,
        'minLearningRate',
        'must not exceed baseLearningRate',
      );
    }
  }

  final double baseLearningRate;
  final double minLearningRate;

  /// Steps spanned by the half cosine, boundary included.
  final int totalSteps;

  @override
  double rateAt(int step) {
    final clamped = step > totalSteps ? totalSteps : step;
    final phase = math.pi * clamped / totalSteps;
    return minLearningRate +
        (baseLearningRate - minLearningRate) * (1 + math.cos(phase)) / 2;
  }

  @override
  String toString() =>
      'CosineAnnealingSchedule(base: $baseLearningRate, '
      'totalSteps: $totalSteps, min: $minLearningRate)';
}

/// Linearly ramps [schedule] from [startFactor] to its full rate over
/// [warmupSteps], then defers to it unchanged.
///
/// The wrapped schedule always sees the same global step, never a shifted one.
/// Warmup is a multiplier on the curve the caller already chose, so composing
/// warmup with cosine annealing does not silently move the annealing horizon.
final class LinearWarmupSchedule extends LearningRateSchedule {
  LinearWarmupSchedule({
    required this.schedule,
    required this.warmupSteps,
    this.startFactor = 0,
  }) {
    if (warmupSteps <= 0) {
      throw ArgumentError.value(warmupSteps, 'warmupSteps', 'must be positive');
    }
    if (!startFactor.isFinite || startFactor < 0 || startFactor > 1) {
      throw ArgumentError.value(
        startFactor,
        'startFactor',
        'must be finite in [0, 1]',
      );
    }
  }

  final LearningRateSchedule schedule;

  /// Steps spent ramping. Step [warmupSteps] is the first unscaled step.
  final int warmupSteps;

  /// Multiplier applied at step zero.
  final double startFactor;

  @override
  double rateAt(int step) {
    if (step >= warmupSteps) return schedule(step);
    final factor = startFactor + (1 - startFactor) * (step / warmupSteps);
    return schedule(step) * factor;
  }

  @override
  String toString() =>
      'LinearWarmupSchedule(warmupSteps: $warmupSteps, '
      'startFactor: $startFactor, schedule: $schedule)';
}

void _positive(double value, String name) {
  if (!value.isFinite || value <= 0) {
    throw ArgumentError.value(value, name, 'must be finite and positive');
  }
}
