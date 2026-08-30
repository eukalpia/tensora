import 'package:tensora/tensora.dart' as core;
import 'package:tensora_nn/tensora_nn.dart' as nn;
import 'package:tensora_optim/tensora_optim.dart' as optim;

import 'history.dart';

/// Produces a loss Tensor from a prediction and its target.
///
/// The returned Tensor is owned by the caller of this function, which is always
/// [Trainer]; `nn.Losses.mse` and `nn.MSELoss().call` both match.
typedef LossFunction =
    core.Tensor Function(core.Tensor prediction, core.Tensor target);

/// Materializes one element of a batch source into the tensors a step needs.
///
/// Ownership of both returned tensors transfers to [Trainer], which disposes
/// them once the step completes or fails. A function that fails part way must
/// dispose whatever it already created; nothing else can see those handles.
typedef BatchMaterializer<B> = TrainingBatch Function(B batch);

/// The input/target Tensor pair for a single step.
///
/// Both tensors are owned by the [Trainer] that receives them. Passing a tensor
/// the caller intends to reuse across steps is a use-after-dispose bug; clone
/// it instead.
final class TrainingBatch {
  TrainingBatch({required this.input, required this.target}) {
    if (input.isDisposed) {
      throw ArgumentError.value(input, 'input', 'must not be disposed');
    }
    if (target.isDisposed) {
      throw ArgumentError.value(target, 'target', 'must not be disposed');
    }
  }

  final core.Tensor input;
  final core.Tensor target;
}

/// Which pass a [TrainingEvent] came from.
enum TrainingPhase { training, evaluation }

/// Metric names [Trainer] writes into its [TrainingHistory].
abstract final class TrainerMetrics {
  /// Loss of one training batch, stepped by the global optimizer step.
  static const String batchLoss = 'loss';

  /// Mean training loss of one epoch, stepped by the epoch index.
  static const String epochLoss = 'epoch_loss';

  /// Mean evaluation loss of one epoch, stepped by the epoch index.
  static const String evaluationLoss = 'eval_loss';
}

/// Progress report for one completed batch.
///
/// Events are delivered after every Tensor the batch created has been released,
/// so a callback can run arbitrary code — including creating tensors — without
/// interleaving with a live autograd graph.
final class TrainingEvent {
  const TrainingEvent({
    required this.phase,
    required this.epoch,
    required this.batch,
    required this.globalStep,
    required this.loss,
  });

  final TrainingPhase phase;

  /// Zero-based epoch index.
  final int epoch;

  /// Zero-based batch index inside this epoch and phase.
  final int batch;

  /// Optimizer steps applied before this event. Evaluation does not advance it.
  final int globalStep;

  final double loss;

  @override
  String toString() =>
      'TrainingEvent(${phase.name}, epoch: $epoch, batch: $batch, '
      'globalStep: $globalStep, loss: $loss)';
}

/// Aggregate result of one epoch.
final class EpochSummary {
  const EpochSummary({
    required this.epoch,
    required this.trainingLoss,
    required this.trainingBatches,
    required this.evaluationLoss,
    required this.evaluationBatches,
  });

  final int epoch;

  /// Mean loss over [trainingBatches]. A cancelled epoch reports the mean over
  /// the batches that actually ran.
  final double trainingLoss;
  final int trainingBatches;

  /// Mean evaluation loss, or null when this epoch ran no evaluation pass.
  final double? evaluationLoss;
  final int evaluationBatches;

  @override
  String toString() =>
      'EpochSummary(epoch: $epoch, trainingLoss: $trainingLoss, '
      'evaluationLoss: $evaluationLoss)';
}

/// Everything [Trainer.fit] observed.
final class TrainingRun {
  TrainingRun({
    required this.history,
    required Iterable<EpochSummary> epochs,
    required this.completedSteps,
    required this.cancelled,
  }) : epochs = List<EpochSummary>.unmodifiable(epochs);

  final TrainingHistory history;
  final List<EpochSummary> epochs;

  /// Optimizer steps applied across the whole run.
  final int completedSteps;

  /// Whether the run stopped early because its cancellation token was set.
  final bool cancelled;
}

/// Cooperative stop signal for a running [Trainer.fit].
///
/// [Trainer.fit] is synchronous, so a token is set from inside a progress
/// callback — early stopping on an evaluation metric is the motivating case —
/// or before the run starts.
final class TrainingCancellation {
  bool _cancelled = false;

  bool get isCancelled => _cancelled;

  void cancel() => _cancelled = true;
}

/// A loss value that no metric can carry and no optimizer should consume.
///
/// [Trainer] reads the loss before `backward()`, so a diverged step is
/// reported without having been applied to the model.
final class NonFiniteLossException implements Exception {
  const NonFiniteLossException(this.value, {required this.phase});

  final double value;
  final TrainingPhase phase;

  @override
  String toString() =>
      'NonFiniteLossException: ${phase.name} produced a loss of $value. '
      'The step was not applied.';
}

/// Runs the epoch/batch loop that callers otherwise hand-write.
///
/// The trainer exists for one reason that a hand-written loop keeps getting
/// wrong: every intermediate Tensor a step allocates — the materialized batch,
/// the prediction, and the loss — is released on the success path *and* on the
/// exception path, before the failure propagates. A leaked native handle is
/// invisible until the process runs out of accelerator memory, so disposal is
/// not left to the caller's `finally`.
///
/// The trainer owns the tensors of a step and nothing else. It never disposes
/// the model, the optimizer, or a batch source; those outlive the run.
///
/// `fit` is deliberately synchronous. Native execution is synchronous, Tensor
/// wrappers are isolate-local, and suspending inside a live autograd graph
/// would let unrelated code observe half-built state. Callers that need a
/// non-blocking loop run the trainer in its own isolate.
///
/// [B] is the batch-source element type. `tensora_train` cannot depend on
/// `tensora_data` (the workspace dependency contract forbids it), so the
/// trainer consumes a plain `Iterable<B>`; `DataLoader.batches()` satisfies it
/// directly with `B` bound to `Batch<T>`.
final class Trainer<B> {
  Trainer({
    required this.model,
    required this.optimizer,
    required this.lossFunction,
    required this.materializeBatch,
  });

  final nn.Module model;
  final optim.Optimizer optimizer;
  final LossFunction lossFunction;
  final BatchMaterializer<B> materializeBatch;

  /// Trains for [epochs] passes over [trainingBatches].
  ///
  /// [trainingBatches] and [evaluationBatches] are iterated once per epoch and
  /// must therefore be re-iterable; a source that yields nothing on a later pass
  /// is reported as an error rather than silently producing an empty epoch.
  ///
  /// An evaluation pass switches the model to `eval()` and restores the mode it
  /// had on entry, including when a batch throws.
  ///
  /// Cancellation is checked between batches and between epochs, never between
  /// `backward()` and `step()`: stopping there would discard a computed
  /// gradient and leave the model in a state no caller asked for.
  TrainingRun fit({
    required Iterable<B> trainingBatches,
    Iterable<B>? evaluationBatches,
    int epochs = 1,
    void Function(TrainingEvent event)? onEvent,
    void Function(EpochSummary summary)? onEpoch,
    TrainingCancellation? cancellation,
  }) {
    if (epochs < 1) {
      throw ArgumentError.value(epochs, 'epochs', 'must be positive');
    }

    bool isCancelled() => cancellation?.isCancelled ?? false;

    final points = <MetricPoint>[];
    final summaries = <EpochSummary>[];
    var globalStep = 0;
    var cancelled = isCancelled();

    model.train();
    for (var epoch = 0; epoch < epochs && !cancelled; epoch++) {
      var batches = 0;
      var lossTotal = 0.0;
      for (final batch in trainingBatches) {
        if (isCancelled()) {
          cancelled = true;
          break;
        }
        final loss = _runBatch(batch, phase: TrainingPhase.training);
        points.add(
          MetricPoint(
            name: TrainerMetrics.batchLoss,
            step: globalStep,
            value: loss,
          ),
        );
        lossTotal += loss;
        final event = TrainingEvent(
          phase: TrainingPhase.training,
          epoch: epoch,
          batch: batches,
          globalStep: globalStep,
          loss: loss,
        );
        globalStep += 1;
        batches += 1;
        onEvent?.call(event);
      }

      if (batches == 0) {
        if (cancelled) break;
        throw StateError(
          'The training batch source yielded no batches for epoch $epoch. '
          'A batch source must be re-iterable across epochs.',
        );
      }

      final trainingLoss = lossTotal / batches;
      points.add(
        MetricPoint(
          name: TrainerMetrics.epochLoss,
          step: epoch,
          value: trainingLoss,
        ),
      );

      var evaluationBatchCount = 0;
      double? evaluationLoss;
      if (evaluationBatches != null && !isCancelled()) {
        final result = _evaluate(
          evaluationBatches,
          epoch: epoch,
          globalStep: globalStep,
          onEvent: onEvent,
          isCancelled: isCancelled,
        );
        evaluationBatchCount = result.batches;
        if (result.batches > 0) {
          evaluationLoss = result.loss;
          points.add(
            MetricPoint(
              name: TrainerMetrics.evaluationLoss,
              step: epoch,
              value: result.loss,
            ),
          );
        }
      }

      final summary = EpochSummary(
        epoch: epoch,
        trainingLoss: trainingLoss,
        trainingBatches: batches,
        evaluationLoss: evaluationLoss,
        evaluationBatches: evaluationBatchCount,
      );
      summaries.add(summary);
      onEpoch?.call(summary);
      cancelled = cancelled || isCancelled();
    }

    return TrainingRun(
      history: TrainingHistory(points),
      epochs: summaries,
      completedSteps: globalStep,
      cancelled: cancelled,
    );
  }

  /// Runs a standalone evaluation pass and returns its mean loss.
  ///
  /// The model's training mode is restored before this returns.
  double evaluate(Iterable<B> batches) {
    final result = _evaluate(
      batches,
      epoch: 0,
      globalStep: 0,
      onEvent: null,
      isCancelled: () => false,
    );
    if (result.batches == 0) {
      throw StateError('The evaluation batch source yielded no batches.');
    }
    return result.loss;
  }

  _EvaluationResult _evaluate(
    Iterable<B> batches, {
    required int epoch,
    required int globalStep,
    required void Function(TrainingEvent event)? onEvent,
    required bool Function() isCancelled,
  }) {
    final wasTraining = model.isTraining;
    model.eval();
    var count = 0;
    var total = 0.0;
    try {
      for (final batch in batches) {
        if (isCancelled()) break;
        final loss = _runBatch(batch, phase: TrainingPhase.evaluation);
        total += loss;
        final event = TrainingEvent(
          phase: TrainingPhase.evaluation,
          epoch: epoch,
          batch: count,
          globalStep: globalStep,
          loss: loss,
        );
        count += 1;
        onEvent?.call(event);
      }
    } finally {
      if (wasTraining) model.train();
    }
    return _EvaluationResult(
      loss: count == 0 ? 0 : total / count,
      batches: count,
    );
  }

  /// Executes one batch and releases every Tensor it created.
  ///
  /// The loss is read before `backward()` so a non-finite value is reported
  /// without ever reaching the parameters.
  double _runBatch(B batch, {required TrainingPhase phase}) {
    final training = phase == TrainingPhase.training;
    if (training) optimizer.zeroGrad();

    final materialized = materializeBatch(batch);
    final owned = <core.Tensor?>[
      materialized.input,
      materialized.target,
      null,
      null,
    ];
    try {
      final prediction = model(materialized.input);
      owned[2] = prediction;
      final loss = lossFunction(prediction, materialized.target);
      owned[3] = loss;

      final value = _scalarLoss(loss, phase);
      if (training) {
        loss.backward();
        optimizer.step();
      }

      final failure = _releaseAll(owned);
      if (failure != null) throw failure;
      return value;
    } catch (error, stackTrace) {
      _releaseAll(owned);
      Error.throwWithStackTrace(error, stackTrace);
    }
  }

  double _scalarLoss(core.Tensor loss, TrainingPhase phase) {
    if (loss.numel != 1) {
      throw core.InvalidShapeException(
        'A training loss must reduce to a single value, but the loss function '
        'returned ${loss.shape}.',
        operation: 'trainer.loss',
      );
    }
    final value = loss.toList().single;
    if (!value.isFinite) {
      throw NonFiniteLossException(value, phase: phase);
    }
    return value;
  }
}

final class _EvaluationResult {
  const _EvaluationResult({required this.loss, required this.batches});

  final double loss;
  final int batches;
}

/// Releases every tracked Tensor in reverse creation order.
///
/// Every entry is attempted even after one release fails, so a single bad
/// handle cannot strand the rest. The first failure is returned rather than
/// thrown: on the exception path the original error is the root cause and must
/// not be replaced by cleanup noise.
Object? _releaseAll(List<core.Tensor?> tensors) {
  Object? failure;
  for (var index = tensors.length - 1; index >= 0; index--) {
    final tensor = tensors[index];
    if (tensor == null) continue;
    try {
      tensor.dispose();
    } catch (error) {
      failure ??= error;
    }
  }
  return failure;
}
