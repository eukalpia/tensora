import 'package:tensora_train/tensora_train.dart';
import 'package:test/test.dart';

final class _Sample {
  const _Sample(this.features, this.label);

  final List<double> features;
  final double label;
}

typedef _Batch = List<_Sample>;

List<_Batch> _trainingBatches() => <_Batch>[
  <_Sample>[
    const _Sample(<double>[0, 0], 0),
    const _Sample(<double>[1, 0], 1),
  ],
  <_Sample>[
    const _Sample(<double>[0, 1], 1),
    const _Sample(<double>[1, 1], 2),
  ],
];

List<_Batch> _evaluationBatches() => <_Batch>[
  <_Sample>[
    const _Sample(<double>[2, 0], 2),
    const _Sample(<double>[0, 2], 2),
  ],
];

TrainingBatch _materialize(_Batch batch) {
  final features = <double>[];
  final labels = <double>[];
  for (final sample in batch) {
    features.addAll(sample.features);
    labels.add(sample.label);
  }
  final input = Tensor.fromList(features, shape: Shape(<int>[batch.length, 2]));
  try {
    final target = Tensor.fromList(
      labels,
      shape: Shape(<int>[batch.length, 1]),
    );
    return TrainingBatch(input: input, target: target);
  } catch (_) {
    input.dispose();
    rethrow;
  }
}

/// Materializes a target the model's output can never match in shape, so the
/// loss call fails while the prediction is already live.
TrainingBatch _materializeMismatched(_Batch batch) {
  final input = Tensor.fromList(<double>[
    for (final sample in batch) ...sample.features,
  ], shape: Shape(<int>[batch.length, 2]));
  try {
    final target = Tensor.zeros(Shape(<int>[batch.length, 2]));
    return TrainingBatch(input: input, target: target);
  } catch (_) {
    input.dispose();
    rethrow;
  }
}

/// Records the training mode observed inside a forward pass.
final class _ModeProbe extends Module {
  final List<bool> observedModes = <bool>[];

  @override
  Tensor forward(Tensor input) {
    observedModes.add(isTraining);
    return input;
  }

  @override
  String get internalDiagnosticLabel => '_ModeProbe()';
}

List<List<double>> _parameterValues(Module model) {
  final values = <List<double>>[];
  for (final parameter in model.parameters) {
    final snapshot = parameter.snapshot();
    try {
      values.add(snapshot.toList());
    } finally {
      snapshot.dispose();
    }
  }
  return values;
}

void _expectValuesEqual(
  List<List<double>> actual,
  List<List<double>> expected,
) {
  expect(actual, hasLength(expected.length));
  for (var outer = 0; outer < expected.length; outer++) {
    expect(actual[outer], hasLength(expected[outer].length));
    for (var inner = 0; inner < expected[outer].length; inner++) {
      expect(actual[outer][inner], closeTo(expected[outer][inner], 1e-9));
    }
  }
}

bool _nativeTrainingAvailable() {
  try {
    return TensoraRuntime.trainingAvailable;
  } catch (_) {
    return false;
  }
}

void main() {
  final skipReason =
      _nativeTrainingAvailable()
          ? null
          : 'requires a training-enabled Tensora native runtime '
              '(set TENSORA_NATIVE_LIBRARY)';

  group('Trainer', skip: skipReason, () {
    late Linear model;
    late SGD optimizer;

    setUp(() {
      TensoraRuntime.manualSeed(7);
      model = Linear(inFeatures: 2, outFeatures: 1);
      optimizer = SGD(parameters: model.parameters, learningRate: 0.05);
    });

    tearDown(() {
      optimizer.dispose();
      model.dispose();
    });

    Trainer<_Batch> trainerWith({
      LossFunction? lossFunction,
      BatchMaterializer<_Batch>? materializer,
      Module? module,
    }) => Trainer<_Batch>(
      model: module ?? model,
      optimizer: optimizer,
      lossFunction: lossFunction ?? Losses.mse,
      materializeBatch: materializer ?? _materialize,
    );

    test('the live-handle counter these tests rely on is sensitive', () {
      final baseline = TensoraRuntime.liveTensorCount;
      final tensor = Tensor.zeros(Shape(<int>[2, 2]));

      expect(TensoraRuntime.liveTensorCount, greaterThan(baseline));

      tensor.dispose();
      expect(TensoraRuntime.liveTensorCount, baseline);
    });

    test('runs the epoch loop and records every metric', () {
      final trainer = trainerWith();

      final run = trainer.fit(trainingBatches: _trainingBatches(), epochs: 3);

      expect(run.completedSteps, 6);
      expect(run.cancelled, isFalse);
      expect(run.epochs, hasLength(3));

      final batchLosses = run.history.forMetric(TrainerMetrics.batchLoss);
      expect(batchLosses, hasLength(6));
      expect(batchLosses.map((point) => point.step), <int>[0, 1, 2, 3, 4, 5]);

      final epochLosses = run.history.forMetric(TrainerMetrics.epochLoss);
      expect(epochLosses, hasLength(3));
      expect(epochLosses.map((point) => point.step), <int>[0, 1, 2]);
      expect(run.history.forMetric(TrainerMetrics.evaluationLoss), isEmpty);

      for (var epoch = 0; epoch < 3; epoch++) {
        expect(run.epochs[epoch].epoch, epoch);
        expect(run.epochs[epoch].trainingBatches, 2);
        expect(run.epochs[epoch].evaluationLoss, isNull);
        expect(
          run.epochs[epoch].trainingLoss,
          closeTo(epochLosses[epoch].value, 1e-12),
        );
      }
    });

    test('actually optimizes: epoch loss falls', () {
      final trainer = trainerWith();

      final run = trainer.fit(trainingBatches: _trainingBatches(), epochs: 25);

      expect(
        run.epochs.last.trainingLoss,
        lessThan(run.epochs.first.trainingLoss),
      );
    });

    test('leaks no native tensor across a full run with evaluation', () {
      final trainer = trainerWith();
      final baseline = TensoraRuntime.liveTensorCount;

      final run = trainer.fit(
        trainingBatches: _trainingBatches(),
        evaluationBatches: _evaluationBatches(),
        epochs: 4,
      );

      expect(TensoraRuntime.liveTensorCount, baseline);
      expect(run.completedSteps, 8);
      expect(
        run.history.forMetric(TrainerMetrics.evaluationLoss),
        hasLength(4),
      );
      expect(run.epochs.first.evaluationBatches, 1);
      expect(run.epochs.first.evaluationLoss, isNotNull);
    });

    test('leaks no native tensor when the loss function throws', () {
      final trainer = trainerWith(
        lossFunction:
            (Tensor prediction, Tensor target) =>
                throw StateError('loss exploded'),
      );
      final baseline = TensoraRuntime.liveTensorCount;

      expect(
        () => trainer.fit(trainingBatches: _trainingBatches()),
        throwsStateError,
      );

      expect(TensoraRuntime.liveTensorCount, baseline);
    });

    test('leaks no native tensor when the loss shape is rejected', () {
      final trainer = trainerWith(materializer: _materializeMismatched);
      final baseline = TensoraRuntime.liveTensorCount;

      expect(
        () => trainer.fit(trainingBatches: _trainingBatches()),
        throwsA(isA<InvalidShapeException>()),
      );

      expect(TensoraRuntime.liveTensorCount, baseline);
    });

    test('rejects a loss that does not reduce to a single value', () {
      final trainer = trainerWith(
        lossFunction:
            (Tensor prediction, Tensor target) => prediction.add(target),
      );
      final baseline = TensoraRuntime.liveTensorCount;

      expect(
        () => trainer.fit(trainingBatches: _trainingBatches()),
        throwsA(
          isA<InvalidShapeException>().having(
            (error) => error.operation,
            'operation',
            'trainer.loss',
          ),
        ),
      );

      expect(TensoraRuntime.liveTensorCount, baseline);
    });

    test('reports a non-finite loss before it reaches the parameters', () {
      final trainer = trainerWith(
        lossFunction:
            (Tensor prediction, Tensor target) =>
                Tensor.fromList(<double>[double.nan], shape: Shape(<int>[1])),
      );
      final before = _parameterValues(model);
      final baseline = TensoraRuntime.liveTensorCount;

      expect(
        () => trainer.fit(trainingBatches: _trainingBatches()),
        throwsA(
          isA<NonFiniteLossException>()
              .having((error) => error.value.isNaN, 'value.isNaN', isTrue)
              .having((error) => error.phase, 'phase', TrainingPhase.training),
        ),
      );

      expect(TensoraRuntime.liveTensorCount, baseline);
      _expectValuesEqual(_parameterValues(model), before);
    });

    test('evaluates in eval mode and restores the previous mode', () {
      final probe = _ModeProbe();
      final composed = Sequential(
        children: <Module>[
          Linear(inFeatures: 2, outFeatures: 4),
          ReLU(),
          Linear(inFeatures: 4, outFeatures: 1),
          probe,
        ],
      );
      addTearDown(composed.dispose);
      final composedOptimizer = SGD(
        parameters: composed.parameters,
        learningRate: 0.05,
      );
      addTearDown(composedOptimizer.dispose);
      final trainer = Trainer<_Batch>(
        model: composed,
        optimizer: composedOptimizer,
        lossFunction: Losses.mse,
        materializeBatch: _materialize,
      );
      final baseline = TensoraRuntime.liveTensorCount;

      final run = trainer.fit(
        trainingBatches: _trainingBatches(),
        evaluationBatches: _evaluationBatches(),
        epochs: 2,
      );

      expect(probe.observedModes, <bool>[true, true, false, true, true, false]);
      expect(composed.isTraining, isTrue);
      expect(run.completedSteps, 4);
      expect(TensoraRuntime.liveTensorCount, baseline);
    });

    test('restores training mode when an evaluation batch throws', () {
      final trainer = trainerWith(
        lossFunction: (Tensor prediction, Tensor target) {
          if (!model.isTraining) throw StateError('evaluation exploded');
          return Losses.mse(prediction, target);
        },
      );
      final baseline = TensoraRuntime.liveTensorCount;

      expect(
        () => trainer.fit(
          trainingBatches: _trainingBatches(),
          evaluationBatches: _evaluationBatches(),
        ),
        throwsStateError,
      );

      expect(model.isTraining, isTrue);
      expect(TensoraRuntime.liveTensorCount, baseline);
    });

    test('evaluate does not change parameters and leaks nothing', () {
      final trainer = trainerWith();
      final before = _parameterValues(model);
      final baseline = TensoraRuntime.liveTensorCount;

      final loss = trainer.evaluate(_evaluationBatches());

      expect(loss.isFinite, isTrue);
      expect(model.isTraining, isTrue);
      expect(TensoraRuntime.liveTensorCount, baseline);
      _expectValuesEqual(_parameterValues(model), before);
    });

    test('evaluate rejects an empty batch source', () {
      final trainer = trainerWith();

      expect(() => trainer.evaluate(<_Batch>[]), throwsStateError);
    });

    test('cancels between batches without leaking', () {
      final trainer = trainerWith();
      final cancellation = TrainingCancellation();
      final baseline = TensoraRuntime.liveTensorCount;
      final events = <TrainingEvent>[];

      final run = trainer.fit(
        trainingBatches: _trainingBatches(),
        epochs: 10,
        cancellation: cancellation,
        onEvent: (event) {
          events.add(event);
          cancellation.cancel();
        },
      );

      expect(run.cancelled, isTrue);
      expect(run.completedSteps, 1);
      expect(events, hasLength(1));
      expect(events.single.phase, TrainingPhase.training);
      expect(events.single.epoch, 0);
      expect(events.single.batch, 0);
      expect(events.single.globalStep, 0);
      expect(run.epochs, hasLength(1));
      expect(run.epochs.single.trainingBatches, 1);
      expect(TensoraRuntime.liveTensorCount, baseline);
    });

    test('cancels before the first step', () {
      final trainer = trainerWith();
      final cancellation = TrainingCancellation()..cancel();

      final run = trainer.fit(
        trainingBatches: _trainingBatches(),
        epochs: 5,
        cancellation: cancellation,
      );

      expect(run.cancelled, isTrue);
      expect(run.completedSteps, 0);
      expect(run.epochs, isEmpty);
      expect(run.history.points, isEmpty);
    });

    test('skips the evaluation pass when cancelled during training', () {
      final trainer = trainerWith();
      final cancellation = TrainingCancellation();

      final run = trainer.fit(
        trainingBatches: _trainingBatches(),
        evaluationBatches: _evaluationBatches(),
        epochs: 3,
        cancellation: cancellation,
        onEvent: (event) {
          if (event.globalStep == 1) cancellation.cancel();
        },
      );

      expect(run.cancelled, isTrue);
      expect(run.completedSteps, 2);
      expect(run.epochs.single.evaluationBatches, 0);
      expect(run.epochs.single.evaluationLoss, isNull);
      expect(run.history.forMetric(TrainerMetrics.evaluationLoss), isEmpty);
    });

    test('cancels between evaluation batches without leaking', () {
      final trainer = trainerWith();
      final cancellation = TrainingCancellation();
      final baseline = TensoraRuntime.liveTensorCount;
      final evaluationEvents = <TrainingEvent>[];

      final run = trainer.fit(
        trainingBatches: _trainingBatches(),
        evaluationBatches: <_Batch>[
          ..._evaluationBatches(),
          ..._evaluationBatches(),
        ],
        epochs: 3,
        cancellation: cancellation,
        onEvent: (event) {
          if (event.phase != TrainingPhase.evaluation) return;
          evaluationEvents.add(event);
          cancellation.cancel();
        },
      );

      expect(evaluationEvents, hasLength(1));
      expect(run.cancelled, isTrue);
      expect(run.epochs.single.evaluationBatches, 1);
      expect(run.epochs.single.evaluationLoss, isNotNull);
      expect(
        run.history.forMetric(TrainerMetrics.evaluationLoss),
        hasLength(1),
      );
      expect(TensoraRuntime.liveTensorCount, baseline);
    });

    test('reports a batch source that yields nothing', () {
      final trainer = trainerWith();

      expect(
        () => trainer.fit(trainingBatches: <_Batch>[]),
        throwsA(
          isA<StateError>().having(
            (error) => error.message,
            'message',
            contains('re-iterable'),
          ),
        ),
      );
    });

    test('reports an exhausted batch source on a later epoch', () {
      final trainer = trainerWith();
      final source = _OneShotBatches(_trainingBatches());

      expect(
        () => trainer.fit(trainingBatches: source, epochs: 2),
        throwsA(
          isA<StateError>().having(
            (error) => error.message,
            'message',
            contains('epoch 1'),
          ),
        ),
      );
    });

    test('delivers epoch summaries to the epoch callback', () {
      final trainer = trainerWith();
      final summaries = <EpochSummary>[];

      final run = trainer.fit(
        trainingBatches: _trainingBatches(),
        evaluationBatches: _evaluationBatches(),
        epochs: 2,
        onEpoch: summaries.add,
      );

      expect(summaries, hasLength(2));
      expect(summaries, run.epochs);
      expect(summaries.first.evaluationLoss, isNotNull);
    });

    test('reports evaluation events without advancing the global step', () {
      final trainer = trainerWith();
      final events = <TrainingEvent>[];

      trainer.fit(
        trainingBatches: _trainingBatches(),
        evaluationBatches: _evaluationBatches(),
        epochs: 1,
        onEvent: events.add,
      );

      expect(events, hasLength(3));
      expect(events[2].phase, TrainingPhase.evaluation);
      expect(events[2].globalStep, 2);
      expect(events[2].batch, 0);
    });

    test('never disposes the model or the optimizer', () {
      final trainer = trainerWith();

      trainer.fit(trainingBatches: _trainingBatches(), epochs: 2);

      expect(model.isDisposed, isFalse);
      expect(optimizer.isDisposed, isFalse);
      expect(model.isTraining, isTrue);
    });

    test('rejects a non-positive epoch count', () {
      final trainer = trainerWith();

      expect(
        () => trainer.fit(trainingBatches: _trainingBatches(), epochs: 0),
        throwsArgumentError,
      );
      expect(
        () => trainer.fit(trainingBatches: _trainingBatches(), epochs: -1),
        throwsArgumentError,
      );
    });

    test('TrainingBatch rejects a disposed tensor', () {
      final input = Tensor.zeros(Shape(<int>[1, 2]));
      final target = Tensor.zeros(Shape(<int>[1, 1]));
      addTearDown(target.dispose);
      input.dispose();

      expect(
        () => TrainingBatch(input: input, target: target),
        throwsArgumentError,
      );

      final live = Tensor.zeros(Shape(<int>[1, 2]));
      addTearDown(live.dispose);
      final disposedTarget = Tensor.zeros(Shape(<int>[1, 1]))..dispose();
      expect(
        () => TrainingBatch(input: live, target: disposedTarget),
        throwsArgumentError,
      );
    });
  });
}

/// A batch source that yields its batches once and is empty afterwards.
final class _OneShotBatches extends Iterable<_Batch> {
  _OneShotBatches(this._batches);

  final List<_Batch> _batches;
  bool _consumed = false;

  @override
  Iterator<_Batch> get iterator {
    if (_consumed) return const <_Batch>[].iterator;
    _consumed = true;
    return _batches.iterator;
  }
}
