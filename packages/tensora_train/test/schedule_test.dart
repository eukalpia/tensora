import 'dart:math' as math;

import 'package:tensora_train/tensora_train.dart';
import 'package:test/test.dart';

void main() {
  group('ConstantSchedule', () {
    test('holds one rate for every step', () {
      final schedule = ConstantSchedule(0.05);

      expect(schedule(0), 0.05);
      expect(schedule(1), 0.05);
      expect(schedule(1000000), 0.05);
      expect(schedule.learningRate, 0.05);
      expect(schedule.toString(), 'ConstantSchedule(0.05)');
    });

    test('rejects a non-positive or non-finite rate', () {
      expect(() => ConstantSchedule(0), throwsArgumentError);
      expect(() => ConstantSchedule(-1), throwsArgumentError);
      expect(() => ConstantSchedule(double.nan), throwsArgumentError);
      expect(() => ConstantSchedule(double.infinity), throwsArgumentError);
    });
  });

  group('StepDecaySchedule', () {
    test('is a staircase that drops on period boundaries', () {
      final schedule = StepDecaySchedule(
        baseLearningRate: 1,
        stepSize: 3,
        gamma: 0.5,
      );

      expect(schedule.sample(9), <double>[
        1,
        1,
        1,
        0.5,
        0.5,
        0.5,
        0.25,
        0.25,
        0.25,
      ]);
    });

    test('gamma of one holds the base rate', () {
      final schedule = StepDecaySchedule(
        baseLearningRate: 0.2,
        stepSize: 1,
        gamma: 1,
      );

      expect(schedule.sample(4), <double>[0.2, 0.2, 0.2, 0.2]);
    });

    test('rejects invalid arguments', () {
      expect(
        () => StepDecaySchedule(baseLearningRate: 0, stepSize: 1, gamma: 0.5),
        throwsArgumentError,
      );
      expect(
        () => StepDecaySchedule(baseLearningRate: 1, stepSize: 0, gamma: 0.5),
        throwsArgumentError,
      );
      expect(
        () => StepDecaySchedule(baseLearningRate: 1, stepSize: -1, gamma: 0.5),
        throwsArgumentError,
      );
      expect(
        () => StepDecaySchedule(baseLearningRate: 1, stepSize: 1, gamma: 0),
        throwsArgumentError,
      );
      expect(
        () => StepDecaySchedule(
          baseLearningRate: 1,
          stepSize: 1,
          gamma: double.nan,
        ),
        throwsArgumentError,
      );
    });
  });

  group('ExponentialDecaySchedule', () {
    test('multiplies by gamma on every step', () {
      final schedule = ExponentialDecaySchedule(
        baseLearningRate: 1,
        gamma: 0.9,
      );
      final rates = schedule.sample(4);

      expect(rates[0], closeTo(1, 1e-12));
      expect(rates[1], closeTo(0.9, 1e-12));
      expect(rates[2], closeTo(0.81, 1e-12));
      expect(rates[3], closeTo(0.729, 1e-12));
    });

    test('decays monotonically and stays positive', () {
      final schedule = ExponentialDecaySchedule(
        baseLearningRate: 0.01,
        gamma: 0.5,
      );
      final rates = schedule.sample(40);

      for (var index = 1; index < rates.length; index++) {
        expect(rates[index], lessThan(rates[index - 1]));
        expect(rates[index], greaterThan(0));
      }
    });

    test('rejects invalid arguments', () {
      expect(
        () => ExponentialDecaySchedule(baseLearningRate: -1, gamma: 0.9),
        throwsArgumentError,
      );
      expect(
        () => ExponentialDecaySchedule(baseLearningRate: 1, gamma: -0.1),
        throwsArgumentError,
      );
      expect(
        () => ExponentialDecaySchedule(
          baseLearningRate: 1,
          gamma: double.infinity,
        ),
        throwsArgumentError,
      );
    });
  });

  group('CosineAnnealingSchedule', () {
    test('follows the half cosine from base to floor', () {
      final schedule = CosineAnnealingSchedule(
        baseLearningRate: 1,
        totalSteps: 4,
      );

      expect(schedule(0), closeTo(1, 1e-12));
      expect(schedule(1), closeTo((1 + math.cos(math.pi / 4)) / 2, 1e-12));
      expect(schedule(2), closeTo(0.5, 1e-12));
      expect(schedule(3), closeTo((1 + math.cos(3 * math.pi / 4)) / 2, 1e-12));
      expect(schedule(4), closeTo(0, 1e-12));
    });

    test('clamps past the planned horizon instead of cycling back up', () {
      final schedule = CosineAnnealingSchedule(
        baseLearningRate: 1,
        totalSteps: 4,
        minLearningRate: 0.1,
      );

      expect(schedule(4), closeTo(0.1, 1e-12));
      expect(schedule(5), closeTo(0.1, 1e-12));
      expect(schedule(4000), closeTo(0.1, 1e-12));
    });

    test('interpolates between the floor and the base', () {
      final schedule = CosineAnnealingSchedule(
        baseLearningRate: 1.1,
        totalSteps: 4,
        minLearningRate: 0.1,
      );

      expect(schedule(0), closeTo(1.1, 1e-12));
      expect(schedule(2), closeTo(0.6, 1e-12));
    });

    test('never leaves the closed floor-to-base interval', () {
      final schedule = CosineAnnealingSchedule(
        baseLearningRate: 0.3,
        totalSteps: 17,
        minLearningRate: 0.05,
      );

      for (final rate in schedule.sample(40)) {
        expect(rate, greaterThanOrEqualTo(0.05 - 1e-12));
        expect(rate, lessThanOrEqualTo(0.3 + 1e-12));
      }
    });

    test('rejects invalid arguments', () {
      expect(
        () => CosineAnnealingSchedule(baseLearningRate: 1, totalSteps: 0),
        throwsArgumentError,
      );
      expect(
        () => CosineAnnealingSchedule(baseLearningRate: 0, totalSteps: 4),
        throwsArgumentError,
      );
      expect(
        () => CosineAnnealingSchedule(
          baseLearningRate: 1,
          totalSteps: 4,
          minLearningRate: -0.1,
        ),
        throwsArgumentError,
      );
      expect(
        () => CosineAnnealingSchedule(
          baseLearningRate: 1,
          totalSteps: 4,
          minLearningRate: 2,
        ),
        throwsArgumentError,
      );
      expect(
        () => CosineAnnealingSchedule(
          baseLearningRate: 1,
          totalSteps: 4,
          minLearningRate: double.nan,
        ),
        throwsArgumentError,
      );
    });
  });

  group('LinearWarmupSchedule', () {
    test('ramps to the full rate and then hands over exactly', () {
      final schedule = LinearWarmupSchedule(
        schedule: ConstantSchedule(1),
        warmupSteps: 4,
      );

      expect(schedule.sample(6), <double>[0, 0.25, 0.5, 0.75, 1, 1]);
    });

    test('starts at startFactor rather than zero when asked', () {
      final schedule = LinearWarmupSchedule(
        schedule: ConstantSchedule(1),
        warmupSteps: 2,
        startFactor: 0.5,
      );

      expect(schedule.sample(3), <double>[0.5, 0.75, 1]);
    });

    test('scales the wrapped schedule on the same step axis', () {
      final inner = CosineAnnealingSchedule(baseLearningRate: 1, totalSteps: 8);
      final schedule = LinearWarmupSchedule(schedule: inner, warmupSteps: 2);

      expect(schedule(0), closeTo(0, 1e-12));
      expect(schedule(1), closeTo(inner(1) * 0.5, 1e-12));
      expect(schedule(2), closeTo(inner(2), 1e-12));
      expect(schedule(7), closeTo(inner(7), 1e-12));
    });

    test('composes over a decaying schedule without moving its horizon', () {
      final inner = StepDecaySchedule(
        baseLearningRate: 1,
        stepSize: 2,
        gamma: 0.5,
      );
      final schedule = LinearWarmupSchedule(schedule: inner, warmupSteps: 2);

      expect(schedule.sample(6), <double>[0, 0.5, 0.5, 0.5, 0.25, 0.25]);
    });

    test('nests', () {
      final schedule = LinearWarmupSchedule(
        schedule: LinearWarmupSchedule(
          schedule: ConstantSchedule(1),
          warmupSteps: 2,
        ),
        warmupSteps: 2,
        startFactor: 0.5,
      );

      expect(schedule(0), closeTo(0, 1e-12));
      expect(schedule(1), closeTo(0.5 * 0.75, 1e-12));
      expect(schedule(2), closeTo(1, 1e-12));
    });

    test('rejects invalid arguments', () {
      expect(
        () =>
            LinearWarmupSchedule(schedule: ConstantSchedule(1), warmupSteps: 0),
        throwsArgumentError,
      );
      expect(
        () => LinearWarmupSchedule(
          schedule: ConstantSchedule(1),
          warmupSteps: -3,
        ),
        throwsArgumentError,
      );
      expect(
        () => LinearWarmupSchedule(
          schedule: ConstantSchedule(1),
          warmupSteps: 2,
          startFactor: -0.1,
        ),
        throwsArgumentError,
      );
      expect(
        () => LinearWarmupSchedule(
          schedule: ConstantSchedule(1),
          warmupSteps: 2,
          startFactor: 1.5,
        ),
        throwsArgumentError,
      );
      expect(
        () => LinearWarmupSchedule(
          schedule: ConstantSchedule(1),
          warmupSteps: 2,
          startFactor: double.nan,
        ),
        throwsArgumentError,
      );
    });
  });

  group('LearningRateSchedule contract', () {
    test('rejects a negative step for every schedule', () {
      final schedules = <LearningRateSchedule>[
        ConstantSchedule(1),
        StepDecaySchedule(baseLearningRate: 1, stepSize: 2, gamma: 0.5),
        ExponentialDecaySchedule(baseLearningRate: 1, gamma: 0.5),
        CosineAnnealingSchedule(baseLearningRate: 1, totalSteps: 4),
        LinearWarmupSchedule(schedule: ConstantSchedule(1), warmupSteps: 2),
      ];

      for (final schedule in schedules) {
        expect(() => schedule(-1), throwsArgumentError, reason: '$schedule');
      }
    });

    test('sample mirrors call and is immutable', () {
      final schedule = ExponentialDecaySchedule(
        baseLearningRate: 0.5,
        gamma: 0.8,
      );
      final rates = schedule.sample(5);

      expect(rates, hasLength(5));
      for (var step = 0; step < rates.length; step++) {
        expect(rates[step], schedule(step));
      }
      expect(() => rates.add(0), throwsUnsupportedError);
      expect(schedule.sample(0), isEmpty);
      expect(() => schedule.sample(-1), throwsArgumentError);
    });

    test('a custom schedule only has to implement rateAt', () {
      final schedule = _ReciprocalSchedule();

      expect(schedule(0), 1);
      expect(schedule(1), 0.5);
      expect(schedule(3), 0.25);
      expect(() => schedule(-2), throwsArgumentError);
    });
  });
}

final class _ReciprocalSchedule extends LearningRateSchedule {
  @override
  double rateAt(int step) => 1 / (step + 1);
}
