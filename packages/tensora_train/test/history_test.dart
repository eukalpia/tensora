import 'package:tensora_train/tensora_train.dart';
import 'package:test/test.dart';

void main() {
  test('training history is immutable and filterable', () {
    final initial = TrainingHistory();
    final loss0 = MetricPoint(name: 'loss', step: 0, value: 2.0);
    final loss1 = MetricPoint(name: 'loss', step: 1, value: 1.0);
    final accuracy = MetricPoint(name: 'accuracy', step: 1, value: 0.5);
    final history = initial.add(loss0).add(loss1).add(accuracy);

    expect(initial.points, isEmpty);
    expect(history.forMetric('loss'), <MetricPoint>[loss0, loss1]);
    expect(() => history.points.add(loss0), throwsUnsupportedError);
  });

  test('metric points validate public invariants', () {
    expect(
      () => MetricPoint(name: '', step: 0, value: 1),
      throwsArgumentError,
    );
    expect(
      () => MetricPoint(name: 'loss', step: -1, value: 1),
      throwsArgumentError,
    );
    expect(
      () => MetricPoint(name: 'loss', step: 0, value: double.nan),
      throwsArgumentError,
    );
  });
}
