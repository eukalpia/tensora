/// One recorded scalar observation.
///
/// A metric point is validated at construction so a history can never carry a
/// value that a plot, a checkpoint policy, or an early-stopping rule would have
/// to defend against later.
final class MetricPoint {
  MetricPoint({required this.name, required this.step, required this.value}) {
    if (name.trim().isEmpty) {
      throw ArgumentError.value(name, 'name', 'must not be empty');
    }
    if (step < 0) {
      throw ArgumentError.value(step, 'step', 'must be non-negative');
    }
    if (!value.isFinite) {
      throw ArgumentError.value(value, 'value', 'must be finite');
    }
  }

  final String name;
  final int step;
  final double value;

  @override
  bool operator ==(Object other) =>
      other is MetricPoint &&
      other.name == name &&
      other.step == step &&
      other.value == value;

  @override
  int get hashCode => Object.hash(name, step, value);

  @override
  String toString() => 'MetricPoint($name, step: $step, value: $value)';
}

/// An immutable, append-only record of training metrics.
///
/// [add] copies, so appending point by point is quadratic. Producers that
/// already hold every point — [Trainer] among them — build the list first and
/// construct the history once.
final class TrainingHistory {
  TrainingHistory([Iterable<MetricPoint> points = const <MetricPoint>[]])
    : _points = List<MetricPoint>.unmodifiable(points);

  final List<MetricPoint> _points;

  List<MetricPoint> get points => _points;

  TrainingHistory add(MetricPoint point) =>
      TrainingHistory(<MetricPoint>[..._points, point]);

  List<MetricPoint> forMetric(String name) => List<MetricPoint>.unmodifiable(
    _points.where((point) => point.name == name),
  );
}
