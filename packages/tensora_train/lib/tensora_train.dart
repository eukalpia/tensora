/// Training orchestration contracts for Tensora.
library;

export 'package:tensora/tensora.dart' show Device, Tensor, TensoraRuntime;
export 'package:tensora_nn/tensora_nn.dart';
export 'package:tensora_optim/tensora_optim.dart';

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
}

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
