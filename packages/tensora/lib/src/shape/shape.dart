/// Immutable tensor dimensions with validated element-count arithmetic.
final class Shape {
  Shape(Iterable<int> dimensions) {
    final copied = List<int>.unmodifiable(dimensions);
    if (copied.length > maxRank) {
      throw ArgumentError.value(
        copied.length,
        'dimensions',
        'Tensor rank cannot exceed $maxRank.',
      );
    }

    var count = 1;
    for (var index = 0; index < copied.length; index++) {
      final dimension = copied[index];
      if (dimension <= 0) {
        throw ArgumentError.value(
          dimension,
          'dimensions[$index]',
          'Tensor dimensions must be positive.',
        );
      }
      if (count > maxNumel ~/ dimension) {
        throw ArgumentError.value(
          copied,
          'dimensions',
          'Tensor element count exceeds the supported int64 range.',
        );
      }
      count *= dimension;
    }

    _dimensions = copied;
    _numel = count;
  }

  /// Maximum rank accepted by the Milestone 1 runtime.
  static const int maxRank = 32;

  /// Maximum element count accepted before native allocation.
  static const int maxNumel = 0x7fffffffffffffff;

  late final List<int> _dimensions;
  late final int _numel;

  /// Number of dimensions. Rank zero represents a scalar.
  int get rank => _dimensions.length;

  /// Dimensions in row-major order.
  List<int> get dimensions => _dimensions;

  /// Total number of tensor elements.
  int get numel => _numel;

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    if (other is! Shape || other.rank != rank) return false;
    for (var index = 0; index < rank; index++) {
      if (_dimensions[index] != other._dimensions[index]) return false;
    }
    return true;
  }

  @override
  int get hashCode => Object.hashAll(_dimensions);

  @override
  String toString() => 'Shape($_dimensions)';
}
