/// Deterministic data primitives for Tensora.
abstract interface class Dataset<T> {
  int get length;
  T operator [](int index);
}

final class ListDataset<T> implements Dataset<T> {
  ListDataset(Iterable<T> values) : _values = List<T>.unmodifiable(values);

  final List<T> _values;

  @override
  int get length => _values.length;

  List<T> get values => _values;

  @override
  T operator [](int index) {
    RangeError.checkValidIndex(index, _values, 'index');
    return _values[index];
  }
}

final class Batch<T> {
  Batch({required this.startIndex, required Iterable<T> values})
    : values = List<T>.unmodifiable(values) {
    if (startIndex < 0) {
      throw ArgumentError.value(startIndex, 'startIndex', 'must be non-negative');
    }
    if (this.values.isEmpty) {
      throw ArgumentError.value(values, 'values', 'must not be empty');
    }
  }

  final int startIndex;
  final List<T> values;

  int get length => values.length;
}

final class DataLoader<T> {
  DataLoader(this.dataset, {required this.batchSize, this.dropLast = false}) {
    if (batchSize <= 0) {
      throw ArgumentError.value(batchSize, 'batchSize', 'must be positive');
    }
  }

  final Dataset<T> dataset;
  final int batchSize;
  final bool dropLast;

  Iterable<Batch<T>> batches() sync* {
    for (var start = 0; start < dataset.length; start += batchSize) {
      final remaining = dataset.length - start;
      if (dropLast && remaining < batchSize) return;
      final count = remaining < batchSize ? remaining : batchSize;
      yield Batch<T>(
        startIndex: start,
        values: List<T>.generate(count, (offset) => dataset[start + offset]),
      );
    }
  }
}
