import 'package:tensora_data/tensora_data.dart';
import 'package:test/test.dart';

void main() {
  test('ListDataset snapshots input and validates indexes', () {
    final source = <int>[1, 2, 3];
    final dataset = ListDataset<int>(source);
    source[0] = 99;

    expect(dataset.length, 3);
    expect(dataset.values, <int>[1, 2, 3]);
    expect(dataset[1], 2);
    expect(() => dataset[3], throwsRangeError);
    expect(() => dataset.values.add(4), throwsUnsupportedError);
  });

  test('DataLoader batches deterministically and supports dropLast', () {
    final dataset = ListDataset<int>(<int>[0, 1, 2, 3, 4]);
    final batches = DataLoader<int>(dataset, batchSize: 2).batches().toList();

    expect(batches.map((batch) => batch.values).toList(), <List<int>>[
      <int>[0, 1],
      <int>[2, 3],
      <int>[4],
    ]);

    final dropped =
        DataLoader<int>(
          dataset,
          batchSize: 2,
          dropLast: true,
        ).batches().toList();
    expect(dropped.map((batch) => batch.values).toList(), <List<int>>[
      <int>[0, 1],
      <int>[2, 3],
    ]);
  });

  test('batching validates invalid construction', () {
    final dataset = ListDataset<int>(<int>[1]);
    expect(() => DataLoader<int>(dataset, batchSize: 0), throwsArgumentError);
    expect(
      () => Batch<int>(startIndex: -1, values: const <int>[1]),
      throwsArgumentError,
    );
    expect(
      () => Batch<int>(startIndex: 0, values: const <int>[]),
      throwsArgumentError,
    );
  });
}
