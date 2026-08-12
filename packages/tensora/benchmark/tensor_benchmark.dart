import 'dart:io';
import 'dart:math' as math;

import 'package:tensora/src/native/native_runtime.dart';
import 'package:tensora/tensora.dart';

List<double> _sorted(List<double> values) => [...values]..sort();

double _percentile(List<double> samples, double percentile) {
  final values = _sorted(samples);
  if (values.isEmpty) return 0;
  final index = percentile * (values.length - 1);
  final lower = index.floor();
  final upper = index.ceil();
  if (lower == upper) return values[lower];
  final fraction = index - lower;
  return values[lower] * (1 - fraction) + values[upper] * fraction;
}

void _measure(
  String name, {
  required int warmup,
  required int iterations,
  required void Function() body,
}) {
  for (var index = 0; index < warmup; index++) {
    body();
  }

  final samples = <double>[];
  for (var index = 0; index < iterations; index++) {
    final stopwatch = Stopwatch()..start();
    body();
    stopwatch.stop();
    samples.add(stopwatch.elapsedMicroseconds.toDouble());
  }

  stdout.writeln(
    '$name warmup=$warmup iterations=$iterations '
    'median_us=${_percentile(samples, 0.50).toStringAsFixed(2)} '
    'p95_us=${_percentile(samples, 0.95).toStringAsFixed(2)}',
  );
}

void main(List<String> arguments) {
  final smoke = arguments.contains('--smoke');
  final large = arguments.contains('--large');
  final runtime = NativeRuntime.instance;

  stdout.writeln('Tensora Dart/FFI benchmark');
  stdout.writeln('os=${Platform.operatingSystem} ${Platform.operatingSystemVersion}');
  stdout.writeln('processors=${Platform.numberOfProcessors}');
  stdout.writeln('dart=${Platform.version.split('\n').first}');
  stdout.writeln('native=${runtime.libraryPath}');
  stdout.writeln('mode=${smoke ? 'smoke' : large ? 'large' : 'standard'}');

  _measure(
    'ffi_noop',
    warmup: smoke ? 50 : 1000,
    iterations: smoke ? 500 : 20000,
    body: runtime.noop,
  );

  final creationShape = Shape([64, 64]);
  final creationValues = List<num>.filled(creationShape.numel, 1);
  _measure(
    'tensor_from_list_64x64',
    warmup: smoke ? 1 : 5,
    iterations: smoke ? 3 : 25,
    body: () {
      final tensor = Tensor.fromList(creationValues, shape: creationShape);
      tensor.dispose();
    },
  );

  final left = Tensor.ones(creationShape);
  final right = Tensor.full(creationShape, 2);
  _measure(
    'elementwise_add_64x64',
    warmup: smoke ? 1 : 10,
    iterations: smoke ? 3 : 100,
    body: () {
      final result = left.add(right);
      result.dispose();
    },
  );

  final sizes = smoke
      ? const [64]
      : large
          ? const [64, 256, 1024]
          : const [64, 256];
  for (final size in sizes) {
    final shape = Shape([size, size]);
    final a = Tensor.ones(shape);
    final b = Tensor.full(shape, 0.5);
    final iterations = smoke ? 2 : math.max(2, 512 ~/ size);
    _measure(
      'matmul_${size}x$size',
      warmup: 1,
      iterations: iterations,
      body: () {
        final result = a.matmul(b);
        result.dispose();
      },
    );
    b.dispose();
    a.dispose();
  }

  _measure(
    'native_to_dart_64x64',
    warmup: smoke ? 1 : 5,
    iterations: smoke ? 3 : 50,
    body: () {
      final values = left.toList();
      if (values.length != creationShape.numel) {
        throw StateError('Unexpected extraction length ${values.length}.');
      }
    },
  );

  right.dispose();
  left.dispose();

  stdout.writeln('live_tensors=${runtime.liveTensorCount()}');
  stdout.writeln('live_storage_bytes=${runtime.liveStorageBytes()}');
}
