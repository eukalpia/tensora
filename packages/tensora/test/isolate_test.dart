import 'dart:isolate';

import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

void _openReceivePort(SendPort readyPort) {
  final receivePort = ReceivePort();
  readyPort.send(receivePort.sendPort);
  receivePort.listen((_) {});
}

void main() {
  test('Tensor wrappers cannot cross Dart isolate boundaries', () async {
    final tensor = Tensor.ones(Shape([2, 2]));
    final readyPort = ReceivePort();
    final isolate = await Isolate.spawn(_openReceivePort, readyPort.sendPort);
    final targetPort = await readyPort.first as SendPort;

    try {
      expect(
        () => targetPort.send(tensor),
        throwsA(isA<ArgumentError>()),
      );
    } finally {
      isolate.kill(priority: Isolate.immediate);
      readyPort.close();
      tensor.dispose();
    }
  });
}
