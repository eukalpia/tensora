import 'package:flutter_test/flutter_test.dart';
import 'package:tensora_flutter/tensora_flutter.dart';

void main() {
  test('controller enforces lifecycle', () {
    var released = 0;
    final first = Object();
    final second = Object();
    final controller = TensorController<Object>(
      value: first,
      disposeValue: (_) => released++,
    );
    controller.replace(second);
    expect(released, 1);
    expect(controller.take(), same(second));
    controller.dispose();
    expect(released, 1);
    expect(controller.take, throwsStateError);
  });
}
