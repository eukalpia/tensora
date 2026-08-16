import 'package:tensora/tensora.dart';
import 'package:tensora_nn/tensora_nn.dart' as nn;
import 'package:test/test.dart';

void main() {
  test(
    'StateDict releases earlier snapshots when a later parameter is disposed',
    () {
      final layer = nn.Linear(inFeatures: 1, outFeatures: 1);
      addTearDown(layer.dispose);

      final parameters = layer.parameters;
      expect(parameters, hasLength(2));
      parameters.last.dispose();
      final baseline = TensoraRuntime.liveTensorCount;

      expect(layer.stateDict, throwsA(isA<DisposedTensorException>()));
      expect(TensoraRuntime.liveTensorCount, baseline);
    },
  );
}
