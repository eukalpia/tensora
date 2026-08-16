import 'package:tensora_nn/tensora_nn.dart';
import 'package:test/test.dart';

void main() {
  test('NN V2 public namespace exposes declarative building blocks', () {
    final Module model = Sequential(children: <Module>[Identity()]);
    final Linear Function({
      required int inFeatures,
      required int outFeatures,
      bool bias,
    })
    linearFactory = Linear.new;
    final List<Module> activations = <Module>[
      ReLU(),
      Sigmoid(),
      Tanh(),
      GELU(),
      SiLU(),
      SwiGLU(),
    ];
    final MSELoss mse = MSELoss();
    final CrossEntropyLoss crossEntropy = CrossEntropyLoss();

    expect(model, isA<Sequential>());
    expect(linearFactory, isNotNull);
    expect(activations, hasLength(6));
    expect(mse, isA<MSELoss>());
    expect(crossEntropy, isA<CrossEntropyLoss>());
  });
}
