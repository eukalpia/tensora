import 'package:tensora_nn/tensora_nn.dart';
import 'package:tensora_optim/tensora_optim.dart';
import 'package:test/test.dart';

void main() {
  test('optimizer API is parameter-oriented', () {
    expect(() => AdamW(parameters: const <Parameter>[]), throwsArgumentError);
    expect(() => SGD(parameters: const <Parameter>[]), throwsArgumentError);
    expect(() => Adam(parameters: const <Parameter>[]), throwsArgumentError);
  });

  test('ParameterGroup rejects an empty parameter collection', () {
    expect(
      () => ParameterGroup(parameters: const <Parameter>[]),
      throwsArgumentError,
    );
  });

  test(
    'group factories reject an empty group collection before native work',
    () {
      expect(
        () => AdamW.groups(groups: const <ParameterGroup>[]),
        throwsArgumentError,
      );
      expect(
        () => SGD.groups(groups: const <ParameterGroup>[]),
        throwsArgumentError,
      );
      expect(
        () => Adam.groups(groups: const <ParameterGroup>[]),
        throwsArgumentError,
      );
    },
  );
}
