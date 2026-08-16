import 'package:tensora/tensora.dart';

void main() {
  final a = Tensor.fromList([1.0, 2.0, 3.0, 4.0], shape: Shape([2, 2]));

  final b = Tensor.fromList([5.0, 6.0, 7.0, 8.0], shape: Shape([2, 2]));

  final result = a.matmul(b);

  print(result.shape);
  print(result.toList());

  result.dispose();
  b.dispose();
  a.dispose();
}
