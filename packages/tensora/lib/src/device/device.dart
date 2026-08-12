/// A device on which Tensora tensors are stored and executed.
final class Device {
  const Device._(this.name);

  /// Host CPU memory and execution.
  static const Device cpu = Device._('cpu');

  /// Stable user-facing device name.
  final String name;

  @override
  String toString() => 'Device.$name';
}
