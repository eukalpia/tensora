/// A device on which Tensora tensors are stored and executed.
final class Device {
  const Device._(this._kind, this.index);

  /// Host CPU memory and execution.
  static const Device cpu = Device._(_DeviceKind.cpu, 0);

  /// A zero-based CUDA device.
  factory Device.cuda(int index) {
    if (index < 0) {
      throw ArgumentError.value(index, 'index', 'must be non-negative');
    }
    return Device._(_DeviceKind.cuda, index);
  }

  final _DeviceKind _kind;

  /// Zero-based device index. CPU always uses index zero.
  final int index;

  /// Stable user-facing device name.
  String get name => isCpu ? 'cpu' : 'cuda:$index';

  /// Whether this value identifies the host CPU.
  bool get isCpu => _kind == _DeviceKind.cpu;

  /// Whether this value identifies a CUDA device.
  bool get isCuda => _kind == _DeviceKind.cuda;

  @override
  bool operator ==(Object other) =>
      other is Device && other._kind == _kind && other.index == index;

  @override
  int get hashCode => Object.hash(_kind, index);

  @override
  String toString() => isCpu ? 'Device.cpu' : 'Device.cuda($index)';
}

enum _DeviceKind { cpu, cuda }
