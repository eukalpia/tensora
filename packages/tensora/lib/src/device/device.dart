/// A device on which Tensora tensors are stored and executed.
final class Device {
  const Device._(this._kind, this.index);

  /// Host CPU memory and execution.
  static const Device cpu = Device._(_DeviceKind.cpu, 0);

  /// The macOS Metal Performance Shaders device.
  static const Device mps = Device._(_DeviceKind.mps, 0);

  /// A zero-based NVIDIA CUDA device.
  factory Device.cuda(int index) => _indexed(_DeviceKind.cuda, index, 'cuda');

  /// A zero-based Intel XPU device.
  factory Device.xpu(int index) => _indexed(_DeviceKind.xpu, index, 'xpu');

  /// A zero-based AMD HIP/ROCm device.
  factory Device.hip(int index) => _indexed(_DeviceKind.hip, index, 'hip');

  static Device _indexed(_DeviceKind kind, int index, String name) {
    if (index < 0) {
      throw ArgumentError.value(index, 'index', 'must be non-negative');
    }
    return Device._(kind, index);
  }

  final _DeviceKind _kind;

  /// Zero-based device index. CPU and MPS always use index zero.
  final int index;

  /// Stable user-facing device name.
  String get name => switch (_kind) {
    _DeviceKind.cpu => 'cpu',
    _DeviceKind.cuda => 'cuda:$index',
    _DeviceKind.mps => 'mps',
    _DeviceKind.xpu => 'xpu:$index',
    _DeviceKind.hip => 'hip:$index',
  };

  /// Whether this value identifies the host CPU.
  bool get isCpu => _kind == _DeviceKind.cpu;

  /// Whether this value identifies an NVIDIA CUDA device.
  bool get isCuda => _kind == _DeviceKind.cuda;

  /// Whether this value identifies the macOS MPS device.
  bool get isMps => _kind == _DeviceKind.mps;

  /// Whether this value identifies an Intel XPU device.
  bool get isXpu => _kind == _DeviceKind.xpu;

  /// Whether this value identifies an AMD HIP/ROCm device.
  bool get isHip => _kind == _DeviceKind.hip;

  @override
  bool operator ==(Object other) =>
      other is Device && other._kind == _kind && other.index == index;

  @override
  int get hashCode => Object.hash(_kind, index);

  @override
  String toString() => switch (_kind) {
    _DeviceKind.cpu => 'Device.cpu',
    _DeviceKind.cuda => 'Device.cuda($index)',
    _DeviceKind.mps => 'Device.mps',
    _DeviceKind.xpu => 'Device.xpu($index)',
    _DeviceKind.hip => 'Device.hip($index)',
  };
}

enum _DeviceKind { cpu, cuda, mps, xpu, hip }
