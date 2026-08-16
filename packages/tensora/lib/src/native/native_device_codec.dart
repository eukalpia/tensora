import '../device/device.dart';

/// Converts a public Tensora [Device] into the stable C ABI device code.
int nativeDeviceCode(Device device) => nativeDeviceCodeForName(device.name);

/// Converts a stable device name into the corresponding C ABI device code.
///
/// Kept separate from [nativeDeviceCode] so malformed/future names can be
/// validated deterministically without constructing an impossible Device.
int nativeDeviceCodeForName(String name) {
  final separator = name.indexOf(':');
  final kind = separator < 0 ? name : name.substring(0, separator);
  return switch (kind) {
    'cpu' => 1,
    'cuda' => 2,
    'mps' => 3,
    'xpu' => 4,
    'hip' => 5,
    _ => throw UnsupportedError('Unknown Tensora device name "$name".'),
  };
}
