import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import '../device/device.dart';
import '../dtype/dtype.dart';
import '../errors/tensora_exception.dart';
import '../shape/shape.dart';
import '../tensor/tensor.dart';
import 'finalizer_release.dart';

const int _quoteCodeUnit = 0x22;
const int _backslashCodeUnit = 0x5c;
const int _colonCodeUnit = 0x3a;
const int _openBraceCodeUnit = 0x7b;
const int _closeBraceCodeUnit = 0x7d;

final Finalizer<RandomAccessFile> _safetensorsFinalizer =
    Finalizer<RandomAccessFile>(closeSafetensorsHandleFromFinalizer);

/// A dtype tag that may appear in a safetensors header.
///
/// The full published tag set is modelled, not only the tags Tensora can
/// materialize. A checkpoint that mixes supported and unsupported weights must
/// still be fully describable, and every tag's width is needed to check a
/// declared byte span against a declared shape.
enum SafetensorsDType {
  /// One byte per boolean element.
  boolean('BOOL', byteWidth: 1, tensoraDType: DType.boolean),

  /// Unsigned eight-bit integer.
  uint8('U8', byteWidth: 1, tensoraDType: DType.uint8),

  /// Signed eight-bit integer.
  int8('I8', byteWidth: 1, tensoraDType: DType.int8),

  /// Eight-bit float with a five-bit exponent. Tensora has no equivalent dtype.
  float8E5M2('F8_E5M2', byteWidth: 1),

  /// Eight-bit float with a four-bit exponent. Tensora has no equivalent dtype.
  float8E4M3('F8_E4M3', byteWidth: 1),

  /// Signed sixteen-bit integer.
  int16('I16', byteWidth: 2, tensoraDType: DType.int16),

  /// Unsigned sixteen-bit integer. Tensora has no equivalent dtype.
  uint16('U16', byteWidth: 2),

  /// IEEE-754 half-precision floating point.
  float16('F16', byteWidth: 2, tensoraDType: DType.float16),

  /// Brain floating point with an eight-bit exponent.
  bfloat16('BF16', byteWidth: 2, tensoraDType: DType.bfloat16),

  /// Signed thirty-two-bit integer.
  int32('I32', byteWidth: 4, tensoraDType: DType.int32),

  /// Unsigned thirty-two-bit integer. Tensora has no equivalent dtype.
  uint32('U32', byteWidth: 4),

  /// IEEE-754 single-precision floating point.
  float32('F32', byteWidth: 4, tensoraDType: DType.float32),

  /// Signed sixty-four-bit integer.
  int64('I64', byteWidth: 8, tensoraDType: DType.int64),

  /// Unsigned sixty-four-bit integer. Tensora has no equivalent dtype.
  uint64('U64', byteWidth: 8),

  /// IEEE-754 double-precision floating point.
  float64('F64', byteWidth: 8, tensoraDType: DType.float64);

  const SafetensorsDType(
    this.headerName, {
    required this.byteWidth,
    this.tensoraDType,
  });

  /// Tag exactly as it is spelled in a header's `dtype` field.
  final String headerName;

  /// Storage width of one element in bytes.
  final int byteWidth;

  /// Equivalent Tensora dtype, or `null` when Tensora models no such dtype.
  final DType? tensoraDType;

  /// Whether [SafetensorsFile.readTensor] can decode this tag today.
  ///
  /// Deliberately narrower than [tensoraDType]: this tracks what the reader's
  /// decoder actually implements, not whichever dtypes native tensor storage
  /// grows to support. Widening it without widening the decoder would silently
  /// reinterpret bytes.
  bool get isMaterializable => this == float32;

  /// Returns the tag named [headerName], or `null` when the tag is unknown.
  static SafetensorsDType? fromHeaderName(String headerName) {
    for (final dtype in values) {
      if (dtype.headerName == headerName) return dtype;
    }
    return null;
  }

  @override
  String toString() => 'SafetensorsDType.$name';
}

/// One tensor as described by a safetensors header.
///
/// An entry carries no tensor data. It exists so a caller can inspect what a
/// checkpoint contains — including weights Tensora cannot yet materialize —
/// before deciding which byte ranges are worth reading.
final class SafetensorsEntry {
  const SafetensorsEntry._({
    required this.name,
    required this.dtype,
    required this.shape,
    required this.byteOffset,
    required this.byteLength,
  });

  /// Tensor name as spelled in the header.
  final String name;

  /// Storage dtype declared by the header.
  final SafetensorsDType dtype;

  /// Row-major dimensions declared by the header.
  final Shape shape;

  /// Start of this tensor's bytes, relative to the start of the data buffer.
  final int byteOffset;

  /// Length of this tensor's bytes, validated against [shape] and [dtype].
  final int byteLength;

  /// Number of tensor elements.
  int get numel => shape.numel;

  /// Whether [SafetensorsFile.readTensor] can materialize this entry.
  bool get isMaterializable => dtype.isMaterializable;

  @override
  String toString() =>
      'SafetensorsEntry(name: $name, dtype: ${dtype.headerName}, '
      'shape: $shape, bytes: $byteLength@$byteOffset)';
}

/// A safetensors weight file opened for lazy, on-demand tensor reads.
///
/// Opening parses and fully validates the header; tensor bytes are read only
/// when [readTensor] is called. Checkpoints regularly exceed available memory,
/// so discovering what a file contains must not require materializing it.
///
/// A safetensors file is untrusted input — it arrives from a model hub, a user
/// download, or a bundle. Every structural defect is rejected as a
/// [MalformedArtifactException] at open time, before any header-derived byte
/// range is used, so no header can steer a read outside the data buffer and a
/// caller can guard a whole load with one catch. Only offsets that survived
/// validation are ever passed to the file handle.
///
/// The file handle is released by [dispose]; using the file afterwards throws
/// [InvalidArgumentException].
@pragma('vm:isolate-unsendable')
final class SafetensorsFile {
  SafetensorsFile._(
    this._handle, {
    required this.path,
    required this.tensors,
    required this.metadata,
    required int dataOffset,
  }) : _dataOffset = dataOffset {
    _safetensorsFinalizer.attach(this, _handle, detach: this);
  }

  /// Opens the safetensors file at [path] and validates its header.
  ///
  /// Throws [FileSystemException] when the path cannot be opened, and
  /// [MalformedArtifactException] when the file is not a structurally valid
  /// safetensors container. The file handle is closed before either escapes.
  factory SafetensorsFile.open(String path) {
    final handle = File(path).openSync();
    try {
      final fileLength = handle.lengthSync();
      final header = _readHeader(handle, fileLength);
      final dataOffset = _lengthPrefixBytes + header.byteLength;
      final parsed = _parseHeader(header.text, fileLength - dataOffset);
      return SafetensorsFile._(
        handle,
        path: path,
        tensors: Map<String, SafetensorsEntry>.unmodifiable(parsed.tensors),
        metadata: Map<String, String>.unmodifiable(parsed.metadata),
        dataOffset: dataOffset,
      );
    } catch (_) {
      handle.closeSync();
      rethrow;
    }
  }

  /// Largest header this reader will accept.
  ///
  /// The declared header length is attacker-controlled and is consumed before
  /// anything else in the file has been inspected, so it is bounded first. The
  /// file-length check alone is not a bound: a large file may still declare
  /// that all of it is JSON. A header describes tensors rather than tensor
  /// data, so 100 MB — the ceiling used by the reference implementation — is
  /// far past any legitimate checkpoint.
  static const int maxHeaderBytes = 100000000;

  static const int _lengthPrefixBytes = 8;
  static const String _metadataKey = '__metadata__';

  final RandomAccessFile _handle;
  final int _dataOffset;
  bool _disposed = false;

  /// Path this file was opened from.
  final String path;

  /// Header entries in file order, keyed by tensor name.
  final Map<String, SafetensorsEntry> tensors;

  /// Free-form `__metadata__` string map, empty when the header omits it.
  final Map<String, String> metadata;

  /// Tensor names in header order.
  Iterable<String> get names => tensors.keys;

  /// Whether deterministic file-handle release has completed.
  bool get isDisposed => _disposed;

  /// Reads the tensor named [name] into Tensora-owned storage on [device].
  ///
  /// Only `F32` is decoded. Any other dtype throws
  /// [UnsupportedOperationException] rather than reinterpreting its bytes,
  /// because a silently mistyped weight tensor produces a model that runs and
  /// is wrong. Unknown names throw [InvalidArgumentException].
  Tensor readTensor(String name, {Device device = Device.cpu}) {
    _ensureOpen('readTensor');
    final entry = tensors[name];
    if (entry == null) {
      throw InvalidArgumentException(
        'Safetensors file does not contain a tensor named "$name".',
        operation: 'safetensors.readTensor',
      );
    }
    if (!entry.dtype.isMaterializable) {
      throw UnsupportedOperationException(
        'Tensor "$name" is stored as ${entry.dtype.headerName}. Tensora can '
        'currently materialize only F32 safetensors tensors.',
        operation: 'safetensors.readTensor',
      );
    }

    _handle.setPositionSync(_dataOffset + entry.byteOffset);
    final bytes = _handle.readSync(entry.byteLength);
    if (bytes.length != entry.byteLength) {
      throw MalformedArtifactException(
        'Reading tensor "$name" needed ${entry.byteLength} bytes but the file '
        'supplied ${bytes.length}; it was truncated after it was opened.',
        operation: 'safetensors.readTensor',
      );
    }

    // safetensors data is little-endian whatever host wrote it, so each element
    // is decoded explicitly instead of viewing the buffer in host order.
    final view = ByteData.sublistView(bytes);
    final values = Float32List(entry.numel);
    for (var index = 0; index < values.length; index++) {
      values[index] = view.getFloat32(
        index * SafetensorsDType.float32.byteWidth,
        Endian.little,
      );
    }
    return Tensor.fromList(values, shape: entry.shape, device: device);
  }

  /// Deterministically closes the underlying file handle.
  ///
  /// The attached finalizer is only a safety net. Waiting for it would hold the
  /// checkpoint open for an unbounded time, which on Windows blocks another
  /// process from replacing the file.
  void dispose() {
    if (_disposed) return;
    _handle.closeSync();
    _safetensorsFinalizer.detach(this);
    _disposed = true;
  }

  void _ensureOpen(String operation) {
    if (_disposed) {
      throw InvalidArgumentException(
        'Safetensors file has already been disposed.',
        operation: 'safetensors.$operation',
      );
    }
  }

  static ({int byteLength, String text}) _readHeader(
    RandomAccessFile handle,
    int fileLength,
  ) {
    if (fileLength < _lengthPrefixBytes) {
      _reject(
        'File holds $fileLength bytes, too few for the '
        '$_lengthPrefixBytes-byte header length prefix.',
      );
    }

    final prefix = handle.readSync(_lengthPrefixBytes);
    if (prefix.length != _lengthPrefixBytes) {
      _reject('File ended inside the header length prefix.');
    }
    final headerLength = ByteData.sublistView(
      prefix,
    ).getUint64(0, Endian.little);

    // A uint64 above 2^63 arrives as a negative Dart int; both that and any
    // merely enormous value are refused before a single byte is allocated.
    if (headerLength < 0 || headerLength > maxHeaderBytes) {
      _reject(
        'Header declares an unusable length; the supported maximum is '
        '$maxHeaderBytes bytes.',
      );
    }
    if (headerLength == 0) {
      _reject('Header length is zero, so the file carries no header.');
    }
    if (headerLength > fileLength - _lengthPrefixBytes) {
      _reject(
        'Header declares $headerLength bytes but only '
        '${fileLength - _lengthPrefixBytes} bytes follow the length prefix.',
      );
    }

    final headerBytes = handle.readSync(headerLength);
    if (headerBytes.length != headerLength) {
      _reject(
        'Header declares $headerLength bytes but the file supplied '
        '${headerBytes.length}.',
      );
    }

    final String text;
    try {
      text = utf8.decode(headerBytes);
    } on FormatException catch (error) {
      _reject('Header is not valid UTF-8: ${error.message}');
    }
    return (byteLength: headerLength, text: text);
  }

  static ({Map<String, SafetensorsEntry> tensors, Map<String, String> metadata})
  _parseHeader(String text, int dataLength) {
    final Object? decoded;
    try {
      decoded = json.decode(text) as Object?;
    } on FormatException catch (error) {
      _reject('Header is not valid JSON: ${error.message}');
    }
    final root = _requireObject(decoded, 'Header');

    final duplicate = _duplicateHeaderKey(text);
    if (duplicate != null) {
      _reject(
        'Header declares "$duplicate" twice inside one JSON object, so the '
        'entry it names is ambiguous.',
      );
    }

    final metadata = <String, String>{};
    final tensors = <String, SafetensorsEntry>{};
    for (final entry in root.entries) {
      final name = entry.key;
      if (name == _metadataKey) {
        final rawMetadata = _requireObject(entry.value, '"$_metadataKey"');
        for (final field in rawMetadata.entries) {
          final value = field.value;
          if (value is! String) {
            _reject(
              '"$_metadataKey" entry "${field.key}" is not a string; '
              '$_metadataKey maps names to strings only.',
            );
          }
          metadata[field.key] = value;
        }
        continue;
      }
      tensors[name] = _parseEntry(name, entry.value, dataLength);
    }
    return (tensors: tensors, metadata: metadata);
  }

  static SafetensorsEntry _parseEntry(
    String name,
    Object? value,
    int dataLength,
  ) {
    final description = _requireObject(value, 'Tensor "$name"');

    final rawDType = description['dtype'];
    if (rawDType is! String) {
      _reject('Tensor "$name" declares no string "dtype".');
    }
    final dtype = SafetensorsDType.fromHeaderName(rawDType);
    if (dtype == null) {
      _reject('Tensor "$name" declares unknown dtype "$rawDType".');
    }

    final rawShape = description['shape'];
    if (rawShape is! List<Object?>) {
      _reject('Tensor "$name" declares no "shape" list.');
    }
    if (rawShape.length > Shape.maxRank) {
      _reject(
        'Tensor "$name" declares rank ${rawShape.length}, above the supported '
        'maximum of ${Shape.maxRank}.',
      );
    }
    final dimensions = <int>[];
    var numel = 1;
    for (var index = 0; index < rawShape.length; index++) {
      final dimension = _requireInt(
        rawShape[index],
        'dimension $index',
        name: name,
      );
      if (dimension <= 0) {
        _reject(
          'Tensor "$name" declares dimension $index as $dimension; '
          'dimensions must be positive.',
        );
      }
      if (numel > Shape.maxNumel ~/ dimension) {
        _reject(
          'Tensor "$name" declares an element count that overflows the '
          'supported int64 range.',
        );
      }
      numel *= dimension;
      dimensions.add(dimension);
    }

    final rawOffsets = description['data_offsets'];
    if (rawOffsets is! List<Object?> || rawOffsets.length != 2) {
      _reject('Tensor "$name" declares no two-element "data_offsets" list.');
    }
    final begin = _requireInt(rawOffsets[0], 'start offset', name: name);
    final end = _requireInt(rawOffsets[1], 'end offset', name: name);
    if (begin < 0) {
      _reject(
        'Tensor "$name" starts at $begin, before the data buffer; offsets are '
        'relative to the buffer and can never reach into the header.',
      );
    }
    if (end < begin) {
      _reject('Tensor "$name" ends at $end, before its start at $begin.');
    }
    if (end > dataLength) {
      _reject(
        'Tensor "$name" ends at $end, past the $dataLength-byte data buffer.',
      );
    }

    if (numel > Shape.maxNumel ~/ dtype.byteWidth) {
      _reject(
        'Tensor "$name" declares a byte size that overflows the supported '
        'int64 range.',
      );
    }
    final declared = numel * dtype.byteWidth;
    final span = end - begin;
    if (declared != span) {
      _reject(
        'Tensor "$name" spans $span bytes, but $dimensions elements of '
        '${dtype.headerName} occupy $declared bytes.',
      );
    }

    return SafetensorsEntry._(
      name: name,
      dtype: dtype,
      shape: Shape(dimensions),
      byteOffset: begin,
      byteLength: span,
    );
  }

  static Map<String, Object?> _requireObject(Object? value, String subject) {
    if (value is! Map<String, Object?>) {
      _reject('$subject is not a JSON object.');
    }
    return value;
  }

  static int _requireInt(Object? value, String field, {required String name}) {
    if (value is! int) {
      _reject('Tensor "$name" declares a non-integer $field.');
    }
    return value;
  }

  static Never _reject(String message) {
    throw MalformedArtifactException(message, operation: 'safetensors.open');
  }

  @override
  String toString() {
    final state = _disposed ? ', disposed' : '';
    return 'SafetensorsFile(path: $path, tensors: ${tensors.length}$state)';
  }
}

/// Returns the first key [headerJson] declares twice inside one JSON object.
///
/// `json.decode` keeps only the last value for a repeated key, so a header that
/// names one tensor twice decodes into a perfectly valid map with one of the
/// two definitions silently discarded — and the discarded one is the writer's
/// choice, not the reader's. Recovering that requires re-scanning the text the
/// decoder already accepted; because it did accept it, the scan can assume
/// balanced braces and terminated strings.
///
/// Only object braces open a scope: in valid JSON a `"key":` pair can never
/// appear inside an array, so the innermost open object is always the owner of
/// a key that is found.
String? _duplicateHeaderKey(String headerJson) {
  final scopes = <Set<String>>[];
  var index = 0;
  while (index < headerJson.length) {
    final code = headerJson.codeUnitAt(index);
    if (code == _quoteCodeUnit) {
      final end = _endOfJsonString(headerJson, index);
      var next = end;
      while (next < headerJson.length &&
          _isJsonWhitespace(headerJson.codeUnitAt(next))) {
        next++;
      }
      final isKey =
          next < headerJson.length &&
          headerJson.codeUnitAt(next) == _colonCodeUnit;
      if (isKey && scopes.isNotEmpty) {
        // Decoding the raw token collapses escapes, so a name spelled with a
        // `\u` escape is recognized as the same name as its plain spelling.
        final key = json.decode(headerJson.substring(index, end)) as Object?;
        if (key is String && !scopes.last.add(key)) return key;
      }
      index = end;
      continue;
    }
    if (code == _openBraceCodeUnit) {
      scopes.add(<String>{});
    } else if (code == _closeBraceCodeUnit && scopes.isNotEmpty) {
      scopes.removeLast();
    }
    index++;
  }
  return null;
}

int _endOfJsonString(String source, int start) {
  var index = start + 1;
  while (index < source.length) {
    final code = source.codeUnitAt(index);
    if (code == _backslashCodeUnit) {
      index += 2;
      continue;
    }
    if (code == _quoteCodeUnit) return index + 1;
    index++;
  }
  return source.length;
}

bool _isJsonWhitespace(int code) =>
    code == 0x20 || code == 0x09 || code == 0x0a || code == 0x0d;
