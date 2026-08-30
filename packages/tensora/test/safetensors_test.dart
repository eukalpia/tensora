import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:tensora/tensora.dart';
import 'package:test/test.dart';

/// Encodes the eight-byte little-endian header length prefix.
List<int> lengthPrefix(int value) {
  final prefix = ByteData(8)..setUint64(0, value, Endian.little);
  return prefix.buffer.asUint8List();
}

/// Concatenates the three physical sections of a safetensors container.
///
/// Malformed fixtures are assembled from these raw parts so a test can lie
/// about the header length independently of the header it actually stores.
Uint8List assemble(
  List<int> prefix,
  List<int> header, [
  List<int> data = const <int>[],
]) => Uint8List.fromList(<int>[...prefix, ...header, ...data]);

/// Assembles a container whose declared header length matches [header].
Uint8List container(String header, [List<int> data = const <int>[]]) {
  final bytes = utf8.encode(header);
  return assemble(lengthPrefix(bytes.length), bytes, data);
}

/// Encodes [values] as densely packed little-endian float32.
List<int> f32(List<double> values) {
  final data = ByteData(values.length * 4);
  for (var index = 0; index < values.length; index++) {
    data.setFloat32(index * 4, values[index], Endian.little);
  }
  return data.buffer.asUint8List();
}

Matcher rejects(String messageFragment) => throwsA(
  isA<MalformedArtifactException>()
      .having((error) => error.operation, 'operation', 'safetensors.open')
      .having((error) => error.message, 'message', contains(messageFragment)),
);

void main() {
  late Directory temporaryDirectory;
  var fixtureIndex = 0;

  setUpAll(() {
    temporaryDirectory = Directory.systemTemp.createTempSync(
      'tensora_safetensors_',
    );
  });

  tearDownAll(() {
    // On Windows this also proves no fixture is still held open: a directory
    // containing an open file cannot be removed.
    temporaryDirectory.deleteSync(recursive: true);
  });

  String writeFixture(List<int> bytes) {
    final path =
        '${temporaryDirectory.path}${Platform.pathSeparator}'
        'fixture-${fixtureIndex++}.safetensors';
    File(path).writeAsBytesSync(bytes);
    return path;
  }

  SafetensorsFile openFixture(List<int> bytes) {
    final file = SafetensorsFile.open(writeFixture(bytes));
    addTearDown(file.dispose);
    return file;
  }

  group('dtype tags', () {
    test('every published tag maps to a stable width and Tensora dtype', () {
      final expected = <SafetensorsDType, (String, int, DType?)>{
        SafetensorsDType.boolean: ('BOOL', 1, DType.boolean),
        SafetensorsDType.uint8: ('U8', 1, DType.uint8),
        SafetensorsDType.int8: ('I8', 1, DType.int8),
        SafetensorsDType.float8E5M2: ('F8_E5M2', 1, null),
        SafetensorsDType.float8E4M3: ('F8_E4M3', 1, null),
        SafetensorsDType.int16: ('I16', 2, DType.int16),
        SafetensorsDType.uint16: ('U16', 2, null),
        SafetensorsDType.float16: ('F16', 2, DType.float16),
        SafetensorsDType.bfloat16: ('BF16', 2, DType.bfloat16),
        SafetensorsDType.int32: ('I32', 4, DType.int32),
        SafetensorsDType.uint32: ('U32', 4, null),
        SafetensorsDType.float32: ('F32', 4, DType.float32),
        SafetensorsDType.int64: ('I64', 8, DType.int64),
        SafetensorsDType.uint64: ('U64', 8, null),
        SafetensorsDType.float64: ('F64', 8, DType.float64),
      };

      expect(SafetensorsDType.values, expected.keys.toList());
      for (final entry in expected.entries) {
        final dtype = entry.key;
        expect(dtype.headerName, entry.value.$1);
        expect(dtype.byteWidth, entry.value.$2);
        expect(dtype.tensoraDType, entry.value.$3);
        expect(SafetensorsDType.fromHeaderName(dtype.headerName), dtype);
        expect(dtype.toString(), 'SafetensorsDType.${dtype.name}');
      }
      expect(SafetensorsDType.fromHeaderName('F32X'), isNull);
    });

    test('only F32 is materializable', () {
      expect(
        SafetensorsDType.values.where((dtype) => dtype.isMaterializable),
        <SafetensorsDType>[SafetensorsDType.float32],
      );
    });
  });

  group('header inspection', () {
    test('a single-tensor file round trips through a Tensor', () {
      final file = openFixture(
        container(
          '{"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}',
          f32(<double>[1, 2, 3, 4]),
        ),
      );

      expect(file.names, <String>['weight']);
      expect(file.metadata, isEmpty);
      expect(file.isDisposed, isFalse);
      expect(file.toString(), contains('tensors: 1'));

      final entry = file.tensors['weight']!;
      expect(entry.name, 'weight');
      expect(entry.dtype, SafetensorsDType.float32);
      expect(entry.shape, Shape(<int>[2, 2]));
      expect(entry.numel, 4);
      expect(entry.byteOffset, 0);
      expect(entry.byteLength, 16);
      expect(entry.isMaterializable, isTrue);
      expect(entry.toString(), contains('bytes: 16@0'));

      final tensor = file.readTensor('weight');
      addTearDown(tensor.dispose);
      expect(tensor.shape, Shape(<int>[2, 2]));
      expect(tensor.dtype, DType.float32);
      expect(tensor.device, Device.cpu);
      expect(tensor.toList(), <double>[1, 2, 3, 4]);
    });

    test('non-square tensors keep row-major order at their own offsets', () {
      final file = openFixture(
        container(
          '{"first":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},'
          '"second":{"dtype":"F32","shape":[3,1],"data_offsets":[24,36]}}',
          <int>[
            ...f32(<double>[1, 2, 3, 4, 5, 6]),
            ...f32(<double>[7, 8, 9]),
          ],
        ),
      );

      expect(file.names, <String>['first', 'second']);

      // Reading the later tensor first proves the reader seeks rather than
      // streaming the buffer in declaration order.
      final second = file.readTensor('second');
      final first = file.readTensor('first');
      addTearDown(first.dispose);
      addTearDown(second.dispose);

      expect(first.shape, Shape(<int>[2, 3]));
      expect(first.toList(), <double>[1, 2, 3, 4, 5, 6]);
      expect(second.shape, Shape(<int>[3, 1]));
      expect(second.toList(), <double>[7, 8, 9]);
    });

    test('rank-zero tensors materialize as scalars', () {
      final file = openFixture(
        container(
          '{"scale":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}',
          f32(<double>[0.5]),
        ),
      );

      final tensor = file.readTensor('scale');
      addTearDown(tensor.dispose);
      expect(tensor.shape.rank, 0);
      expect(tensor.numel, 1);
      expect(tensor.toList(), <double>[0.5]);
    });

    test('negative, fractional and extreme float32 values survive', () {
      final values = <double>[-1.5, 0, 3.4028234663852886e38, -0.0009765625];
      final file = openFixture(
        container(
          '{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}',
          f32(values),
        ),
      );

      final tensor = file.readTensor('w');
      addTearDown(tensor.dispose);
      expect(tensor.toList(), values);
    });

    test('__metadata__ is exposed but never treated as a tensor', () {
      final file = openFixture(
        container(
          '{"__metadata__":{"format":"pt","author":"tensora"},'
          '"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}',
          f32(<double>[42]),
        ),
      );

      expect(file.metadata, <String, String>{
        'format': 'pt',
        'author': 'tensora',
      });
      expect(file.names, <String>['w']);
      expect(file.tensors.containsKey('__metadata__'), isFalse);

      final tensor = file.readTensor('w');
      addTearDown(tensor.dispose);
      expect(tensor.toList(), <double>[42]);
    });

    test('a header carrying only __metadata__ describes no tensors', () {
      final file = openFixture(container('{"__metadata__":{"format":"pt"}}'));

      expect(file.tensors, isEmpty);
      expect(file.metadata, <String, String>{'format': 'pt'});
    });

    test('the returned header maps are unmodifiable snapshots', () {
      final file = openFixture(
        container(
          '{"__metadata__":{"format":"pt"},'
          '"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}',
          f32(<double>[1]),
        ),
      );

      expect(() => file.tensors.remove('w'), throwsUnsupportedError);
      expect(() => file.metadata.clear(), throwsUnsupportedError);
    });

    test('unsupported dtypes are described but refuse to materialize', () {
      final file = openFixture(
        container(
          '{"supported":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
          '"half":{"dtype":"BF16","shape":[2,2],"data_offsets":[4,12]},'
          '"tokens":{"dtype":"I64","shape":[2],"data_offsets":[12,28]},'
          '"exotic":{"dtype":"U16","shape":[1],"data_offsets":[28,30]}}',
          List<int>.filled(30, 0),
        ),
      );

      expect(file.names, <String>['supported', 'half', 'tokens', 'exotic']);
      expect(file.tensors['half']!.dtype, SafetensorsDType.bfloat16);
      expect(file.tensors['half']!.shape, Shape(<int>[2, 2]));
      expect(file.tensors['tokens']!.dtype, SafetensorsDType.int64);
      expect(file.tensors['exotic']!.dtype, SafetensorsDType.uint16);
      expect(file.tensors['exotic']!.dtype.tensoraDType, isNull);
      expect(file.tensors['half']!.isMaterializable, isFalse);

      for (final name in <String>['half', 'tokens', 'exotic']) {
        expect(
          () => file.readTensor(name),
          throwsA(
            isA<UnsupportedOperationException>()
                .having(
                  (error) => error.operation,
                  'operation',
                  'safetensors.readTensor',
                )
                .having((error) => error.message, 'message', contains(name)),
          ),
          reason: name,
        );
      }
    });

    test('unknown tensor names are rejected', () {
      final file = openFixture(
        container(
          '{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}',
          f32(<double>[1]),
        ),
      );

      expect(
        () => file.readTensor('missing'),
        throwsA(
          isA<InvalidArgumentException>().having(
            (error) => error.operation,
            'operation',
            'safetensors.readTensor',
          ),
        ),
      );
    });
  });

  group('rejection of hostile headers', () {
    test('a file too small for the length prefix', () {
      expect(
        () => SafetensorsFile.open(writeFixture(<int>[0, 0, 0, 0, 0, 0, 0])),
        rejects('too few for the'),
      );
    });

    test('a header length above the documented ceiling', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            assemble(
              lengthPrefix(SafetensorsFile.maxHeaderBytes + 1),
              utf8.encode('{}'),
            ),
          ),
        ),
        rejects('unusable length'),
      );
    });

    test('a header length whose uint64 exceeds the signed range', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(assemble(List<int>.filled(8, 0xff), utf8.encode('{}'))),
        ),
        rejects('unusable length'),
      );
    });

    test('a zero header length', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(assemble(lengthPrefix(0), utf8.encode('{}'))),
        ),
        rejects('Header length is zero'),
      );
    });

    test('a header length past the end of the file', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(assemble(lengthPrefix(1000), utf8.encode('{}'))),
        ),
        rejects('but only 2 bytes follow'),
      );
    });

    test('a header that is not valid UTF-8', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            assemble(lengthPrefix(4), <int>[0xff, 0xfe, 0xfd, 0xfc]),
          ),
        ),
        rejects('not valid UTF-8'),
      );
    });

    test('a header that is not valid JSON', () {
      expect(
        () => SafetensorsFile.open(writeFixture(container('{"w":'))),
        rejects('not valid JSON'),
      );
    });

    test('a header whose JSON root is not an object', () {
      expect(
        () => SafetensorsFile.open(writeFixture(container('[1,2,3]'))),
        rejects('Header is not a JSON object'),
      );
    });

    test('a tensor description that is not an object', () {
      expect(
        () => SafetensorsFile.open(writeFixture(container('{"w":5}'))),
        rejects('Tensor "w" is not a JSON object'),
      );
    });

    test('a missing or non-string dtype', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(container('{"w":{"shape":[1],"data_offsets":[0,4]}}')),
        ),
        rejects('no string "dtype"'),
      );
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            container('{"w":{"dtype":32,"shape":[1],"data_offsets":[0,4]}}'),
          ),
        ),
        rejects('no string "dtype"'),
      );
    });

    test('an unknown dtype tag', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            container('{"w":{"dtype":"F9","shape":[1],"data_offsets":[0,4]}}'),
          ),
        ),
        rejects('unknown dtype "F9"'),
      );
    });

    test('a missing or non-list shape', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(container('{"w":{"dtype":"F32","data_offsets":[0,4]}}')),
        ),
        rejects('no "shape" list'),
      );
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            container('{"w":{"dtype":"F32","shape":2,"data_offsets":[0,4]}}'),
          ),
        ),
        rejects('no "shape" list'),
      );
    });

    test('a non-integer dimension', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            container(
              '{"w":{"dtype":"F32","shape":[2.5],"data_offsets":[0,4]}}',
            ),
          ),
        ),
        rejects('non-integer dimension 0'),
      );
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            container(
              '{"w":{"dtype":"F32","shape":["2"],"data_offsets":[0,4]}}',
            ),
          ),
        ),
        rejects('non-integer dimension 0'),
      );
    });

    test('a non-positive dimension', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            container(
              '{"w":{"dtype":"F32","shape":[2,0],"data_offsets":[0,0]}}',
            ),
          ),
        ),
        rejects('dimension 1 as 0'),
      );
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            container(
              '{"w":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}',
            ),
          ),
        ),
        rejects('dimension 0 as -1'),
      );
    });

    test('a rank above the supported maximum', () {
      final dimensions = List<String>.filled(Shape.maxRank + 1, '1').join(',');
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            container(
              '{"w":{"dtype":"F32","shape":[$dimensions],'
              '"data_offsets":[0,4]}}',
            ),
          ),
        ),
        rejects('declares rank ${Shape.maxRank + 1}'),
      );
    });

    test('an element count that overflows', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            container(
              '{"w":{"dtype":"F32","shape":[3037000500,3037000500],'
              '"data_offsets":[0,4]}}',
            ),
          ),
        ),
        rejects('element count that overflows'),
      );
    });

    test('a byte size that overflows once scaled by the dtype width', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            container(
              '{"w":{"dtype":"I64","shape":[4611686018427387904],'
              '"data_offsets":[0,0]}}',
            ),
          ),
        ),
        rejects('byte size that overflows'),
      );
    });

    test('malformed data_offsets', () {
      for (final offsets in <String>['4', '[0]', '[0,4,8]', '{}']) {
        expect(
          () => SafetensorsFile.open(
            writeFixture(
              container(
                '{"w":{"dtype":"F32","shape":[1],"data_offsets":$offsets}}',
              ),
            ),
          ),
          rejects('no two-element "data_offsets" list'),
          reason: offsets,
        );
      }
    });

    test('non-integer data_offsets', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            container(
              '{"w":{"dtype":"F32","shape":[1],"data_offsets":["0",4]}}',
            ),
          ),
        ),
        rejects('non-integer start offset'),
      );
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            container(
              '{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4.5]}}',
            ),
          ),
        ),
        rejects('non-integer end offset'),
      );
    });

    test('a negative offset that would reach back into the header', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            container(
              '{"w":{"dtype":"F32","shape":[1],"data_offsets":[-4,0]}}',
              f32(<double>[1]),
            ),
          ),
        ),
        rejects('before the data buffer'),
      );
    });

    test('a reversed offset pair', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            container(
              '{"w":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}}',
              f32(<double>[1, 2, 3]),
            ),
          ),
        ),
        rejects('ends at 4, before its start at 8'),
      );
    });

    test('an offset past the end of the data buffer', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            container(
              '{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,64]}}',
              f32(<double>[1, 2, 3, 4]),
            ),
          ),
        ),
        rejects('past the 16-byte data buffer'),
      );
    });

    test('a byte span that disagrees with shape times dtype width', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            container(
              '{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,8]}}',
              f32(<double>[1, 2]),
            ),
          ),
        ),
        rejects('spans 8 bytes, but [2, 2] elements of F32 occupy 16 bytes'),
      );
    });

    test('a duplicate tensor name', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            container(
              '{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
              '"w":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}',
              f32(<double>[1, 2]),
            ),
          ),
        ),
        rejects('declares "w" twice'),
      );
    });

    test('a duplicate tensor name spelled with a JSON escape', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            container(
              '{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
              r'"\u0077":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}',
              f32(<double>[1, 2]),
            ),
          ),
        ),
        rejects('declares "w" twice'),
      );
    });

    test('a duplicate key nested inside a tensor description', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(
            container(
              '{"w":{"dtype":"F32","shape":[1],"shape":[2],'
              '"data_offsets":[0,8]}}',
              f32(<double>[7, 8]),
            ),
          ),
        ),
        rejects('declares "shape" twice'),
      );
    });

    test('a __metadata__ value that is not an object', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(container('{"__metadata__":"pt"}')),
        ),
        rejects('"__metadata__" is not a JSON object'),
      );
    });

    test('a __metadata__ field that is not a string', () {
      expect(
        () => SafetensorsFile.open(
          writeFixture(container('{"__metadata__":{"format":7}}')),
        ),
        rejects('entry "format" is not a string'),
      );
    });
  });

  group('lifecycle', () {
    test('reads and disposal return liveTensorCount to its baseline', () {
      final baseline = TensoraRuntime.liveTensorCount;
      final file = SafetensorsFile.open(
        writeFixture(
          container(
            '{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},'
            '"b":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}',
            <int>[
              ...f32(<double>[1, 2]),
              ...f32(<double>[3, 4]),
            ],
          ),
        ),
      );

      final a = file.readTensor('a');
      final b = file.readTensor('b', device: Device.cpu);
      expect(TensoraRuntime.liveTensorCount, baseline + 2);

      a.dispose();
      b.dispose();
      file.dispose();

      expect(file.isDisposed, isTrue);
      expect(file.toString(), contains('disposed'));
      expect(TensoraRuntime.liveTensorCount, baseline);
    });

    test('disposal is idempotent and later reads are refused', () {
      final file = SafetensorsFile.open(
        writeFixture(
          container(
            '{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}',
            f32(<double>[1]),
          ),
        ),
      );

      file.dispose();
      file.dispose();

      expect(
        () => file.readTensor('w'),
        throwsA(
          isA<InvalidArgumentException>().having(
            (error) => error.operation,
            'operation',
            'safetensors.readTensor',
          ),
        ),
      );
    });

    test('a rejected open leaks no tensor handles', () {
      final baseline = TensoraRuntime.liveTensorCount;
      expect(
        () => SafetensorsFile.open(writeFixture(container('{"w":5}'))),
        rejects('not a JSON object'),
      );
      expect(TensoraRuntime.liveTensorCount, baseline);
    });

    test('a missing path surfaces as a FileSystemException', () {
      expect(
        () => SafetensorsFile.open(
          '${temporaryDirectory.path}${Platform.pathSeparator}absent.st',
        ),
        throwsA(isA<FileSystemException>()),
      );
    });
  });
}
