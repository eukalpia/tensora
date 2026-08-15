import 'dart:collection';

import 'package:tensora/tensora.dart' as core;

/// Immutable native-backed model state snapshot.
final class StateDict {
  /// @nodoc
  StateDict.fromOwned(Map<String, core.Tensor> entries)
    : _entries = Map<String, core.Tensor>.unmodifiable(entries);

  final Map<String, core.Tensor> _entries;
  bool _disposed = false;

  int get length {
    _ensureLive();
    return _entries.length;
  }

  bool get isEmpty => length == 0;

  Iterable<String> get keys {
    _ensureLive();
    return _entries.keys;
  }

  UnmodifiableMapView<String, core.Tensor> get entries {
    _ensureLive();
    return UnmodifiableMapView<String, core.Tensor>(_entries);
  }

  core.Tensor? operator [](String key) {
    _ensureLive();
    return _entries[key];
  }

  bool get isDisposed => _disposed;

  void dispose() {
    if (_disposed) return;
    for (final tensor in _entries.values) {
      tensor.dispose();
    }
    _disposed = true;
  }

  void _ensureLive() {
    if (_disposed) {
      throw StateError('StateDict has already been disposed.');
    }
  }
}

/// Structured result returned by state restoration.
final class StateLoadResult {
  StateLoadResult({
    Iterable<String> missingKeys = const <String>[],
    Iterable<String> unexpectedKeys = const <String>[],
  }) : missingKeys = List<String>.unmodifiable(missingKeys),
       unexpectedKeys = List<String>.unmodifiable(unexpectedKeys);

  final List<String> missingKeys;
  final List<String> unexpectedKeys;

  bool get isSuccess => missingKeys.isEmpty && unexpectedKeys.isEmpty;
}
