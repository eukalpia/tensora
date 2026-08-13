import 'package:flutter/foundation.dart';

typedef ValueDisposer<T extends Object> = void Function(T value);

final class TensorController<T extends Object> extends ChangeNotifier {
  TensorController({T? value, required ValueDisposer<T> disposeValue})
    : _value = value,
      _disposeValue = disposeValue;

  T? _value;
  final ValueDisposer<T> _disposeValue;
  bool _disposed = false;

  T? get value => _value;
  bool get isDisposed => _disposed;

  void replace(T? next, {bool disposePrevious = true}) {
    _ensureLive();
    if (identical(next, _value)) return;
    final previous = _value;
    _value = next;
    if (disposePrevious && previous != null) {
      _disposeValue(previous);
    }
    notifyListeners();
  }

  T? take() {
    _ensureLive();
    final current = _value;
    if (current == null) return null;
    _value = null;
    notifyListeners();
    return current;
  }

  @override
  void dispose() {
    if (_disposed) return;
    final current = _value;
    _value = null;
    if (current != null) {
      _disposeValue(current);
    }
    _disposed = true;
    super.dispose();
  }

  void _ensureLive() {
    if (_disposed) {
      throw StateError('TensorController has already been disposed.');
    }
  }
}
